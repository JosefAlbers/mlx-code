# Copyright 2026 J Joe
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import tempfile
import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step

stream_logger = logging.getLogger("stream")
stream_logger.setLevel(logging.DEBUG)
s_handler = logging.FileHandler("mlx_stream.log", mode='w')
s_handler.setFormatter(logging.Formatter("%(message)s"))
s_handler.terminator = ""
stream_logger.addHandler(s_handler)
trace_logger = logging.getLogger("trace")
trace_logger.setLevel(logging.DEBUG)
t_handler = logging.FileHandler("mlx_trace.log", mode='w')
t_handler.setFormatter(logging.Formatter("【%(message)s\n】\n"))
trace_logger.addHandler(t_handler)
gen_lock = threading.Lock()
prompt_cache = {}

def parse_tool(tools, names):
    qwen_tools = []
    for tool in tools:
        if names is not None and tool["name"] not in names:
            continue
        qwen_tool = {
            # "type": "function",
            # "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("input_schema", {
                    "type": "object",
                    "properties": {}
                })
            # }
        }
        # params = qwen_tool["function"]["parameters"]
        params = qwen_tool["parameters"]
        params.pop("$schema", None)
        qwen_tools.append(qwen_tool)
    return qwen_tools

def encode(body, tokenizer, system, names, skips):
    msgs = []
    sys_parts = []
    if isinstance(system, str):
        env = "\n".join(l.strip() for l in next((b["text"] for b in body.get("system", []) if "Primary working directory" in b.get("text", "")), "").splitlines() if "Primary working directory" in l or "Shell:" in l)
        sys_parts.append(system.replace("{env}", env))
    else:
        raw_system = body.get("system")
        if isinstance(raw_system, str) and raw_system.strip():
            sys_parts.append(raw_system.strip())
        elif isinstance(raw_system, list):
            for block in raw_system:
                if block.get("type") != "text":
                    continue
                text = block.get("text", "").strip()
                if re.match(r'^x-anthropic-billing-header:\s?.*;$', text) and '\n' not in text:
                    continue
                if text:
                    sys_parts.append(text)
    if sys_parts:
        msgs.append({"role": "system", "content": "\n\n".join(sys_parts)})
    calls = {}
    def skip(text, show_skipped=False):
        if skips is None:
            return text
        lines = []
        for pattern in skips:
            found = re.findall(pattern, text)
            if found:
                lines.append(f"{pattern}\n" + "\n".join(found))
        if lines and show_skipped:
            trace_logger.debug("\n".join(["S"]+lines))
        for pattern in skips:
            text = re.sub(pattern, "", text)
        return text
    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            content = [dict(type='text', text=content)]
        parts = {}
        for block in content:
            t = block.get("type")
            if t == "text":
                parts['content'] = parts.get('content', '') + skip(block['text'])
            elif t == "thinking":
                parts['reasoning_content'] = block['thinking']
            elif t == "tool_use":
                calls[block["id"]] = block
            elif t == "tool_result":
                tu = calls.get(block["tool_use_id"])
                rc = block.get("content", "")
                if isinstance(rc, list):
                    rc = skip("\n".join(c.get("text", "") for c in rc if c.get("type") == "text"))
                msgs.append({"role": "tool", "name": tu['name'], "content": f"{tu['input']}\n{rc}"})
        if parts:
            msgs.append({"role": role}|parts)
    if not msgs[-1].get('content', '').strip():
        return None
    return tokenizer.apply_chat_template(msgs, tools = parse_tool(body.get("tools", []), names), tokenize=False, add_generation_prompt=True)

def decode(raw_text, tokenizer, parse_think, single_think=False):
    # think_id = tokenizer.convert_tokens_to_ids("<think>")
    def escape(text):
        def repl(match):
            inner = match.group(1)
            inner = inner.replace('<', '‹').replace('>', '›')
            return f'`{inner}`'
        return re.sub(r'`([^\n`]*)`', repl, text)
    raw_text = escape(raw_text)
    raw_text = '<think>' + raw_text if (c := raw_text.find('</think>')) != -1 and ((o := raw_text.find('<think>')) == -1 or c < o) else raw_text
    blocks = []
    if parse_think:
        parts = re.split(r'(<think>.*?</think>)', raw_text, flags=re.DOTALL, maxsplit=1 if single_think else 0)
    else:
        parts = [raw_text]

    for part in parts:
        if not part:
            continue 
        if parse_think and not single_think and part.startswith('<think>') and part.endswith('</think>'):
            thinking_content = part[7:-8].strip()
            if thinking_content:
                blocks.append({"type": "thinking", "thinking": thinking_content})
        else:
            blocks.append({"type": "text", "text": re.sub(r'</?think>', '‹think›', part)}) #: show tool call
            tool_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
            for match in tool_pattern.finditer(part):
                content = match.group(1).strip()
                if not ("<function=" in content and "<parameter=" in content and "</parameter>" in content):
                    continue
                fn_match = re.search(r"<function=([^\s>]+)>", content)
                if fn_match:
                    name = fn_match.group(1)
                    params = re.findall(r"<parameter=([^\s>]+)>\s*(.*?)\s*</parameter>", content, re.DOTALL)
                    args = {k: v.strip() for k, v in params}
                    blocks.append({
                        "type": "tool_use",
                        "id": f"toolu_{uuid.uuid4().hex[:8]}",
                        "name": name,
                        "input": args,
                    })
    stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    return blocks, stop_reason

def blocks_to_sse(blocks: list[dict], msg_id: str, in_tokens: int, out_tokens: int, stop_reason='end_turn') -> bytes:
    def event(name: str, data: dict) -> bytes:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
    out = bytearray()
    out += event("message_start", {"type": "message_start", "message": {
        "id": msg_id, "type": "message", "role": "assistant",
        "model": "local", "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": in_tokens, "output_tokens": 0},
    }})
    for i, block in enumerate(blocks):
        bt = block["type"]
        if bt == "text":
            out += event("content_block_start", {"type": "content_block_start", "index": i,
                "content_block": {"type": "text", "text": ""}})
            out += event("content_block_delta", {"type": "content_block_delta", "index": i,
                "delta": {"type": "text_delta", "text": block["text"]}})
        elif bt == "thinking":
            out += event("content_block_start", {"type": "content_block_start", "index": i,
                "content_block": {"type": "thinking", "thinking": ""}})
            out += event("content_block_delta", {"type": "content_block_delta", "index": i,
                "delta": {"type": "thinking_delta", "thinking": block["thinking"]}})
        elif bt == "tool_use":
            out += event("content_block_start", {"type": "content_block_start", "index": i,
                "content_block": {"type": "tool_use", "id": block["id"],
                    "name": block["name"], "input": {}}})
            out += event("content_block_delta", {"type": "content_block_delta", "index": i,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}})
        out += event("content_block_stop", {"type": "content_block_stop", "index": i})
    out += event("message_delta", {"type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": out_tokens}})
    out += event("message_stop", {"type": "message_stop"})
    return bytes(out)

def dmca(p_str):
    symbols = ["▲", "△", "▶", "▷", "▼", "▽", "◀", "◁", "◆", "◇"]
    def mask_text(text):
        return re.sub(r"\S", lambda _: random.choice(symbols), text)
    pattern1 = r"(<\|im_start\|>system\n)(.*?)(<\|im_end\|>)"
    def mask_system(match):
        return match.group(1) + mask_text(match.group(2)) + match.group(3)
    p_str = re.sub(pattern1, mask_system, p_str, flags=re.DOTALL)
    block_patterns = [
        r"(?m)^<system-reminder>[\s\S]*?^</system-reminder>\s*",
        r"(?m)^\[SUGGESTION MODE[\s\S]*",
    ]
    for pattern in block_patterns:
        p_str = re.sub(pattern, lambda m: mask_text(m.group(0)), p_str)
    return p_str

def generate(model, tokenizer, prompt, hook=None, max_tokens=256, helper_max_tokens=64, **kwargs):
    global prompt_cache
    if prompt is None:
        return '', 0, 0
    if not isinstance(tokenizer, mlx_lm.tokenizer_utils.TokenizerWrapper):
        tokenizer = mlx_lm.tokenizer_utils.TokenizerWrapper(tokenizer)
    detokenizer = tokenizer.detokenizer
    if isinstance(prompt, str):
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        prompt_s = prompt
        prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    else:
        prompt_s = tokenizer.decode(prompt)
    stream_logger.debug(dmca(prompt_s))
    common_len = 0
    if prompt_cache.get('cache', None):
        for p, h in zip(prompt, prompt_cache['hx']):
            if p == h:
                common_len += 1
            else:
                break
        common_len = min(common_len, len(prompt) - 1)
    else:
        prompt_cache['hx'] = []
        prompt_cache['cache'] = mlx_lm.models.cache.make_prompt_cache(model)
    hx_len = len(prompt_cache['hx']) 
    trim_len = hx_len - common_len
    mlx_lm.models.cache.trim_prompt_cache(prompt_cache['cache'], trim_len)
    token_gen = generate_step(
        mx.array(prompt[common_len:]),
        model,
        prompt_cache=prompt_cache['cache'],
        max_tokens=max_tokens,
        **kwargs,
    )
    text = ""
    tic_non = time.perf_counter()
    gens = []
    for token, _ in token_gen:
        gens.append(token)
        if token in tokenizer.eos_token_ids:
            break
        detokenizer.add_token(token)
        seg = detokenizer.last_segment
        stream_logger.debug(seg)
        text += seg
        if len(gens) == 1:
            tic_inp = time.perf_counter()
            if prompt_cache.get('file_name'): 
                _fn = prompt_cache.pop('file_name')
                mlx_lm.models.cache.save_prompt_cache(_fn, prompt_cache['cache'], metadata=dict(model_name=prompt_cache['model_name'], hx=json.dumps(prompt+gens)))
        if len(gens) >= max_tokens:
            break
    tic_out = time.perf_counter()
    detokenizer.finalize()
    text += detokenizer.last_segment
    prompt_cache['hx'] = prompt+gens
    trace_logger.debug(f'G {hx_len} {len(prompt)} {common_len} {trim_len} {len(gens)}\n=== TPS ===\n- Processed {len(prompt)} input tokens in {tic_inp-tic_non:.0f} seconds ({len(prompt)/(tic_inp-tic_non):.0f} tokens per second)\n- Generated {len(gens)} new tokens in {tic_out-tic_inp:.0f} seconds ({len(gens)/(tic_out-tic_inp):.0f} tokens per second)\n\n=== INP ===\n{dmca(prompt_s)}\n=== OUT ===\n{text}')
    return text, len(prompt), len(gens)

def make_handler(model, tokenizer, system, names, skips, parse_think=True):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send_json(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self.send_json(200, {"data": [{"id": "local", "object": "model",
                    "created": int(time.time()), "owned_by": "local"}]})
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == "/v1/messages/count_tokens":
                self.send_json(200, {"input_tokens": 0})
                return
            if path != "/v1/messages":
                self.send_json(404, {"error": f"unknown endpoint {path}"})
                return
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            prompt = encode(body, tokenizer, system, names, skips)
            with gen_lock:
                raw, in_tokens, out_tokens = generate(model, tokenizer, prompt=prompt, max_tokens=body.get("max_tokens", 8192))
            blocks, stop_reason = decode(raw, tokenizer, parse_think=parse_think)
            msg_id = f"msg_{uuid.uuid4().hex}"
            sse = blocks_to_sse(blocks, msg_id, in_tokens, out_tokens, stop_reason)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(sse)))
            self.end_headers()
            try:
                self.wfile.write(sse)
                self.wfile.flush()
            except BrokenPipeError:
                pass
    return Handler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3.5-4B-OptiQ-4bit")
    # parser.add_argument("--model", default="mlx-community/Qwen3.5-2B-OptiQ-4bit")
    # parser.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-MLX-bf16")
    parser.add_argument("--system", type=str, default='')
    # parser.add_argument("--system", type=str, default='# Env\n{env}')
    # parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--cache", type=str, default='cache/cache.safetensors')
    # parser.add_argument("--names", nargs="+", default=[])
    parser.add_argument("--names", nargs="+", default=['Read','Edit','Write','Grep','Glob','Bash','Agent','Skill'])
    # parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--skips", nargs="+", default=[
        r'(?m)^\[SUGGESTION MODE[\s\S]*',
        r'(?m)^<system-reminder>[\s\S]*?^</system-reminder>\s*',
    ])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--home", default=tempfile.mkdtemp())
    parser.add_argument("--work", default=os.getcwd())
    args, claude_args = parser.parse_known_args()
    global prompt_cache
    if os.path.exists(args.cache):
        cache, metadata = mlx_lm.models.cache.load_prompt_cache(args.cache, return_metadata=True)
        mx.eval(cache)
        model_name = metadata.pop("model_name", "")
        tokens_str = metadata.pop("hx", "[]")
        tokens = json.loads(tokens_str)
        prompt_cache = dict(cache=cache, hx=tokens, model_name=model_name)
        if prompt_cache.get('model_name') != args.model:
            prompt_cache = dict(model_name=args.model)
    else:
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
        prompt_cache = dict(model_name=args.model)
    prompt_cache['file_name']=args.cache
    model, tokenizer = mlx_lm.load(args.model)
    server = HTTPServer((args.host, args.port), make_handler(model, tokenizer, args.system, args.names, args.skips))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://{args.host}:{args.port}"
    env["ANTHROPIC_AUTH_TOKEN"] = "local"
    env["ANTHROPIC_MODEL"] = args.model
    env["ANTHROPIC_SMALL_FAST_MODEL"] = args.model
    env["HOME"] = args.home 
    def mirror_workspace(src: str, dst: str):
        for root, dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            os.makedirs(os.path.join(dst, rel), exist_ok=True)
            for f in files:
                os.link(os.path.join(root, f), os.path.join(dst, rel, f))
    workspace = os.path.join(args.home, "workspace")
    mirror_workspace(args.work, workspace)
    sys.exit(subprocess.run(["claude"] + claude_args, env=env, cwd=workspace).returncode)

if __name__ == "__main__":
    main()

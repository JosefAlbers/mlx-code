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
import numpy as np
import hashlib
import contextlib
import functools
import mlx.nn as nn
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

generation_stream = mx.new_stream(mx.default_device())
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
dict_cache = {}

def hash_tokens(tokens):
    arr = np.array(tokens, dtype=np.uint32)
    return hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()

def get_common_len(a, b):
    common_len = 0
    for p, h in zip(a, b):
        if p == h:
            common_len += 1
        else:
            break
    return common_len

@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    if not mx.metal.is_available():
        try:
            yield
        finally:
            pass
    else:
        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
        )
        max_rec_size = mx.device_info()["max_recommended_working_set_size"]
        if model_bytes > 0.9 * max_rec_size:
            model_mb = model_bytes // 2**20
            max_rec_mb = max_rec_size // 2**20
            print(f"{model_mb=} {max_rec_mb=}")
        old_limit = mx.set_wired_limit(max_rec_size)
        try:
            yield
        finally:
            if streams is not None:
                for s in streams:
                    mx.synchronize(s)
            else:
                mx.synchronize()
            mx.set_wired_limit(old_limit)

def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)

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
    trace_logger.debug(body)
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
        return None, -1
    apply_chat_template = lambda x: tokenizer.apply_chat_template(x, tools = parse_tool(body.get("tools", []), names), tokenize=False, add_generation_prompt=True)
    full_s = apply_chat_template(msgs)
    last_user_idx = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=None)
    if last_user_idx is None:
        return None, -1
    p_msgs = msgs[:last_user_idx] + [dict(role='user', content='h' if msgs[last_user_idx]['content'][0] != 'h' else 'i')]
    prfx_s = apply_chat_template(p_msgs)
    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
    full = tokenizer.encode(full_s, add_special_tokens=add_special_tokens)
    prfx = tokenizer.encode(prfx_s, add_special_tokens=add_special_tokens)
    save_at = get_common_len(full, prfx)
    stream_logger.debug(f'{save_at}\n{dmca(tokenizer.decode(full[:save_at]))}\n---\n{dmca(tokenizer.decode(full[save_at:]))}')
    return full, save_at

def decode(raw_text, tokenizer, parse_think, single_think=False):
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
                    "name": block["name"], "input": {}} })
            out += event("content_block_delta", {"type": "content_block_delta", "index": i,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}})
        out += event("content_block_stop", {"type": "content_block_stop", "index": i})
    out += event("message_delta", {"type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": out_tokens}})
    out += event("message_stop", {"type": "message_stop"})
    return bytes(out)

def dmca(p_str):
    if True: #: False for recording
        return p_str
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

def make_handler(model_name, system, names, skips, parse_think=True):
    model, tokenizer = mlx_lm.load(model_name)
    if not isinstance(tokenizer, mlx_lm.tokenizer_utils.TokenizerWrapper):
        tokenizer = mlx_lm.tokenizer_utils.TokenizerWrapper(tokenizer)
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
            prompt, save_at = encode(body, tokenizer, system, names, skips)
            with gen_lock:
                raw, in_tokens, out_tokens = generate(model, tokenizer, prompt=prompt, save_at=save_at, max_tokens=body.get("max_tokens", 8192))
            blocks, stop_reason = decode(raw, tokenizer, parse_think=parse_think)
            msg_id = f"msg_{uuid.uuid4().hex}"
            sse = blocks_to_sse(blocks, msg_id, in_tokens, out_tokens, stop_reason)
            trace_logger.debug(sse)
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

def load_dict_cache(cache_path):
    cache, metadata = mlx_lm.models.cache.load_prompt_cache(cache_path, return_metadata=True)
    mx.eval(cache)
    model_name = metadata.pop("model_name", "")
    tokens_str = metadata.pop("hx", "[]")
    tokens = json.loads(tokens_str)
    global dict_cache
    dict_cache |= dict(cache=cache, hx=tokens, model_name=model_name)

def save_dict_cache(cache_path, metadata, prompt_cache):
    mlx_lm.models.cache.save_prompt_cache(cache_path, prompt_cache, metadata=metadata)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", default=None)
    # parser.add_argument("--harness", default="claude")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-4B-OptiQ-4bit")
    # parser.add_argument("--model", default="mlx-community/Qwen3.5-2B-OptiQ-4bit")
    # parser.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-MLX-bf16")
    # parser.add_argument("--system", type=str, default='')
    # parser.add_argument("--system", type=str, default='# Env\n{env}')
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--cache", type=str, default='cache')
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
    parser.add_argument("--nocc", action="store_true", help="Disable Claude Code subprocess and run server only")
    args, harness_args = parser.parse_known_args()
    Path(args.cache).mkdir(parents=True, exist_ok=True)
    global dict_cache
    dict_cache = dict(model_name=args.model, cache_dir = args.cache)
    server = HTTPServer((args.host, args.port), make_handler(args.model, args.system, args.names, args.skips))
    if args.nocc:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.server_close()
    else:
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
        if args.harness is None:
            from pie import run_repl
            run_repl(base_url=f"http://{args.host}:{args.port}/v1")
        else:
            sys.exit(subprocess.run([args.harness] + harness_args, env=env, cwd=workspace).returncode)

def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    input_embeddings: Optional[mx.array] = None,
    save_at: int = -1,
    save_fn = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    if input_embeddings is not None:
        if not does_model_support_input_embeddings(model):
            raise ValueError("Model does not support input embeddings.")
        elif len(prompt) > 0 and len(prompt) != len(input_embeddings):
            raise ValueError(f"{len(input_embeddings)=} {len(prompt)=}")
    elif len(prompt) == 0:
        raise ValueError("Either input_embeddings or prompt (or both) must be provided.")

    tokens = None

    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )

    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _model_call(input_tokens: mx.array, input_embeddings: Optional[mx.array]):
        if input_embeddings is not None:
            return model(
                input_tokens, cache=prompt_cache, input_embeddings=input_embeddings
            )
        else:
            return model(input_tokens, cache=prompt_cache)

    def _step(input_tokens: mx.array, input_embeddings: Optional[mx.array] = None):
        nonlocal tokens

        with mx.stream(generation_stream):
            logits = _model_call(
                input_tokens=input_tokens[None],
                input_embeddings=(
                    input_embeddings[None] if input_embeddings is not None else None
                ),
            )

            logits = logits[:, -1, :]

            if logits_processors and len(input_tokens) > 0:
                tokens = (
                    mx.concat([tokens, input_tokens])
                    if tokens is not None
                    else input_tokens
                )
                for processor in logits_processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            return sampled, logprobs.squeeze(0)

    with mx.stream(generation_stream):
        total_prompt_tokens = (
            len(input_embeddings) if input_embeddings is not None else len(prompt)
        )
        prompt_processed_tokens = 0
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        while total_prompt_tokens - prompt_processed_tokens > 1:
            remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
            n_to_process = min(prefill_step_size, remaining)
            if prompt_processed_tokens < save_at:
                n_to_process = min(n_to_process, save_at - prompt_processed_tokens)

            _model_call(
                input_tokens=prompt[:n_to_process][None],
                input_embeddings=(
                    input_embeddings[:n_to_process][None]
                    if input_embeddings is not None
                    else None
                ),
            )
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            prompt_processed_tokens += n_to_process
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt = prompt[n_to_process:]
            input_embeddings = (
                input_embeddings[n_to_process:]
                if input_embeddings is not None
                else input_embeddings
            )
            mx.clear_cache()
            if save_fn is not None and prompt_processed_tokens == save_at:
                save_fn(prompt_cache)

        y, logprobs = _step(input_tokens=prompt, input_embeddings=input_embeddings)

    mx.async_eval(y, logprobs)
    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if n == max_tokens:
            break
        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1

def generate(model, tokenizer, prompt, save_at, hook=None, max_tokens=256, helper_max_tokens=64, **kwargs):
    if prompt is None:
        return '', 0, 0
    global dict_cache
    detokenizer = tokenizer.detokenizer
    text = ''
    gens = []
    common_len = 0
    hx_len = None
    trim_len = None
    save_fn = None
    if not dict_cache.get('cache'):
        ckpt_path = Path(dict_cache['cache_dir'])/f'{"".join(c for c in dict_cache["model_name"] if c.isalnum())}_{save_at}_{hash_tokens(prompt[:save_at])}.safetensors'
        if os.path.exists(ckpt_path):
            load_dict_cache(ckpt_path)
        else:
            dict_cache |= dict(cache=mlx_lm.models.cache.make_prompt_cache(model), hx=[])
            save_fn = functools.partial(save_dict_cache, ckpt_path, dict(model_name=dict_cache['model_name'], hx=json.dumps(prompt[:save_at+1])))

    if (hx := dict_cache.get('hx')):
        _hx = hx[:-1]
        common_len = get_common_len(prompt, _hx)
        hx_len = len(_hx)
        trim_len = hx_len - common_len
        if trim_len > 0:
            if all(c.is_trimmable() for c in dict_cache['cache']):
                mlx_lm.models.cache.trim_prompt_cache(dict_cache['cache'], trim_len)
            else:
                ckpt_path = Path(dict_cache['cache_dir'])/f'{"".join(c for c in dict_cache["model_name"] if c.isalnum())}_{save_at}_{hash_tokens(prompt[:save_at])}.safetensors'
                if os.path.exists(ckpt_path):
                    load_dict_cache(ckpt_path)
                    common_len = save_at
                else:
                    dict_cache |= dict(cache=mlx_lm.models.cache.make_prompt_cache(model), hx=[])
                    save_fn = functools.partial(save_dict_cache, ckpt_path, dict(model_name=dict_cache['model_name'], hx=json.dumps(prompt[:save_at+1])))
                    common_len = 0

    if save_at > common_len and not all(c.is_trimmable() for c in dict_cache['cache']):
        ckpt_path = Path(dict_cache['cache_dir'])/f'{"".join(c for c in dict_cache["model_name"] if c.isalnum())}_{save_at}_{hash_tokens(prompt[:save_at])}.safetensors'
        save_fn = functools.partial(save_dict_cache, ckpt_path, dict(model_name=dict_cache['model_name'], hx=json.dumps(prompt[:save_at+1])))
    else:
        save_at = -1

    if common_len==len(prompt):
        _last_gen = dict_cache['hx'][common_len]
        prompt_arr = mx.array([_last_gen])
        gens.append(_last_gen)
        detokenizer.add(_last_gen)
    else:
        prompt_arr = mx.array(prompt[common_len:])

    token_gen = generate_step(
        prompt_arr,
        model,
        prompt_cache=dict_cache['cache'],
        max_tokens=max_tokens,
        save_at=save_at-common_len,
        save_fn=save_fn,
        **kwargs,
    )
    tic_non = time.perf_counter()
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
        if len(gens) >= max_tokens:
            break
    tic_out = time.perf_counter()
    detokenizer.finalize()
    text += detokenizer.last_segment
    dict_cache['hx'] = prompt+gens
    trace_logger.debug(f'G {hx_len} {len(prompt)} {common_len} {trim_len} {len(gens)}\n=== TPS ===\n- Processed {len(prompt)} input tokens in {tic_inp-tic_non:.0f} seconds ({len(prompt)/(tic_inp-tic_non):.0f} tokens per second)\n- Generated {len(gens)} new tokens in {tic_out-tic_inp:.0f} seconds ({len(gens)/(tic_out-tic_inp):.0f} tokens per second)\n\n=== INP ===\n{dmca(tokenizer.decode(prompt))}\n=== OUT ===\n{text}')
    return text, len(prompt), len(gens)

if __name__ == "__main__":
    main()

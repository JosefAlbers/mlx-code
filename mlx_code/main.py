# Copyright 2026 J Joe

# {{{utl

import json, uuid
from dataclasses import dataclass, field, replace
from typing import Any, Literal

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
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer, HTTPServer
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

import json
from datetime import datetime, timezone

_LOG_RECORD_BUILTIN_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "file": record.pathname,
            "function": record.funcName,
            "line": record.lineno,
        }

        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _LOG_RECORD_BUILTIN_KEYS
        }
        if extras:
            log_entry["extra"] = extras

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)

def setup_logger(log_file="app.log.json", console=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

        if console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(JsonFormatter())
            logger.addHandler(stream_handler)

    return logger

logger = setup_logger("log.json")

generation_stream = mx.new_thread_local_stream(mx.default_device())
gen_lock = threading.Lock()
abort_ev = threading.Event()
# }}}utl
# {{{enc
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class Message:
    role: str 
    content: str | None = None
    thinking: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: tuple[str, str] | None = None

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict

def _norm_params(fxn: dict) -> dict:
    params = dict(
        fxn.get("parameters")
        or fxn.get("parametersJsonSchema") 
        or fxn.get("input_schema")
        or {"type": "object", "properties": {}}
    )
    params.pop("$schema", None)
    return params

def _safe_json(x) -> dict:
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x)
    except Exception:
        return {}

def _tool_schema(tool: Tool) -> dict:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
    }

def _copy_msg(m: Message) -> Message:
    return Message(
        role=m.role,
        content=m.content,
        thinking=m.thinking,
        tool_calls=list(m.tool_calls),
        tool_result=m.tool_result,
    )

def _skip(text, skips=None, show_skipped=False):
    if text is None or skips is None:
        return text
    lines = []
    for pattern in skips:
        found = re.findall(pattern, text)
        if found:
            lines.append(f"{pattern}\n" + "\n".join(found))
    if lines and show_skipped:
        logger.debug('\n---\n'.join(lines))
    for pattern in skips:
        text = re.sub(pattern, "", text)
    return text

def normalize(
    tools: list[Tool],
    messages: list[Message],
    strict: bool = False,
) -> tuple[list[Tool], list[Message]]:
    tool_names = {t.name for t in tools}
    known_calls: dict[str, ToolCall] = {}
    seen_ids: set[str] = set()
    out: list[Message] = []
    orphans: list[Message] = []

    def fresh_id(base: str | None) -> str:
        if base and base not in seen_ids:
            seen_ids.add(base)
            return base
        while True:
            nid = str(uuid.uuid4())[:8]
            if nid not in seen_ids:
                seen_ids.add(nid)
                return nid

    def append_msg(msg: Message) -> None:
        if (
            out
            and not msg.tool_calls
            and not msg.tool_result
            and not out[-1].tool_calls
            and not out[-1].tool_result
            and out[-1].role == msg.role
        ):
            prev = out[-1]
            out[-1] = Message(
                role=prev.role,
                content="\n".join(filter(None, [prev.content, msg.content])) or None,
                thinking="\n".join(filter(None, [prev.thinking, msg.thinking])) or None,
            )
        else:
            out.append(msg)

    for raw in messages:
        msg = _copy_msg(raw)

        if msg.tool_result and msg.role != "tool":
            if strict:
                raise ValueError(f"tool_result on role={msg.role!r}; expected 'tool'")
            msg.role = "tool"

        if msg.tool_calls:
            cleaned = []
            for tc in msg.tool_calls:
                if tc.name not in tool_names:
                    if strict:
                        raise ValueError(f"tool_call references unknown tool: {tc.name!r}")
                    continue
                tc = ToolCall(fresh_id(tc.id), tc.name, tc.arguments)
                known_calls[tc.id] = tc
                cleaned.append(tc)
            msg.tool_calls = cleaned
            if not msg.tool_calls and not msg.content:
                continue

        if msg.tool_result:
            call_id, _ = msg.tool_result
            if call_id not in known_calls:
                if strict:
                    raise ValueError(f"tool_result references unknown call id: {call_id!r}")
                orphans.append(msg)
                continue

        append_msg(msg)

    for msg in orphans:
        call_id, _ = msg.tool_result
        if call_id in known_calls:
            append_msg(msg)

    return tools, out

def parse_gemini(body: dict) -> tuple[list[Tool], list[Message]]:
    tools = []
    for tool_group in body.get("tools", []):
        for f in tool_group.get("functionDeclarations", []):
            tools.append(Tool(
                name=f["name"],
                description=f.get("description", ""),
                parameters=_norm_params(f),
            ))

    messages = []

    sys = body.get("systemInstruction") or body.get("system_instruction")
    if sys:
        parts = sys.get("parts", [])
        text = "\n".join(p["text"] for p in parts if "text" in p)
        if text:
            messages.append(Message(role="system", content=text))

    for content in body.get("contents", []):
        role = content.get("role", "user")

        if role == "system":
            text = "\n".join(
                p["text"] for p in content.get("parts", []) if "text" in p
            )
            if text:
                if messages and messages[0].role == "system":
                    messages[0] = Message(
                        role="system",
                        content="\n\n".join(filter(None, [messages[0].content, text]))
                    )
                else:
                    messages.insert(0, Message(role="system", content=text))
            continue

        parts = content.get("parts", [])
        text_parts, thinking_parts, tool_calls, tool_results = [], [], [], []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            if "thought" in part:
                thinking_parts.append(part.get("thinking") or part.get("text", ""))
            if "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(ToolCall(
                    id=fc.get("id") or fc["name"],
                    name=fc["name"],
                    arguments=fc.get("args", {}),
                ))
            if "functionResponse" in part:
                fr = part["functionResponse"]
                response_content = fr.get("response", {})
                if isinstance(response_content, dict):
                    response_content = json.dumps(response_content)
                tool_results.append((
                    fr.get("id") or fr["name"],
                    response_content,
                ))

        canonical_role = "assistant" if role == "model" else "user"

        if tool_results:
            for tr in tool_results:
                messages.append(Message(role="tool", tool_result=tr))
        elif tool_calls:
            messages.append(Message(
                role=canonical_role,
                content="\n".join(text_parts) or None,
                thinking="\n".join(thinking_parts) or None,
                tool_calls=tool_calls,
            ))
        else:
            messages.append(Message(
                role=canonical_role,
                content="\n".join(text_parts) or None,
                thinking="\n".join(thinking_parts) or None,
            ))

    return tools, messages

def parse_default(body: dict) -> tuple[list[Tool], list[Message]]:
    tools = [
        Tool(
            name=(f := t.get("function", t))["name"],
            description=f.get("description", ""),
            parameters=_norm_params(f),
        )
        for t in body.get("tools", [])
    ]

    def _extract_parts(content) -> tuple[str, str]:
        """Returns (text, thinking). Handles str, flat block lists."""
        if isinstance(content, str):
            return content, ""
        text_parts, thinking_parts = [], []
        for b in content:
            btype = b.get("type")
            if btype == "thinking":
                thinking_parts.append(b.get("thinking") or b.get("text", ""))
            elif btype == "text":
                text_parts.append(b.get("text", ""))
        return "\n".join(text_parts), "\n".join(thinking_parts)

    messages = []
    for m in body.get("messages", []):
        role, content = m["role"], m.get("content")

        if role == "tool":
            messages.append(Message(
                role="tool",
                tool_result=(m["tool_call_id"], _extract_parts(content or "")[0]),
            ))

        elif role == "assistant" and m.get("tool_calls"):
            text, thinking = _extract_parts(content) if content else ("", "")
            messages.append(Message(
                role="assistant",
                content=text or None,
                thinking=thinking or m.get("thinking") or None,
                tool_calls=[
                    ToolCall(tc["id"], tc["function"]["name"], _safe_json(tc["function"]["arguments"]))
                    for tc in m["tool_calls"]
                ],
            ))

        else:
            text, thinking = _extract_parts(content or "")
            messages.append(Message(
                role=role,
                content=text or None,
                thinking=thinking or m.get("thinking") or None,
            ))

    return tools, messages


def parse_codex(body: dict) -> tuple[list[Tool], list[Message]]:
    role_map = {"developer": "system"}
    tools = [
        Tool(
            name=t["name"],
            description=t.get("description", ""),
            parameters=_norm_params(t),
        )
        for t in body.get("tools", [])
        if t.get("type") == "function" and "name" in t
    ]

    def _extract_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if block.get("type") in ("input_text", "output_text", "text")
            )
        return ""

    messages = []
    raw = body.get("input", body.get("messages", []))

    for m in raw:
        mtype = m.get("type")

        if mtype == "message":
            role = role_map.get(m.get("role", "user"), m.get("role", "user"))
            content = _extract_text(m.get("content", ""))
            messages.append(Message(role=role, content=content))

        elif mtype == "function_call":
            messages.append(Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        m["call_id"],
                        m["name"],
                        _safe_json(m.get("arguments", "{}")),
                    )
                ],
            ))

        elif mtype == "function_call_output":
            messages.append(Message(
                role="tool",
                tool_result=(m["call_id"], m.get("output", "")),
            ))

    return tools, messages


def parse_claude(body: dict) -> tuple[list[Tool], list[Message]]:
    tools = [
        Tool(t["name"], t.get("description", ""), _norm_params(t))
        for t in body.get("tools", [])
    ]
    messages = []

    sys = body.get("system")
    if isinstance(sys, str) and sys.strip():
        messages.append(Message(role="system", content=sys.strip()))
    elif isinstance(sys, list):
        text = "\n\n".join(b["text"] for b in sys if b.get("type") == "text" and not b["text"].startswith("x-anthropic-billing-header:"))
        if text:
            messages.append(Message(role="system", content=text))

    for m in body.get("messages", []):
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        text_parts, thinking_parts, tool_calls, tool_results = [], [], [], []

        for block in content:
            t = block.get("type")
            if t == "text":
                text_parts.append(block["text"])
            elif t == "thinking":
                thinking_parts.append(block.get("thinking") or block.get("text", ""))
            elif t == "tool_use":
                tool_calls.append(ToolCall(block["id"], block["name"], block.get("input", {})))
            elif t == "tool_result":
                rc = block.get("content", "")
                if isinstance(rc, list):
                    rc = "\n".join(c.get("text", "") for c in rc if c.get("type") == "text")
                tool_results.append((block["tool_use_id"], rc))

        if tool_results:
            for tr in tool_results:
                messages.append(Message(role="tool", tool_result=tr))
        elif tool_calls:
            messages.append(Message(
                role=role,
                content="\n".join(text_parts) or None,
                thinking="\n".join(thinking_parts) or None,
                tool_calls=tool_calls,
            ))
        else:
            messages.append(Message(
                role=role,
                content="\n".join(text_parts) or None,
                thinking="\n".join(thinking_parts) or None,
            ))

    return tools, messages

PARSERS: dict[str, Any] = {
    "codex": parse_codex,
    "claude": parse_claude,
    "gemini": parse_gemini,
    "default": parse_default,
}

def render_default(tools: list[Tool], messages: list[Message]) -> dict:
    out: list[dict] = []
    tc_map: dict[str, ToolCall] = {}

    for msg in messages:
        if msg.role == "tool":
            call_id, content = msg.tool_result
            tc = tc_map.get(call_id)
            m: dict[str, Any] = {"role": "tool", "tool_call_id": call_id, "content": content}
            if tc:
                m["name"] = tc.name 
            out.append(m)


        else:
            m = {"role": msg.role}
            if msg.content:
                m["content"] = msg.content
            if msg.thinking:
                m["reasoning_content"] = msg.thinking
            out.append(m)
            if msg.tool_calls:
                tc_map.update({tc.id: tc for tc in msg.tool_calls})

    body: dict[str, Any] = {"messages": out}
    if tools:
        body["tools"] = [_tool_schema(t) for t in tools]
    return body


def render_claude(tools: list[Tool], messages: list[Message]) -> dict:
    out: list[dict] = []
    sys_content = None
    tc_map: dict[str, ToolCall] = {}

    for msg in messages:
        if msg.role == "system":
            sys_content = msg.content
            continue

        if msg.role == "tool":
            call_id, content = msg.tool_result
            tc = tc_map.get(call_id)
            block: dict[str, Any] = {"type": "tool_result", "tool_use_id": call_id, "content": content}
            if tc:
                block["name"] = tc.name
            out.append({"role": "user", "content": [block]})

        elif msg.tool_calls:
            tc_map.update({tc.id: tc for tc in msg.tool_calls})
            blocks: list[dict] = []
            if msg.thinking:
                blocks.append({"type": "thinking", "thinking": msg.thinking})
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            blocks += [
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                for tc in msg.tool_calls
            ]
            out.append({"role": "assistant", "content": blocks})

        else:
            blocks = []
            if msg.thinking:
                blocks.append({"type": "thinking", "thinking": msg.thinking})
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            out.append({"role": msg.role, "content": blocks or ""})

    body: dict[str, Any] = {"messages": out}
    if sys_content:
        body["system"] = sys_content
    if tools:
        body["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]
    return body

def render_gemini(tools: list[Tool], messages: list[Message]) -> dict:
    contents: list[dict] = []
    sys_content = None
    tc_map: dict[str, ToolCall] = {}

    pending_results: list[dict] = []

    def flush_results():
        if pending_results:
            contents.append({"role": "user", "parts": list(pending_results)})
            pending_results.clear()

    for msg in messages:
        if msg.role == "system":
            sys_content = msg.content
            continue

        if msg.role == "tool":
            call_id, content = msg.tool_result
            tc = tc_map.get(call_id)
            try:
                response_body = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                response_body = {"result": content}

            fr: dict[str, Any] = {
                "name": tc.name if tc else call_id,
                "response": response_body,
            }
            if tc and tc.id != tc.name:
                fr["id"] = tc.id   
            pending_results.append({"functionResponse": fr})

        else:
            flush_results()
            role = "model" if msg.role == "assistant" else msg.role
            parts: list[dict] = []

            if msg.thinking:
                parts.append({"thought": msg.thinking})
            if msg.content:
                parts.append({"text": msg.content})
            if msg.tool_calls:
                tc_map.update({tc.id: tc for tc in msg.tool_calls})
                for tc in msg.tool_calls:
                    fc: dict[str, Any] = {"name": tc.name, "args": tc.arguments}
                    if tc.id != tc.name:
                        fc["id"] = tc.id
                    parts.append({"functionCall": fc})

            if parts:
                contents.append({"role": role, "parts": parts})

    flush_results()

    body: dict[str, Any] = {"contents": contents}
    if sys_content:
        body["system_instruction"] = {"parts": [{"text": sys_content}]}
    if tools:
        body["tools"] = [{"functionDeclarations": [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in tools
        ]}]
    return body


RENDERERS: dict[str, Any] = {
    "claude": render_claude,
    "gemini": render_gemini,
    "default": render_default,
}

def translate(
    body: dict,
    src: str,
    dst: str,
    *,
    system_override: str | None = None,
    tool_names: list[str] | None = None,
    skips: list[str] | None = None,
    strict: bool = False,
    **kwargs,
) -> Any:
    tools, messages = PARSERS[src](body)
    if tool_names is not None:
        missing = set(tool_names) - {t.name for t in tools}
        if missing and strict:
            raise ValueError(f"tool_names requested but not in body: {missing}")
        tools = [t for t in tools if t.name in tool_names]
    if system_override is not None:
        messages = [replace(m, content=system_override) if m.role == "system" else m for m in messages]
    logger.debug(f'{messages=}\n\n{tools=}')
    if skips is not None:
        messages = [replace(m, content=_skip(m.content, skips)) for m in messages]
    tools, messages = normalize(tools, messages, strict=strict)
    return RENDERERS[dst](tools, messages)

def encode(body, api, tokenizer, system_override, tool_names, skips, strict=False):
    body = translate(body, api, "default", system_override=system_override, tool_names=tool_names, skips=skips, strict=strict)
    tools = body.pop("tools", None)
    msgs = body.pop('messages', None)
    if not msgs or not msgs[-1].get('content', '').strip():
        return '', None
    apply_chat_template = lambda x: tokenizer.apply_chat_template(x, tools = tools or None, tokenize=False, add_generation_prompt=True)
    full_s = apply_chat_template(msgs)
    add_special_tokens = tokenizer.bos_token is None or not full_s.startswith(tokenizer.bos_token)
    full = tokenizer.encode(full_s, add_special_tokens=add_special_tokens)
    ckpts = []
    for last_user_idx in (i for i, m in enumerate(msgs) if m.get("role") == "user"):
        p_msgs = msgs[:last_user_idx] + [dict(role='user', content='h' if msgs[last_user_idx]['content'][0] != 'h' else 'i')]
        prfx_s = apply_chat_template(p_msgs)
        prfx = tokenizer.encode(prfx_s, add_special_tokens=add_special_tokens)
        ckpts.append(get_common_len(full, prfx))
    logger.info(f'{ckpts=}\n'+'\n'.join([f"{tokenizer.decode(full[i:j])}\n---{j}" for i, j in zip([0]+sorted(ckpts), sorted(ckpts)+[len(full)])]))
    return full, sorted(ckpts, reverse=True)
# }}}enc
# {{{gen
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

def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)

class PromptCache:
    def __init__(self, model, model_name, cache_dir):
        self.model = model
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache = None
        self.hx = []
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, prompt, ckpts):
        cl = get_common_len(self.hx, prompt)
        logger.info(f'{ckpts=} {len(prompt)=} {len(self.hx)=} {cl=}')
        if self.cache is not None:
            if len(self.hx)==cl:
                logger.debug('cont')
                return cl
            if all(c.is_trimmable() for c in self.cache):
                logger.debug(f'trim {len(self.hx)-cl}')
                mlx_lm.models.cache.trim_prompt_cache(self.cache, len(self.hx)-cl)
                self.hx = prompt[:cl]
                return cl
        for ckpt in ckpts:
            if self.get_path(prompt[:ckpt]).exists():
                self.load(prompt[:ckpt])
                return ckpt
        logger.debug('anew')
        self.cache = mlx_lm.models.cache.make_prompt_cache(self.model)
        self.hx = []
        return 0

    def get_path(self, prompt):
        safe_name = "".join(c for c in self.model_name if c.isalnum())
        token_hash = hash_tokens(prompt)
        return self.cache_dir / f"{safe_name}_{len(prompt)}_{token_hash}.safetensors"

    def load(self, prompt):
        path = self.get_path(prompt)
        logger.debug(path)
        self.cache, metadata = mlx_lm.models.cache.load_prompt_cache(path, return_metadata=True)
        self.hx = json.loads(metadata.pop("hx", "[]"))
        mx.eval(self.cache)

    def save(self, hx, cache, ppt=None):
        if ppt is None or ppt == len(hx) - len(self.hx):
            path = self.get_path(hx)
            logger.debug(path)
            metadata = dict(model_name=self.model_name, hx=json.dumps(hx))
            mlx_lm.models.cache.save_prompt_cache(path, cache, metadata=metadata)
            return 0
        return len(hx)-ppt


def generate(model, tokenizer, prompt, ckpts, pc, max_tokens=256, **kwargs):
    if ckpts is None:
        return 
    _eos_token_ids = set(tokenizer.eos_token_ids)
    _eos_token_ids.add(tokenizer.eos_token_id)

    detokenizer = tokenizer.detokenizer
    gens = []

    cl = pc(prompt, ckpts)
    prompt_arr = mx.array(prompt[cl:])
    save_fn = None
    if ckpts[0] > cl:
        save_fn = functools.partial(pc.save, prompt[:ckpts[0]])
        logger.debug(f'save_fn {ckpts[0]}')

    token_gen = generate_step(prompt_arr, model, prompt_cache=pc.cache, max_tokens=max_tokens, save_fn=save_fn, **kwargs)
    tic_non = time.perf_counter()
    tic_inp = time.perf_counter()
    for token, _ in token_gen:
        gens.append(token)
        if token in _eos_token_ids:
            break
        detokenizer.add_token(token)
        if abort_ev.is_set():
            break
        seg = detokenizer.last_segment
        if len(gens) == 1:
            tic_inp = time.perf_counter()
            logger.debug(f'pmt\nProcessed {len(prompt)} input tokens in {tic_inp-tic_non:.0f} seconds ({len(prompt)/(tic_inp-tic_non):.0f} tokens per second)')
        if seg:
            yield seg
    pc.hx = prompt + gens
    tic_out = time.perf_counter()
    logger.info(f'gen\n{tokenizer.decode(gens)}\nGenerated {len(gens)} new tokens in {tic_out-tic_inp:.0f} seconds ({len(gens)/(tic_out-tic_inp):.0f} tokens per second)')
    detokenizer.finalize()
    if seg := detokenizer.last_segment:
        yield seg

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
            if save_fn is not None: 
                if (saved := save_fn(prompt_cache, prompt_processed_tokens)) > 0:
                    n_to_process = min(n_to_process, saved)

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
# }}}gen
# {{{dec
def _parse_tools_xml(part):
    blocks = []
    tool_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    for match in tool_pattern.finditer(part):
        content = match.group(1).strip()
        if not "<function=" in content:
            continue
        fn_match = re.search(r"<function=([^\s>]+)>", content)
        if fn_match:
            name = fn_match.group(1)
            params = re.findall(r"<parameter=([^\s>]+)>\s*(.*?)\s*</parameter>", content, re.DOTALL)
            blocks.append({
                "id": f"toolu_{uuid.uuid4().hex[:8]}",
                "name": name,
                "input": {k: v for k, v in params},
            })
    return blocks

class BaseAdapter:
    def __init__(self, msg_id, in_tokens):
        self.msg_id = msg_id
        self.in_tokens = in_tokens

    def sse(self, data):
        return f"data: {json.dumps(data)}\n\n".encode()

    def start(self):
        return b""

    def text(self, state, text):
        return b""

    def tool(self, tool):
        return b""

    def end(self, has_tool):
        return b""

class ClaudeAdapter(BaseAdapter):
    def __init__(self, msg_id, in_tokens):
        super().__init__(msg_id, in_tokens)
        self.index = 0
        self.open = False
        self.state = None
        self.out_tokens = 0

    def _event(self, name, data):
        return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

    def start(self):
        return self._event("message_start", {
            "type": "message_start",
            "message": {
                "id": self.msg_id,
                "type": "message",
                "role": "assistant",
                "model": "local",
                "content": [],
                "usage": {"input_tokens": self.in_tokens, "output_tokens": 0}
            }
        })

    def _start_block(self, state, extra=None):
        self.open = True
        cb = {"type": state}
        if state in ("text", "thinking"):
            cb[state] = ""
        if extra:
            cb.update(extra)

        return self._event("content_block_start", {
            "type": "content_block_start",
            "index": self.index,
            "content_block": cb
        })

    def _delta(self, state, text):
        return self._event("content_block_delta", {
            "type": "content_block_delta",
            "index": self.index,
            "delta": {
                "type": f"{state}_delta",
                state: text
            }
        })

    def _stop_block(self):
        self.open = False
        e = self._event("content_block_stop", {
            "type": "content_block_stop",
            "index": self.index
        })
        self.index += 1
        return e

    def text(self, state, text):
        out = b""

        if not self.open or self.state != state:
            if self.open:
                out += self._stop_block()
            out += self._start_block(state)
            self.state = state

        if text:
            out += self._delta(state, text)
            self.out_tokens += 1

        return out

    def tool(self, tool):
        out = b""

        if self.open:
            out += self._stop_block()

        out += self._start_block("tool_use", {
            "id": tool["id"],
            "name": tool["name"],
            "input": {}
        })

        out += self._event("content_block_delta", {
            "type": "content_block_delta",
            "index": self.index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps(tool["input"])
            }
        })

        out += self._stop_block()
        return out

    def end(self, has_tool):
        out = b""

        if self.open:
            out += self._stop_block()

        out += self._event("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": "tool_use" if has_tool else "end_turn",
                "stop_sequence": None
            },
            "usage": {
                "output_tokens": self.out_tokens
            }
        })

        out += self._event("message_stop", {"type": "message_stop"})
        return out

class DefaultAdapter(BaseAdapter):
    def __init__(self, msg_id, in_tokens):
        super().__init__(msg_id, in_tokens)
        self.created = int(time.time())
        self.tool_index = 0

    def chunk(self, delta, finish_reason=None):
        return self.sse({
            "id": self.msg_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": "local",
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        })

    def start(self):
        return self.chunk({"role": "assistant", "content": ""})

    def text(self, state, text):
        if not text:
            return b""
        if state == "thinking":
            return self.chunk({"thinking": text})
        return self.chunk({"content": text})

    def tool(self, tool):
        i = self.tool_index
        self.tool_index += 1

        return (
            self.chunk({
                "tool_calls": [{
                    "index": i,
                    "id": tool["id"],
                    "type": "function",
                    "function": {"name": tool["name"], "arguments": ""}
                }]
            }) +
            self.chunk({
                "tool_calls": [{
                    "index": i,
                    "function": {"arguments": json.dumps(tool["input"])}
                }]
            })
        )

    def end(self, has_tool):
        return (
            self.chunk({}, finish_reason="tool_calls" if has_tool else "stop") +
            b"data: [DONE]\n\n"
        )

class CodexAdapter(BaseAdapter):
    def __init__(self, msg_id, in_tokens):
        super().__init__(msg_id, in_tokens)
        self.created = int(time.time())
        self.seq = 0
        self.item_id = f"item_{msg_id}"

    def _next_seq(self):
        self.seq += 1
        return self.seq

    def start(self):
        created = self.sse({
            "type": "response.created",
            "sequence_number": self._next_seq(),
            "response": {
                "id": self.msg_id,
                "object": "response",
                "created_at": self.created,
                "model": "local",
                "status": "in_progress",
                "output": [],
                "tools": [],
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "metadata": {},
                "parallel_tool_calls": True,
                "temperature": 1.0,
                "tool_choice": "auto",
                "top_p": 1.0,
            }
        })
        item_added = self.sse({
            "type": "response.output_item.added",
            "sequence_number": self._next_seq(),
            "output_index": 0,
            "item": {
                "id": self.item_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            }
        })
        part_added = self.sse({
            "type": "response.content_part.added",
            "sequence_number": self._next_seq(),
            "item_id": self.item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        })
        return created + item_added + part_added

    def text(self, state, text):
        if state == "thinking" or not text:
            return b""
        return self.sse({
            "type": "response.output_text.delta",
            "sequence_number": self._next_seq(),
            "item_id": self.item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        })

    def tool(self, tool):
        tool_item_id = f"tool_{tool['id']}"
        item_added = self.sse({
            "type": "response.output_item.added",
            "sequence_number": self._next_seq(),
            "output_index": 1,
            "item": {
                "id": tool_item_id,
                "type": "function_call",
                "name": tool["name"],
                "call_id": tool["id"],
                "arguments": "",
                "status": "in_progress",
            }
        })
        args = json.dumps(tool["input"])
        args_delta = self.sse({
            "type": "response.function_call_arguments.delta",
            "sequence_number": self._next_seq(),
            "item_id": tool_item_id,
            "output_index": 1,
            "delta": args,
        })
        args_done = self.sse({
            "type": "response.function_call_arguments.done",
            "sequence_number": self._next_seq(),
            "item_id": tool_item_id,
            "output_index": 1,
            "name": tool["name"],
            "arguments": args,
        })
        item_done = self.sse({
            "type": "response.output_item.done",
            "sequence_number": self._next_seq(),
            "output_index": 1,
            "item": {
                "id": tool_item_id,
                "type": "function_call",
                "name": tool["name"],
                "call_id": tool["id"],
                "arguments": args,
                "status": "completed",
            }
        })
        return item_added + args_delta + args_done + item_done

    def end(self, has_tool):
        part_done = self.sse({
            "type": "response.content_part.done",
            "sequence_number": self._next_seq(),
            "item_id": self.item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        })
        item_done = self.sse({
            "type": "response.output_item.done",
            "sequence_number": self._next_seq(),
            "output_index": 0,
            "item": {
                "id": self.item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            }
        })
        completed = self.sse({
            "type": "response.completed",
            "sequence_number": self._next_seq(),
            "response": {
                "id": self.msg_id,
                "object": "response",
                "created_at": self.created,
                "model": "local",
                "status": "completed",
                "output": [],
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "metadata": {},
                "parallel_tool_calls": True,
                "temperature": 1.0,
                "tool_choice": "auto",
                "top_p": 1.0,
                "usage": {
                    "input_tokens": self.in_tokens,
                    "output_tokens": 0,
                    "total_tokens": self.in_tokens,
                },
            }
        })
        return part_done + item_done + completed

class GeminiAdapter(BaseAdapter):
    def chunk(self, parts=None, finish_reason=None):
        return self.sse({
            "candidates": [{
                "content": {"role": "model", "parts": parts or []},
                "finishReason": finish_reason
            }],
            "usageMetadata": {
                "promptTokenCount": self.in_tokens,
                "candidatesTokenCount": 0
            }
        })

    def text(self, state, text):
        if not text:
            return b""
        if state == "thinking":
            return self.chunk([{"thought": text}])
        return self.chunk([{"text": text}])

    def tool(self, tool):
        return self.chunk([{
            "functionCall": {
                "id": tool["id"],
                "name": tool["name"],
                "args": tool["input"]
            }
        }])

    def end(self, has_tool):
        return self.chunk(finish_reason="TOOL_USE" if has_tool else "STOP")

def stream_sse(format_type, seg_gen, msg_id, in_tokens, think_tags=None):
    adapters = {
        "claude": ClaudeAdapter,
        "codex": CodexAdapter,
        "gemini": GeminiAdapter,
        "default": DefaultAdapter,
    }

    adapter = adapters.get(format_type, CodexAdapter)(msg_id, in_tokens)

    yield adapter.start()

    state = "thinking"
    buf = ""
    think_tags = ["<think>", "</think>"] if think_tags is None else think_tags

    for seg in seg_gen:
        buf += seg
        while any(t in seg for t in think_tags):
            if state == "text" and think_tags[0] in seg:
                before, _, seg = seg.partition(think_tags[0])
                if before:
                    yield adapter.text("text", before)
                state = "thinking"

            if state == "thinking" and think_tags[1] in seg:
                before, _, seg = seg.partition(think_tags[1])
                if before:
                    yield adapter.text("thinking", before)
                state = "text"

        if seg:
            yield adapter.text(state, seg)

    if tools := _parse_tools_xml(buf):
        for tool in tools:
            yield adapter.tool(tool)
        yield adapter.end(True)
    else:
        yield adapter.end(False)
# }}}dec
# {{{ser
def make_handler(model_name, api, cache_dir, system, names, skips, parse_think=True):
    model, tokenizer = mlx_lm.load(model_name)
    pc = PromptCache(model, model_name=model_name, cache_dir=cache_dir)
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
                self.send_json(200, {
                    "data": [{
                        "id": "local",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                    }]
                })
            else:
                self.send_json(404, {"error": "not found"})
        def do_POST(self):
            try:
                path = self.path.split("?")[0].rstrip("/")
                if path == "/v1/messages/count_tokens":
                    self.send_json(200, {"input_tokens": 0})
                    return
                if path.startswith("/v1beta/models/"):
                    if not ("alt=sse" in self.path or "streamGenerateContent" in path):
                        n = int(self.headers.get("Content-Length", 0))
                        _ = self.rfile.read(n)
                        dummy = json.dumps({
                            "candidates": [{"content": {"role": "model", "parts": [{"text": '{"complexity_reasoning":"local","complexity_score":50}'}]}, "finishReason": "STOP"}],
                            "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0}
                        }).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(dummy)))
                        self.end_headers()
                        self.wfile.write(dummy)
                        return
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                logger.info(f'{self.path=}\n{json.dumps(body, indent=2)}')
                abort_ev.set()
                with gen_lock:
                    abort_ev.clear()
                    prompt, ckpts = encode(body, api, tokenizer, system, names, skips)
                    seg_gen = generate(model, tokenizer, prompt=prompt, ckpts=ckpts, pc=pc, max_tokens=body.get("max_tokens", 8192))
                    msg_id = f"msg_{uuid.uuid4().hex}"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    try:
                        for chunk in stream_sse(api, seg_gen, msg_id, len(prompt)):
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    except BrokenPipeError:
                        seg_gen.close()
            except Exception as e:
                logger.exception(f'{self.path=}')
                raise
    return Handler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="mlx-community/Qwen3.5-4B-OptiQ-4bit")
    parser.add_argument("-a", "--api", type=str, default="default")
    parser.add_argument("-H", "--harness", default=None)
    parser.add_argument("-s", "--system", type=str, default='You are a helpful assistant')
    # parser.add_argument("--tools", nargs="+", default=['Bash', 'run_shell_command', 'exec_command'])
    parser.add_argument("--tools", nargs="+", default=None, help="Allow all tools if None (default None)")
    parser.add_argument("--cache", type=str, default='cache')
    parser.add_argument("--work", default=os.getcwd())
    parser.add_argument("--nocc", action="store_true", help="Disable Claude Code subprocess and run server only")
    parser.add_argument("--port", type=int, default=None, help="8000 if None (default None)")
    parser.add_argument("--skips", nargs="+", default=[
        r'(?m)^\[SUGGESTION MODE[\s\S]*',
        r'(?m)^<system-reminder>[\s\S]*?^</system-reminder>\s*',
    ])
    parser.add_argument("--host", default="127.0.0.1")
    args, harness_args = parser.parse_known_args()
    logger.info(f'{args=} {harness_args=}')
    api = args.harness if args.harness else args.api
    port = args.port if args.port is not None else 8000
    server = None
    while server is None:
        try:
            server = HTTPServer(
                (args.host, port),
                make_handler(args.model, api, args.cache, args.system, args.tools, args.skips),
            )
        except OSError as e:
            if e.errno == 48 or e.errno == 98: 
                if args.port is not None:
                    logger.error(f"Port {port} is already in use.")
                    sys.exit(1)
                port += 1
            else:
                raise e
    url = f"http://{args.host}:{port}"
    if args.nocc:
        print(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.server_close()
    else:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        env = os.environ.copy()

        with tempfile.TemporaryDirectory() as _home:
            home = Path(_home)
            env["HOME"] = str(home)
            env["SHELL"] = "/bin/bash"
            env["GOOGLE_GEMINI_BASE_URL"]=url
            env["GEMINI_API_KEY"]="mc"

            env.pop("OPENAI_API_KEY", None)
            codex_dir = home/".codex"
            codex_dir.mkdir(parents=True, exist_ok=True)
            config_path = (codex_dir/"config.toml").write_text(f"""
[model_providers.local]
name = "openai"
base_url = "{url}/v1"
api_key = "mc"

[profiles.local]
model_provider = "local"
model = "gpt-5.4-mini"
""".strip())

            env["ANTHROPIC_BASE_URL"] = url
            env["ANTHROPIC_AUTH_TOKEN"] = "mc"
            env["ANTHROPIC_MODEL"] = args.model
            def mirror_workspace(src: str, dst: str):
                for root, dirs, files in os.walk(src):
                    rel = os.path.relpath(root, src)
                    os.makedirs(os.path.join(dst, rel), exist_ok=True)
                    for f in files:
                        os.link(os.path.join(root, f), os.path.join(dst, rel, f))
            workspace = home/"workspace"
            mirror_workspace(args.work, workspace)
            if args.harness is None:
                from .pie import run_repl
                run_repl(base_url=url, provider=api, cwd=str(workspace), env=env)
            else:
                if args.harness == "codex":
                    harness_args += ['--profile', 'local']
                sys.exit(subprocess.run([args.harness] + harness_args, env=env, cwd=workspace).returncode)

if __name__ == "__main__":
    main()
# }}}ser

try:
    import mlx_lm
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    raise SystemExit('pip install mlx-lm')
import json
import uuid
from dataclasses import dataclass, field, replace
import random
import tempfile
import argparse
import logging
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer, HTTPServer
from pathlib import Path
from array import array
import hashlib
import contextlib
import functools
from datetime import datetime, timezone
from typing import Any, Callable, Generator, List, Optional, Tuple, Union
from .gits import create_worktree, commit_worktree
from .util import setup_logger
logger = setup_logger('.log.json')
generation_stream = mx.new_thread_local_stream(mx.default_device())
gen_lock = threading.Lock()
abort_ev = threading.Event()

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
    params = dict(fxn.get('parameters') or fxn.get('parametersJsonSchema') or fxn.get('input_schema') or {'type': 'object', 'properties': {}})
    params.pop('$schema', None)
    return params

def _safe_json(x) -> dict:
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x)
    except Exception:
        return {}

def _tool_schema(tool: Tool) -> dict:
    return {'type': 'function', 'function': {'name': tool.name, 'description': tool.description, 'parameters': tool.parameters}}

def _copy_msg(m: Message) -> Message:
    return Message(role=m.role, content=m.content, thinking=m.thinking, tool_calls=list(m.tool_calls), tool_result=m.tool_result)

def _skip(text, skips=None, show_skipped=False):
    if text is None or skips is None:
        return text
    lines = []
    for pattern in skips:
        found = re.findall(pattern, text)
        if found:
            lines.append(f'{pattern}\n' + '\n'.join(found))
    if lines and show_skipped:
        logger.debug('\n---\n'.join(lines))
    for pattern in skips:
        text = re.sub(pattern, '', text)
    return text

def parse_gemini(body: dict) -> tuple[list[Tool], list[Message]]:
    tools = []
    for tool_group in body.get('tools', []):
        for f in tool_group.get('functionDeclarations', []):
            tools.append(Tool(f['name'], f.get('description', ''), _norm_params(f)))
    messages = []
    sys = body.get('systemInstruction') or body.get('system_instruction')
    if sys:
        text = '\n'.join((p['text'] for p in sys.get('parts', []) if 'text' in p))
        if text:
            messages.append(Message(role='system', content=text))
    for content in body.get('contents', []):
        role = content.get('role', 'user')
        if role == 'system':
            text = '\n'.join((p['text'] for p in content.get('parts', []) if 'text' in p))
            if text:
                if messages and messages[0].role == 'system':
                    messages[0] = Message(role='system', content='\n\n'.join(filter(None, [messages[0].content, text])))
                else:
                    messages.insert(0, Message(role='system', content=text))
            continue
        parts = content.get('parts', [])
        text_parts, thinking_parts, tool_calls, tool_results = ([], [], [], [])
        for part in parts:
            if 'text' in part:
                text_parts.append(part['text'])
            if 'thought' in part:
                thinking_parts.append(part.get('thinking') or part.get('thought'))
            if 'functionCall' in part:
                fc = part['functionCall']
                tc = ToolCall(id=fc.get('id') or fc['name'], name=fc['name'], arguments=fc.get('args', {}))
                tool_calls.append(tc)
            if 'functionResponse' in part:
                fr = part['functionResponse']
                call_id = fr.get('id') or fr['name']
                output = fr.get('response', {})
                if isinstance(output, dict):
                    output = json.dumps(output)
                tool_results.append((call_id, output))
        canonical_role = 'assistant' if role == 'model' else 'user'
        if tool_results:
            for tr in tool_results:
                messages.append(Message(role='tool', tool_result=tr))
        elif tool_calls:
            messages.append(Message(role=canonical_role, content='\n'.join(text_parts) or None, thinking='\n'.join(thinking_parts).strip() or None, tool_calls=tool_calls))
        else:
            messages.append(Message(role=canonical_role, content='\n'.join(text_parts) or None, thinking='\n'.join(thinking_parts).strip() or None))
    return (tools, messages)

def parse_default(body: dict) -> tuple[list[Tool], list[Message]]:
    tools = [Tool(name=(f := t.get('function', t))['name'], description=f.get('description', ''), parameters=_norm_params(f)) for t in body.get('tools', [])]

    def _extract_parts(content) -> tuple[str, str]:
        if isinstance(content, str):
            return (content, '')
        text_parts, thinking_parts = ([], [])
        for b in content:
            btype = b.get('type')
            if btype == 'thinking':
                thinking_parts.append(b.get('thinking') or b.get('reasoning_content') or b.get('text', ''))
            elif btype == 'text':
                text_parts.append(b.get('text', ''))
        return ('\n'.join(text_parts), '\n'.join(thinking_parts))
    messages = []
    for m in body.get('messages', []):
        role, content = (m['role'], m.get('content'))
        if role == 'tool':
            call_id = m['tool_call_id']
            output = _extract_parts(content or '')[0]
            messages.append(Message(role='tool', tool_result=(call_id, output)))
        elif role == 'assistant' and m.get('tool_calls'):
            text, thinking = _extract_parts(content) if content else ('', '')
            tcs = [ToolCall(tc['id'], tc['function']['name'], _safe_json(tc['function']['arguments'])) for tc in m['tool_calls']]
            messages.append(Message(role='assistant', content=text or None, thinking=thinking or m.get('reasoning_content') or m.get('thinking') or None, tool_calls=tcs))
        else:
            text, thinking = _extract_parts(content or '')
            messages.append(Message(role=role, content=text or None, thinking=thinking or m.get('reasoning_content') or m.get('thinking') or None))
    return (tools, messages)

def parse_codex(body: dict) -> tuple[list[Tool], list[Message]]:
    tools = [Tool(t['name'], t.get('description', ''), _norm_params(t)) for t in body.get('tools', []) if t.get('type') == 'function' and 'name' in t]

    def _extract_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return ''.join((block.get('text', '') for block in content if block.get('type') in ('input_text', 'output_text', 'text')))
        return ''
    messages = []
    role_map = {'developer': 'system'}
    for m in body.get('input', body.get('messages', [])):
        mtype = m.get('type')
        if mtype == 'message':
            role = role_map.get(m.get('role', 'user'), m.get('role', 'user'))
            content = _extract_text(m.get('content', ''))
            if content.strip():
                messages.append(Message(role=role, content=content))
        elif mtype == 'function_call':
            tc = ToolCall(m['call_id'], m['name'], _safe_json(m.get('arguments', '{}')))
            messages.append(Message(role='assistant', tool_calls=[tc]))
        elif mtype == 'function_call_output':
            call_id = m['call_id']
            output = m.get('output', '')
            messages.append(Message(role='tool', tool_result=(call_id, output)))
    return (tools, messages)

def parse_claude(body: dict) -> tuple[list[Tool], list[Message]]:
    tools = [Tool(t['name'], t.get('description', ''), _norm_params(t)) for t in body.get('tools', [])]
    messages = []
    sys = body.get('system')
    if isinstance(sys, str) and sys.strip():
        messages.append(Message(role='system', content=sys.strip()))
    elif isinstance(sys, list):
        text = '\n\n'.join((b['text'] for b in sys if b.get('type') == 'text' and (not b['text'].startswith('x-anthropic-billing-header:'))))
        if text:
            messages.append(Message(role='system', content=text))
    for m in body.get('messages', []):
        role = m['role']
        content = m['content']
        if isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        text_parts, thinking_parts, tool_calls, tool_results = ([], [], [], [])
        for block in content:
            t = block.get('type')
            if t == 'text':
                text_parts.append(block['text'])
            elif t == 'thinking':
                thinking_parts.append(block.get('thinking') or block.get('text', ''))
            elif t == 'tool_use':
                tc = ToolCall(block['id'], block['name'], block.get('input', {}))
                tool_calls.append(tc)
            elif t == 'tool_result':
                call_id = block['tool_use_id']
                rc = block.get('content', '')
                if isinstance(rc, list):
                    rc = '\n'.join((c.get('text', '') for c in rc if c.get('type') == 'text'))
                tool_results.append((call_id, rc))
        if tool_results:
            for tr in tool_results:
                messages.append(Message(role='tool', tool_result=tr))
        elif tool_calls:
            messages.append(Message(role=role, content='\n'.join(text_parts) or None, thinking='\n'.join(thinking_parts) or None, tool_calls=tool_calls))
        else:
            messages.append(Message(role=role, content='\n'.join(text_parts) or None, thinking='\n'.join(thinking_parts) or None))
    return (tools, messages)
PARSERS: dict[str, Any] = {'codex': parse_codex, 'claude': parse_claude, 'gemini': parse_gemini, 'noapi': parse_default}

def render_default(tools: list[Tool], messages: list[Message], render_tc=False) -> dict:
    out: list[dict] = []
    tc_map = {}
    for msg in messages:
        if msg.role == 'tool':
            call_id, content = msg.tool_result
            tc = tc_map.get(call_id)
            m = {'role': 'tool', 'tool_call_id': call_id, 'content': f'<input>{json.dumps({'name': tc.name, 'arguments': tc.arguments})}</input>\n{content}' if tc and (not render_tc) else content}
            out.append(m)
        else:
            m = {}
            if msg.content:
                m['content'] = msg.content
            if msg.thinking:
                m['reasoning_content'] = msg.thinking
            if msg.tool_calls:
                tc_map.update({tc.id: tc for tc in msg.tool_calls})
                if render_tc:
                    m['tool_calls'] = [{'id': tc.id, 'type': 'function', 'function': {'name': tc.name, 'arguments': tc.arguments}} for tc in msg.tool_calls]
            if m:
                out.append(m | {'role': msg.role})
    body: dict[str, Any] = {'messages': out}
    if tools:
        body['tools'] = [_tool_schema(t) for t in tools]
    return body

def render_claude(tools: list[Tool], messages: list[Message]) -> dict:
    out: list[dict] = []
    sys_content = None
    tc_map: dict[str, ToolCall] = {}
    for msg in messages:
        if msg.role == 'system':
            sys_content = msg.content
            continue
        if msg.role == 'tool':
            call_id, content = msg.tool_result
            tc = tc_map.get(call_id)
            block: dict[str, Any] = {'type': 'tool_result', 'tool_use_id': call_id, 'content': content}
            if tc:
                block['name'] = tc.name
            out.append({'role': 'user', 'content': [block]})
        elif msg.tool_calls:
            tc_map.update({tc.id: tc for tc in msg.tool_calls})
            blocks: list[dict] = []
            if msg.thinking:
                blocks.append({'type': 'thinking', 'thinking': msg.thinking})
            if msg.content:
                blocks.append({'type': 'text', 'text': msg.content})
            blocks += [{'type': 'tool_use', 'id': tc.id, 'name': tc.name, 'input': tc.arguments} for tc in msg.tool_calls]
            out.append({'role': 'assistant', 'content': blocks})
        else:
            blocks = []
            if msg.thinking:
                blocks.append({'type': 'thinking', 'thinking': msg.thinking})
            if msg.content:
                blocks.append({'type': 'text', 'text': msg.content})
            out.append({'role': msg.role, 'content': blocks or ''})
    body: dict[str, Any] = {'messages': out}
    if sys_content:
        body['system'] = sys_content
    if tools:
        body['tools'] = [{'name': t.name, 'description': t.description, 'input_schema': t.parameters} for t in tools]
    return body

def render_gemini(tools: list[Tool], messages: list[Message]) -> dict:
    contents: list[dict] = []
    sys_content = None
    tc_map: dict[str, ToolCall] = {}
    pending_results: list[dict] = []

    def flush_results():
        if pending_results:
            contents.append({'role': 'user', 'parts': list(pending_results)})
            pending_results.clear()
    for msg in messages:
        if msg.role == 'system':
            sys_content = msg.content
            continue
        if msg.role == 'tool':
            call_id, content = msg.tool_result
            tc = tc_map.get(call_id)
            try:
                response_body = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                response_body = {'result': content}
            fr: dict[str, Any] = {'name': tc.name if tc else call_id, 'response': response_body}
            if tc and tc.id != tc.name:
                fr['id'] = tc.id
            pending_results.append({'functionResponse': fr})
        else:
            flush_results()
            role = 'model' if msg.role == 'assistant' else msg.role
            parts: list[dict] = []
            if msg.thinking:
                parts.append({'thought': msg.thinking})
            if msg.content:
                parts.append({'text': msg.content})
            if msg.tool_calls:
                tc_map.update({tc.id: tc for tc in msg.tool_calls})
                for tc in msg.tool_calls:
                    fc: dict[str, Any] = {'name': tc.name, 'args': tc.arguments}
                    if tc.id != tc.name:
                        fc['id'] = tc.id
                    parts.append({'functionCall': fc})
            if parts:
                contents.append({'role': role, 'parts': parts})
    flush_results()
    body: dict[str, Any] = {'contents': contents}
    if sys_content:
        body['system_instruction'] = {'parts': [{'text': sys_content}]}
    if tools:
        body['tools'] = [{'functionDeclarations': [{'name': t.name, 'description': t.description, 'parameters': t.parameters} for t in tools]}]
    return body
RENDERERS: dict[str, Any] = {'claude': render_claude, 'gemini': render_gemini, 'noapi': render_default}

def translate(body: dict, src: str, dst: str, *, system_override: str | None=None, tool_names: list[str] | None=None, skips: list[str] | None=None, strict: bool=False, **kwargs) -> Any:
    tools, messages = PARSERS[src](body)
    if tool_names is not None:
        missing = set(tool_names) - {t.name for t in tools}
        if missing and strict:
            raise ValueError(f'tool_names requested but not in body: {missing}')
        tools = [t for t in tools if t.name in tool_names]
    if system_override is not None:
        messages = [replace(m, content=system_override) if m.role == 'system' else m for m in messages]
    if skips is not None:
        messages = [replace(m, content=_skip(m.content, skips)) for m in messages]
    return RENDERERS[dst](tools, messages)

def encode(body, api, tokenizer, system_override, tool_names, skips, strict=False):
    body = translate(body, api, 'noapi', system_override=system_override, tool_names=tool_names, skips=skips, strict=strict)
    tools = body.pop('tools', None)
    msgs = body.pop('messages', None)
    if not msgs or not msgs[-1].get('content', '').strip():
        return ('', None)
    apply_chat_template = lambda x: tokenizer.apply_chat_template(x, tools=tools or None, tokenize=False, add_generation_prompt=True)
    full_s = apply_chat_template(msgs)
    add_special_tokens = tokenizer.bos_token is None or not full_s.startswith(tokenizer.bos_token)
    full = tokenizer.encode(full_s, add_special_tokens=add_special_tokens)
    ckpts = []
    for last_user_idx in (i for i, m in enumerate(msgs) if m.get('role') == 'user'):
        p_msgs = msgs[:last_user_idx] + [dict(role='user', content='h' if msgs[last_user_idx]['content'][0] != 'h' else 'i')]
        prfx_s = apply_chat_template(p_msgs)
        prfx = tokenizer.encode(prfx_s, add_special_tokens=add_special_tokens)
        ckpts.append(get_common_len(full, prfx))
    logger.debug(f'ckpts={ckpts!r}\n' + '\n'.join([f'{tokenizer.decode(full[i:j])}\n---{j}' for i, j in zip([0] + sorted(ckpts), sorted(ckpts) + [len(full)])]))
    return (full, sorted(ckpts, reverse=True))

def hash_tokens(tokens):
    arr = array('I', tokens)
    return hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()

def is_stuck(tokens, pattern_size=100, min_repeats=3):
    if len(tokens) < pattern_size * min_repeats:
        return False
    pattern = tuple(tokens[-pattern_size:])
    window = tokens[-(pattern_size * min_repeats + pattern_size):]
    positions = []
    limit = len(window) - pattern_size
    for i in range(limit):
        if tuple(window[i:i + pattern_size]) == pattern:
            positions.append(i)
    if len(positions) < min_repeats:
        return False
    intervals = [b - a for a, b in zip(positions, positions[1:])]
    if len(set(intervals[-(min_repeats - 1):])) == 1:
        return True
    return False

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
        if hasattr(c, 'to_quantized') and c.offset >= quantized_kv_start:
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
        logger.debug(f'ckpts={ckpts!r} len(prompt)={len(prompt)!r} len(self.hx)={len(self.hx)!r} cl={cl!r}')
        if self.cache is not None:
            if len(self.hx) == cl:
                logger.debug('cont')
                return cl
            if all((c.is_trimmable() for c in self.cache)):
                logger.debug(f'trim {len(self.hx) - cl}')
                mlx_lm.models.cache.trim_prompt_cache(self.cache, len(self.hx) - cl)
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
        safe_name = ''.join((c for c in self.model_name if c.isalnum()))
        token_hash = hash_tokens(prompt)
        return self.cache_dir / f'{safe_name}_{len(prompt)}_{token_hash}.safetensors'

    def load(self, prompt):
        path = self.get_path(prompt)
        logger.debug(path)
        self.cache, metadata = mlx_lm.models.cache.load_prompt_cache(path, return_metadata=True)
        self.hx = json.loads(metadata.pop('hx', '[]'))
        mx.async_eval(self.cache)

    def save(self, hx, cache, ppt=None):
        if ppt is None or ppt == len(hx) - len(self.hx):
            path = self.get_path(hx)
            logger.debug(path)
            metadata = dict(model_name=self.model_name, hx=json.dumps(hx))
            mlx_lm.models.cache.save_prompt_cache(path, cache, metadata=metadata)
            return 0
        return len(hx) - ppt

def generate(model, tokenizer, prompt, ckpts, pc, max_tokens=256, **kwargs):
    _log_str = ''
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
        logger.debug(f'Save {ckpts[0]}')
    token_gen = generate_step(prompt_arr, model, prompt_cache=pc.cache, max_tokens=max_tokens, save_fn=save_fn, _te=tokenizer.convert_tokens_to_ids('</think>'), **kwargs)
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
            _log_str = f'{len(prompt)} input tokens processed in {tic_inp - tic_non:.0f} seconds ({len(prompt) / (tic_inp - tic_non):.0f} tokens per second; {cl} from {ckpts}):\n\n'
            logger.debug(_log_str)
            _log_str += '\n'.join([f'{tokenizer.decode(prompt[i:j])}' if j == len(prompt) else f'{tokenizer.decode(prompt[i:j])}\n--- {j}' for i, j in zip([0] + sorted(ckpts), sorted(ckpts) + [len(prompt)])])
        if seg:
            yield seg
    pc.hx = prompt + gens
    tic_out = time.perf_counter()
    logger.info(f'{_log_str}\n{len(gens)} new tokens generated in {tic_out - tic_inp:.0f} seconds ({len(gens) / (tic_out - tic_inp):.0f} tokens per second):\n\n{tokenizer.decode(gens)}')
    detokenizer.finalize()
    if (seg := detokenizer.last_segment):
        yield seg

def generate_step(prompt: mx.array, model: nn.Module, *, max_tokens: int=256, sampler: Optional[Callable[[mx.array], mx.array]]=None, logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]]=None, max_kv_size: Optional[int]=None, prompt_cache: Optional[Any]=None, prefill_step_size: int=2048, kv_bits: Optional[int]=None, kv_group_size: int=64, quantized_kv_start: int=0, prompt_progress_callback: Optional[Callable[[int, int], None]]=None, input_embeddings: Optional[mx.array]=None, save_fn=None, _te=None) -> Generator[Tuple[mx.array, mx.array], None, None]:
    if input_embeddings is not None:
        if not does_model_support_input_embeddings(model):
            raise ValueError('Model does not support input embeddings.')
        elif len(prompt) > 0 and len(prompt) != len(input_embeddings):
            raise ValueError(f'len(input_embeddings)={len(input_embeddings)!r} len(prompt)={len(prompt)!r}')
    elif len(prompt) == 0:
        raise ValueError('Either input_embeddings or prompt (or both) must be provided.')
    tokens = None
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(model, max_kv_size=max_kv_size)
    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)
    quantize_cache_fn = functools.partial(maybe_quantize_kv_cache, quantized_kv_start=quantized_kv_start, kv_group_size=kv_group_size, kv_bits=kv_bits)
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _model_call(input_tokens: mx.array, input_embeddings: Optional[mx.array]):
        if input_embeddings is not None:
            return model(input_tokens, cache=prompt_cache, input_embeddings=input_embeddings)
        else:
            return model(input_tokens, cache=prompt_cache)

    def _step(input_tokens: mx.array, input_embeddings: Optional[mx.array]=None):
        nonlocal tokens
        with mx.stream(generation_stream):
            logits = _model_call(input_tokens=input_tokens[None], input_embeddings=input_embeddings[None] if input_embeddings is not None else None)
            logits = logits[:, -1, :]
            if logits_processors and len(input_tokens) > 0:
                tokens = mx.concat([tokens, input_tokens]) if tokens is not None else input_tokens
                for processor in logits_processors:
                    logits = processor(tokens, logits)
            quantize_cache_fn(prompt_cache)
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            return (sampled, logprobs.squeeze(0))
    with mx.stream(generation_stream):
        total_prompt_tokens = len(input_embeddings) if input_embeddings is not None else len(prompt)
        prompt_processed_tokens = 0
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        while total_prompt_tokens - prompt_processed_tokens > 1:
            remaining = total_prompt_tokens - prompt_processed_tokens - 1
            n_to_process = min(prefill_step_size, remaining)
            if save_fn is not None:
                if (saved := save_fn(prompt_cache, prompt_processed_tokens)) > 0:
                    n_to_process = min(n_to_process, saved)
            _model_call(input_tokens=prompt[:n_to_process][None], input_embeddings=input_embeddings[:n_to_process][None] if input_embeddings is not None else None)
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            prompt_processed_tokens += n_to_process
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt = prompt[n_to_process:]
            input_embeddings = input_embeddings[n_to_process:] if input_embeddings is not None else input_embeddings
            mx.clear_cache()
        y, logprobs = _step(input_tokens=prompt, input_embeddings=input_embeddings)
    mx.async_eval(y, logprobs)
    n = 0
    _tl = int(max_tokens * 0.9)
    _td = []
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if n == max_tokens:
            break
        _yi = y.item()
        yield (_yi, logprobs)
        if n % 256 == 0:
            mx.clear_cache()
        y, logprobs = (next_y, next_logprobs)
        if n <= _tl:
            _td.append(_yi)
            if (is_stuck(_td) or n == _tl) and _te not in _td:
                y = mx.array([_te])
                _tl = -1
        n += 1

def _parse_tools_xml(part):
    blocks = []
    tool_pattern = re.compile('<tool_call>(.*?)</tool_call>', re.DOTALL)
    for match in tool_pattern.finditer(part):
        content = match.group(1).strip()
        if not '<function=' in content:
            continue
        fn_match = re.search('<function=([^\\s>]+)>', content)
        if fn_match:
            name = fn_match.group(1)
            params = re.findall('<parameter=([^\\s>]+)>\\s*(.*?)\\s*</parameter>', content, re.DOTALL)
            blocks.append({'id': f'toolu_{uuid.uuid4().hex[:8]}', 'name': name, 'input': {k: v for k, v in params}})
    return blocks

class BaseAdapter:

    def __init__(self, msg_id, in_tokens):
        self.msg_id = msg_id
        self.in_tokens = in_tokens

    def sse(self, data):
        return f'data: {json.dumps(data)}\n\n'.encode()

    def start(self):
        return b''

    def text(self, state, text):
        return b''

    def tool(self, tool):
        return b''

    def end(self, has_tool):
        return b''

class ClaudeAdapter(BaseAdapter):

    def __init__(self, msg_id, in_tokens):
        super().__init__(msg_id, in_tokens)
        self.index = 0
        self.open = False
        self.state = None
        self.out_tokens = 0

    def _event(self, name, data):
        return f'event: {name}\ndata: {json.dumps(data)}\n\n'.encode()

    def start(self):
        return self._event('message_start', {'type': 'message_start', 'message': {'id': self.msg_id, 'type': 'message', 'role': 'assistant', 'model': 'local', 'content': [], 'usage': {'input_tokens': self.in_tokens, 'output_tokens': 0}}})

    def _start_block(self, state, extra=None):
        self.open = True
        cb = {'type': state}
        if state in ('text', 'thinking'):
            cb[state] = ''
        if extra:
            cb.update(extra)
        return self._event('content_block_start', {'type': 'content_block_start', 'index': self.index, 'content_block': cb})

    def _delta(self, state, text):
        return self._event('content_block_delta', {'type': 'content_block_delta', 'index': self.index, 'delta': {'type': f'{state}_delta', state: text}})

    def _stop_block(self):
        self.open = False
        e = self._event('content_block_stop', {'type': 'content_block_stop', 'index': self.index})
        self.index += 1
        return e

    def text(self, state, text):
        out = b''
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
        out = b''
        if self.open:
            out += self._stop_block()
        out += self._start_block('tool_use', {'id': tool['id'], 'name': tool['name'], 'input': {}})
        out += self._event('content_block_delta', {'type': 'content_block_delta', 'index': self.index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(tool['input'])}})
        out += self._stop_block()
        return out

    def end(self, has_tool):
        out = b''
        if self.open:
            out += self._stop_block()
        out += self._event('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use' if has_tool else 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': self.out_tokens}})
        out += self._event('message_stop', {'type': 'message_stop'})
        return out

class DefaultAdapter(BaseAdapter):

    def __init__(self, msg_id, in_tokens):
        super().__init__(msg_id, in_tokens)
        self.created = int(time.time())
        self.tool_index = 0

    def chunk(self, delta, finish_reason=None):
        return self.sse({'id': self.msg_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': 'local', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]})

    def start(self):
        return self.chunk({'role': 'assistant', 'content': ''})

    def text(self, state, text):
        if not text:
            return b''
        if state == 'thinking':
            return self.chunk({'reasoning_content': text})
        return self.chunk({'content': text})

    def tool(self, tool):
        i = self.tool_index
        self.tool_index += 1
        return self.chunk({'tool_calls': [{'index': i, 'id': tool['id'], 'type': 'function', 'function': {'name': tool['name'], 'arguments': ''}}]}) + self.chunk({'tool_calls': [{'index': i, 'function': {'arguments': json.dumps(tool['input'])}}]})

    def end(self, has_tool):
        return self.chunk({}, finish_reason='tool_calls' if has_tool else 'stop') + b'data: [DONE]\n\n'

class CodexAdapter(BaseAdapter):

    def __init__(self, msg_id, in_tokens):
        super().__init__(msg_id, in_tokens)
        self.created = int(time.time())
        self.seq = 0
        self.item_id = f'item_{msg_id}'

    def _next_seq(self):
        self.seq += 1
        return self.seq

    def start(self):
        created = self.sse({'type': 'response.created', 'sequence_number': self._next_seq(), 'response': {'id': self.msg_id, 'object': 'response', 'created_at': self.created, 'model': 'local', 'status': 'in_progress', 'output': [], 'tools': [], 'error': None, 'incomplete_details': None, 'instructions': None, 'metadata': {}, 'parallel_tool_calls': True, 'temperature': 1.0, 'tool_choice': 'auto', 'top_p': 1.0}})
        item_added = self.sse({'type': 'response.output_item.added', 'sequence_number': self._next_seq(), 'output_index': 0, 'item': {'id': self.item_id, 'type': 'message', 'role': 'assistant', 'status': 'in_progress', 'content': []}})
        part_added = self.sse({'type': 'response.content_part.added', 'sequence_number': self._next_seq(), 'item_id': self.item_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': []}})
        return created + item_added + part_added

    def text(self, state, text):
        if state == 'thinking' or not text:
            return b''
        return self.sse({'type': 'response.output_text.delta', 'sequence_number': self._next_seq(), 'item_id': self.item_id, 'output_index': 0, 'content_index': 0, 'delta': text})

    def tool(self, tool):
        tool_item_id = f'tool_{tool['id']}'
        item_added = self.sse({'type': 'response.output_item.added', 'sequence_number': self._next_seq(), 'output_index': 1, 'item': {'id': tool_item_id, 'type': 'function_call', 'name': tool['name'], 'call_id': tool['id'], 'arguments': '', 'status': 'in_progress'}})
        args = json.dumps(tool['input'])
        args_delta = self.sse({'type': 'response.function_call_arguments.delta', 'sequence_number': self._next_seq(), 'item_id': tool_item_id, 'output_index': 1, 'delta': args})
        args_done = self.sse({'type': 'response.function_call_arguments.done', 'sequence_number': self._next_seq(), 'item_id': tool_item_id, 'output_index': 1, 'name': tool['name'], 'arguments': args})
        item_done = self.sse({'type': 'response.output_item.done', 'sequence_number': self._next_seq(), 'output_index': 1, 'item': {'id': tool_item_id, 'type': 'function_call', 'name': tool['name'], 'call_id': tool['id'], 'arguments': args, 'status': 'completed'}})
        return item_added + args_delta + args_done + item_done

    def end(self, has_tool):
        part_done = self.sse({'type': 'response.content_part.done', 'sequence_number': self._next_seq(), 'item_id': self.item_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': []}})
        item_done = self.sse({'type': 'response.output_item.done', 'sequence_number': self._next_seq(), 'output_index': 0, 'item': {'id': self.item_id, 'type': 'message', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': '', 'annotations': []}]}})
        completed = self.sse({'type': 'response.completed', 'sequence_number': self._next_seq(), 'response': {'id': self.msg_id, 'object': 'response', 'created_at': self.created, 'model': 'local', 'status': 'completed', 'output': [], 'error': None, 'incomplete_details': None, 'instructions': None, 'metadata': {}, 'parallel_tool_calls': True, 'temperature': 1.0, 'tool_choice': 'auto', 'top_p': 1.0, 'usage': {'input_tokens': self.in_tokens, 'output_tokens': 0, 'total_tokens': self.in_tokens}}})
        return part_done + item_done + completed

class GeminiAdapter(BaseAdapter):

    def chunk(self, parts=None, finish_reason=None):
        return self.sse({'candidates': [{'content': {'role': 'model', 'parts': parts or []}, 'finishReason': finish_reason}], 'usageMetadata': {'promptTokenCount': self.in_tokens, 'candidatesTokenCount': 0}})

    def text(self, state, text):
        if not text:
            return b''
        if state == 'thinking':
            return self.chunk([{'thought': text}])
        return self.chunk([{'text': text}])

    def tool(self, tool):
        return self.chunk([{'functionCall': {'id': tool['id'], 'name': tool['name'], 'args': tool['input']}}])

    def end(self, has_tool):
        return self.chunk(finish_reason='TOOL_USE' if has_tool else 'STOP')

def stream_sse(format_type, seg_gen, msg_id, in_tokens, think_tags=None):
    adapters = {'claude': ClaudeAdapter, 'codex': CodexAdapter, 'gemini': GeminiAdapter, 'noapi': DefaultAdapter}
    adapter = adapters.get(format_type, CodexAdapter)(msg_id, in_tokens)
    yield adapter.start()
    state = 'thinking'
    buf = ''
    think_tags = ['<think>', '</think>'] if think_tags is None else think_tags
    for seg in seg_gen:
        buf += seg
        while any((t in seg for t in think_tags)):
            if state == 'text' and think_tags[0] in seg:
                before, _, seg = seg.partition(think_tags[0])
                if before:
                    yield adapter.text('text', before)
                state = 'thinking'
            if state == 'thinking' and think_tags[1] in seg:
                before, _, seg = seg.partition(think_tags[1])
                if before:
                    yield adapter.text('thinking', before)
                state = 'text'
        if seg:
            yield adapter.text(state, seg)
    if (tools := _parse_tools_xml(buf)):
        for tool in tools:
            yield adapter.tool(tool)
        yield adapter.end(True)
    else:
        yield adapter.end(False)

def make_handler(model_name, cache_dir, system, names, skips, gwt=None, parse_think=True):
    model, tokenizer = mlx_lm.load(model_name)
    pc = PromptCache(model, model_name=model_name, cache_dir=cache_dir)
    if not isinstance(tokenizer, mlx_lm.tokenizer_utils.TokenizerWrapper):
        tokenizer = mlx_lm.tokenizer_utils.TokenizerWrapper(tokenizer)
    gwt_cell = [gwt]

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, fmt, *args):
            pass

        def send_json(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip('/') == '/v1/models':
                self.send_json(200, {'data': [{'id': 'local', 'object': 'model', 'created': int(time.time()), 'owned_by': 'local'}]})
            else:
                self.send_json(404, {'error': 'not found'})

        def do_POST(self):
            try:
                path = self.path.split('?')[0].rstrip('/')
                if path == '/v1/messages/count_tokens':
                    self.send_json(200, {'input_tokens': 0})
                    return
                if path.startswith('/v1beta/models/'):
                    api = 'gemini'
                    if not ('alt=sse' in self.path or 'streamGenerateContent' in path):
                        n = int(self.headers.get('Content-Length', 0))
                        _ = self.rfile.read(n)
                        dummy = json.dumps({'candidates': [{'content': {'role': 'model', 'parts': [{'text': '{"complexity_reasoning":"local","complexity_score":50}'}]}, 'finishReason': 'STOP'}], 'usageMetadata': {'promptTokenCount': 0, 'candidatesTokenCount': 0}}).encode()
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Content-Length', str(len(dummy)))
                        self.end_headers()
                        self.wfile.write(dummy)
                        return
                elif path.startswith('/v1/messages'):
                    api = 'claude'
                elif path.startswith('/v1/responses'):
                    api = 'codex'
                else:
                    api = 'noapi'
                n = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(n))
                logger.debug(f'self.path={self.path!r}\n{json.dumps(body, indent=2)}')
                with gen_lock:
                    abort_ev.set()
                    abort_ev.clear()
                    prompt, ckpts = encode(body, api, tokenizer, system, names, skips)
                    if ckpts is not None:
                        gwt_cell[0], _ = commit_worktree(gwt_cell[0])
                    seg_gen = generate(model, tokenizer, prompt=prompt, ckpts=ckpts, pc=pc, max_tokens=body.get('max_tokens', 8192))
                    msg_id = f'msg_{uuid.uuid4().hex}'
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    try:
                        for chunk in stream_sse(api, seg_gen, msg_id, len(prompt)):
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    except BrokenPipeError:
                        seg_gen.close()
            except Exception as e:
                logger.exception(f'self.path={self.path!r}')
                raise
    return Handler

def _serve_cache(host, port, model, cache, system, tools, skips, *, fixed_port=False, gwt=None):
    handler = make_handler(model, cache, system, tools, skips, gwt)
    while True:
        try:
            server = HTTPServer((host, port), handler)
            url = f'http://{host}:{port}'
            logger.debug(f'Cache server bound to {url}')
            return (server, url)
        except OSError as e:
            if e.errno in (48, 98):
                if fixed_port:
                    logger.error(f'Port {port} is already in use.')
                    sys.exit(1)
                port += 1
            else:
                raise

def _serve_batch(host, port, model, cache_dir='.cache', *, fixed_port=False):
    import uvicorn
    from .bats import make_batch_app
    import socket
    import time
    app = make_batch_app(model, cache_dir=cache_dir)
    while True:
        try:
            with socket.socket() as s:
                s.bind((host, port))
        except OSError as e:
            if e.errno in (48, 98):
                if fixed_port:
                    logger.error(f'Port {port} is already in use.')
                    sys.exit(1)
                port += 1
            else:
                raise
        else:
            break
    config = uvicorn.Config(app, host=host, port=port, loop='asyncio', log_level='warning')
    uv_server = uvicorn.Server(config)
    t = threading.Thread(target=uv_server.run, daemon=True)
    t.start()
    start_time = time.time()
    notified = False
    while True:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                break
        except OSError:
            if not notified and time.time() - start_time > 3.0:
                logger.info('Waiting for batch server to start (model may be downloading)...')
                notified = True
            time.sleep(0.2)
    url = f'http://{host}:{port}'
    logger.debug(f'Batch server bound to {url}')
    return (uv_server, url)

def main():
    parser = argparse.ArgumentParser(description='mlx-code MAIN')
    parser.add_argument('-p', '--prompt', default=None, help='Initial prompt sent automatically when the REPL starts')
    parser.add_argument('-r', '--resume', default=None, metavar='COMMIT', help='Resume a previous session from the given git commit hash')
    parser.add_argument('-m', '--model', default='mlx-community/Qwen3.5-4B-OptiQ-4bit', help='MLX model path or HuggingFace repo ID (default: Qwen3.5-4B-OptiQ-4bit)')
    parser.add_argument('-l', '--leash', choices=['claude', 'codex', 'gemini', 'noapi', 'none'], default='noapi', help="AI harness to launch against the server; 'noapi' starts the built-in REPL, 'none' runs the server only")
    parser.add_argument('--engine', choices=['cache', 'batch'], default='cache', help="'cache' uses PromptCache + single-sequence (default); 'batch' uses BatchGenerator for concurrent sequences (only compatible with --leash none or noapi)")
    parser.add_argument('--skill', default=None, help='Directory to scan recursively for SKILL.md files')
    parser.add_argument('--tools', nargs='+', default=None, help='Whitelist of tool names to enable; allows all tools when omitted')
    parser.add_argument('--system', type=str, default=None, help='System prompt override passed to the model')
    parser.add_argument('--cache', type=str, default='.cache', help='Directory for the KV-cache (default: .cache)')
    parser.add_argument('--work', default=os.getcwd(), help='Working directory used as the git repo root (default: cwd)')
    parser.add_argument('--host', default='127.0.0.1', help='Host address to bind the server to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=None, help='Port to listen on; auto-increments if already in use (default: 8000)')
    parser.add_argument('--skips', nargs='+', default=['(?m)^\\[SUGGESTION MODE[\\s\\S]*', '(?m)^<system-reminder>[\\s\\S]*?^</system-reminder>\\s*'], help='Regex patterns stripped from model output before it is returned to the client')
    parser.add_argument('--stream', default=None, help='File to stream log into')
    parser.add_argument('--bare', action='store_true', help='Use simple terminal REPL instead of TUI')
    parser.add_argument('--web', action='store_true', help='Use web UI instead of TUI')
    parser.add_argument('--web-port', type=int, default=None, help='Port for web UI (default: inference port + 80)')
    args, leash_args = parser.parse_known_args()
    logger.debug(f'args={args!r} leash_args={leash_args!r}')
    if args.engine == 'batch' and args.leash not in ('none', 'noapi'):
        parser.error('--engine batch only supports --leash none or --leash noapi for now')
    cache = os.path.abspath(args.cache)
    port = args.port if args.port is not None else 8000
    fixed_port = args.port is not None
    with tempfile.TemporaryDirectory(dir='/tmp') as _home:
        env = os.environ.copy()
        home = Path(_home)
        gwt = None if args.leash in ('none', 'noapi') else create_worktree(args.work, worktree_dir=str(home / 'workspace'))
        cwd = args.work if gwt is None else gwt.worktree
        env['HOME'] = _home
        env['SHELL'] = '/bin/bash'
        env['PWD'] = cwd
        if args.engine == 'batch':
            server, url = _serve_batch(args.host, port, args.model, cache_dir=cache, fixed_port=fixed_port)
        else:
            server, url = _serve_cache(host=args.host, port=port, model=args.model, cache=cache, system=None if args.leash in ('none', 'noapi') else args.system, tools=args.tools, skips=args.skips, fixed_port=fixed_port, gwt=gwt)
        if args.leash == 'none':
            if args.engine == 'batch':
                try:
                    threading.Event().wait()
                except KeyboardInterrupt:
                    print('\nShutting down server...')
            else:
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    print('\nShutting down server...')
                    server.server_close()
        else:
            if args.engine == 'cache':
                threading.Thread(target=server.serve_forever, daemon=True).start()
            if args.leash == 'noapi':
                if args.web:
                    from .web import run_web
                    web_port = args.web_port if args.web_port is not None else port + 80
                    run_web(base_url=url, api=args.leash, repo=cwd, env=env, system=args.system, tool_names=args.tools, sdir=args.skill, init_prompt=args.prompt, resume=args.resume, stream=args.stream, host=args.host, port=web_port)
                else:
                    from .repl import run_repl
                    run_repl(base_url=url, api=args.leash, repo=cwd, env=env, system=args.system, tool_names=args.tools, sdir=args.skill, init_prompt=args.prompt, resume=args.resume, stream=args.stream, bare=args.bare)
            else:
                env['GOOGLE_GEMINI_BASE_URL'] = url
                env['GEMINI_API_KEY'] = 'mc'
                env['ANTHROPIC_BASE_URL'] = url
                env['ANTHROPIC_AUTH_TOKEN'] = 'mc'
                env['ANTHROPIC_MODEL'] = args.model
                if args.leash == 'codex':
                    env.pop('OPENAI_API_KEY', None)
                    codex_dir = home / '.codex'
                    codex_dir.mkdir(parents=True, exist_ok=True)
                    (codex_dir / 'config.toml').write_text(f'\n[model_providers.local]\nname = "openai"\nbase_url = "{url}/v1"\napi_key = "mc"\n \n[profiles.local]\nmodel_provider = "local"\nmodel = "gpt-5.4-mini"\n'.strip())
                    leash_args = leash_args + ['--profile', 'local']
                sys.exit(subprocess.run([args.leash] + leash_args, env=env, cwd=cwd).returncode)
if __name__ == '__main__':
    main()
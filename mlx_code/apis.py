from __future__ import annotations
import asyncio
import json
import os
import time
import uuid
import logging
import httpx
from typing import Any, Literal
from .tools import Tool

logger = logging.getLogger(__name__)


class EventStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._result: dict | None = None
        self._task: asyncio.Task | None = None

    def _attach(self, task: asyncio.Task) -> None:
        self._task = task

    def push(self, event: dict) -> None:
        self._queue.put_nowait(event)

    def finish(self, result: dict) -> None:
        self._result = result
        self._queue.put_nowait(None)

    async def result(self) -> dict:
        if self._result is None:
            async for _ in self:
                pass
        assert self._result is not None
        return self._result

    def __aiter__(self) -> "EventStream":
        return self

    async def __anext__(self) -> dict:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item


_REASONING_BUDGET: dict[str, int] = {
    "minimal": 512,
    "low": 1024,
    "medium": 8192,
    "high": 16000,
    "xhigh": 32000,
}


class ClaudeChat:
    def __init__(
        self,
        *,
        model=None,
        api_key=None,
        base_url=None,
        max_tokens=8192,
        temperature=None,
        reasoning: Literal["off", "minimal", "low", "medium", "high", "xhigh"] = "off",
        tool_choice=None,
    ):
        self.model = model or "claude-haiku-4-5"
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.tool_choice = tool_choice
        if not self.api_key:
            logger.debug("No api key")

    def _fmt_content(self, content) -> str | list:
        if isinstance(content, str):
            return content
        out = []
        for b in content:
            t = b["type"]
            if t == "text":
                blk = {"type": "text", "text": b["text"]}
                if b.get("cache_control"):
                    blk["cache_control"] = {"type": b["cache_control"]}
                out.append(blk)
            elif t == "image":
                blk = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": b["mime_type"],
                        "data": b["data"],
                    },
                }
                if b.get("cache_control"):
                    blk["cache_control"] = {"type": b["cache_control"]}
                out.append(blk)
            elif t == "thinking":
                if b["redacted"]:
                    out.append({"type": "redacted_thinking", "data": b["thinking"]})
                else:
                    out.append(
                        {
                            "type": "thinking",
                            "thinking": b["thinking"],
                            "signature": b.get("signature") or "",
                        }
                    )
            elif t == "toolCall":
                out.append(
                    {
                        "type": "tool_use",
                        "id": b["id"],
                        "name": b["name"],
                        "input": b["arguments"],
                    }
                )
        return out

    def _build_messages(self, messages: list[dict]) -> list[dict]:
        out = []
        i = 0
        while i < len(messages):
            m = messages[i]
            if m["role"] == "user":
                out.append({"role": "user", "content": self._fmt_content(m["content"])})
                i += 1
            elif m["role"] == "assistant":
                out.append(
                    {"role": "assistant", "content": self._fmt_content(m["content"])}
                )
                i += 1
            elif m["role"] == "toolResult":
                batch = []
                while i < len(messages) and messages[i]["role"] == "toolResult":
                    tr = messages[i]
                    batch.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tr["tool_call_id"],
                            "content": self._fmt_content(tr["content"]),
                            "is_error": tr["is_error"],
                        }
                    )
                    i += 1
                out.append({"role": "user", "content": batch})
            else:
                i += 1
        return out

    async def stream(
        self, messages: list[dict], system: str, tools: list[Tool]
    ) -> EventStream:
        es = EventStream()
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._build_messages(messages),
            "stream": True,
        }
        if system:
            payload["system"] = [{"type": "text", "text": system}]
        if self.temperature:
            payload["temperature"] = self.temperature
        if self.reasoning != "off":
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": _REASONING_BUDGET[self.reasoning],
            }
        if tools:
            payload["tools"] = [t.schema() for t in tools]
            tc = self.tool_choice
            if tc is not None:
                payload["tool_choice"] = (
                    {"type": "tool", "name": tc["name"]}
                    if isinstance(tc, dict)
                    else {"type": "any"}
                    if tc == "required"
                    else {"type": "none"}
                    if tc == "none"
                    else {"type": "auto"}
                )
        headers = {
            "x-api-key": str(self.api_key),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        logger.debug(json.dumps(payload, indent=2))

        async def _run() -> None:
            msg = {
                "role": "assistant",
                "content": [],
                "stop_reason": "stop",
                "usage": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                "error_message": None,
                "timestamp": int(time.time() * 1000),
            }
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/v1/messages",
                        json=payload,
                        headers=headers,
                    ) as resp:
                        if resp.status_code >= 400:
                            raise RuntimeError(
                                f"HTTP {resp.status_code}: {(await resp.aread()).decode()}"
                            )
                        es.push({"type": "start", "payload": {"partial": msg}})
                        _tool_buf: dict[int, str] = {}
                        _idx: dict[int, int] = {}
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            chunk = json.loads(data)
                            et = chunk.get("type")
                            if et == "message_start":
                                u = chunk.get("message", {}).get("usage", {})
                                msg["usage"]["input"] = u.get("input_tokens", 0)
                                msg["usage"]["cache_read"] = u.get(
                                    "cache_read_input_tokens", 0
                                )
                                msg["usage"]["cache_write"] = u.get(
                                    "cache_creation_input_tokens", 0
                                )
                            elif et == "content_block_start":
                                idx = chunk.get("index", 0)
                                blk = chunk.get("content_block", {})
                                bt = blk.get("type")
                                if bt == "text":
                                    _idx[idx] = len(msg["content"])
                                    msg["content"].append({"type": "text", "text": ""})
                                elif bt == "thinking":
                                    _idx[idx] = len(msg["content"])
                                    msg["content"].append(
                                        {
                                            "type": "thinking",
                                            "thinking": "",
                                            "signature": None,
                                            "redacted": False,
                                        }
                                    )
                                elif bt == "redacted_thinking":
                                    _idx[idx] = len(msg["content"])
                                    msg["content"].append(
                                        {
                                            "type": "thinking",
                                            "thinking": "",
                                            "signature": None,
                                            "redacted": True,
                                        }
                                    )
                                elif bt == "tool_use":
                                    _idx[idx] = len(msg["content"])
                                    msg["content"].append(
                                        {
                                            "type": "toolCall",
                                            "id": blk["id"],
                                            "name": blk["name"],
                                            "arguments": {},
                                        }
                                    )
                                    _tool_buf[idx] = ""
                            elif et == "content_block_delta":
                                idx = chunk.get("index", 0)
                                delta = chunk.get("delta", {})
                                dt = delta.get("type")
                                pos = _idx.get(idx)
                                if pos is None:
                                    continue
                                b = msg["content"][pos]
                                if dt == "text_delta":
                                    d = delta.get("text", "")
                                    b["text"] += d
                                    es.push(
                                        {
                                            "type": "text_delta",
                                            "payload": {"delta": d, "partial": msg},
                                        }
                                    )
                                elif dt == "thinking_delta":
                                    d = delta.get("thinking", "")
                                    b["thinking"] += d
                                    es.push(
                                        {
                                            "type": "thinking_delta",
                                            "payload": {"delta": d, "partial": msg},
                                        }
                                    )
                                elif dt == "signature_delta":
                                    b["signature"] = (
                                        b.get("signature") or ""
                                    ) + delta.get("signature", "")
                                elif dt == "input_json_delta":
                                    _tool_buf[idx] = _tool_buf.get(idx, "") + delta.get(
                                        "partial_json", ""
                                    )
                            elif et == "content_block_stop":
                                idx = chunk.get("index", 0)
                                pos = _idx.get(idx)
                                if pos is not None and pos < len(msg["content"]):
                                    b = msg["content"][pos]
                                    if b["type"] == "toolCall":
                                        raw = _tool_buf.pop(idx, "")
                                        if raw:
                                            try:
                                                b["arguments"] = json.loads(raw)
                                            except json.JSONDecodeError:
                                                pass
                                        es.push(
                                            {
                                                "type": "toolcall_end",
                                                "payload": {
                                                    "tool_call": b,
                                                    "partial": msg,
                                                },
                                            }
                                        )
                            elif et == "message_delta":
                                reason = chunk.get("delta", {}).get("stop_reason")
                                if reason == "tool_use":
                                    msg["stop_reason"] = "tool_use"
                                elif reason == "max_tokens":
                                    msg["stop_reason"] = "length"
                                elif reason:
                                    msg["stop_reason"] = "stop"
                                u = chunk.get("usage", {})
                                if u:
                                    msg["usage"]["output"] = u.get(
                                        "output_tokens", msg["usage"]["output"]
                                    )
                            elif et == "message_stop":
                                es.push(
                                    {
                                        "type": "done",
                                        "payload": {
                                            "reason": msg["stop_reason"],
                                            "message": msg,
                                        },
                                    }
                                )
            except Exception as exc:
                msg["stop_reason"] = "error"
                msg["error_message"] = str(exc)
                es.push({"type": "error", "payload": {"error": msg}})
            es.finish(msg)

        es._attach(asyncio.create_task(_run()))
        return es


class DefaultChat:
    def __init__(
        self,
        *,
        model=None,
        api_key=None,
        base_url=None,
        max_tokens=8192,
        temperature=None,
        tool_choice=None,
    ):
        self.model = model or "noapi"
        self.api_key = api_key or "noapi"
        self.base_url = (base_url or "http://127.0.0.1:8000").rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tool_choice = tool_choice
        if not api_key:
            logger.debug("No api key")

    def _build_messages(self, messages: list[dict], system: str) -> list[dict]:
        out = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            if m["role"] == "user":
                content = m["content"]
                if isinstance(content, str):
                    out.append({"role": "user", "content": content})
                else:
                    out.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": b["text"]}
                                if b["type"] == "text"
                                else {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{b['mime_type']};base64,{b['data']}"
                                    },
                                }
                                for b in content
                            ],
                        }
                    )
            elif m["role"] == "assistant":
                texts = [b for b in m["content"] if b["type"] == "text"]
                calls = [b for b in m["content"] if b["type"] == "toolCall"]
                thinking = [b for b in m["content"] if b["type"] == "thinking"]
                mo: dict[str, Any] = {"role": "assistant"}
                if texts:
                    mo["content"] = "".join((b["text"] for b in texts))
                if thinking:
                    mo["reasoning_content"] = "".join((b["thinking"] for b in thinking))
                if calls:
                    mo["tool_calls"] = [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(c["arguments"]),
                            },
                        }
                        for c in calls
                    ]
                out.append(mo)
            elif m["role"] == "toolResult":
                c = m["content"]
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m["tool_call_id"],
                        "content": c[0]["text"]
                        if len(c) == 1 and c[0]["type"] == "text"
                        else [
                            {"type": "text", "text": b["text"]}
                            if b["type"] == "text"
                            else {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{b['mime_type']};base64,{b['data']}"
                                },
                            }
                            for b in c
                        ],
                    }
                )
        return out

    async def stream(
        self, messages: list[dict], system: str, tools: list[Tool]
    ) -> EventStream:
        es = EventStream()
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._build_messages(messages, system),
            "stream": True,
        }
        if self.temperature:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": {
                            "type": "object",
                            "properties": t.parameters.model_json_schema().get(
                                "properties", {}
                            ),
                            "required": t.parameters.model_json_schema().get(
                                "required", []
                            ),
                        },
                    },
                }
                for t in tools
            ]
            tc = self.tool_choice
            if tc is not None:
                payload["tool_choice"] = (
                    {"type": "function", "function": {"name": tc["name"]}}
                    if isinstance(tc, dict)
                    else tc
                )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        logger.debug(json.dumps(payload, indent=2))

        async def _run() -> None:
            msg = {
                "role": "assistant",
                "content": [],
                "stop_reason": "stop",
                "usage": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                "error_message": None,
                "timestamp": int(time.time() * 1000),
            }
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                        headers=headers,
                    ) as resp:
                        if resp.status_code >= 400:
                            raise RuntimeError(
                                f"HTTP {resp.status_code}: {(await resp.aread()).decode()}"
                            )
                        es.push({"type": "start", "payload": {"partial": msg}})
                        _tc: dict[int, dict] = {}
                        _text_buf = _think_buf = ""
                        finish = None
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:].strip()
                            if data == "[DONE]" or not data:
                                continue
                            chunk = json.loads(data)
                            choice = (chunk.get("choices") or [{}])[0]
                            delta = choice.get("delta", {})
                            finish = choice.get("finish_reason") or finish
                            if r := (
                                delta.get("reasoning_content") or delta.get("thinking")
                            ):
                                _think_buf += r
                                es.push(
                                    {
                                        "type": "thinking_delta",
                                        "payload": {"delta": r, "partial": msg},
                                    }
                                )
                            if d := delta.get("content"):
                                _text_buf += d
                                es.push(
                                    {
                                        "type": "text_delta",
                                        "payload": {"delta": d, "partial": msg},
                                    }
                                )
                            for tcd in delta.get("tool_calls") or []:
                                idx = tcd.get("index", 0)
                                fn = tcd.get("function", {})
                                if idx not in _tc:
                                    _tc[idx] = {
                                        "id": tcd.get("id", str(uuid.uuid4())),
                                        "name": fn.get("name", ""),
                                        "args_buf": fn.get("arguments", ""),
                                    }
                                else:
                                    if "name" in fn:
                                        _tc[idx]["name"] += fn["name"]
                                    if "arguments" in fn:
                                        _tc[idx]["args_buf"] += fn["arguments"]
                            if u := chunk.get("usage"):
                                msg["usage"]["input"] = u.get(
                                    "prompt_tokens", msg["usage"]["input"]
                                )
                                msg["usage"]["output"] = u.get(
                                    "completion_tokens", msg["usage"]["output"]
                                )
                        if _think_buf:
                            msg["content"].append(
                                {
                                    "type": "thinking",
                                    "thinking": _think_buf,
                                    "signature": None,
                                    "redacted": False,
                                }
                            )
                        if _text_buf:
                            msg["content"].append({"type": "text", "text": _text_buf})
                        for _, s in sorted(_tc.items()):
                            try:
                                args = (
                                    json.loads(s["args_buf"]) if s["args_buf"] else {}
                                )
                            except json.JSONDecodeError:
                                args = {}
                            call = {
                                "type": "toolCall",
                                "id": s["id"],
                                "name": s["name"],
                                "arguments": args,
                            }
                            msg["content"].append(call)
                            es.push(
                                {
                                    "type": "toolcall_end",
                                    "payload": {"tool_call": call, "partial": msg},
                                }
                            )
                        msg["stop_reason"] = (
                            "tool_use"
                            if finish == "tool_calls"
                            else "length"
                            if finish in ("length", "max_tokens")
                            else "stop"
                        )
                        es.push(
                            {
                                "type": "done",
                                "payload": {
                                    "reason": msg["stop_reason"],
                                    "message": msg,
                                },
                            }
                        )
            except Exception as exc:
                msg["stop_reason"] = "error"
                msg["error_message"] = str(exc)
                es.push({"type": "error", "payload": {"error": msg}})
            es.finish(msg)

        es._attach(asyncio.create_task(_run()))
        return es


class GeminiChat:
    def __init__(
        self,
        *,
        model=None,
        api_key=None,
        base_url=None,
        max_tokens=8192,
        temperature=None,
        thinking=False,
        thinking_budget=8192,
        tool_choice=None,
    ):
        self.model = model or "gemini-3.1-flash-lite-preview"
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.base_url = (
            base_url or "https://generativelanguage.googleapis.com"
        ).rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.tool_choice = tool_choice
        if not self.api_key:
            logger.debug("No api key")

    def _build_contents(self, messages: list[dict]) -> list[dict]:
        out = []
        pending: list[dict] = []

        def flush():
            if pending:
                out.append({"role": "user", "parts": list(pending)})
                pending.clear()

        for m in messages:
            if m["role"] == "user":
                flush()
                content = m["content"]
                if isinstance(content, str):
                    out.append({"role": "user", "parts": [{"text": content}]})
                else:
                    out.append(
                        {
                            "role": "user",
                            "parts": [
                                {"text": b["text"]}
                                if b["type"] == "text"
                                else {
                                    "inlineData": {
                                        "mimeType": b["mime_type"],
                                        "data": b["data"],
                                    }
                                }
                                for b in content
                            ],
                        }
                    )
            elif m["role"] == "assistant":
                flush()
                parts = []
                for b in m["content"]:
                    if b["type"] == "thinking":
                        parts.append({"thought": b["thinking"]})
                    elif b["type"] == "text":
                        parts.append({"text": b["text"]})
                    elif b["type"] == "toolCall":
                        fc: dict = {"name": b["name"], "args": b["arguments"]}
                        if b["id"] != b["name"]:
                            fc["id"] = b["id"]
                        parts.append({"functionCall": fc})
                if parts:
                    out.append({"role": "model", "parts": parts})
            elif m["role"] == "toolResult":
                c = m["content"]
                try:
                    body = json.loads(c[0]["text"]) if c else {}
                except (json.JSONDecodeError, KeyError):
                    body = {"result": c[0]["text"] if c else ""}
                fr: dict = {"name": m["tool_name"], "response": body}
                if m["tool_call_id"] != m["tool_name"]:
                    fr["id"] = m["tool_call_id"]
                pending.append({"functionResponse": fr})
        flush()
        return out

    async def stream(
        self, messages: list[dict], system: str, tools: list[Tool]
    ) -> EventStream:
        es = EventStream()
        payload: dict[str, Any] = {
            "contents": self._build_contents(messages),
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if self.temperature:
            payload["generationConfig"]["temperature"] = self.temperature
        if self.thinking:
            payload["generationConfig"]["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingBudget": self.thinking_budget,
            }
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": {
                                "type": "object",
                                "properties": t.parameters.model_json_schema().get(
                                    "properties", {}
                                ),
                                "required": t.parameters.model_json_schema().get(
                                    "required", []
                                ),
                            },
                        }
                        for t in tools
                    ]
                }
            ]
            tc = self.tool_choice
            if tc is not None:
                payload["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [tc["name"]],
                    }
                    if isinstance(tc, dict)
                    else {"mode": "ANY"}
                    if tc == "required"
                    else {"mode": "NONE"}
                    if tc == "none"
                    else {"mode": "AUTO"}
                }
        url = f"{self.base_url}/v1beta/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        logger.debug(json.dumps(payload, indent=2))

        async def _run() -> None:
            msg = {
                "role": "assistant",
                "content": [],
                "stop_reason": "stop",
                "usage": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                "error_message": None,
                "timestamp": int(time.time() * 1000),
            }
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        if resp.status_code >= 400:
                            raise RuntimeError(
                                f"HTTP {resp.status_code}: {(await resp.aread()).decode()}"
                            )
                        es.push({"type": "start", "payload": {"partial": msg}})
                        _text = _think = ""
                        _calls: dict[str, dict] = {}
                        finish = None
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:].strip()
                            if not data:
                                continue
                            chunk = json.loads(data)
                            cand = (chunk.get("candidates") or [{}])[0]
                            finish = cand.get("finishReason") or finish
                            for part in cand.get("content", {}).get("parts") or []:
                                if "thought" in part:
                                    _think += part["thought"]
                                    es.push(
                                        {
                                            "type": "thinking_delta",
                                            "payload": {
                                                "delta": part["thought"],
                                                "partial": msg,
                                            },
                                        }
                                    )
                                elif "text" in part:
                                    _text += part["text"]
                                    es.push(
                                        {
                                            "type": "text_delta",
                                            "payload": {
                                                "delta": part["text"],
                                                "partial": msg,
                                            },
                                        }
                                    )
                                elif "functionCall" in part:
                                    fc = part["functionCall"]
                                    cid = fc.get("id") or fc["name"]
                                    if cid not in _calls:
                                        _calls[cid] = {
                                            "name": fc["name"],
                                            "args": fc.get("args", {}),
                                        }
                                    else:
                                        _calls[cid]["args"].update(fc.get("args", {}))
                            if u := chunk.get("usageMetadata"):
                                msg["usage"]["input"] = u.get(
                                    "promptTokenCount", msg["usage"]["input"]
                                )
                                msg["usage"]["output"] = u.get(
                                    "candidatesTokenCount", msg["usage"]["output"]
                                )
                        if _think:
                            msg["content"].append(
                                {
                                    "type": "thinking",
                                    "thinking": _think,
                                    "signature": None,
                                    "redacted": False,
                                }
                            )
                        if _text:
                            msg["content"].append({"type": "text", "text": _text})
                        for cid, s in _calls.items():
                            call = {
                                "type": "toolCall",
                                "id": cid,
                                "name": s["name"],
                                "arguments": s["args"],
                            }
                            msg["content"].append(call)
                            es.push(
                                {
                                    "type": "toolcall_end",
                                    "payload": {"tool_call": call, "partial": msg},
                                }
                            )
                        msg["stop_reason"] = (
                            "length"
                            if finish == "MAX_TOKENS"
                            else "tool_use"
                            if _calls
                            else "stop"
                        )
                        es.push(
                            {
                                "type": "done",
                                "payload": {
                                    "reason": msg["stop_reason"],
                                    "message": msg,
                                },
                            }
                        )
                        es.finish(msg)
                        return
            except Exception as exc:
                msg["stop_reason"] = "error"
                msg["error_message"] = str(exc)
                es.push({"type": "error", "payload": {"error": msg}})
            es.finish(msg)

        es._attach(asyncio.create_task(_run()))
        return es


class CodexChat:
    def __init__(
        self,
        *,
        model=None,
        api_key=None,
        base_url=None,
        max_tokens=8192,
        temperature=None,
        tool_choice=None,
    ):
        self.model = model or "gpt-5.4-mini"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or "https://api.openai.com").rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tool_choice = tool_choice
        if not self.api_key:
            logger.debug("No api key")

    def _build_input(self, messages: list[dict], system: str) -> list[dict]:
        out = []
        if system:
            out.append({"type": "message", "role": "developer", "content": system})
        for m in messages:
            if m["role"] == "user":
                c = m["content"]
                out.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": c
                        if isinstance(c, str)
                        else [
                            {"type": "input_text", "text": b["text"]}
                            if b["type"] == "text"
                            else {
                                "type": "input_image",
                                "image_url": f"data:{b['mime_type']};base64,{b['data']}",
                            }
                            for b in c
                        ],
                    }
                )
            elif m["role"] == "assistant":
                for b in m["content"]:
                    if b["type"] == "text":
                        out.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": b["text"]}],
                            }
                        )
                    elif b["type"] == "toolCall":
                        out.append(
                            {
                                "type": "function_call",
                                "call_id": b["id"],
                                "name": b["name"],
                                "arguments": json.dumps(b["arguments"]),
                            }
                        )
            elif m["role"] == "toolResult":
                out.append(
                    {
                        "type": "function_call_output",
                        "call_id": m["tool_call_id"],
                        "output": m["content"][0]["text"] if m["content"] else "",
                    }
                )
        return out

    async def stream(
        self, messages: list[dict], system: str, tools: list[Tool]
    ) -> EventStream:
        es = EventStream()
        payload: dict[str, Any] = {
            "model": self.model,
            "max_output_tokens": self.max_tokens,
            "input": self._build_input(messages, system),
            "stream": True,
        }
        if self.temperature:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": t.parameters.model_json_schema().get(
                            "properties", {}
                        ),
                        "required": t.parameters.model_json_schema().get(
                            "required", []
                        ),
                    },
                }
                for t in tools
            ]
            tc = self.tool_choice
            if tc is not None:
                payload["tool_choice"] = (
                    {"type": "function", "name": tc["name"]}
                    if isinstance(tc, dict)
                    else tc
                )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        logger.debug(json.dumps(payload, indent=2))

        async def _run() -> None:
            msg = {
                "role": "assistant",
                "content": [],
                "stop_reason": "stop",
                "usage": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                "error_message": None,
                "timestamp": int(time.time() * 1000),
            }
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/v1/responses",
                        json=payload,
                        headers=headers,
                    ) as resp:
                        if resp.status_code >= 400:
                            raise RuntimeError(
                                f"HTTP {resp.status_code}: {(await resp.aread()).decode()}"
                            )
                        es.push({"type": "start", "payload": {"partial": msg}})
                        _text = ""
                        _calls: dict[str, dict] = {}
                        finish = None
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:].strip()
                            if not data:
                                continue
                            chunk = json.loads(data)
                            et = chunk.get("type", "")
                            if et == "response.output_text.delta":
                                d = chunk.get("delta", "")
                                _text += d
                                es.push(
                                    {
                                        "type": "text_delta",
                                        "payload": {"delta": d, "partial": msg},
                                    }
                                )
                            elif et == "response.output_item.added":
                                item = chunk.get("item", {})
                                if item.get("type") == "function_call":
                                    iid = item["id"]
                                    _calls[iid] = {
                                        "name": item.get("name", ""),
                                        "call_id": item.get("call_id", iid),
                                        "args_buf": "",
                                    }
                            elif et == "response.function_call_arguments.delta":
                                iid = chunk.get("item_id", "")
                                if iid in _calls:
                                    _calls[iid]["args_buf"] += chunk.get("delta", "")
                            elif et == "response.completed":
                                finish = chunk.get("response", {}).get("status")
                                if u := chunk.get("response", {}).get("usage"):
                                    msg["usage"]["input"] = u.get(
                                        "input_tokens", msg["usage"]["input"]
                                    )
                                    msg["usage"]["output"] = u.get(
                                        "output_tokens", msg["usage"]["output"]
                                    )
                        if _text:
                            msg["content"].append({"type": "text", "text": _text})
                        for s in _calls.values():
                            try:
                                args = (
                                    json.loads(s["args_buf"]) if s["args_buf"] else {}
                                )
                            except json.JSONDecodeError:
                                args = {}
                            call = {
                                "type": "toolCall",
                                "id": s["call_id"],
                                "name": s["name"],
                                "arguments": args,
                            }
                            msg["content"].append(call)
                            es.push(
                                {
                                    "type": "toolcall_end",
                                    "payload": {"tool_call": call, "partial": msg},
                                }
                            )
                        msg["stop_reason"] = (
                            "tool_use"
                            if _calls
                            else "length"
                            if finish == "incomplete"
                            else "stop"
                        )
                        es.push(
                            {
                                "type": "done",
                                "payload": {
                                    "reason": msg["stop_reason"],
                                    "message": msg,
                                },
                            }
                        )
            except Exception as exc:
                msg["stop_reason"] = "error"
                msg["error_message"] = str(exc)
                es.push({"type": "error", "payload": {"error": msg}})
            es.finish(msg)

        es._attach(asyncio.create_task(_run()))
        return es


def resolve_api(api, *, model, api_key, base_url):
    if not isinstance(api, str):
        return api
    if api == "claude":
        return ClaudeChat(model=model, api_key=api_key, base_url=base_url)
    if api == "gemini":
        return GeminiChat(model=model, api_key=api_key, base_url=base_url)
    if api == "codex":
        return CodexChat(model=model, api_key=api_key, base_url=base_url)
    return DefaultChat(model=model, api_key=api_key, base_url=base_url)

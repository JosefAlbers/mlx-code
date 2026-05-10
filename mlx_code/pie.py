# Based on pi (https://github.com/badlogic/pi-mono) by Mario Zechner (MIT License)
#
# Copyright 2026 J Joe
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import asyncio
import argparse
import fnmatch
import json
import os
import pathlib
import re
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, TypeAlias, Union
import httpx
from pydantic import BaseModel, Field, ValidationError
from .ledger import create_worktree, commit_worktree #.
import sys, tty, termios
import logging

logger = logging.getLogger(__name__) 


@dataclass
class TextContent:
    text: str
    type: Literal["text"] = "text"
    cache_control: str | None = None


@dataclass
class ThinkingContent:
    thinking: str
    type: Literal["thinking"] = "thinking"
    signature: str | None = None
    redacted: bool = False


@dataclass
class ImageContent:
    data: str 
    mime_type: str
    type: Literal["image"] = "image"
    cache_control: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["toolCall"] = "toolCall"

@dataclass
class UserMessage:
    content: str | list[TextContent | ImageContent]
    role: Literal["user"] = "user"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

@dataclass
class AssistantMessage:
    content: list[TextContent | ThinkingContent | ToolCall] = field(default_factory=list)
    stop_reason: "StopReason" = "stop"
    usage: "Usage" = field(default_factory=lambda: Usage())
    error_message: str | None = None
    role: Literal["assistant"] = "assistant"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

@dataclass
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent]
    is_error: bool = False
    role: Literal["toolResult"] = "toolResult"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


Message: TypeAlias = Union[UserMessage, AssistantMessage, ToolResultMessage]
StopReason: TypeAlias = Literal["stop", "length", "tool_use", "error", "aborted"]

@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

EventType: TypeAlias = Literal[
    "start",
    "text_delta",
    "thinking_delta",
    "toolcall_end",
    "done",
    "error",
]


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)


class EventStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._result: AssistantMessage | None = None
        self._task: asyncio.Task | None = None

    def _attach(self, task: asyncio.Task) -> None:
        self._task = task

    def push(self, event: Event) -> None:
        self._queue.put_nowait(event)

    def finish(self, result: AssistantMessage) -> None:
        self._result = result
        self._queue.put_nowait(None)

    async def result(self) -> AssistantMessage:
        if self._result is None:
            async for _ in self:
                pass
        assert self._result is not None
        return self._result

    def __aiter__(self) -> "EventStream":
        return self

    async def __anext__(self) -> Event:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

@dataclass
class ToolResult:
    content: list[TextContent | ImageContent]
    is_error: bool = False


class Tool(ABC):
    name: str
    description: str
    parameters: type[BaseModel]

    @abstractmethod
    async def execute(
        self,
        params: BaseModel,
        signal: asyncio.Event | None = None,
    ) -> ToolResult: ...

    def schema(self) -> dict[str, Any]:
        s = self.parameters.model_json_schema()
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": s.get("properties", {}),
                "required": s.get("required", []),
                **( {"$defs": s["$defs"]} if "$defs" in s else {} ),
            },
        }


def validate_tool_call(tool: Tool, call: ToolCall) -> BaseModel:
    try:
        return tool.parameters.model_validate(call.arguments)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        raise ValueError(f"Invalid arguments for '{tool.name}': {details}")

_REASONING_BUDGET: dict[str, int] = {
    "minimal": 512,
    "low":     1_024,
    "medium":  8_192,
    "high":    16_000,
    "xhigh":   32_000,
}

class ClaudeChat:
    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 8_192,
        temperature: float | None = None,
        reasoning: Literal["off", "minimal", "low", "medium", "high", "xhigh"] = "off",
        tool_choice: Any = None,
    ) -> None:
        self.model = "claude-haiku-4-5" if model is None else model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY") if api_key is None else api_key
        if not self.api_key:
            logger.debug('No api key')
        self.base_url = "https://api.anthropic.com" if base_url is None else base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.tool_choice = tool_choice

    def _fmt_content(
        self,
        content: str | list,
    ) -> str | list[dict]:
        if isinstance(content, str):
            return content
        out = []
        for item in content:
            if isinstance(item, TextContent):
                blk: dict = {"type": "text", "text": item.text}
                if item.cache_control:
                    blk["cache_control"] = {"type": item.cache_control}
                out.append(blk)
            elif isinstance(item, ImageContent):
                blk = {
                    "type": "image",
                    "source": {"type": "base64", "media_type": item.mime_type, "data": item.data},
                }
                if item.cache_control:
                    blk["cache_control"] = {"type": item.cache_control}
                out.append(blk)
            elif isinstance(item, ThinkingContent):
                if item.redacted:
                    out.append({"type": "redacted_thinking", "data": item.thinking})
                else:
                    out.append({"type": "thinking", "thinking": item.thinking, "signature": item.signature or ""})
            elif isinstance(item, ToolCall):
                out.append({"type": "tool_use", "id": item.id, "name": item.name, "input": item.arguments})
        return out

    def _build_messages(self, messages: list[Message]) -> list[dict]:
        out = []
        i = 0
        while i < len(messages):
            m = messages[i]
            if isinstance(m, UserMessage):
                out.append({"role": "user", "content": self._fmt_content(m.content)})
                i += 1
            elif isinstance(m, AssistantMessage):
                out.append({"role": "assistant", "content": self._fmt_content(m.content)})
                i += 1
            elif isinstance(m, ToolResultMessage):
                batch = []
                while i < len(messages) and isinstance(messages[i], ToolResultMessage):
                    tr = messages[i]
                    assert isinstance(tr, ToolResultMessage)
                    batch.append({
                        "type": "tool_result",
                        "tool_use_id": tr.tool_call_id,
                        "content": self._fmt_content(tr.content),
                        "is_error": tr.is_error,
                    })
                    i += 1
                out.append({"role": "user", "content": batch})
            else:
                i += 1
        return out

    async def stream(
        self,
        messages: list[Message],
        system: str,
        tools: list[Tool],
    ) -> EventStream:
        es = EventStream()

        payload: dict[str, Any] = {
            "model":      self.model,
            "max_tokens": self.max_tokens,
            "messages":   self._build_messages(messages),
            "stream":     True,
        }
        if system:
            payload["system"] = [{"type": "text", "text": system}]
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning != "off":
            payload["thinking"] = {"type": "enabled", "budget_tokens": _REASONING_BUDGET[self.reasoning]}
        if tools:
            payload["tools"] = [t.schema() for t in tools]
            if self.tool_choice is not None:
                tc = self.tool_choice
                if isinstance(tc, dict):
                    payload["tool_choice"] = {"type": "tool", "name": tc["name"]}
                elif tc == "required":
                    payload["tool_choice"] = {"type": "any"}
                elif tc == "none":
                    payload["tool_choice"] = {"type": "none"}
                else:
                    payload["tool_choice"] = {"type": "auto"}

        headers = {
            "x-api-key":         self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }

        logger.debug(json.dumps(payload, indent=2))

        async def _run() -> None:
            msg = AssistantMessage()
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST", f"{self.base_url}/v1/messages", json=payload, headers=headers) as resp:
                        if resp.status_code >= 400:
                            body = await resp.aread()
                            raise RuntimeError(f"HTTP {resp.status_code}: {body.decode()}")

                        es.push(Event("start", {"partial": msg}))

                        _tool_buf: dict[int, str] = {}
                        _block_idx: dict[int, int] = {}

                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            chunk = json.loads(data)
                            etype = chunk.get("type")

                            if etype == "message_start":
                                u = chunk.get("message", {}).get("usage", {})
                                msg.usage.input       = u.get("input_tokens", 0)
                                msg.usage.cache_read  = u.get("cache_read_input_tokens", 0)
                                msg.usage.cache_write = u.get("cache_creation_input_tokens", 0)

                            elif etype == "content_block_start":
                                idx = chunk.get("index", 0)
                                blk = chunk.get("content_block", {})
                                btype = blk.get("type")
                                if btype == "text":
                                    _block_idx[idx] = len(msg.content)
                                    msg.content.append(TextContent(text=""))
                                elif btype == "thinking":
                                    _block_idx[idx] = len(msg.content)
                                    msg.content.append(ThinkingContent(thinking=""))
                                elif btype == "redacted_thinking":
                                    _block_idx[idx] = len(msg.content)
                                    msg.content.append(ThinkingContent(thinking="", redacted=True))
                                elif btype == "tool_use":
                                    _block_idx[idx] = len(msg.content)
                                    msg.content.append(ToolCall(id=blk["id"], name=blk["name"], arguments={}))
                                    _tool_buf[idx] = ""

                            elif etype == "content_block_delta":
                                idx = chunk.get("index", 0)
                                delta = chunk.get("delta", {})
                                dtype = delta.get("type")
                                pos = _block_idx.get(idx)
                                if pos is None:
                                    continue

                                if dtype == "text_delta":
                                    text = delta.get("text", "")
                                    blk = msg.content[pos]
                                    if isinstance(blk, TextContent):
                                        blk.text += text
                                    es.push(Event("text_delta", {"delta": text, "partial": msg}))

                                elif dtype == "thinking_delta":
                                    chunk_text = delta.get("thinking", "")
                                    blk = msg.content[pos]
                                    if isinstance(blk, ThinkingContent):
                                        blk.thinking += chunk_text
                                    es.push(Event("thinking_delta", {"delta": chunk_text, "partial": msg}))

                                elif dtype == "signature_delta":
                                    sig = delta.get("signature", "")
                                    blk = msg.content[pos]
                                    if isinstance(blk, ThinkingContent):
                                        blk.signature = (blk.signature or "") + sig

                                elif dtype == "input_json_delta":
                                    _tool_buf[idx] = _tool_buf.get(idx, "") + delta.get("partial_json", "")

                            elif etype == "content_block_stop":
                                idx = chunk.get("index", 0)
                                pos = _block_idx.get(idx)
                                if pos is not None and pos < len(msg.content):
                                    blk = msg.content[pos]
                                    if isinstance(blk, ToolCall):
                                        raw = _tool_buf.pop(idx, "")
                                        if raw:
                                            try:
                                                blk.arguments = json.loads(raw)
                                            except json.JSONDecodeError:
                                                pass
                                        es.push(Event("toolcall_end", {"tool_call": blk, "partial": msg}))

                            elif etype == "message_delta":
                                delta = chunk.get("delta", {})
                                reason = delta.get("stop_reason")
                                if reason == "tool_use":
                                    msg.stop_reason = "tool_use"
                                elif reason == "max_tokens":
                                    msg.stop_reason = "length"
                                elif reason:
                                    msg.stop_reason = "stop"
                                u = chunk.get("usage", {})
                                if u:
                                    msg.usage.output = u.get("output_tokens", msg.usage.output)

                            elif etype == "message_stop":
                                es.push(Event("done", {"reason": msg.stop_reason, "message": msg}))

            except Exception as exc:
                msg.stop_reason = "error"
                msg.error_message = str(exc)
                es.push(Event("error", {"error": msg}))

            es.finish(msg)

        task = asyncio.create_task(_run())
        es._attach(task)
        return es

class DefaultChat:
    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 8_192,
        temperature: float | None = None,
        tool_choice: Any = None,
    ) -> None:
        self.model = "jj" if model is None else model
        if api_key is None:
            logger.debug('No api key')
        self.api_key = "jj" if api_key is None else api_key
        self.base_url = "http://127.0.0.1:8000" if base_url is None else base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tool_choice = tool_choice

    def _build_messages(self, messages: list[Message], system: str) -> list[dict]:
        out = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            if isinstance(m, UserMessage):
                if isinstance(m.content, str):
                    out.append({"role": "user", "content": m.content})
                else:
                    parts = []
                    for item in m.content:
                        if isinstance(item, TextContent):
                            parts.append({"type": "text", "text": item.text})
                        elif isinstance(item, ImageContent):
                            parts.append({"type": "image_url", "image_url": {
                                "url": f"data:{item.mime_type};base64,{item.data}"
                            }})
                    out.append({"role": "user", "content": parts})
            elif isinstance(m, AssistantMessage):
                text_parts = [b for b in m.content if isinstance(b, TextContent)]
                tool_calls = [b for b in m.content if isinstance(b, ToolCall)]
                thinking_parts = [b for b in m.content if isinstance(b, ThinkingContent)]
                msg: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    msg["content"] = "".join(b.text for b in text_parts)
                if thinking_parts:
                    msg["reasoning_content"] = "".join(b.thinking for b in thinking_parts)
                if tool_calls:
                    msg["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                        for tc in tool_calls
                    ]
                out.append(msg)
            elif isinstance(m, ToolResultMessage):
                content_val: str | list
                if len(m.content) == 1 and isinstance(m.content[0], TextContent):
                    content_val = m.content[0].text
                else:
                    content_val = [
                        {"type": "text", "text": b.text} if isinstance(b, TextContent)
                        else {"type": "image_url", "image_url": {"url": f"data:{b.mime_type};base64,{b.data}"}}
                        for b in m.content
                    ]
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": content_val})
        return out

    async def stream(
        self,
        messages: list[Message],
        system: str,
        tools: list[Tool],
    ) -> EventStream:
        es = EventStream()

        payload: dict[str, Any] = {
            "model":      self.model,
            "max_tokens": self.max_tokens,
            "messages":   self._build_messages(messages, system),
            "stream":     True,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": t.parameters.model_json_schema().get("properties", {}),
                        "required": t.parameters.model_json_schema().get("required", []),
                    },
                }}
                for t in tools
            ]
            if self.tool_choice is not None:
                tc = self.tool_choice
                if isinstance(tc, dict):
                    payload["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
                else:
                    payload["tool_choice"] = tc

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        logger.debug(json.dumps(payload, indent=2))

        async def _run() -> None:
            msg = AssistantMessage()
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload, headers=headers) as resp:
                        if resp.status_code >= 400:
                            body = await resp.aread()
                            raise RuntimeError(f"HTTP {resp.status_code}: {body.decode()}")

                        es.push(Event("start", {"partial": msg}))

                        _tc_state: dict[int, dict] = {}
                        _text_buf = ""
                        _thinking_buf = ""          
                        finish_reason: str | None = None

                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:].strip()
                            if data == "[DONE]":
                                break
                            if not data:
                                continue
                            chunk = json.loads(data)
                            choice = (chunk.get("choices") or [{}])[0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason") or finish_reason

                            reasoning = delta.get("reasoning_content") or delta.get("thinking")
                            if reasoning:
                                _thinking_buf += reasoning
                                es.push(Event("thinking_delta", {"delta": reasoning, "partial": msg}))

                            if delta.get("content"):
                                text = delta["content"]
                                _text_buf += text
                                es.push(Event("text_delta", {"delta": text, "partial": msg}))

                            for tcd in delta.get("tool_calls") or []:
                                idx = tcd.get("index", 0)
                                if idx not in _tc_state:
                                    fn = tcd.get("function", {})
                                    _tc_state[idx] = {
                                        "id":       tcd.get("id", str(uuid.uuid4())),
                                        "name":     fn.get("name", ""),
                                        "args_buf": fn.get("arguments", ""),
                                    }
                                else:
                                    fn = tcd.get("function", {})
                                    if "name" in fn:
                                        _tc_state[idx]["name"] += fn["name"]
                                    if "arguments" in fn:
                                        _tc_state[idx]["args_buf"] += fn["arguments"]

                            u = chunk.get("usage") or {}
                            if u:
                                msg.usage.input  = u.get("prompt_tokens", msg.usage.input)
                                msg.usage.output = u.get("completion_tokens", msg.usage.output)

                        if _thinking_buf:
                            msg.content.append(ThinkingContent(thinking=_thinking_buf))
                        if _text_buf:
                            msg.content.append(TextContent(text=_text_buf))

                        for _, state in sorted(_tc_state.items()):
                            try:
                                args = json.loads(state["args_buf"]) if state["args_buf"] else {}
                            except json.JSONDecodeError:
                                args = {}
                            call = ToolCall(id=state["id"], name=state["name"], arguments=args)
                            msg.content.append(call)
                            es.push(Event("toolcall_end", {"tool_call": call, "partial": msg}))

                        if finish_reason == "tool_calls":
                            msg.stop_reason = "tool_use"
                        elif finish_reason in ("length", "max_tokens"):
                            msg.stop_reason = "length"
                        else:
                            msg.stop_reason = "stop"

                        es.push(Event("done", {"reason": msg.stop_reason, "message": msg}))

            except Exception as exc:
                msg.stop_reason = "error"
                msg.error_message = str(exc)
                es.push(Event("error", {"error": msg}))

            es.finish(msg)

        task = asyncio.create_task(_run())
        es._attach(task)
        return es

class GeminiChat:
    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 8_192,
        temperature: float | None = None,
        thinking: bool = False, 
        thinking_budget: int = 8_192,
        tool_choice: Any = None, 
    ) -> None:
        self.model = "gemini-3.1-flash-lite-preview" if model is None else model
        self.api_key = os.environ.get("GEMINI_API_KEY") if api_key is None else api_key
        if not self.api_key:
            logger.debug('No api key')
        self.base_url = "https://generativelanguage.googleapis.com" if base_url is None else base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.tool_choice = tool_choice

    def _build_contents(self, messages: list[Message]) -> list[dict]:
        out = []
        pending_fn_responses: list[dict] = []

        def flush_responses() -> None:
            if pending_fn_responses:
                out.append({"role": "user", "parts": list(pending_fn_responses)})
                pending_fn_responses.clear()

        for m in messages:
            if isinstance(m, UserMessage):
                flush_responses()
                if isinstance(m.content, str):
                    out.append({"role": "user", "parts": [{"text": m.content}]})
                else:
                    parts = []
                    for item in m.content:
                        if isinstance(item, TextContent):
                            parts.append({"text": item.text})
                        elif isinstance(item, ImageContent):
                            parts.append({
                                "inlineData": {
                                    "mimeType": item.mime_type,
                                    "data": item.data,
                                }
                            })
                    out.append({"role": "user", "parts": parts})

            elif isinstance(m, AssistantMessage):
                flush_responses()
                parts = []
                for block in m.content:
                    if isinstance(block, ThinkingContent):
                        parts.append({"thought": block.thinking})
                    elif isinstance(block, TextContent):
                        parts.append({"text": block.text})
                    elif isinstance(block, ToolCall):
                        fc: dict[str, Any] = {"name": block.name, "args": block.arguments}
                        if block.id != block.name:
                            fc["id"] = block.id
                        parts.append({"functionCall": fc})
                if parts:
                    out.append({"role": "model", "parts": parts})

            elif isinstance(m, ToolResultMessage):
                try:
                    response_body = json.loads(m.content[0].text) if m.content else {}
                except (json.JSONDecodeError, AttributeError):
                    response_body = {"result": m.content[0].text if m.content else ""}
                fr: dict[str, Any] = {
                    "name": m.tool_name,
                    "response": response_body,
                }
                if m.tool_call_id != m.tool_name:
                    fr["id"] = m.tool_call_id
                pending_fn_responses.append({"functionResponse": fr})

        flush_responses()
        return out

    def _tool_schema(self, tool: Tool) -> dict:
        s = tool.parameters.model_json_schema()
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": s.get("properties", {}),
                "required": s.get("required", []),
            },
        }

    async def stream(
        self,
        messages: list[Message],
        system: str,
        tools: list[Tool],
    ) -> EventStream:

        es = EventStream()

        payload: dict[str, Any] = {
            "contents": self._build_contents(messages),
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if self.temperature is not None:
            payload["generationConfig"]["temperature"] = self.temperature
        if self.thinking:
            payload["generationConfig"]["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingBudget": self.thinking_budget,
            }
        if tools:
            payload["tools"] = [{"functionDeclarations": [self._tool_schema(t) for t in tools]}]
            tc = self.tool_choice
            if tc is not None:
                if isinstance(tc, dict):
                    payload["toolConfig"] = {"functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [tc["name"]],
                    }}
                elif tc == "required":
                    payload["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
                elif tc == "none":
                    payload["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
                else:
                    payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        url = f"{self.base_url}/v1beta/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        logger.debug(json.dumps(payload, indent=2))

        async def _run() -> None:
            msg = AssistantMessage()
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST", url, json=payload, headers=headers) as resp:
                        if resp.status_code >= 400:
                            body = await resp.aread()
                            raise RuntimeError(f"HTTP {resp.status_code}: {body.decode()}")

                        es.push(Event("start", {"partial": msg}))

                        _text_buf = ""
                        _thinking_buf = ""
                        _fn_calls: dict[str, dict] = {}  
                        finish_reason: str | None = None

                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:].strip()
                            if not data:
                                continue
                            chunk = json.loads(data)

                            candidate = (chunk.get("candidates") or [{}])[0]
                            finish_reason = candidate.get("finishReason") or finish_reason
                            parts = candidate.get("content", {}).get("parts") or []

                            for part in parts:
                                if "thought" in part:
                                    delta = part["thought"]
                                    _thinking_buf += delta
                                    es.push(Event("thinking_delta", {"delta": delta, "partial": msg}))
                                elif "text" in part:
                                    delta = part["text"]
                                    _text_buf += delta
                                    es.push(Event("text_delta", {"delta": delta, "partial": msg}))
                                elif "functionCall" in part:
                                    fc = part["functionCall"]
                                    name = fc["name"]
                                    call_id = fc.get("id") or name
                                    if call_id not in _fn_calls:
                                        _fn_calls[call_id] = {"name": name, "args": fc.get("args", {})}
                                    else:
                                        _fn_calls[call_id]["args"].update(fc.get("args", {}))

                            u = chunk.get("usageMetadata") or {}
                            if u:
                                msg.usage.input  = u.get("promptTokenCount", msg.usage.input)
                                msg.usage.output = u.get("candidatesTokenCount", msg.usage.output)

                        if _thinking_buf:
                            msg.content.append(ThinkingContent(thinking=_thinking_buf))
                        if _text_buf:
                            msg.content.append(TextContent(text=_text_buf))
                        for call_id, state in _fn_calls.items():
                            call = ToolCall(id=call_id, name=state["name"], arguments=state["args"])
                            msg.content.append(call)
                            es.push(Event("toolcall_end", {"tool_call": call, "partial": msg}))

                        if finish_reason == "MAX_TOKENS":
                            msg.stop_reason = "length"
                        elif _fn_calls:
                            msg.stop_reason = "tool_use"
                        else:
                            msg.stop_reason = "stop"

                        es.push(Event("done", {"reason": msg.stop_reason, "message": msg}))
                        es.finish(msg)

            except Exception as exc:
                msg.stop_reason = "error"
                msg.error_message = str(exc)
                es.push(Event("error", {"error": msg}))
                es.finish(msg)

        task = asyncio.create_task(_run())
        es._attach(task)
        return es

class CodexChat:
    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 8_192,
        temperature: float | None = None,
        tool_choice: Any = None,
    ) -> None:
        self.model = "gpt-5.4-mini" if model is None else model
        self.api_key = os.environ.get("OPENAI_API_KEY") if api_key is None else api_key
        if not self.api_key:
            logger.debug('No api key')
        self.base_url = "https://api.openai.com" if base_url is None else base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tool_choice = tool_choice

    def _build_input(self, messages: list[Message], system: str) -> list[dict]:
        out = []
        if system:
            out.append({"type": "message", "role": "developer", "content": system})

        for m in messages:
            if isinstance(m, UserMessage):
                if isinstance(m.content, str):
                    content = m.content
                else:
                    content = [
                        {"type": "input_text", "text": item.text}
                        if isinstance(item, TextContent)
                        else {"type": "input_image", "image_url": f"data:{item.mime_type};base64,{item.data}"}
                        for item in m.content
                    ]
                out.append({"type": "message", "role": "user", "content": content})

            elif isinstance(m, AssistantMessage):
                for block in m.content:
                    if isinstance(block, TextContent):
                        out.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": block.text}],
                        })
                    elif isinstance(block, ToolCall):
                        out.append({
                            "type": "function_call",
                            "call_id": block.id,
                            "name": block.name,
                            "arguments": json.dumps(block.arguments),
                        })

            elif isinstance(m, ToolResultMessage):
                out.append({
                    "type": "function_call_output",
                    "call_id": m.tool_call_id,
                    "output": m.content[0].text if m.content else "",
                })

        return out

    async def stream(
        self,
        messages: list[Message],
        system: str,
        tools: list[Tool],
    ) -> EventStream:
        es = EventStream()

        payload: dict[str, Any] = {
            "model":      self.model,
            "max_output_tokens": self.max_tokens,
            "input":      self._build_input(messages, system),
            "stream":     True,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": t.parameters.model_json_schema().get("properties", {}),
                        "required":   t.parameters.model_json_schema().get("required", []),
                    },
                }
                for t in tools
            ]
            if self.tool_choice is not None:
                tc = self.tool_choice
                if isinstance(tc, dict):
                    payload["tool_choice"] = {"type": "function", "name": tc["name"]}
                else:
                    payload["tool_choice"] = tc

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        logger.debug(json.dumps(payload, indent=2))

        async def _run() -> None:
            msg = AssistantMessage()
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST", f"{self.base_url}/v1/responses",
                        json=payload, headers=headers
                    ) as resp:
                        if resp.status_code >= 400:
                            body = await resp.aread()
                            raise RuntimeError(f"HTTP {resp.status_code}: {body.decode()}")

                        es.push(Event("start", {"partial": msg}))

                        _text_buf = ""
                        _fn_calls: dict[str, dict] = {}
                        finish_reason: str | None = None

                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:].strip()
                            if not data:
                                continue
                            chunk = json.loads(data)
                            etype = chunk.get("type", "")

                            if etype == "response.output_text.delta":
                                delta = chunk.get("delta", "")
                                _text_buf += delta
                                es.push(Event("text_delta", {"delta": delta, "partial": msg}))

                            elif etype == "response.output_item.added":
                                item = chunk.get("item", {})
                                if item.get("type") == "function_call":
                                    iid = item["id"]
                                    _fn_calls[iid] = {
                                        "name":     item.get("name", ""),
                                        "call_id":  item.get("call_id", iid),
                                        "args_buf": "",
                                    }

                            elif etype == "response.function_call_arguments.delta":
                                iid = chunk.get("item_id", "")
                                if iid in _fn_calls:
                                    _fn_calls[iid]["args_buf"] += chunk.get("delta", "")

                            elif etype == "response.completed":
                                response = chunk.get("response", {})
                                finish_reason = response.get("status") 
                                u = response.get("usage") or {}
                                if u:
                                    msg.usage.input  = u.get("input_tokens", msg.usage.input)
                                    msg.usage.output = u.get("output_tokens", msg.usage.output)

                        if _text_buf:
                            msg.content.append(TextContent(text=_text_buf))

                        for state in _fn_calls.values():
                            try:
                                args = json.loads(state["args_buf"]) if state["args_buf"] else {}
                            except json.JSONDecodeError:
                                args = {}
                            call = ToolCall(id=state["call_id"], name=state["name"], arguments=args)
                            msg.content.append(call)
                            es.push(Event("toolcall_end", {"tool_call": call, "partial": msg}))

                        if _fn_calls:
                            msg.stop_reason = "tool_use"
                        elif finish_reason == "incomplete":
                            msg.stop_reason = "length"
                        else:
                            msg.stop_reason = "stop"

                        es.push(Event("done", {"reason": msg.stop_reason, "message": msg}))

            except Exception as exc:
                msg.stop_reason = "error"
                msg.error_message = str(exc)
                es.push(Event("error", {"error": msg}))

            es.finish(msg)

        task = asyncio.create_task(_run())
        es._attach(task)
        return es

AgentEventType: TypeAlias = Literal[
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "text_delta",
    "thinking_delta",
    "tool_start",
    "tool_result",
    "tool_end",
    "error",
]

@dataclass
class AgentEvent:
    type: AgentEventType
    payload: dict[str, Any] = field(default_factory=dict)

class API:
    async def stream(
        self,
        messages: list[Message],
        system: str,
        tools: list[Tool],
    ) -> EventStream: ...

class Agent:
    def __init__(
        self,
        api,
        system,
        tools = None,
        model = None,
        api_key = None,
        base_url = None,
        gwt=None,
    ) -> None:
        if api == "claude":
            api = ClaudeChat(model=model, api_key=api_key, base_url=base_url)
        elif api == "gemini":
            api = GeminiChat(model=model, api_key=api_key, base_url=base_url)
        elif api == "codex":
            api = CodexChat(model=model, api_key=api_key, base_url=base_url)
        else:
            api = DefaultChat(model=model, api_key=api_key, base_url=base_url)
        self.api = api
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._api = api
        self.system = system
        tools = tools or []
        self.tools = tools + [AgentTool(self)]
        self.messages: list[Message] = []
        self._signal: asyncio.Event | None = None
        self._listeners: set[Callable] = set()
        self.gwt = gwt

    async def run(self, prompt: str) -> AssistantMessage:
        if self._signal is None or not self._signal.is_set():
            self._signal = None
        self.messages.append(UserMessage(content=prompt))
        return await self._loop()

    def abort(self) -> None:
        if self._signal is None:
            self._signal = asyncio.Event()
        self._signal.set()

    def branch(self) -> "Agent":
        child = Agent(api=self.api, system=self.system, tools=list(self.tools), model=self.model,api_key=self.api_key, base_url=self.base_url, gwt=self.gwt)
        child.messages = list(self.messages)
        return child

    def subscribe(self, fn: Callable[[AgentEvent], Any]) -> Callable:
        self._listeners.add(fn)
        return lambda: self._listeners.discard(fn)

    async def _emit(self, event: AgentEvent) -> None:
        for fn in list(self._listeners):
            result = fn(event)
            if asyncio.iscoroutine(result):
                await result

    async def _loop(self) -> AssistantMessage:
        await self._emit(AgentEvent("agent_start"))
        final: AssistantMessage = AssistantMessage(stop_reason="error",
                                                    error_message="no turns ran")
        while True:
            await self._emit(AgentEvent("turn_start"))

            es = await self._api.stream(self.messages, self.system, self.tools)

            async for event in es:
                if event.type == "text_delta":
                    await self._emit(AgentEvent("text_delta", event.payload))
                elif event.type == "thinking_delta":
                    await self._emit(AgentEvent("thinking_delta", event.payload))
                elif event.type == "error":
                    await self._emit(AgentEvent("error", event.payload))

            final = await es.result()
            logger.debug(final)
            self.messages.append(final)

            await self._emit(AgentEvent("turn_end", {"message": final}))

            if final.stop_reason in ("error", "aborted"):
                break

            if self._signal and self._signal.is_set():
                final.stop_reason = "aborted"
                break

            tool_calls = [b for b in final.content if isinstance(b, ToolCall)]
            if not tool_calls:
                break

            results = await self._execute_tools(tool_calls)
            self.gwt = commit_worktree(self.gwt)
            logger.debug(results)
            self.messages.extend(results)

            if self._signal and self._signal.is_set():
                final.stop_reason = "aborted"
                break

        await self._emit(AgentEvent("agent_end", {"message": final}))
        return final

    async def _execute_tools(self, calls: list[ToolCall]) -> list[ToolResultMessage]:
        return list(await asyncio.gather(*[self._execute_one(c) for c in calls]))

    async def _execute_one(self, call: ToolCall) -> ToolResultMessage:
        await self._emit(AgentEvent("tool_start", {"name": call.name, "args": call.arguments}))

        tool = next((t for t in self.tools if t.name == call.name), None)
        if tool is None:
            result = ToolResult(
                content=[TextContent(f"Tool '{call.name}' not found")],
                is_error=True,
            )
        else:
            try:
                params = validate_tool_call(tool, call)
                result = await tool.execute(params, self._signal)
            except Exception as exc:
                result = ToolResult(content=[TextContent(str(exc))], is_error=True)

        msg = ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            content=result.content,
            is_error=result.is_error,
        )

        await self._emit(AgentEvent("tool_result", {"message": msg}))

        await self._emit(AgentEvent("tool_end", {
            "name": call.name, "is_error": result.is_error, "result": msg,
        }))
        return msg

_MAX_BYTES = 50 * 1024
_MAX_LINES = 2_000
_FILE_LOCKS: dict[str, asyncio.Lock] = {}
_FILE_LOCKS_GUARD = asyncio.Lock()

async def _file_lock(path: str) -> asyncio.Lock:
    async with _FILE_LOCKS_GUARD:
        if path not in _FILE_LOCKS:
            _FILE_LOCKS[path] = asyncio.Lock()
        return _FILE_LOCKS[path]

def _truncate(text: str, label: str = "") -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) > _MAX_LINES:
        text = "".join(lines[:_MAX_LINES])
        text += f"\n[truncated at {_MAX_LINES} lines{': ' + label if label else ''}]"
    encoded = text.encode()
    if len(encoded) > _MAX_BYTES:
        text = encoded[:_MAX_BYTES].decode(errors="replace")
        text += f"\n[truncated at 50 KB{': ' + label if label else ''}]"
    return text

def _resolve(path: str, cwd: str) -> str:
    path = path.lstrip("@")
    base = pathlib.Path(cwd).resolve()
    candidate = (base / path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError(f"Path '{path}' escapes working directory")
    return str(candidate)

def _gitignore_patterns(dirpath: str) -> list[str]:
    patterns: list[str] = []
    current = pathlib.Path(dirpath).resolve()
    visited: list[pathlib.Path] = []
    while True:
        visited.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for d in reversed(visited):
        gi = d / ".gitignore"
        if gi.exists():
            try:
                for ln in gi.read_text(encoding="utf-8", errors="replace").splitlines():
                    ln = ln.strip()
                    if ln and not ln.startswith("#") and not ln.startswith("!"):
                        patterns.append(ln)
            except OSError:
                pass
    return patterns


def _is_gitignored(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(name + "/", p) for p in patterns)

class ReadParams(BaseModel):
    path: str = Field(description="File path to read (relative to cwd)")
    offset: int | None = Field(default=None, description="Start line (1-based)")
    limit: int | None = Field(default=None, description="Max lines to read")

class ReadTool(Tool):
    name = "Read"
    description = "Read a file. Use offset/limit for large files instead of reading the whole thing."
    parameters = ReadParams

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, params: ReadParams, signal=None) -> ToolResult:
        path = _resolve(params.path, self.cwd)
        try:
            lines = pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        except FileNotFoundError:
            raise ValueError(f"File not found: {params.path}")
        except IsADirectoryError:
            raise ValueError(f"Path is a directory: {params.path}")
        start = max(0, (params.offset - 1) if params.offset else 0)
        end = (start + params.limit) if params.limit else len(lines)
        header = f"# {params.path}  (lines {start+1}–{min(end, len(lines))} of {len(lines)})\n"
        return ToolResult(content=[TextContent(_truncate(header + "".join(lines[start:end]), params.path))])

class WriteParams(BaseModel):
    path: str = Field(description="File path to create or overwrite (relative to cwd)")
    content: str = Field(description="Full file content")

class WriteTool(Tool):
    name = "Write"
    description = "Create or overwrite a file. Prefer 'edit' for small changes to existing files."
    parameters = WriteParams

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, params: WriteParams, signal=None) -> ToolResult:
        path = _resolve(params.path, self.cwd)
        lock = await _file_lock(path)
        async with lock:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            existed = pathlib.Path(path).exists()
            pathlib.Path(path).write_text(params.content, encoding="utf-8")
        action = "overwrote" if existed else "created"
        lines = params.content.count("\n") + 1
        return ToolResult(content=[TextContent(f"{action} {params.path} ({lines} lines)")])

class EditParams(BaseModel):
    path: str = Field(description="File path to edit (relative to cwd)")
    old_text: str = Field(description="Exact text to replace (must appear exactly once)")
    new_text: str = Field(description="Replacement text")

class EditTool(Tool):
    name = "Edit"
    description = "Replace an exact unique string in a file. Read the file first if unsure of exact text."
    parameters = EditParams

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, params: EditParams, signal=None) -> ToolResult:
        path = _resolve(params.path, self.cwd)
        lock = await _file_lock(path)
        async with lock:
            try:
                original = pathlib.Path(path).read_text(encoding="utf-8")
            except FileNotFoundError:
                raise ValueError(f"File not found: {params.path}")
            count = original.count(params.old_text)
            if count == 0:
                raise ValueError(f"old_text not found in {params.path}")
            if count > 1:
                raise ValueError(f"old_text appears {count} times in {params.path}; make it more specific")
            pathlib.Path(path).write_text(original.replace(params.old_text, params.new_text, 1), encoding="utf-8")
        return ToolResult(content=[TextContent(
            f"edited {params.path}: replaced {params.old_text.count(chr(10))+1} line(s) "
            f"with {params.new_text.count(chr(10))+1} line(s)"
        )])

class BashParams(BaseModel):
    command: str = Field(description="Shell command to execute")
    timeout: int = Field(default=120, description="Timeout in seconds")

class BashTool(Tool):
    name = "Bash"
    description = "Run a shell command, get stdout+stderr. Prefer read/grep/find/ls for file exploration."
    parameters = BashParams

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, params: BashParams, signal=None) -> ToolResult:
        proc = await asyncio.create_subprocess_shell(
            params.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.cwd,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=params.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise ValueError(f"Command timed out after {params.timeout}s")
        output = stdout.decode(errors="replace")
        exit_code = proc.returncode or 0
        # result = f"$ {params.command}\n{output}" # □
        result = str(output) + ' '
        if exit_code != 0:
            result += f"\n[exit code {exit_code}]"
        return ToolResult(content=[TextContent(_truncate(result, params.command))])

class GrepParams(BaseModel):
    pattern: str = Field(description="Regex or literal pattern")
    path: str | None = Field(default=None, description="Directory or file (default: cwd)")
    glob: str | None = Field(default=None, description="File glob filter e.g. '*.py'")
    ignore_case: bool = Field(default=False)
    literal: bool = Field(default=False, description="Treat pattern as literal string")
    context: int | None = Field(default=None, description="Lines of context around each match")
    limit: int = Field(default=100, description="Max matches to return")

class GrepTool(Tool):
    name = "Grep"
    description = "Search files for a pattern. Respects .gitignore."
    parameters = GrepParams

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, params: GrepParams, signal=None) -> ToolResult:
        search_root = _resolve(params.path, self.cwd) if params.path else self.cwd
        flags = re.IGNORECASE if params.ignore_case else 0
        try:
            compiled = re.compile(re.escape(params.pattern) if params.literal else params.pattern, flags)
        except re.error as exc:
            raise ValueError(f"Invalid regex: {exc}")

        search_path = pathlib.Path(search_root)
        if search_path.is_file():
            file_list = [search_path]
        else:
            file_list = []
            for dirpath, dirnames, filenames in os.walk(search_root):
                gi = _gitignore_patterns(dirpath)
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and not _is_gitignored(d, gi)]
                for fname in filenames:
                    if _is_gitignored(fname, gi):
                        continue
                    if params.glob and not fnmatch.fnmatch(fname, params.glob):
                        continue
                    file_list.append(pathlib.Path(dirpath) / fname)

        matches: list[str] = []
        count = 0
        ctx = params.context or 0

        for fpath in file_list:
            if count >= params.limit:
                break
            try:
                file_lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except (OSError, IsADirectoryError):
                continue
            rel = os.path.relpath(str(fpath), self.cwd)
            emitted: set[int] = set()
            for i, line in enumerate(file_lines):
                if count >= params.limit:
                    break
                if compiled.search(line):
                    for j in range(max(0, i - ctx), min(len(file_lines), i + ctx + 1)):
                        if j not in emitted:
                            sep = ":" if j == i else "-"
                            matches.append(f"{rel}:{j+1}{sep}{file_lines[j]}")
                            emitted.add(j)
                    if ctx:
                        matches.append("--")
                    count += 1

        if not matches:
            text = f"No matches for '{params.pattern}'"
        else:
            text = "\n".join(matches)
            if count >= params.limit:
                text += f"\n[truncated at {params.limit} matches]"
        return ToolResult(content=[TextContent(text)])


class FindParams(BaseModel):
    path: str | None = Field(default=None, description="Directory to search (default: cwd)")
    pattern: str | None = Field(default=None, description="Filename glob pattern e.g. '*.py'")
    type: Literal["file", "dir", "any"] = Field(default="any")
    limit: int = Field(default=200)


class FindTool(Tool):
    name = "Find"
    description = "Find files or directories by name pattern. Respects .gitignore."
    parameters = FindParams

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, params: FindParams, signal=None) -> ToolResult:
        search_root = _resolve(params.path, self.cwd) if params.path else self.cwd
        results: list[str] = []
        for dirpath, dirnames, filenames in os.walk(search_root):
            gi = _gitignore_patterns(dirpath)
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and not _is_gitignored(d, gi)]
            if params.type in ("dir", "any"):
                for d in dirnames:
                    if params.pattern and not fnmatch.fnmatch(d, params.pattern):
                        continue
                    results.append(os.path.relpath(os.path.join(dirpath, d), self.cwd) + "/")
            if params.type in ("file", "any"):
                for f in filenames:
                    if _is_gitignored(f, gi):
                        continue
                    if params.pattern and not fnmatch.fnmatch(f, params.pattern):
                        continue
                    results.append(os.path.relpath(os.path.join(dirpath, f), self.cwd))
            if len(results) >= params.limit:
                results = results[:params.limit]
                results.append(f"[truncated at {params.limit} results]")
                break
        text = "\n".join(results) if results else "No results"
        return ToolResult(content=[TextContent(text)])


class LsParams(BaseModel):
    path: str | None = Field(default=None, description="Directory to list (default: cwd)")


class LsTool(Tool):
    name = "Ls"
    description = "List directory contents. Respects .gitignore."
    parameters = LsParams

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, params: LsParams, signal=None) -> ToolResult:
        target = _resolve(params.path, self.cwd) if params.path else self.cwd
        p = pathlib.Path(target)
        if not p.is_dir():
            raise ValueError(f"Not a directory: {params.path or '.'}")
        gi = _gitignore_patterns(target)
        entries = []
        for entry in sorted(p.iterdir()):
            if entry.name.startswith(".") or _is_gitignored(entry.name, gi):
                continue
            entries.append(entry.name + ("/" if entry.is_dir() else ""))
        return ToolResult(content=[TextContent("\n".join(entries) if entries else "(empty)")])


class ReadTreeParams(BaseModel):
    path: str = Field(description="File or directory to inspect (relative to cwd)")
    symbols: list[str] = Field(
        default_factory=list,
        description=(
            "Symbol names to look up, e.g. [\"MyClass\", \"my_fn\", \"ClassName.method\"]. "
            "Omit (or pass []) for outline mode. "
            "Dotted names are accepted — the last component is matched and results "
            "are labelled with the full qualified name."
        ),
    )
    depth: int = Field(
        default=1,
        description="Outline mode: nesting depth (1=top-level only, 2=classes+methods).",
    )
    kinds: list[str] = Field(
        default_factory=list,
        description=(
            "Filter symbol-lookup results by kind. "
            "Valid values: definition, assignment, call, import, reference. "
            "Default: definition + assignment + call + import (no bare references)."
        ),
    )
    lang: str | None = Field(
        default=None,
        description="Override language detection, e.g. 'py', 'ts', 'go'.",
    )

    @classmethod
    def model_validate(cls, obj, **kw):
        if isinstance(obj, dict) and isinstance(obj.get("symbols"), str):
            raw = obj["symbols"].strip()
            try:
                parsed = json.loads(raw)
                obj = {**obj, "symbols": parsed if isinstance(parsed, list) else [parsed]}
            except (json.JSONDecodeError, ValueError):
                obj = {**obj, "symbols": [raw] if raw else []}
        return super().model_validate(obj, **kw)


class ReadTreeTool(Tool):
    name = "ReadTree"
    description = (
        "Inspect source code using tree-sitter. Works for any supported language "
        "(Python, JS/TS, Go, Rust, Java, C/C++, Ruby, and more). Two modes:\n"
        "  OUTLINE (no symbols): returns the symbol tree of a file or directory — "
        "class/function/method/var names with line ranges. Use this first to "
        "orient yourself before reading or editing code.\n"
        "  SYMBOL LOOKUP (symbols=[...]): returns the full source body of every "
        "definition of those names, plus every call/assignment site with context. "
        "Output is paste-safe for use as old_str in an Edit call. "
        "Accepts dotted names like 'ClassName.method' to narrow results."
    )
    parameters = ReadTreeParams

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, params: ReadTreeParams, signal=None) -> ToolResult:
        from .sittree import (  #.
            outline_path, print_outline,
            search_symbols, format_search_results,
            resolve_lang_ext, Capability,
        )
        import io

        target   = pathlib.Path(_resolve(params.path, self.cwd))
        lang_ext = resolve_lang_ext(params.lang) if params.lang else None

        if not params.symbols:
            items = outline_path(target, lang_ext=lang_ext, max_depth=params.depth)
            if not items:
                return ToolResult(
                    content=[TextContent(f"No symbols found in: {params.path}")],
                    is_error=True,
                )
            buf = io.StringIO()
            print_outline(items, use_color=False, file=buf)
            return ToolResult(content=[TextContent(buf.getvalue())])

        default_kinds = {"definition", "assignment", "call", "import"}
        filter_kinds  = set(params.kinds) if params.kinds else default_kinds
        include_refs  = "reference" in filter_kinds

        all_usages, capabilities = search_symbols(
            params.symbols,
            target,
            lang_ext=lang_ext,
            include_references=include_refs,
        )

        warnings: list[str] = []
        for sym, cap in capabilities.items():
            if cap is Capability.NONE:
                warnings.append(f"# warning: no grammar support for '{sym}' in this file type")
            elif cap is Capability.PARTIAL:
                warnings.append(f"# warning: '{sym}' matched by generic scan (kind may be inaccurate)")

        if filter_kinds != {"definition", "assignment", "call", "import", "reference"}:
            all_usages = [u for u in all_usages if u.kind in filter_kinds]

        if not all_usages:
            msg = f"No usages found for: {params.symbols}"
            if warnings:
                msg = "\n".join(warnings) + "\n" + msg
            return ToolResult(content=[TextContent(msg)], is_error=True)

        text = format_search_results(all_usages, raw=True, show_refs=include_refs)
        if warnings:
            text = "\n".join(warnings) + "\n\n" + text
        return ToolResult(content=[TextContent(text)])


class GetSkillParams(BaseModel):
    name: str = Field(description="Skill name to retrieve")

class GetSkillTool(Tool):
    name = "GetSkill"
    description = "Retrieve the full instructions for a skill by name."
    parameters = GetSkillParams

    def __init__(self, skills: list[dict]) -> None:
        self._skills = {s["name"]: s["content"] for s in skills}

    async def execute(self, params: GetSkillParams, signal=None) -> ToolResult:
        content = self._skills.get(params.name)
        if content is None:
            return ToolResult(content=[TextContent(f"Skill '{params.name}' not found.")], is_error=True)
        return ToolResult(content=[TextContent(content)])

class AgentParams(BaseModel):
    task: str = Field(description=( "Clear autonomous task for the sub-agent. Should include objective, constraints, expected output, and relevant context."))
    api: str | None = Field(default=None, description=("API/backend override for the sub-agent (e.g. claude, gemini, codex). Defaults to parent agent backend."))
    system: str | None = Field(default=None, description=("Optional replacement system prompt for the sub-agent. If omitted, inherits parent system prompt."))
    tools: list[str] | str | None = Field(default=None, description=("Restrict available tools for the sub-agent. Example: ['Read', 'Edit', 'Bash']. If omitted, inherits all parent tools."))
    # ref:      str       = Field(default="", description="Git branch or sha to start from. Empty = current branch.") # □ todo
 
 
class AgentTool(Tool):
    name = "Agent"

    description = """
Spawn an autonomous sub-agent in an isolated git worktree.

The sub-agent is a full agent instance with:
- its own conversation history
- its own reasoning loop
- its own tool execution cycle
- its own git worktree/branch
- inherited or overridden system prompt
- restricted or inherited tools

Use this tool to delegate complex, multi-step, or parallelizable work.

Good use cases:
- implementing a self-contained feature
- investigating bugs
- performing deep codebase research
- running long tool chains
- generating/refactoring code
- testing alternative implementations
- summarizing large subsystems
- parallel exploration of different approaches

Avoid using Agent for:
- trivial single-step operations
- simple reads/searches
- tiny edits
- tasks requiring tight conversational continuity
- tasks where direct tool usage is cheaper/simpler

The sub-agent does NOT stream intermediate reasoning or tool usage back.
Only the final response is returned.

The sub-agent operates independently and may:
- read/write/edit files
- run bash commands
- invoke tools repeatedly
- spawn additional agents (if Agent tool is allowed)

Context inheritance:
- system prompt defaults to parent system
- tools default to inherited tools
- model/api settings default to parent
- git state/worktree is isolated

Tool restriction:
Pass a limited tool list to constrain the sub-agent.
This is strongly recommended for focused tasks.

Examples:
- Investigate failing tests and explain root cause
- Implement parser refactor using only Read/Edit/Bash
- Analyze auth flow and summarize architecture
- Search entire repo for dead code candidates
- Prototype an alternative implementation in isolation

Delegation tips:
- Give concrete goals
- Specify constraints
- Define expected output
- Restrict tools when possible
- Prefer focused subtasks over vague broad objectives
- Prefer multiple specialized agents over one giant task
"""
    parameters = AgentParams
 
    def __init__(self, parent) -> None:
        self.parent = parent
 
    async def execute(self, params, signal=None) -> ToolResult:
        api = params.api if params.api is not None else self.parent.api
        system = params.system if params.system is not None else self.parent.system
        tools_raw = params.tools
        if isinstance(tools_raw, str):
            tools_raw = json.loads(tools_raw)
        if tools_raw is not None:
            tools_raw = [t for t in tools_raw if isinstance(t, str)]
        tools = [t for t in self.parent.tools if t.name.lower() in {n.lower() for n in params.tools}] if params.tools is not None else self.parent.tools
        agent = Agent(api=api, system=system, tools=tools, model=self.parent.model, api_key=self.parent.api_key, base_url=self.parent.base_url, gwt=self.parent.gwt) # □ todo
        try:
            result = await agent.run(params.task)
            texts = [b.text for b in result.content if isinstance(b, TextContent)]
            return ToolResult(content=[TextContent(''.join(texts).strip())])
        except Exception as e:
            return ToolResult(content=[TextContent(f"Agent failed: {e}")], is_error=True)

def collect_tools(tools: str | None = None, skills: list[dict] | None = None, cwd: str | None = None) -> list[Tool]:
    available_tools = [ReadTool(cwd), WriteTool(cwd), EditTool(cwd), BashTool(cwd), GrepTool(cwd), FindTool(cwd), LsTool(cwd), ReadTreeTool(cwd)]
    if skills:
        available_tools.append(GetSkillTool(skills))
    if tools is not None:
        requested_names = {name.lower() for name in tools}
        available_tools = [
            t for t in available_tools
            if t.name.lower() in requested_names
        ]
        found_names = {t.name.lower() for t in available_tools}
        for req in requested_names:
            if req not in found_names:
                logger.warning(f"Tool '{req}' is not recognized.")
    return available_tools

_REPL_HELP = """\
Commands:
  /help          — show this message
  /clear         — clear conversation history
  /history       — print message history
  /tools         — list active tools
  /branch        — spawn a branched sub-agent and run a one-shot prompt
  /abort         — signal abort after next tool call
  exit / quit    — end the session
"""

def read_input(prompt: str = "\033[32m≫\033[0m ") -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    sys.stdout.write("\x1b[?2004h" + prompt)
    sys.stdout.flush()

    buf = []
    esc = ""
    pasting = False

    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)

            if esc or ch == "\x1b":
                esc += ch
                if esc == "\x1b[200~":   pasting = True;  esc = ""
                elif esc == "\x1b[201~": pasting = False; esc = ""
                elif len(esc) >= 6:      buf.extend(esc); esc = ""
                continue

            if ch == "\x03": raise KeyboardInterrupt
            if ch == "\x04": raise EOFError

            if ch in ("\r", "\n"):
                if pasting:
                    buf.append("\n")
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                else:
                    line = "".join(buf)
                    if line.endswith("\\"):
                        buf[-1] = "\n"  
                        sys.stdout.write("\r\n... ")
                        sys.stdout.flush()
                    else:
                        sys.stdout.write("\r\n")
                        sys.stdout.flush()
                        break
                continue

            if ch == "\x7f":
                if buf and not pasting:
                    buf.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue

            buf.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()

    return "".join(buf).strip()

async def repl(
    api: Any,
    system: str,
    tools: list[Tool] | None = None,
    skills: list[dict] | None = None,
    api_key = None,
    base_url = None, 
    gwt = None,
) -> None:
    is_tty = sys.stdin.isatty()
    agent = Agent(api, system, tools=tools, api_key=api_key, base_url=base_url, gwt=gwt)
    loop = asyncio.get_running_loop()

    _suppress = False
    last_block = ""
    trail = ""  

    async def on_event(event: AgentEvent) -> None:
        nonlocal _suppress, last_block, trail

        def emit(delta, block, prefix="", suffix=""):
            nonlocal last_block, trail
            changed = last_block != block
            if changed:
                trail = ""
                delta = delta.lstrip("\n")
                if not delta:
                    return          
                if last_block:
                    print("\n", end="")
                last_block = block
            else:
                last_block = block
                if trail:
                    print(trail, end="")
                trail = ""
            rstripped = delta.rstrip("\n")
            trail = delta[len(rstripped):]
            if rstripped:
                print(prefix + rstripped + suffix, end="", flush=True)

        if event.type == "text_delta":
            delta = event.payload.get("delta", "")
            if "<tool_call>" in delta:
                before, _, delta = delta.partition("<tool_call>")
                if before:
                    emit(before, "text")
                _suppress = True
            if "</tool_call>" in delta:
                _, _, delta = delta.partition("</tool_call>")
                _suppress = False
            if not delta or _suppress:
                return
            emit(delta, "text")

        elif is_tty:
            if event.type == "thinking_delta":
                delta = event.payload.get("delta", "")
                if not delta:
                    return
                emit(delta, "thinking", prefix="\033[2m", suffix="\033[0m")

            elif event.type == "tool_start":
                text = event.payload['name'] + " "
                if event.payload['args']:
                    text += json.dumps(event.payload['args']) + " "
                emit(text, "tool", prefix="\033[33m", suffix="\033[0m")

            elif event.type == "tool_end":
                if event.payload.get("is_error"):
                    print(" \033[31m(error)\033[0m", flush=True)

            elif event.type == "error":
                err = event.payload.get("error")
                print(f"\n\033[31m[error]\033[0m {getattr(err, 'error_message', str(err))}\n")

            elif event.type == "agent_end":
                last_block = ""

    agent.subscribe(on_event)
    if is_tty:
        print("pie REPL  •  type /help for commands, Ctrl-D or 'exit' to quit.\n")

    while True:
        if is_tty:
            try:
                user_input = await loop.run_in_executor(None, lambda: read_input().strip())
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                return
        else:
            user_input = sys.stdin.read()

        if not user_input:
            continue

        last_block = ""
        logger.debug(user_input)

        if user_input.startswith("/"):
            cmd, _, arg = user_input.partition(" ")
            cmd = cmd.lower()
            if cmd == "/help":
                print(_REPL_HELP)
            elif cmd == "/clear":
                agent.messages.clear()
                print("[history cleared]")
            elif cmd == "/history":
                def truncate(text, limit=80):
                    text = text.replace('\n', ' ').strip()
                    return (text[:limit] + "‥") if len(text) > limit else text

                for i, m in enumerate(agent.messages):
                    if isinstance(m, UserMessage):
                        content = m.content if isinstance(m.content, str) else str(m.content)
                        print(f"  {i:2d} [user] {truncate(content)}")
                        
                    elif isinstance(m, AssistantMessage):
                        text = "".join(b.text for b in m.content if isinstance(b, TextContent))
                        print(f"  {i:2d} [assistant] {truncate(text)}")
                        
                    elif isinstance(m, ToolResultMessage):
                        text = m.content[0].text if m.content else ""
                        print(f"  {i:2d} [tool:{m.tool_name}] {truncate(text, limit=60)}")
            elif cmd == "/tools":
                for t in agent.tools:
                    print(f"  {t.name} — {t.description}")
            elif cmd == "/branch":
                prompt = arg.strip() or "Summarise what we have discussed."
                print(f"[branching: '{prompt}']")
                child = agent.branch()
                result = await child.run(prompt)
                texts = [b.text for b in result.content if isinstance(b, TextContent)]
                print("\n[branch result]:", "".join(texts).strip())
            elif cmd == "/abort":
                agent.abort()
                print("[abort signalled]")
            else:
                print(f"Unknown command: {cmd}  (try /help)")
            continue

        if is_tty:
            if user_input.lower() in {"exit", "quit"}:
                print("Bye!")
                break
            print("\033[34mπ\033[0m ", end="", flush=True)

        await agent.run(user_input)
        print()
        if not is_tty:
            break

def collect_skills(skills_dir, skills=None):
    skills = [] if skills is None else skills
    if skills_dir is not None:
        root = pathlib.Path(skills_dir)
        if root.exists():
            for skill_dir in sorted(root.iterdir()):
                md = skill_dir / "SKILL.md"
                if not md.is_file():
                    continue
                text = md.read_text(encoding="utf-8", errors="replace")
                name = skill_dir.name
                description = ""
                if text.startswith("---"):
                    end = text.find("---", 3)
                    if end != -1:
                        fm = text[3:end]
                        n = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
                        if n:
                            name = n.group(1).strip()
                        m = re.search(r"^description:\s*(.+)$", fm, re.MULTILINE)
                        if m:
                            description = m.group(1).strip()
                        text = text[end+3:].strip()
                        if not n and not m:
                            continue
                        skills.append({"name": name, "description": description, "content": text})

    skill_prompt = "Available skills (use GetSkill to load full instructions when needed):\n" + '\n'.join(f"- {s['name']}: {s['description']}" for s in skills) if skills else ''
    return skills, skill_prompt

def harness(
    *,
    base_url: str | None = None,
    model: str | None = None,
    api: Literal["claude", "codex", "gemini", "deepseek", "pie"] = "pie",
    system: str = "",
    sdir: str| None = None,
    skills: list[dict] | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    tools: list[str] | None = None,
    api_key: str | None = None,
    gwt = None,
) -> None:
    cwd = os.getcwd() if cwd is None else os.path.abspath(cwd)
    sdir = cwd if sdir is None else os.path.abspath(sdir)
    if env is not None:
        os.environ.clear()
        os.environ.update(env)
    os.chdir(cwd)
    skills, skill_prompt = collect_skills(sdir, skills)
    tools = collect_tools(tools, skills, cwd)
    system = '\n\n'.join(filter(None, [system, skill_prompt]))

    try:
        asyncio.run(repl(api, system, tools=tools, skills=skills, api_key=api_key, base_url=base_url, gwt=gwt))
    except KeyboardInterrupt:
        print("\nExiting...")


def main():
    from .main import setup_logger #.
    setup_logger(log_file='log.json', console=True)
    parser = argparse.ArgumentParser(description="Pie REPL")
    parser.add_argument("-a", "--api", choices=["claude", "codex", "gemini", "deepseek", "pie"], default="pie", help="API Provider")
    parser.add_argument("-m", "--model", type=str, default=None, help="Model name")
    parser.add_argument("-t", "--tools", nargs='+', help="List of tools to enable (e.g., Bash Read Ls). Defaults to all.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Base URL for the API")
    parser.add_argument("--system", type=str, default="", help="System prompt")
    parser.add_argument("--skill", type=str, default=None, help="Directory to scan for skills")
    parser.add_argument("--cwd", type=str, default=None, help="Current working directory")
    parser.add_argument("--key", type=str, default=None, help="API key")
    args = parser.parse_args()
    logger.debug(args)

    url = args.url
    model = args.model
    tools = args.tools

    api_key = args.key

    if args.api in ["deepseek", "gemini"]:
        if args.api=="deepseek":
            api_key = os.environ.get('DEEPSEEK_API_KEY') if api_key is None else api_key
            url = "https://api.deepseek.com" if api_key else url
            model = "deepseek-v4-flash" if model is None else model
        elif args.api=='gemini':
            api_key = os.environ.get('GEMINI_API_KEY') if api_key is None else api_key
            url = "https://generativelanguage.googleapis.com" if api_key else url
            model = "gemini-3.1-flash-lite-preview" if model is None else model
        tools = [] if tools is None else tools # □

    import tempfile
    repo = args.cwd or os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        gwt = create_worktree(repo, worktree_dir=os.path.join(tmp, 'workspace'))

        harness(
            api=args.api,
            system=args.system,
            cwd=gwt.worktree,       
            model=model,
            base_url=url,
            tools=tools,
            sdir=args.skill,
            api_key=api_key,
            gwt=gwt,        
        )

if __name__ == "__main__":
    main()

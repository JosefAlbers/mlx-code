# Based on pi (https://github.com/badlogic/pi-mono) by Mario Zechner (MIT License)
#
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

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
    Union,
    TypeVar,
    Generic,
    cast
)

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]
Role = Literal["system", "user", "assistant", "toolResult"]

@dataclass(slots=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: dict[str, float] = field(default_factory=lambda: {"total": 0.0})


@dataclass(slots=True)
class TextContent:
    text: str
    type: Literal["text"] = "text"


@dataclass(slots=True)
class ThinkingContent:
    thinking: str
    type: Literal["thinking"] = "thinking"
    redacted: bool = False
    signature: str | None = None


@dataclass(slots=True)
class ImageContent:
    data: str  
    mime_type: str
    type: Literal["image"] = "image"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["toolCall"] = "toolCall"


AssistantContent = Union[TextContent, ThinkingContent, ToolCall]


@dataclass(slots=True)
class UserMessage:
    content: str | list[Union[TextContent, ImageContent]]
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    role: Literal["user"] = "user"


@dataclass(slots=True)
class AssistantMessage:
    api: str
    provider: str
    model: str
    content: list[AssistantContent] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    role: Literal["assistant"] = "assistant"
    response_id: str | None = None


@dataclass(slots=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: list[Union[TextContent, ImageContent]]
    is_error: bool = False
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    role: Literal["toolResult"] = "toolResult"
    details: Any = None


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]

TParams = TypeVar("TParams", bound=BaseModel)
TDetails = TypeVar("TDetails")

@dataclass(slots=True)
class AgentToolResult(Generic[TDetails]):
    content: list[Union[TextContent, ImageContent]]
    details: TDetails

AgentToolUpdateCallback = Callable[[AgentToolResult[Any]], Awaitable[None]]

from abc import ABC, abstractmethod

class AgentTool(ABC, Generic[TParams, TDetails]):
    name: str
    description: str
    parameters: type[TParams]
    label: str | None = None

    @abstractmethod
    async def execute(
        self,
        tool_call_id: str,
        params: TParams,
        signal: asyncio.Event | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult[TDetails]: ...

    def prepare_arguments(self, args: Any) -> dict[str, Any]:
        """Optional hook to shim/fix raw tool-call arguments before validation."""
        return cast(dict[str, Any], args)

def validate_tool_arguments(tool: AgentTool[TParams, TDetails], tool_call: ToolCall) -> TParams:
    """Validates tool call arguments against the tool's Pydantic model."""
    try:
        prepared_args = tool.prepare_arguments(tool_call.arguments)
        return tool.parameters.model_validate(prepared_args)
    except ValidationError as e:
        error_details = "\n".join(f"  - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors())
        error_msg = (
            f"Validation failed for tool '{tool_call.name}':\n{error_details}\n\n"
            f"Received arguments:\n{json.dumps(tool_call.arguments, indent=2)}"
        )
        raise ValueError(error_msg)

@dataclass(slots=True)
class Model:
    id: str
    provider: str
    api: str
    name: str = ""
    baseUrl: str = ""
    reasoning: bool = False
    contextWindow: int = 4096
    maxTokens: int = 1024


@dataclass(slots=True)
class ProviderOptions:
    api_key: str | None = None
    base_url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning: Literal["off", "minimal", "low", "medium", "high", "xhigh"] = "off"


EventType = Literal[
    "start",
    "text_start",
    "text_delta",
    "text_end",
    "thinking_start",
    "thinking_delta",
    "thinking_end",
    "toolcall_start",
    "toolcall_delta",
    "toolcall_end",
    "done",
    "error",
]


@dataclass(slots=True)
class Event:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)


class AssistantEventStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._result: AssistantMessage | None = None

    def push(self, event: Event) -> None:
        self._queue.put_nowait(event)

    def end(self, result: AssistantMessage) -> None:
        self._result = result
        self._queue.put_nowait(None)

    async def result(self) -> AssistantMessage:
        if self._result is None:
            async for _ in self:
                pass
        assert self._result is not None
        return self._result

    def __aiter__(self) -> "AssistantEventStream":
        return self

    async def __anext__(self) -> Event:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item


class Provider(Protocol):
    async def stream(
        self, model: Model, context: Context, options: ProviderOptions | None = None
    ) -> AssistantEventStream: ...


@dataclass(slots=True)
class Context:
    model_id: str
    provider: str
    api: str
    messages: list[Message] = field(default_factory=list)
    system_prompt: str = ""
    tools: list[AgentTool] = field(default_factory=list)


@dataclass(slots=True)
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: Any
    context: Context

@dataclass(slots=True)
class BeforeToolCallResult:
    block: bool = False
    reason: str | None = None

@dataclass(slots=True)
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: Any
    result: AgentToolResult[Any]
    is_error: bool
    context: Context

@dataclass(slots=True)
class AfterToolCallResult:
    content: list[Union[TextContent, ImageContent]] | None = None
    details: Any = None
    is_error: bool | None = None

@dataclass(slots=True)
class AgentConfig:
    tool_execution: Literal["sequential", "parallel"] = "parallel"
    convert_to_llm: Callable[[list[Message]], Awaitable[list[Message]]] | None = None
    transform_context: Callable[[list[Message]], Awaitable[list[Message]]] | None = None
    before_tool_call: (
        Callable[[BeforeToolCallContext, asyncio.Event | None], Awaitable[BeforeToolCallResult | None]] | None
    ) = None
    after_tool_call: (
        Callable[[AfterToolCallContext, asyncio.Event | None], Awaitable[AfterToolCallResult | None]] | None
    ) = None
    get_steering_messages: Callable[[], Awaitable[list[Message]]] | None = None
    get_follow_up_messages: Callable[[], Awaitable[list[Message]]] | None = None

@dataclass(slots=True)
class AgentEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

AgentEventSink = Callable[[AgentEvent], Awaitable[None]]

async def default_convert(messages: list[Message]) -> list[Message]:
    return [
        m for m in messages if getattr(m, "role", None) in {"user", "assistant", "toolResult"}
    ]

async def run_agent_loop(
    context: Context,
    emit: AgentEventSink,
    registry: ProviderRegistry,
    tools: list[AgentTool] | None = None,
    config: AgentConfig | None = None,
    provider_options: ProviderOptions | None = None,
    signal: asyncio.Event | None = None,
) -> list[Message]:
    cfg = config or AgentConfig()
    new_messages: list[Message] = []
    current_context = context
    if tools:
        current_context.tools = tools

    await emit(AgentEvent(type="agent_start"))

    model = Model(id=context.model_id, provider=context.provider, api=context.api)

    pending_steering = (
        await cfg.get_steering_messages() if cfg.get_steering_messages else []
    )

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_steering:
            await emit(AgentEvent(type="turn_start"))
            first_turn = False

            if pending_steering:
                for msg in pending_steering:
                    await emit(AgentEvent(type="message_start", payload={"message": msg}))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                    await emit(AgentEvent(type="message_end", payload={"message": msg}))
                pending_steering = []

            transformed = (
                await cfg.transform_context(current_context.messages)
                if cfg.transform_context
                else current_context.messages
            )
            llm_messages = (
                await cfg.convert_to_llm(transformed)
                if cfg.convert_to_llm
                else await default_convert(transformed)
            )

            llm_context = Context(
                model_id=context.model_id,
                provider=context.provider,
                api=context.api,
                system_prompt=context.system_prompt,
                messages=llm_messages,
                tools=context.tools,
            )
            
            stream = await registry.stream(model, llm_context, provider_options)

            assistant: AssistantMessage | None = None
            added_partial = False
            
            async for event in stream:
                if event.type == "start":
                    assistant = event.payload["partial"]
                    current_context.messages.append(assistant)
                    added_partial = True
                    await emit(AgentEvent(type="message_start", payload={"message": assistant}))
                elif event.type in {"text_delta", "thinking_delta", "toolcall_delta", "toolcall_start", "toolcall_end"}:
                    if assistant:
                        assistant = event.payload["partial"]
                        current_context.messages[-1] = assistant
                        await emit(
                            AgentEvent(
                                type="message_update",
                                payload={"message": assistant, "assistant_message_event": event},
                            )
                        )
                elif event.type == "error":
                    await emit(AgentEvent(type="error", payload=event.payload))

            final_assistant = await stream.result()
            if added_partial:
                current_context.messages[-1] = final_assistant
            else:
                current_context.messages.append(final_assistant)
                await emit(AgentEvent(type="message_start", payload={"message": final_assistant}))
            
            new_messages.append(final_assistant)
            await emit(AgentEvent(type="message_end", payload={"message": final_assistant}))

            if final_assistant.stop_reason in {"error", "aborted"}:
                await emit(AgentEvent(type="turn_end", payload={"message": final_assistant, "tool_results": []}))
                await emit(AgentEvent(type="agent_end", payload={"messages": new_messages}))
                return new_messages

            tool_calls = [c for c in final_assistant.content if isinstance(c, ToolCall)]
            has_more_tool_calls = len(tool_calls) > 0
            
            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                tool_results = await execute_tool_calls(current_context, final_assistant, tool_calls, cfg, signal, emit)
                for res in tool_results:
                    current_context.messages.append(res)
                    new_messages.append(res)

            await emit(AgentEvent(type="turn_end", payload={"message": final_assistant, "tool_results": tool_results}))
            pending_steering = (await cfg.get_steering_messages()) if cfg.get_steering_messages else []

        follow_ups = (await cfg.get_follow_up_messages()) if cfg.get_follow_up_messages else []
        if not follow_ups:
            break
        pending_steering = follow_ups

    await emit(AgentEvent(type="agent_end", payload={"messages": new_messages}))
    return new_messages

async def execute_tool_calls(
    context: Context,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    if config.tool_execution == "sequential":
        return await execute_tool_calls_sequential(context, assistant_message, tool_calls, config, signal, emit)
    return await execute_tool_calls_parallel(context, assistant_message, tool_calls, config, signal, emit)

async def execute_tool_calls_sequential(
    context: Context,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    results = []
    for call in tool_calls:
        results.append(await handle_single_tool_call(context, assistant_message, call, config, signal, emit))
    return results

async def execute_tool_calls_parallel(
    context: Context,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    tasks = []
    for call in tool_calls:
        tasks.append(handle_single_tool_call(context, assistant_message, call, config, signal, emit))
    return list(await asyncio.gather(*tasks))

async def handle_single_tool_call(
    context: Context,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> ToolResultMessage:
    await emit(AgentEvent(type="tool_execution_start", payload={
        "tool_call_id": tool_call.id,
        "tool_name": tool_call.name,
        "args": tool_call.arguments
    }))

    tool = next((t for t in context.tools if t.name == tool_call.name), None)
    if not tool:
        return await finalize_tool_result(
            tool_call, 
            AgentToolResult(content=[TextContent(text=f"Tool {tool_call.name} not found")], details={"error": "not found"}), 
            True, emit
        )

    try:
        validated_args = validate_tool_arguments(tool, tool_call)
        
        if config.before_tool_call:
            before_ctx = BeforeToolCallContext(assistant_message, tool_call, validated_args, context)
            before_res = await config.before_tool_call(before_ctx, signal)
            if before_res and before_res.block:
                return await finalize_tool_result(
                    tool_call,
                    AgentToolResult(content=[TextContent(text=before_res.reason or "Tool blocked")], details={"error": "blocked"}),
                    True, emit
                )

        async def on_update(partial: AgentToolResult[Any]):
            await emit(AgentEvent(type="tool_execution_update", payload={
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                "partial_result": partial
            }))

        result = await tool.execute(tool_call.id, validated_args, signal, on_update)
        is_error = False

        if config.after_tool_call:
            after_ctx = AfterToolCallContext(assistant_message, tool_call, validated_args, result, is_error, context)
            after_res = await config.after_tool_call(after_ctx, signal)
            if after_res:
                if after_res.content is not None: result.content = after_res.content
                if after_res.details is not None: result.details = after_res.details
                if after_res.is_error is not None: is_error = after_res.is_error

        return await finalize_tool_result(tool_call, result, is_error, emit)

    except Exception as e:
        return await finalize_tool_result(
            tool_call,
            AgentToolResult(content=[TextContent(text=str(e))], details={"error": str(e)}),
            True, emit
        )

async def finalize_tool_result(
    tool_call: ToolCall,
    result: AgentToolResult[Any],
    is_error: bool,
    emit: AgentEventSink
) -> ToolResultMessage:
    await emit(AgentEvent(type="tool_execution_end", payload={
        "tool_call_id": tool_call.id,
        "tool_name": tool_call.name,
        "result": result,
        "is_error": is_error
    }))

    msg = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details=result.details,
        is_error=is_error
    )
    
    await emit(AgentEvent(type="message_start", payload={"message": msg}))
    await emit(AgentEvent(type="message_end", payload={"message": msg}))
    return msg


class FauxProvider:
    async def stream(
        self, model: Model, context: Context, options: ProviderOptions | None = None
    ) -> AssistantEventStream:
        stream = AssistantEventStream()
        msg = AssistantMessage(api=model.api, provider=model.provider, model=model.id)
        
        last_msg = context.messages[-1] if context.messages else None
        
        if isinstance(last_msg, ToolResultMessage):
            stream.push(Event(type="start", payload={"partial": msg}))
            stream.push(Event(type="text_start", payload={"partial": msg}))
            text = f"I received the result for {last_msg.tool_name}: {last_msg.content[0].text if last_msg.content else 'empty'}"
            msg.content.append(TextContent(text=text))
            stream.push(Event(type="text_delta", payload={"delta": text, "partial": msg}))
            stream.push(Event(type="text_end", payload={"partial": msg}))
            stream.push(Event(type="done", payload={"reason": "stop", "message": msg}))
        else:
            stream.push(Event(type="start", payload={"partial": msg}))
            last_user = next((m for m in reversed(context.messages) if isinstance(m, UserMessage)), None)
            text = "ACK"
            if last_user:
                content = last_user.content if isinstance(last_user.content, str) else (last_user.content[0].text if last_user.content else "")
                text = f"ACK: {content}"

            if "call_tool" in text:
                tool_name = next(
                    (t.name for t in context.tools if t.name in text),
                    context.tools[0].name if context.tools else "echo"
                )
                stream.push(Event(type="text_start", payload={"partial": msg}))
                msg.content.append(TextContent(text=f"I will call the {tool_name} tool now."))
                stream.push(Event(type="text_delta", payload={"delta": f"I will call the {tool_name} tool now.", "partial": msg}))
                stream.push(Event(type="text_end", payload={"partial": msg}))
                
                call = ToolCall(id=str(uuid.uuid4()), name=tool_name, arguments={"message": text})
                msg.content.append(call)
                msg.stop_reason = "toolUse"
                stream.push(Event(type="toolcall_start", payload={"partial": msg}))
                stream.push(Event(type="toolcall_end", payload={"toolCall": call, "partial": msg}))
                stream.push(Event(type="done", payload={"reason": "toolUse", "message": msg}))
            else:
                stream.push(Event(type="text_start", payload={"partial": msg}))
                msg.content.append(TextContent(text=text))
                stream.push(Event(type="text_delta", payload={"delta": text, "partial": msg}))
                stream.push(Event(type="text_end", payload={"partial": msg}))
                stream.push(Event(type="done", payload={"reason": "stop", "message": msg}))

        stream.end(msg)
        return stream


class OpenAIResponsesProvider:
    async def stream(
        self, model: Model, context: Context, options: ProviderOptions | None = None
    ) -> AssistantEventStream:
        stream = AssistantEventStream()
        opts = options or ProviderOptions()
        base_url = opts.base_url or "https://api.openai.com/v1"

        def format_content(content):
            if isinstance(content, str): return content
            return " ".join(c.text for c in content if isinstance(c, TextContent))

        async with httpx.AsyncClient(timeout=opts.timeout_seconds) as client:
            try:
                req_body = {
                    "model": model.id,
                    "input": [{"role": getattr(m, "role", "user"), "content": format_content(m.content)} for m in context.messages],
                }
                if context.system_prompt: req_body["instructions"] = context.system_prompt
                
                resp = await client.post(
                    f"{base_url}/responses",
                    json=req_body,
                    headers={"Authorization": f"Bearer {opts.api_key}"},
                )
                if resp.status_code >= 400:
                    raise Exception(f"HTTP {resp.status_code}: {await resp.aread()}")
                
                data = resp.json()
                text = data.get("output_text", "")

                msg = AssistantMessage(api=model.api, provider=model.provider, model=model.id)
                stream.push(Event(type="start", payload={"partial": msg}))
                stream.push(Event(type="text_start", payload={"partial": msg}))
                msg.content.append(TextContent(text=text))
                stream.push(Event(type="text_delta", payload={"delta": text, "partial": msg}))
                stream.push(Event(type="text_end", payload={"partial": msg}))
                stream.push(Event(type="done", payload={"reason": "stop", "message": msg}))
                stream.end(msg)
            except Exception as e:
                msg = AssistantMessage(api=model.api, provider=model.provider, model=model.id, stop_reason="error", error_message=str(e))
                stream.push(Event(type="error", payload={"reason": "error", "error": msg}))
                stream.end(msg)
        return stream


class AnthropicMessagesProvider:
    async def stream(
        self, model: Model, context: Context, options: ProviderOptions | None = None
    ) -> AssistantEventStream:
        stream = AssistantEventStream()
        opts = options or ProviderOptions()
        if not opts.api_key:
            msg = AssistantMessage(api=model.api, provider=model.provider, model=model.id, stop_reason="error", error_message="missing api key")
            stream.push(Event(type="error", payload={"reason": "error", "error": msg}))
            stream.end(msg)
            return stream

        base_url = opts.base_url or "https://api.anthropic.com/v1"

        def format_content(content):
            if isinstance(content, str): return content
            res = []
            for item in content:
                if isinstance(item, TextContent): res.append({"type": "text", "text": item.text})
                elif isinstance(item, ImageContent): res.append({"type": "image", "source": {"type": "base64", "media_type": item.mime_type, "data": item.data}})
                elif isinstance(item, ToolCall): res.append({"type": "tool_use", "id": item.id, "name": item.name, "input": item.arguments})
            return res

        messages = []
        for m in context.messages:
            role = getattr(m, "role", None)
            if role == "user": messages.append({"role": "user", "content": format_content(m.content)})
            elif role == "assistant": messages.append({"role": "assistant", "content": format_content(m.content)})
            elif role == "toolResult":
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": format_content(m.content),
                        "is_error": getattr(m, "is_error", False)
                    }]
                })

        payload = {
            "model": model.id,
            "max_tokens": opts.max_tokens or 1024,
            "messages": messages,
            "stream": True,
        }
        if context.system_prompt: payload["system"] = [{"type": "text", "text": context.system_prompt}]
        if opts.temperature is not None: payload["temperature"] = opts.temperature
        
        if context.tools:
            anthropic_tools = []
            for t in context.tools:
                schema = t.parameters.model_json_schema()
                anthropic_tools.append({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", [])
                    }
                })
            payload["tools"] = anthropic_tools

        async def run():
            msg = AssistantMessage(api=model.api, provider=model.provider, model=model.id)
            try:
                async with httpx.AsyncClient(timeout=opts.timeout_seconds) as client:
                    async with client.stream(
                        "POST", f"{base_url}/messages", json=payload,
                        headers={
                            "x-api-key": opts.api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                    ) as resp:
                        if resp.status_code >= 400:
                            raise Exception(f"HTTP {resp.status_code}: {await resp.aread()}")

                        stream.push(Event(type="start", payload={"partial": msg}))
                        _tool_arg_buf: dict[int, str] = {}  
                        _block_index = 0
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "): continue
                            data_str = line[6:]
                            if data_str == "[DONE]": break
                            chunk = json.loads(data_str)
                            event_type = chunk.get("type")

                            if event_type == "content_block_start":
                                _block_index = chunk.get("index", _block_index)
                                block = chunk.get("content_block", {})
                                if block.get("type") == "text":
                                    stream.push(Event(type="text_start", payload={"partial": msg}))
                                    msg.content.append(TextContent(text=""))
                                elif block.get("type") == "tool_use":
                                    call = ToolCall(id=block["id"], name=block["name"], arguments={})
                                    msg.content.append(call)
                                    _tool_arg_buf[_block_index] = ""
                                    stream.push(Event(type="toolcall_start", payload={"partial": msg}))
                            elif event_type == "content_block_delta":
                                idx = chunk.get("index", _block_index)
                                delta = chunk.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if msg.content and isinstance(msg.content[-1], TextContent):
                                        msg.content[-1].text += text
                                    stream.push(Event(type="text_delta", payload={"delta": text, "partial": msg}))
                                elif delta.get("type") == "input_json_delta":
                                    _tool_arg_buf[idx] = _tool_arg_buf.get(idx, "") + delta.get("partial_json", "")
                            elif event_type == "content_block_stop":
                                idx = chunk.get("index", _block_index)
                                if msg.content and isinstance(msg.content[-1], TextContent):
                                    stream.push(Event(type="text_end", payload={"partial": msg}))
                                elif msg.content and isinstance(msg.content[-1], ToolCall):
                                    raw = _tool_arg_buf.pop(idx, "")
                                    if raw:
                                        try:
                                            msg.content[-1].arguments = json.loads(raw)
                                        except json.JSONDecodeError:
                                            pass
                                    stream.push(Event(type="toolcall_end", payload={"partial": msg}))
                            elif event_type == "message_delta":
                                if "stop_reason" in chunk.get("delta", {}):
                                    reason = chunk["delta"]["stop_reason"]
                                    msg.stop_reason = "toolUse" if reason == "tool_use" else "stop"
                                if "usage" in chunk:
                                    usage = chunk["usage"]
                                    msg.usage.input = usage.get("input_tokens", 0)
                                    msg.usage.output = usage.get("output_tokens", 0)
                            elif event_type == "message_stop":
                                stream.push(Event(type="done", payload={"reason": msg.stop_reason, "message": msg}))

                stream.end(msg)
            except Exception as e:
                msg.stop_reason = "error"
                msg.error_message = str(e)
                stream.push(Event(type="error", payload={"reason": "error", "error": msg}))
                stream.end(msg)

        asyncio.create_task(run())
        return stream


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {
            "faux": FauxProvider(),
            "openai-responses": OpenAIResponsesProvider(),
            "anthropic-messages": AnthropicMessagesProvider(),
        }

    def register(self, api: str, provider: Provider) -> None:
        self._providers[api] = provider

    async def stream(
        self, model: Model, context: Context, options: ProviderOptions | None = None
    ) -> AssistantEventStream:
        provider = self._providers.get(model.api)
        if not provider: raise ValueError(f"Provider for API {model.api} not found")
        return await provider.stream(model, context, options)


class Agent:
    def __init__(self, context: Context, registry: ProviderRegistry | None = None) -> None:
        self.context = context
        self.registry = registry or ProviderRegistry()
        self.tools: list[AgentTool] = []
        self.config = AgentConfig()
        self.provider_options = ProviderOptions()
        self._listeners: set[Callable[[AgentEvent], Awaitable[None]]] = set()

    def subscribe(self, listener: Callable[[AgentEvent], Awaitable[None]]) -> Callable[[], None]:
        self._listeners.add(listener)
        return lambda: self._listeners.remove(listener)

    async def _emit(self, event: AgentEvent) -> None:
        if not self._listeners: return
        await asyncio.gather(*(l(event) for l in self._listeners))

    async def prompt(self, text: str) -> list[Message]:
        msg = UserMessage(content=text)
        self.context.messages.append(msg)
        await self._emit(AgentEvent(type="message_start", payload={"message": msg}))
        await self._emit(AgentEvent(type="message_end", payload={"message": msg}))
        return await self.continue_run()

    async def continue_run(self) -> list[Message]:
        return await run_agent_loop(
            self.context,
            self._emit,
            self.registry,
            self.tools,
            self.config,
            self.provider_options,
        )

class EchoParams(BaseModel):
    message: str = Field(description="The message to echo back")

class EchoTool(AgentTool):
    name = "echo"
    description = "Echos back the input message"
    parameters = EchoParams
    label = "Echo Tool"

    async def execute(self, tool_call_id, params: EchoParams, signal=None, on_update=None) -> AgentToolResult[dict]:
        if on_update:
            await on_update(AgentToolResult(content=[TextContent(text="[Thinking about echoing...]")], details={}))
            await asyncio.sleep(0.1)
        return AgentToolResult(
            content=[TextContent(text=f"Echo: {params.message}")],
            details={"echoed": params.message}
        )

import os
import re
import shlex
import fnmatch
import subprocess
import pathlib

_MAX_BYTES = 50 * 1024   # 50 KB
_MAX_LINES = 2000

def _truncate(text: str, label: str = "") -> str:
    """Hard-cap output to 50 KB / 2000 lines, whichever comes first."""
    lines = text.splitlines(keepends=True)
    if len(lines) > _MAX_LINES:
        lines = lines[:_MAX_LINES]
        text = "".join(lines) + f"\n[truncated at {_MAX_LINES} lines{': ' + label if label else ''}]"
    if len(text.encode()) > _MAX_BYTES:
        text = text.encode()[:_MAX_BYTES].decode(errors="replace") + f"\n[truncated at 50 KB{': ' + label if label else ''}]"
    return text

def _resolve(path: str, cwd: str) -> str:
    """Resolve path relative to cwd, preventing directory traversal above cwd."""
    base = pathlib.Path(cwd).resolve()
    resolved = (base / path).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        raise ValueError(f"Path '{path}' escapes working directory '{cwd}'")
    return str(resolved)



class ReadParams(BaseModel):
    path: str = Field(description="File path to read (relative to cwd)")
    offset: int | None = Field(default=None, description="Start line (1-based, inclusive)")
    limit: int | None = Field(default=None, description="Max number of lines to read")

class ReadTool(AgentTool):
    name = "read"
    description = (
        "Read the contents of a file. "
        "Supports optional line-range with offset (1-based) and limit. "
        "Use this instead of cat or sed."
    )
    parameters = ReadParams
    label = "Read"

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, tool_call_id, params: ReadParams, signal=None, on_update=None) -> AgentToolResult[dict]:
        path = _resolve(params.path, self.cwd)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except FileNotFoundError:
            raise ValueError(f"File not found: {params.path}")
        except IsADirectoryError:
            raise ValueError(f"Path is a directory: {params.path}")

        start = (params.offset - 1) if params.offset else 0
        start = max(0, start)
        end = (start + params.limit) if params.limit else len(lines)
        sliced = lines[start:end]
        text = "".join(sliced)
        total = len(lines)
        header = f"# {params.path}  (lines {start+1}-{min(end, total)} of {total})\n"
        return AgentToolResult(
            content=[TextContent(text=_truncate(header + text, params.path))],
            details={"path": path, "total_lines": total}
        )



class WriteParams(BaseModel):
    path: str = Field(description="File path to create or overwrite (relative to cwd)")
    content: str = Field(description="Full content to write")

class WriteTool(AgentTool):
    name = "write"
    description = (
        "Create a new file or completely overwrite an existing file. "
        "Use for new files or full rewrites; prefer edit for surgical changes."
    )
    parameters = WriteParams
    label = "Write"

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, tool_call_id, params: WriteParams, signal=None, on_update=None) -> AgentToolResult[dict]:
        path = _resolve(params.path, self.cwd)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        existed = os.path.exists(path)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(params.content)
        action = "overwrote" if existed else "created"
        lines = params.content.count("\n") + 1
        return AgentToolResult(
            content=[TextContent(text=f"{action} {params.path} ({lines} lines)")],
            details={"path": path, "action": action, "lines": lines}
        )



class EditParams(BaseModel):
    path: str = Field(description="File path to edit (relative to cwd)")
    old_text: str = Field(description="Exact text to find (must match exactly once)")
    new_text: str = Field(description="Replacement text")

class EditTool(AgentTool):
    name = "edit"
    description = (
        "Make a surgical edit to a file by replacing an exact string occurrence. "
        "old_text must appear exactly once in the file. "
        "Use read to inspect the file before editing."
    )
    parameters = EditParams
    label = "Edit"

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, tool_call_id, params: EditParams, signal=None, on_update=None) -> AgentToolResult[dict]:
        path = _resolve(params.path, self.cwd)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                original = fh.read()
        except FileNotFoundError:
            raise ValueError(f"File not found: {params.path}")

        count = original.count(params.old_text)
        if count == 0:
            raise ValueError(f"old_text not found in {params.path}")
        if count > 1:
            raise ValueError(
                f"old_text appears {count} times in {params.path}; "
                "make it more specific so it matches exactly once"
            )

        updated = original.replace(params.old_text, params.new_text, 1)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(updated)

        old_lines = params.old_text.count("\n") + 1
        new_lines = params.new_text.count("\n") + 1
        return AgentToolResult(
            content=[TextContent(text=f"edited {params.path}: replaced {old_lines} line(s) with {new_lines} line(s)")],
            details={"path": path, "old_lines": old_lines, "new_lines": new_lines}
        )



class BashParams(BaseModel):
    command: str = Field(description="Shell command to execute")
    timeout: int | None = Field(default=120, description="Timeout in seconds (default 120)")

class BashTool(AgentTool):
    name = "bash"
    description = (
        "Execute a bash command synchronously and return stdout+stderr. "
        "Use grep/find/ls tools for file exploration instead of bash. "
        "Commands run in cwd. No background processes."
    )
    parameters = BashParams
    label = "Bash"

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, tool_call_id, params: BashParams, signal=None, on_update=None) -> AgentToolResult[dict]:
        timeout = params.timeout or 120
        try:
            proc = await asyncio.create_subprocess_shell(
                params.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.cwd,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                raise ValueError(f"Command timed out after {timeout}s: {params.command}")
            output = stdout.decode(errors="replace")
            exit_code = proc.returncode
        except Exception as e:
            raise ValueError(f"bash error: {e}") from e

        result = f"$ {params.command}\n{output}"
        if exit_code != 0:
            result += f"\n[exit code {exit_code}]"

        is_error = exit_code != 0
        return AgentToolResult(
            content=[TextContent(text=_truncate(result, params.command))],
            details={"exit_code": exit_code, "command": params.command}
        )



class GrepParams(BaseModel):
    pattern: str = Field(description="Regex or literal pattern to search for")
    path: str | None = Field(default=None, description="Directory or file to search (default: cwd)")
    glob: str | None = Field(default=None, description="File glob filter e.g. '*.py'")
    ignore_case: bool = Field(default=False, description="Case-insensitive match")
    literal: bool = Field(default=False, description="Treat pattern as a literal string")
    context: int | None = Field(default=None, description="Lines of context before/after match")
    limit: int | None = Field(default=100, description="Max matches to return")

class GrepTool(AgentTool):
    name = "grep"
    description = (
        "Search files for a regex or literal pattern. "
        "Respects .gitignore. Faster than using bash grep for codebase exploration."
    )
    parameters = GrepParams
    label = "Grep"

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, tool_call_id, params: GrepParams, signal=None, on_update=None) -> AgentToolResult[dict]:
        search_root = _resolve(params.path, self.cwd) if params.path else self.cwd
        limit = params.limit if params.limit is not None else 100

        try:
            raw_pattern = params.pattern if params.literal else params.pattern
            flags = re.IGNORECASE if params.ignore_case else 0
            if params.literal:
                compiled = re.compile(re.escape(params.pattern), flags)
            else:
                compiled = re.compile(params.pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

        matches: list[str] = []
        count = 0

        def _matches_glob(filename: str) -> bool:
            if not params.glob:
                return True
            return fnmatch.fnmatch(filename, params.glob)

        def _is_gitignored(dirpath: str, entries: list[str]) -> set[str]:
            """Simple .gitignore line matching (prefix/exact only)."""
            gitignore = os.path.join(dirpath, ".gitignore")
            ignored: set[str] = set()
            if os.path.exists(gitignore):
                with open(gitignore) as f:
                    patterns = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                for entry in entries:
                    for p in patterns:
                        if fnmatch.fnmatch(entry, p) or fnmatch.fnmatch(entry + "/", p):
                            ignored.add(entry)
            return ignored

        search_path = pathlib.Path(search_root)

        if search_path.is_file():
            file_list = [search_path]
        else:
            file_list = []
            for dirpath, dirnames, filenames in os.walk(search_root):
                ignored = _is_gitignored(dirpath, dirnames + filenames)
                dirnames[:] = [d for d in dirnames if d not in ignored and not d.startswith(".")]
                for fname in filenames:
                    if fname not in ignored and _matches_glob(fname):
                        file_list.append(pathlib.Path(dirpath) / fname)

        for fpath in file_list:
            if count >= limit:
                break
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    file_lines = fh.readlines()
            except (OSError, IsADirectoryError):
                continue

            rel = os.path.relpath(str(fpath), self.cwd)
            ctx = params.context or 0

            for i, line in enumerate(file_lines):
                if count >= limit:
                    break
                if compiled.search(line):
                    start = max(0, i - ctx)
                    end = min(len(file_lines), i + ctx + 1)
                    for j in range(start, end):
                        sep = ":" if j == i else "-"
                        matches.append(f"{rel}:{j+1}{sep}{file_lines[j].rstrip()}")
                    if ctx > 0:
                        matches.append("--")
                    count += 1

        if not matches:
            text = f"No matches for '{params.pattern}'"
        else:
            text = "\n".join(matches)
            if count >= limit:
                text += f"\n[truncated at {limit} matches]"

        return AgentToolResult(
            content=[TextContent(text=_truncate(text))],
            details={"match_count": count, "pattern": params.pattern}
        )



class FindParams(BaseModel):
    pattern: str = Field(description="Glob pattern e.g. '**/*.py'")
    path: str | None = Field(default=None, description="Search root directory (default: cwd)")
    limit: int | None = Field(default=1000, description="Max results")

class FindTool(AgentTool):
    name = "find"
    description = (
        "Find files matching a glob pattern. "
        "Respects .gitignore. Use instead of bash find for codebase exploration."
    )
    parameters = FindParams
    label = "Find"

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, tool_call_id, params: FindParams, signal=None, on_update=None) -> AgentToolResult[dict]:
        search_root = _resolve(params.path, self.cwd) if params.path else self.cwd
        limit = params.limit if params.limit is not None else 1000

        results: list[str] = []

        def _is_gitignored_name(name: str, dirpath: str) -> bool:
            gitignore = os.path.join(dirpath, ".gitignore")
            if os.path.exists(gitignore):
                with open(gitignore) as f:
                    patterns = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                for p in patterns:
                    if fnmatch.fnmatch(name, p):
                        return True
            return False

        pattern = params.pattern
        simple = pattern.lstrip("*/")

        for dirpath, dirnames, filenames in os.walk(search_root):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and not _is_gitignored_name(d, dirpath)
            ]
            for fname in filenames:
                if len(results) >= limit:
                    break
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, self.cwd)
                if fnmatch.fnmatch(fname, simple) or fnmatch.fnmatch(rel, pattern):
                    results.append(rel)
            if len(results) >= limit:
                break

        if not results:
            text = f"No files matching '{params.pattern}'"
        else:
            text = "\n".join(sorted(results))
            if len(results) >= limit:
                text += f"\n[truncated at {limit} results]"

        return AgentToolResult(
            content=[TextContent(text=_truncate(text))],
            details={"count": len(results), "pattern": params.pattern}
        )



class LsParams(BaseModel):
    path: str | None = Field(default=None, description="Directory to list (default: cwd)")
    limit: int | None = Field(default=500, description="Max entries")

class LsTool(AgentTool):
    name = "ls"
    description = (
        "List directory contents with file sizes and types. "
        "Respects .gitignore. Use instead of bash ls for codebase exploration."
    )
    parameters = LsParams
    label = "Ls"

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    async def execute(self, tool_call_id, params: LsParams, signal=None, on_update=None) -> AgentToolResult[dict]:
        dir_path = _resolve(params.path, self.cwd) if params.path else self.cwd
        limit = params.limit if params.limit is not None else 500

        if not os.path.isdir(dir_path):
            raise ValueError(f"Not a directory: {params.path or '.'}")

        gitignore_patterns: list[str] = []
        gi = os.path.join(dir_path, ".gitignore")
        if os.path.exists(gi):
            with open(gi) as f:
                gitignore_patterns = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        def is_ignored(name: str) -> bool:
            for p in gitignore_patterns:
                if fnmatch.fnmatch(name, p):
                    return True
            return False

        entries = []
        try:
            raw = sorted(os.listdir(dir_path))
        except PermissionError:
            raise ValueError(f"Permission denied: {params.path or '.'}")

        for name in raw:
            if is_ignored(name):
                continue
            full = os.path.join(dir_path, name)
            try:
                stat = os.stat(full)
                is_dir = os.path.isdir(full)
                size = stat.st_size
                entries.append((name, is_dir, size))
            except OSError:
                entries.append((name, False, 0))

        truncated = entries[:limit]
        rel_dir = os.path.relpath(dir_path, self.cwd)
        lines = [f"{rel_dir}/"]
        for name, is_dir, size in truncated:
            if is_dir:
                lines.append(f"  {name}/")
            else:
                size_str = f"{size:,}" if size < 1024 * 1024 else f"{size / (1024*1024):.1f}M"
                lines.append(f"  {name}  ({size_str} bytes)")

        if len(entries) > limit:
            lines.append(f"[truncated at {limit} entries, {len(entries)} total]")

        return AgentToolResult(
            content=[TextContent(text="\n".join(lines))],
            details={"path": dir_path, "count": len(truncated)}
        )



def create_coding_tools(cwd: str | None = None) -> list[AgentTool]:
    """read + write + edit + bash — the default coding agent set."""
    return [ReadTool(cwd), WriteTool(cwd), EditTool(cwd), BashTool(cwd)]

def create_read_only_tools(cwd: str | None = None) -> list[AgentTool]:
    """read + grep + find + ls — safe exploration without modification."""
    return [ReadTool(cwd), GrepTool(cwd), FindTool(cwd), LsTool(cwd)]

def create_all_tools(cwd: str | None = None) -> list[AgentTool]:
    """All 7 tools."""
    return [ReadTool(cwd), WriteTool(cwd), EditTool(cwd), BashTool(cwd),
            GrepTool(cwd), FindTool(cwd), LsTool(cwd)]



def _ok(label: str) -> None:
    print(f"  ✓ {label}")

def _fail(label: str, err: Any) -> None:
    print(f"  ✗ {label}: {err}")

async def simulate():
    """
    Smoke-tests all 7 pi coding tools against a real temp directory,
    plus the original echo/validation simulation.
    """
    import tempfile

    print("=== pie.py simulation ===\n")

    print("--- Part 1: Echo / Faux-Agent ---")
    context = Context(
        model_id="faux-model",
        provider="faux-provider",
        api="faux",
        system_prompt="You are a helpful assistant with tools."
    )
    agent = Agent(context)
    agent.tools = [EchoTool()]

    async def logger(event: AgentEvent):
        if event.type == "agent_start": print("[Agent Start]")
        elif event.type == "turn_start": print("  [Turn Start]")
        elif event.type == "message_start":
            msg = event.payload["message"]
            print(f"    [Message Start] Role: {getattr(msg, 'role', '?')}")
        elif event.type == "message_update":
            ev = event.payload.get("assistant_message_event")
            if ev and ev.type == "text_delta":
                print(f"      [Stream Delta] {ev.payload.get('delta')}")
        elif event.type == "message_end":
            msg = event.payload["message"]
            if isinstance(msg, AssistantMessage):
                text = "".join(c.text for c in msg.content if isinstance(c, TextContent))
                calls = [c for c in msg.content if isinstance(c, ToolCall)]
                if text: print(f"    [Assistant Text End] {text}")
                if calls: print(f"    [Assistant requested {len(calls)} tool calls]")
            elif isinstance(msg, ToolResultMessage):
                print(f"    [Tool Result End] {msg.tool_name}: {msg.content[0].text}")
        elif event.type == "tool_execution_start":
            print(f"      [Tool Start] {event.payload['tool_name']}({event.payload['args']})")
        elif event.type == "tool_execution_update":
            print(f"      [Tool Update] {event.payload['partial_result'].content[0].text}")
        elif event.type == "tool_execution_end":
            print(f"      [Tool End] {event.payload['tool_name']} (Error: {event.payload['is_error']})")
        elif event.type == "agent_end": print("[Agent End]")

    agent.subscribe(logger)
    print("\n> User: Please echo 'Hello world' by calling the tool call_tool:echo")
    await agent.prompt("Please echo 'Hello world' by calling the tool call_tool:echo")

    print("\n--- Validation Error ---")
    try:
        validate_tool_arguments(EchoTool(), ToolCall(id="x", name="echo", arguments={}))
    except ValueError as e:
        print(f"Caught expected error:\n{e}")

    print("\n\n--- Part 2: Coding Tools ---")

    with tempfile.TemporaryDirectory() as tmp:
        cwd = tmp

        async def run(tool: AgentTool, **kwargs: Any) -> str:
            params_model = tool.parameters(**kwargs)
            result = await tool.execute("test-id", params_model)
            return result.content[0].text

        write = WriteTool(cwd)
        read  = ReadTool(cwd)
        edit  = EditTool(cwd)
        bash  = BashTool(cwd)
        grep  = GrepTool(cwd)
        find  = FindTool(cwd)
        ls    = LsTool(cwd)

        try:
            out = await run(write, path="hello.py", content="print('hello')\nprint('world')\n")
            assert "created" in out
            _ok("write: create new file")
        except Exception as e:
            _fail("write: create new file", e)

        try:
            out = await run(write, path="hello.py", content="print('hi')\n")
            assert "overwrote" in out
            _ok("write: overwrite existing file")
        except Exception as e:
            _fail("write: overwrite", e)

        try:
            out = await run(read, path="hello.py")
            assert "print('hi')" in out
            _ok("read: full file")
        except Exception as e:
            _fail("read: full file", e)

        await run(write, path="lines.txt", content="\n".join(f"line{i}" for i in range(1, 21)) + "\n")
        try:
            out = await run(read, path="lines.txt", offset=5, limit=3)
            assert "line5" in out and "line7" in out and "line8" not in out
            _ok("read: offset + limit")
        except Exception as e:
            _fail("read: offset + limit", e)

        try:
            await run(read, path="no_such_file.py")
            _fail("read: missing file", "no error raised")
        except ValueError:
            _ok("read: missing file raises ValueError")

        await run(write, path="src.py", content="def foo():\n    return 1\n\ndef bar():\n    return 2\n")
        try:
            out = await run(edit, path="src.py", old_text="return 1", new_text="return 42")
            assert "edited" in out
            content = await run(read, path="src.py")
            assert "return 42" in content and "return 1" not in content
            _ok("edit: replace unique string")
        except Exception as e:
            _fail("edit: replace unique string", e)

        try:
            await run(edit, path="src.py", old_text="NOSUCHSTRING", new_text="x")
            _fail("edit: missing old_text", "no error raised")
        except ValueError:
            _ok("edit: missing old_text raises ValueError")

        await run(write, path="dup.py", content="x = 1\nx = 1\n")
        try:
            await run(edit, path="dup.py", old_text="x = 1", new_text="x = 2")
            _fail("edit: duplicate old_text", "no error raised")
        except ValueError:
            _ok("edit: duplicate old_text raises ValueError")

        try:
            out = await run(bash, command="echo pie_test_ok")
            assert "pie_test_ok" in out
            _ok("bash: echo command")
        except Exception as e:
            _fail("bash: echo command", e)

        try:
            out = await run(bash, command="exit 1")
            assert "exit code 1" in out
            _ok("bash: non-zero exit code captured")
        except Exception as e:
            _fail("bash: non-zero exit", e)

        await run(write, path="a.py", content="def foo():\n    pass\ndef bar():\n    pass\n")
        await run(write, path="b.py", content="class Foo:\n    def foo(self): pass\n")
        try:
            out = await run(grep, pattern="def foo")
            assert "a.py" in out
            _ok("grep: basic pattern across files")
        except Exception as e:
            _fail("grep: basic pattern", e)

        try:
            out = await run(grep, pattern="class", glob="*.py")
            assert "Foo" in out
            _ok("grep: glob filter")
        except Exception as e:
            _fail("grep: glob filter", e)

        try:
            out = await run(grep, pattern="DEF FOO", ignore_case=True)
            assert "foo" in out.lower()
            _ok("grep: ignore_case")
        except Exception as e:
            _fail("grep: ignore_case", e)

        try:
            out = await run(grep, pattern="def.foo", literal=True)
            assert "No matches" in out
            _ok("grep: literal (no regex)")
        except Exception as e:
            _fail("grep: literal", e)

        try:
            out = await run(grep, pattern="def foo", context=1)
            assert "--" in out  
            _ok("grep: context lines")
        except Exception as e:
            _fail("grep: context lines", e)

        os.makedirs(os.path.join(cwd, "sub"), exist_ok=True)
        await run(write, path="sub/c.py", content="# sub\n")
        try:
            out = await run(find, pattern="*.py")
            assert "a.py" in out and "sub/c.py" in out
            _ok("find: glob pattern")
        except Exception as e:
            _fail("find: glob pattern", e)

        try:
            out = await run(find, pattern="*.txt")
            assert "lines.txt" in out
            _ok("find: txt files")
        except Exception as e:
            _fail("find: txt files", e)

        try:
            out = await run(ls)
            assert "a.py" in out or "hello.py" in out
            _ok("ls: default cwd")
        except Exception as e:
            _fail("ls: default cwd", e)

        try:
            out = await run(ls, path="sub")
            assert "c.py" in out
            _ok("ls: subdirectory")
        except Exception as e:
            _fail("ls: subdirectory", e)

        try:
            coding = create_coding_tools(cwd)
            assert len(coding) == 4
            assert {t.name for t in coding} == {"read", "write", "edit", "bash"}
            _ok("create_coding_tools: 4 tools, correct names")
        except Exception as e:
            _fail("create_coding_tools", e)

        try:
            ro = create_read_only_tools(cwd)
            assert len(ro) == 4
            assert {t.name for t in ro} == {"read", "grep", "find", "ls"}
            _ok("create_read_only_tools: 4 tools, correct names")
        except Exception as e:
            _fail("create_read_only_tools", e)

        try:
            all_t = create_all_tools(cwd)
            assert len(all_t) == 7
            _ok("create_all_tools: 7 tools")
        except Exception as e:
            _fail("create_all_tools", e)

    print("\n=== Simulation Complete ===")


async def repl(base_url, api, system_prompt):
    context = Context(model_id="local-mod", provider="local-pro", api=api, system_prompt=system_prompt)
    agent = Agent(context)
    agent.provider_options.base_url = base_url
    agent.provider_options.api_key = "local-key"
    agent.tools = [EchoTool()]

    async def on_event(event):
        if event.type == "message_update":
            provider_ev = event.payload.get("assistant_message_event")
            if provider_ev and provider_ev.type == "text_delta":
                print(provider_ev.payload.get("delta", ""), end="", flush=True)
        elif event.type == "error":
            print(f"\n[error] {event.payload}")
        elif event.type == "tool_execution_start":
            print(f"\n[executing {event.payload['tool_name']}...]", end="", flush=True)

    agent.subscribe(on_event)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break

        if not user_input: continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!"); break

        print("Assistant: ", end="", flush=True)
        await agent.prompt(user_input)
        print() 


def run_repl(base_url="http://127.0.0.1:8000/v1", system_prompt="You are a helpful assistant.", api="anthropic-messages"):
    asyncio.run(repl(base_url=base_url, system_prompt=system_prompt, api=api))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "simulate":
        asyncio.run(simulate())
    else:
        run_repl()

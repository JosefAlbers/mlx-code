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
)

import httpx

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


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


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
    tools: list[ToolDefinition] = field(default_factory=list)


class FauxProvider:
    async def stream(
        self, model: Model, context: Context, options: ProviderOptions | None = None
    ) -> AssistantEventStream:
        stream = AssistantEventStream()
        msg = AssistantMessage(api=model.api, provider=model.provider, model=model.id)
        stream.push(Event(type="start", payload={"partial": msg}))

        last_user = next(
            (m for m in reversed(context.messages) if isinstance(m, UserMessage)), None
        )
        last_is_tool_result = (
            isinstance(context.messages[-1], ToolResultMessage)
            if context.messages
            else False
        )

        text = "ACK"
        if last_user:
            content = (
                last_user.content
                if isinstance(last_user.content, str)
                else (last_user.content[0].text if last_user.content else "")
            )
            text = f"ACK: {content}"

        if "call_tool" in text and not last_is_tool_result:
            tool_name = text.split(":")[-1] if ":" in text else "echo"
            call = ToolCall(id=str(uuid.uuid4()), name=tool_name, arguments={"input": text})
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
            if isinstance(content, str):
                return content
            return " ".join(c.text for c in content if isinstance(c, TextContent))

        async with httpx.AsyncClient(timeout=opts.timeout_seconds) as client:
            try:
                req_body: dict[str, Any] = {
                    "model": model.id,
                    "input": [
                        {"role": getattr(m, "role", "user"), "content": format_content(m.content)}
                        for m in context.messages
                    ],
                }
                if context.system_prompt:
                    req_body["instructions"] = context.system_prompt
                resp = await client.post(
                    f"{base_url}/responses",
                    json=req_body,
                    headers={"Authorization": f"Bearer {opts.api_key}"},
                )
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {body.decode()}",
                        request=resp.request,
                        response=resp
                    )
                data = resp.json()
                text = data.get("output_text", "")

                msg = AssistantMessage(
                    api=model.api, provider=model.provider, model=model.id
                )
                stream.push(Event(type="start", payload={"partial": msg}))
                stream.push(Event(type="text_start", payload={"partial": msg}))
                msg.content.append(TextContent(text=text))
                stream.push(Event(type="text_delta", payload={"delta": text, "partial": msg}))
                stream.push(Event(type="text_end", payload={"partial": msg}))
                stream.push(Event(type="done", payload={"reason": "stop", "message": msg}))
                stream.end(msg)
            except Exception as e:
                msg = AssistantMessage(
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    stop_reason="error",
                    error_message=str(e),
                )
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
            msg = AssistantMessage(
                api=model.api,
                provider=model.provider,
                model=model.id,
                stop_reason="error",
                error_message="missing api key",
            )
            stream.push(Event(type="error", payload={"reason": "error", "error": msg}))
            stream.end(msg)
            return stream

        base_url = opts.base_url or "https://api.anthropic.com/v1"

        def format_content(content: str | list[AssistantContent | TextContent | ImageContent | ToolCall]):
            if isinstance(content, str):
                return content
            res = []
            for item in content:
                if isinstance(item, TextContent):
                    res.append({"type": "text", "text": item.text})
                elif isinstance(item, ImageContent):
                    res.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item.mime_type,
                            "data": item.data
                        }
                    })
                elif isinstance(item, ToolCall):
                    res.append({
                        "type": "tool_use",
                        "id": item.id,
                        "name": item.name,
                        "input": item.arguments
                    })
            return res

        messages = []
        for m in context.messages:
            role = getattr(m, "role", None)
            if role == "user":
                messages.append({"role": "user", "content": format_content(m.content)})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": format_content(m.content)})
            elif role == "toolResult":
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": format_content(m.content),
                            "is_error": getattr(m, "is_error", False)
                        }
                    ]
                })

        payload = {
            "model": model.id,
            "max_tokens": opts.max_tokens or 1024,
            "messages": messages,
            "stream": True,
        }
        if context.system_prompt:
            payload["system"] = [{"type": "text", "text": context.system_prompt}]
        if opts.temperature is not None:
            payload["temperature"] = opts.temperature

        async def run():
            msg = AssistantMessage(api=model.api, provider=model.provider, model=model.id)
            try:
                async with httpx.AsyncClient(timeout=opts.timeout_seconds) as client:
                    async with client.stream(
                        "POST",
                        f"{base_url}/messages",
                        json=payload,
                        headers={
                            "x-api-key": opts.api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                    ) as resp:
                        if resp.status_code >= 400:
                            body = await resp.aread()
                            raise httpx.HTTPStatusError(
                                f"HTTP {resp.status_code}: {body.decode()}",
                                request=resp.request,
                                response=resp
                            )

                        stream.push(Event(type="start", payload={"partial": msg}))
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            chunk = json.loads(data_str)
                            event_type = chunk.get("type")

                            if event_type == "content_block_start":
                                block = chunk.get("content_block", {})
                                if block.get("type") == "text":
                                    stream.push(Event(type="text_start", payload={"partial": msg}))
                                    msg.content.append(TextContent(text=""))
                                elif block.get("type") == "tool_use":
                                    pass
                            elif event_type == "content_block_delta":
                                delta = chunk.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if msg.content and isinstance(msg.content[-1], TextContent):
                                        msg.content[-1].text += text
                                    stream.push(
                                        Event(
                                            type="text_delta",
                                            payload={"delta": text, "partial": msg},
                                        )
                                    )
                                elif delta.get("type") == "input_json_delta":
                                    pass
                            elif event_type == "content_block_stop":
                                if msg.content and isinstance(msg.content[-1], TextContent):
                                    stream.push(Event(type="text_end", payload={"partial": msg}))
                            elif event_type == "message_stop":
                                stream.push(
                                    Event(type="done", payload={"reason": "stop", "message": msg})
                                )

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
        if not provider:
            raise ValueError(f"Provider for API {model.api} not found")
        return await provider.stream(model, context, options)


@dataclass(slots=True)
class AgentConfig:
    tool_execution: Literal["sequential", "parallel"] = "parallel"
    convert_to_llm: Callable[[list[Message]], Awaitable[list[Message]]] | None = None
    transform_context: Callable[[list[Message]], Awaitable[list[Message]]] | None = None
    before_tool_call: (
        Callable[[str, dict[str, Any], AssistantMessage], Awaitable[bool]] | None
    ) = None
    after_tool_call: (
        Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]] | None
    ) = None
    get_steering_messages: Callable[[], Awaitable[list[Message]]] | None = None
    get_follow_up_messages: Callable[[], Awaitable[list[Message]]] | None = None


class AgentTool(Protocol):
    name: str

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]: ...


async def default_convert(messages: list[Message]) -> list[Message]:
    return [
        m for m in messages if getattr(m, "role", None) in {"user", "assistant", "toolResult"}
    ]


async def run_agent_loop(
    context: Context,
    emit: Callable[[AgentEvent], Awaitable[None]],
    registry: ProviderRegistry,
    tools: dict[str, AgentTool] | None = None,
    config: AgentConfig | None = None,
    provider_options: ProviderOptions | None = None,
) -> list[Message]:
    cfg = config or AgentConfig()
    new_messages: list[Message] = []

    await emit(AgentEvent(type="agent_start"))

    model = Model(id=context.model_id, provider=context.provider, api=context.api)

    while True:
        has_more_tool_calls = True
        pending_steering = (
            await cfg.get_steering_messages() if cfg.get_steering_messages else []
        )

        while has_more_tool_calls or pending_steering:
            await emit(AgentEvent(type="turn_start"))

            if pending_steering:
                for msg in pending_steering:
                    await emit(AgentEvent(type="message_start", payload={"message": msg}))
                    context.messages.append(msg)
                    new_messages.append(msg)
                    await emit(AgentEvent(type="message_end", payload={"message": msg}))
                pending_steering = []

            transformed = (
                await cfg.transform_context(context.messages)
                if cfg.transform_context
                else context.messages
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
            async for event in stream:
                if event.type == "start":
                    assistant = event.payload["partial"]
                    await emit(
                        AgentEvent(type="message_start", payload={"message": assistant})
                    )
                elif event.type == "text_delta":
                    await emit(
                        AgentEvent(
                            type="message_update",
                            payload={"message": assistant, "delta": event.payload.get("delta", ""), "provider_event": event},
                        )
                    )
                elif event.type in {
                    "thinking_delta",
                    "toolcall_start",
                    "toolcall_end",
                }:
                    await emit(
                        AgentEvent(
                            type="message_update",
                            payload={"message": assistant, "provider_event": event},
                        )
                    )
                elif event.type == "error":
                    await emit(AgentEvent(type="error", payload=event.payload))

            assistant = await stream.result()
            context.messages.append(assistant)
            new_messages.append(assistant)
            await emit(AgentEvent(type="message_end", payload={"message": assistant}))

            if assistant.stop_reason in {"error", "aborted"}:
                await emit(AgentEvent(type="turn_end", payload={"message": assistant}))
                await emit(AgentEvent(type="agent_end", payload={"messages": new_messages}))
                return new_messages

            tool_calls = [
                c for c in assistant.content if isinstance(c, ToolCall)
            ]
            has_more_tool_calls = len(tool_calls) > 0
            results: list[ToolResultMessage] = []

            if has_more_tool_calls:

                async def run_call(call: ToolCall):
                    await emit(
                        AgentEvent(
                            type="tool_execution_start",
                            payload={"id": call.id, "name": call.name, "args": call.arguments},
                        )
                    )
                    allowed = True
                    if cfg.before_tool_call:
                        allowed = await cfg.before_tool_call(call.name, call.arguments, assistant)

                    res_data: dict[str, Any]
                    is_error = False
                    if not allowed:
                        res_data = {"error": "blocked by hook"}
                        is_error = True
                    elif not tools or call.name not in tools:
                        res_data = {"error": f"tool {call.name} not found"}
                        is_error = True
                    else:
                        try:
                            res_data = await tools[call.name].execute(call.arguments)
                        except Exception as e:
                            res_data = {"error": str(e)}
                            is_error = True

                    if cfg.after_tool_call:
                        res_data = await cfg.after_tool_call(
                            call.name, call.arguments, res_data
                        )

                    tool_res = ToolResultMessage(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        content=[TextContent(text=json.dumps(res_data))],
                        is_error=is_error,
                        details=res_data,
                    )
                    results.append(tool_res)
                    await emit(
                        AgentEvent(
                            type="tool_execution_end",
                            payload={
                                "id": call.id,
                                "name": call.name,
                                "result": tool_res,
                                "is_error": is_error,
                            },
                        )
                    )

                if cfg.tool_execution == "sequential":
                    for call in tool_calls:
                        await run_call(call)
                else:
                    await asyncio.gather(*(run_call(call) for call in tool_calls))

                for res in results:
                    await emit(AgentEvent(type="message_start", payload={"message": res}))
                    context.messages.append(res)
                    new_messages.append(res)
                    await emit(AgentEvent(type="message_end", payload={"message": res}))

            await emit(AgentEvent(type="turn_end", payload={"message": assistant}))
            pending_steering = (
                await cfg.get_steering_messages() if cfg.get_steering_messages else []
            )

        follow_ups = (
            await cfg.get_follow_up_messages() if cfg.get_follow_up_messages else []
        )
        if not follow_ups:
            break
        pending_steering = follow_ups

    await emit(AgentEvent(type="agent_end", payload={"messages": new_messages}))
    return new_messages


@dataclass(slots=True)
class AgentEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


class Agent:
    def __init__(self, context: Context, registry: ProviderRegistry | None = None) -> None:
        self.context = context
        self.registry = registry or ProviderRegistry()
        self.tools: dict[str, AgentTool] = {}
        self.config = AgentConfig()
        self.provider_options = ProviderOptions()
        self._listeners: set[Callable[[AgentEvent], Awaitable[None]]] = set()

    def subscribe(self, listener: Callable[[AgentEvent], Awaitable[None]]) -> Callable[[], None]:
        self._listeners.add(listener)
        return lambda: self._listeners.remove(listener)

    async def _emit(self, event: AgentEvent) -> None:
        if not self._listeners:
            return
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


async def run_jsonl_bridge() -> None:
    agent: Agent | None = None
    protocol_version = "1.1"

    async def emit_json(data: dict[str, Any]):
        sys.stdout.write(json.dumps(data) + "\n")
        sys.stdout.flush()

    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            cmd_data = json.loads(line)
            cmd = cmd_data.get("cmd")

            if cmd == "init":
                ctx = Context(
                    model_id=cmd_data["model_id"],
                    provider=cmd_data["provider"],
                    api=cmd_data["api"],
                    system_prompt=cmd_data.get("system_prompt", ""),
                )
                agent = Agent(ctx)
                if "api_key" in cmd_data:
                    agent.provider_options.api_key = cmd_data["api_key"]
                await emit_json({"ok": True, "v": protocol_version, "event": "initialized"})
            elif not agent:
                await emit_json({"ok": False, "error": "not initialized"})
            elif cmd == "prompt":

                async def bridge_listener(ev: AgentEvent):
                    await emit_json({"v": protocol_version, "event": ev.type, "payload": str(ev.payload)[:200]})

                unsub = agent.subscribe(bridge_listener)
                await agent.prompt(cmd_data["text"])
                unsub()
                await emit_json({"ok": True, "v": protocol_version, "event": "prompt_done"})
            elif cmd == "state":
                await emit_json(
                    {"ok": True, "v": protocol_version, "message_count": len(agent.context.messages)}
                )
            else:
                await emit_json({"ok": False, "error": f"unknown command {cmd}"})
        except Exception as e:
            await emit_json({"ok": False, "error": str(e)})


async def repl(base_url, api, system_prompt):
    context = Context(
        model_id="local-mod", 
        provider="local-pro",
        api=api,
        system_prompt=system_prompt
    )

    agent = Agent(context)

    agent.provider_options.base_url = base_url
    agent.provider_options.api_key = "local-key"

    async def on_event(event):
        if event.type == "message_update":
            print(event.payload.get("delta", ""), end="", flush=True)
        elif event.type == "error":
            print(f"\n[error] {event.payload}")

    agent.subscribe(on_event)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        print("Assistant: ", end="", flush=True)
        await agent.prompt(user_input)
        print() 

def run_repl(base_url="http://127.0.0.1:8000/v1", system_prompt="You are a helpful assistant.", api="anthropic-messages"):
    asyncio.run(repl(base_url=base_url, system_prompt=system_prompt, api=api))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bridge":
        asyncio.run(run_jsonl_bridge())
    else:
        run_repl()

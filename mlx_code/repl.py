from __future__ import annotations
import asyncio
import argparse
import json
import os
import socket
import urllib.parse
import pathlib
import re
import time
import sys, tty, termios
import logging
import tempfile
from .gits import create_worktree, commit_worktree, resume_worktree
from .tools import Tool, validate_tool_call, DEFAULT_TOOLS
from .apis import resolve_api

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        system=None,
        api=None,
        tool_names=None,
        extra_tool_classes=None,
        model=None,
        api_key=None,
        base_url=None,
        ctx=None,
    ):
        self.api = resolve_api(
            api="default" if api is None else api,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.system = "" if system is None else system
        self.messages: list[dict] = []
        self._signal: asyncio.Event | None = None
        self._listeners: set[Callable] = set()
        self._extra_tool_classes: list[type[Tool]] = extra_tool_classes or []
        self.ctx: dict = {
            "cwd": os.getcwd(),
            "skills": [],
            "gwt": None,
            **(ctx or {}),
            "agent": self,
        }
        self._last_result_sig: str | None = None
        self._same_result_count: int = 0
        tools = [cls(self.ctx) for cls in DEFAULT_TOOLS + self._extra_tool_classes]
        if tool_names is not None:
            name_set = {n.lower() for n in tool_names}
            tools = [t for t in tools if t.name.lower() in name_set]
        self.tools = tools

    def spawn(self, **overrides) -> "Agent":
        kwargs = {
            "api": self.api,
            "system": self.system,
            "extra_tool_classes": self._extra_tool_classes,
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "ctx": {k: v for k, v in self.ctx.items() if k != "agent"},
        }
        kwargs.update(overrides)
        return Agent(**kwargs)

    def branch(self) -> "Agent":
        child = self.spawn(tool_names=[t.name for t in self.tools])
        child.messages = list(self.messages)
        return child

    async def run(self, prompt: str) -> dict:
        await self._wait()
        self._signal = None
        self.messages.append(
            {"role": "user", "content": prompt, "timestamp": int(time.time() * 1000)}
        )
        return await self._loop()

    def abort(self) -> None:
        if self._signal is None:
            self._signal = asyncio.Event()
        self._signal.set()

    def subscribe(self, fn: Callable[[dict], Any]) -> Callable:
        self._listeners.add(fn)
        return lambda: self._listeners.discard(fn)

    async def _wait(self, timeout: float = 30.0, retry_delay: float = 0.2):
        parsed = urllib.parse.urlparse(self.api.base_url)
        host = parsed.hostname
        port = parsed.port
        if port is None:
            return
        logger.debug(f"Holding requests. Checking server status at {host}:{port}...")
        sys.stdout.flush()
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.create_connection((host, port), timeout=0.2):
                    logger.debug(
                        f"Connected after {int(time.time() - start_time)} seconds! Releasing pending requests.\n"
                    )
                    return True
            except (OSError, ConnectionRefusedError):
                time.sleep(retry_delay)
        raise TimeoutError(
            f"Target backend at {self.api.base_url} failed to respond within {timeout} seconds."
        )

    async def _emit(self, event: dict) -> None:
        for fn in list(self._listeners):
            r = fn(event)
            if asyncio.iscoroutine(r):
                await r

    async def _loop(self) -> dict:
        await self._emit({"type": "agent_start", "payload": {}})
        final: dict = {
            "role": "assistant",
            "content": [],
            "stop_reason": "error",
            "error_message": "no turns ran",
            "usage": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
            "timestamp": int(time.time() * 1000),
        }
        while True:
            await self._emit({"type": "turn_start", "payload": {}})
            es = await self.api.stream(self.messages, self.system, self.tools)
            async for event in es:
                if event["type"] in ("text_delta", "thinking_delta", "error"):
                    await self._emit(
                        {"type": event["type"], "payload": event["payload"]}
                    )
            final = await es.result()
            logger.debug(final)
            self.messages.append(final)
            await self._emit({"type": "turn_end", "payload": {"message": final}})
            if final["stop_reason"] in ("error", "aborted"):
                break
            if self._signal and self._signal.is_set():
                final["stop_reason"] = "aborted"
                break
            calls = [b for b in final["content"] if b["type"] == "toolCall"]
            if not calls:
                break
            results = await self._execute_tools(calls)
            result_sig = json.dumps(
                [(r["tool_name"], r["content"]) for r in results], sort_keys=True
            )
            if result_sig == self._last_result_sig:
                self._same_result_count += 1
            else:
                self._last_result_sig = result_sig
                self._same_result_count = 1
            if self._same_result_count >= 2:
                logger.warning(
                    f"Doom-loop detected! Same result {self._same_result_count} times."
                )
                warning_msg = f"\n\n[SYSTEM WARNING: You have received this exact same tool result {self._same_result_count} times in a row. You are likely in a loop. Change your strategy.]\n\n"
                for r in results:
                    if r.get("content") and isinstance(r["content"], list):
                        if r["content"][0]["type"] == "text":
                            r["content"][0]["text"] = (
                                r["content"][0]["text"] + warning_msg
                            )
                        else:
                            r["content"].insert(
                                0, {"type": "text", "text": warning_msg}
                            )
            self.messages.extend(results)
            self.ctx["gwt"] = commit_worktree(self.ctx["gwt"], self.messages)
            if self._signal and self._signal.is_set():
                final["stop_reason"] = "aborted"
                break
        await self._emit({"type": "agent_end", "payload": {"message": final}})
        return final

    async def _execute_tools(self, calls: list[dict]) -> list[dict]:
        return list(await asyncio.gather(*[self._execute_one(c) for c in calls]))

    async def _execute_one(self, call: dict) -> dict:
        await self._emit(
            {
                "type": "tool_start",
                "payload": {"name": call["name"], "args": call["arguments"]},
            }
        )
        tool = next((t for t in self.tools if t.name == call["name"]), None)
        if tool is None:
            result = {
                "content": [
                    {"type": "text", "text": f"Tool '{call['name']}' not found"}
                ],
                "is_error": True,
            }
        else:
            try:
                result = await tool.execute(
                    validate_tool_call(tool, call), self._signal
                )
            except Exception as exc:
                result = {
                    "content": [{"type": "text", "text": str(exc)}],
                    "is_error": True,
                }
        msg = {
            "role": "toolResult",
            "tool_call_id": call["id"],
            "tool_name": call["name"],
            "content": result["content"],
            "is_error": result["is_error"],
            "timestamp": int(time.time() * 1000),
        }
        await self._emit({"type": "tool_result", "payload": {"message": msg}})
        await self._emit(
            {
                "type": "tool_end",
                "payload": {
                    "name": call["name"],
                    "is_error": result["is_error"],
                    "result": msg,
                },
            }
        )
        return msg


_REPL_HELP = "Commands:\n  /help          — show this message\n  /clear         — clear conversation history\n  /history       — print message history\n  /tools         — list active tools\n  /branch        — spawn a branched sub-agent and run a one-shot prompt\n  /abort         — signal abort after next tool call\n  exit / quit    — end the session\n"
import sys
import os
from contextlib import contextmanager


def read_input(prompt: str = "\x1b[32m≫\x1b[0m ") -> str:
    if sys.platform == "win32":
        return _read_input_win(prompt)
    else:
        return _read_input_unix(prompt)


@contextmanager
def _raw_mode(fd):
    import termios, tty

    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_input_unix(prompt: str) -> str:
    fd = sys.stdin.fileno()
    sys.stdout.write("\x1b[?2004h" + prompt)
    sys.stdout.flush()
    buf, esc, pasting = ([], "", False)
    try:
        with _raw_mode(fd):
            while True:
                ch = sys.stdin.read(1)
                if esc or ch == "\x1b":
                    esc += ch
                    if esc == "\x1b[200~":
                        pasting = True
                        esc = ""
                    elif esc == "\x1b[201~":
                        pasting = False
                        esc = ""
                    elif len(esc) >= 6:
                        buf.extend(esc)
                        esc = ""
                    continue
                if ch == "\x03":
                    if buf:
                        sys.stdout.write(" \x1b[31m[ABORT]\x1b[0m\r\n" + prompt)
                        sys.stdout.flush()
                        buf, esc, pasting = ([], "", False)
                        continue
                    raise KeyboardInterrupt
                if ch == "\x04":
                    raise EOFError
                if ch in ("\r", "\n"):
                    if pasting:
                        buf.append("\n")
                        sys.stdout.write("\r\n")
                        sys.stdout.flush()
                    else:
                        line = "".join(buf)
                        if line.endswith("\\"):
                            buf[-1] = "\n"
                            sys.stdout.write("\r\n‥ ")
                            sys.stdout.flush()
                        else:
                            sys.stdout.write("\r\n")
                            sys.stdout.flush()
                            break
                    continue
                if ch in ("\x7f", "\x08"):
                    if buf and (not pasting):
                        buf.pop()
                        sys.stdout.write("\x08 \x08")
                        sys.stdout.flush()
                    continue
                buf.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()
    finally:
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()
    return "".join(buf).strip()


def _read_input_win(prompt: str) -> str:
    import ctypes
    import ctypes.wintypes as wt

    kernel32 = ctypes.windll.kernel32
    STD_INPUT_HANDLE = -10
    KEY_EVENT = 1
    ENABLE_PROCESSED_INPUT = 1
    PASTE_THRESHOLD_MS = 30

    class _UChar(ctypes.Union):
        _fields_ = [("UnicodeChar", ctypes.c_wchar), ("AsciiChar", ctypes.c_char)]

    class _KeyEventRecord(ctypes.Structure):
        _fields_ = [
            ("bKeyDown", wt.BOOL),
            ("wRepeatCount", wt.WORD),
            ("wVirtualKeyCode", wt.WORD),
            ("wVirtualScanCode", wt.WORD),
            ("uChar", _UChar),
            ("dwControlKeyState", wt.DWORD),
        ]

    class _EventUnion(ctypes.Union):
        _fields_ = [("KeyEvent", _KeyEventRecord), ("padding", ctypes.c_byte * 16)]

    class _InputRecord(ctypes.Structure):
        _fields_ = [("EventType", wt.WORD), ("Event", _EventUnion)]

    hin = kernel32.GetStdHandle(STD_INPUT_HANDLE)
    old_mode = wt.DWORD()
    kernel32.GetConsoleMode(hin, ctypes.byref(old_mode))
    kernel32.SetConsoleMode(hin, old_mode.value & ~ENABLE_PROCESSED_INPUT)
    sys.stdout.write(prompt)
    sys.stdout.flush()
    buf, pasting = ([], False)
    last_key_tick = None
    try:
        while True:
            rec = _InputRecord()
            n = wt.DWORD(0)
            kernel32.ReadConsoleInputW(hin, ctypes.byref(rec), 1, ctypes.byref(n))
            if n.value == 0:
                continue
            if rec.EventType != KEY_EVENT:
                continue
            kr = rec.Event.KeyEvent
            if not kr.bKeyDown:
                continue
            ch = kr.uChar.UnicodeChar
            if not ch:
                continue
            now = ctypes.windll.kernel32.GetTickCount()
            if last_key_tick is not None:
                pasting = now - last_key_tick < PASTE_THRESHOLD_MS
            last_key_tick = now
            if ch == "\x03":
                if buf:
                    sys.stdout.write(" \x1b[31m[ABORT]\x1b[0m\n" + prompt)
                    sys.stdout.flush()
                    buf, pasting, last_key_tick = ([], False, None)
                    continue
                raise KeyboardInterrupt
            if ch == "\x04":
                raise EOFError
            if ch in ("\r", "\n"):
                if pasting:
                    buf.append("\n")
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                else:
                    line = "".join(buf)
                    if line.endswith("\\"):
                        buf[-1] = "\n"
                        sys.stdout.write("\n‥ ")
                        sys.stdout.flush()
                    else:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        break
                continue
            if ch in ("\x7f", "\x08"):
                if buf and (not pasting):
                    buf.pop()
                    sys.stdout.write("\x08 \x08")
                    sys.stdout.flush()
                continue
            buf.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
    finally:
        kernel32.SetConsoleMode(hin, old_mode.value)
    return "".join(buf).strip()


async def repl(agent, init_prompt=None) -> None:
    is_tty = sys.stdin.isatty()
    loop = asyncio.get_running_loop()
    _suppress = False
    last_block = ""
    trail = ""

    async def on_event(event: dict) -> None:
        nonlocal _suppress, last_block, trail

        def emit(delta, block, prefix="", suffix=""):
            nonlocal last_block, trail
            if last_block != block:
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
            trail = delta[len(rstripped) :]
            if rstripped:
                print(prefix + rstripped + suffix, end="", flush=True)

        et = event["type"]
        p = event["payload"]
        if et == "text_delta":
            delta = p.get("delta", "")
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
            if et == "thinking_delta":
                if d := p.get("delta"):
                    emit(d, "thinking", "\x1b[2m", "\x1b[0m")
            elif et == "tool_start":
                name_part = "\x1b[1;43;30;1m" + p["name"] + "\x1b[0m"
                args_part = (
                    " \x1b[2;33m" + json.dumps(p["args"]) + "\x1b[0m"
                    if p["args"]
                    else ""
                )
                emit(name_part + args_part + "\n", "tool")
            elif et == "tool_end":
                if p.get("is_error"):
                    print(" \x1b[5;31m→ error\x1b[0m", flush=True)
            elif et == "error":
                err = p.get("error", {})
                print(
                    f"\n\x1b[31m[error]\x1b[0m {(err.get('error_message', str(err)) if isinstance(err, dict) else str(err))}\n"
                )
            elif et == "agent_end":
                last_block = ""

    agent.subscribe(on_event)
    if is_tty:
        print("mlx-code REPL  •  type /help for commands, Ctrl-D or 'exit' to quit.\n")
    init_prompt_processed = False
    while True:
        if init_prompt and (not init_prompt_processed):
            user_input = init_prompt.strip()
            init_prompt_processed = True
            if is_tty:
                print(f"\x1b[32m≫\x1b[0m {user_input}")
        elif is_tty:
            try:
                user_input = await loop.run_in_executor(
                    None, lambda: read_input().strip()
                )
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
                trunc = lambda t, n=80: t[:n] + "‥" if len(t) > n else t
                for i, m in enumerate(agent.messages):
                    if m["role"] == "user":
                        c = (
                            m["content"]
                            if isinstance(m["content"], str)
                            else str(m["content"])
                        )
                        print(
                            f"  {i:2d} [user] {trunc(c.replace(chr(10), ' ').strip())}"
                        )
                    elif m["role"] == "assistant":
                        t = "".join(
                            (b["text"] for b in m["content"] if b["type"] == "text")
                        )
                        print(
                            f"  {i:2d} [assistant] {trunc(t.replace(chr(10), ' ').strip())}"
                        )
                    elif m["role"] == "toolResult":
                        t = m["content"][0]["text"] if m["content"] else ""
                        print(
                            f"  {i:2d} [tool:{m['tool_name']}] {trunc(t.replace(chr(10), ' ').strip(), 60)}"
                        )
            elif cmd == "/tools":
                for t in agent.tools:
                    print(f"  {t.name} — {t.description}")
            elif cmd == "/branch":
                prompt = arg.strip() or "Summarise what we have discussed."
                print(f"[branching: '{prompt}']")
                child = agent.branch()
                result = await child.run(prompt)
                texts = [b["text"] for b in result["content"] if b["type"] == "text"]
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
            print("\x1b[34mπ\x1b[0m ", end="", flush=True)
        await agent.run(user_input)
        print()
        if not is_tty:
            break


def collect_skills(skills_dir, skills=None):
    skills = [] if skills is None else skills
    if skills_dir is not None:
        root = pathlib.Path(skills_dir)
        if root.exists():
            for md in sorted(root.rglob("SKILL.md")):
                text = md.read_text(encoding="utf-8", errors="replace")
                name, description = (md.parent.name, "")
                if text.startswith("---"):
                    end = text.find("---", 3)
                    if end != -1:
                        fm = text[3:end]
                        if n := re.search("^name:\\s*(.+)$", fm, re.MULTILINE):
                            name = n.group(1).strip()
                        if m := re.search("^description:\\s*(.+)$", fm, re.MULTILINE):
                            description = m.group(1).strip()
                        text = text[end + 3 :].strip()
                        if not n and (not m):
                            continue
                        skills.append(
                            {"name": name, "description": description, "content": text}
                        )
    skill_prompt = (
        "Available skills (use GetSkill to load full instructions when needed):\n"
        + "\n".join((f"- {s['name']}: {s['description']}" for s in skills))
        if skills
        else ""
    )
    return (skills, skill_prompt)


def run_repl(
    *,
    base_url=None,
    model=None,
    api: Literal["claude", "codex", "gemini", "deepseek", "noapi"] = "noapi",
    system="",
    sdir=None,
    skills=None,
    env=None,
    tool_names=None,
    extra_tool_classes=None,
    api_key=None,
    gwt=None,
    ctx=None,
    init_prompt=None,
    resume_messages=None,
    repo=None,
    resume=None,
    stream=None,
):
    repo = os.path.abspath(repo or os.getcwd())
    import tempfile

    with tempfile.TemporaryDirectory(dir="/tmp") as _home:
        if env is None:
            env = os.environ.copy()
        env["HOME"] = _home
        env.setdefault("SHELL", "/bin/bash")
        if gwt is None:
            if resume:
                result = resume_worktree(
                    repo, resume, worktree_dir=os.path.join(_home, "workspace")
                )
                if result is None:
                    print(f"[error] Could not resume from commit {resume!r}. Aborting.")
                    return
                gwt, resume_messages = result
                print(f"[resumed worktree at {gwt.worktree} from commit {resume}]")
            else:
                gwt = create_worktree(
                    repo, worktree_dir=os.path.join(_home, "workspace")
                )
        cwd = gwt.worktree if gwt else repo
        env["PWD"] = cwd
        if env is not None:
            os.environ.clear()
            os.environ.update(env)
        os.chdir(cwd)
        sdir = os.path.abspath(sdir or cwd)
        skills, skill_prompt = collect_skills(sdir, skills)
        system = "\n\n".join(filter(None, [system, skill_prompt]))
        merged_ctx = {"cwd": cwd, "skills": skills, "gwt": gwt, **(ctx or {})}
        agent = Agent(
            system=system,
            api=api,
            model=model,
            tool_names=tool_names,
            extra_tool_classes=extra_tool_classes,
            api_key=api_key,
            base_url=base_url,
            ctx=merged_ctx,
        )
        if stream is not None:
            from .stream_log import StreamLogger

            log_path = stream
            fp = open(log_path, "w", buffering=1)
            agent.ctx["_stream_log_fp"] = fp
            agent.ctx["_stream_log_depth"] = 0
            StreamLogger(agent, fp, depth=0, name="base")
            print(f"[streaming log: tail -f {log_path}]")
        if resume_messages:
            agent.messages = list(resume_messages)
            print(f"[resumed {len(resume_messages)} messages from checkpoint]")
        try:
            asyncio.run(repl(agent, init_prompt=init_prompt))
        except KeyboardInterrupt:
            print("\nExiting...")


def main():
    from .util import setup_logger

    setup_logger(log_file=".log.json")
    parser = argparse.ArgumentParser(description="mlx-code REPL")
    parser.add_argument(
        "-a",
        "--api",
        choices=["claude", "codex", "gemini", "deepseek", "noapi"],
        default="noapi",
        help="API backend to use; 'noapi' routes through the local MLX server (default: noapi)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model name or ID; defaults to a sensible value for each API",
    )
    parser.add_argument(
        "-t",
        "--tools",
        nargs="+",
        help="Whitelist of tool names to enable; allows all tools when omitted",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base URL of the API server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--system", default="", help="System prompt prepended to every conversation"
    )
    parser.add_argument(
        "--skill", default=None, help="Directory to scan recursively for SKILL.md files"
    )
    parser.add_argument(
        "--cwd", default=None, help="Working directory / git repo root (default: cwd)"
    )
    parser.add_argument(
        "--key",
        default=None,
        help="API key; falls back to the relevant *_API_KEY env var",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Initial prompt sent automatically when the REPL starts",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="COMMIT",
        help="Resume a previous session from the given git commit hash",
    )
    parser.add_argument("--stream", default=None, help="File to stream log into")
    args = parser.parse_args()
    logger.debug(args)
    url, model, tool_names, api_key = (args.url, args.model, args.tools, args.key)
    if args.api in ["deepseek", "gemini"]:
        if args.api == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY") if api_key is None else api_key
            url = "https://api.deepseek.com" if api_key else url
            model = "deepseek-v4-flash" if model is None else model
        elif args.api == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY") if api_key is None else api_key
            url = "https://generativelanguage.googleapis.com" if api_key else url
            model = "gemini-3.1-flash-lite-preview" if model is None else model
        tool_names = [] if tool_names is None else tool_names
    run_repl(
        api=args.api,
        system=args.system,
        repo=args.cwd,
        model=model,
        base_url=url,
        tool_names=tool_names,
        sdir=args.skill,
        api_key=api_key,
        init_prompt=args.prompt,
        resume=args.resume,
        stream=args.stream,
    )


if __name__ == "__main__":
    main()

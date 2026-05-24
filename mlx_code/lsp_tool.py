from __future__ import annotations
import asyncio
import json
import logging
import os
import pathlib
import re
import shutil
from typing import Any, Literal
from pydantic import BaseModel, Field
from .tools import Tool, resolve_path, tout

logger = logging.getLogger(__name__)
_DEFAULT_CMDS: dict[str, list[list[str]]] = {
    "py": [["ty", "server"], ["pyright-langserver", "--stdio"], ["pylsp"]],
    "ts": [["typescript-language-server", "--stdio"], ["vtsls", "--stdio"]],
    "tsx": [["typescript-language-server", "--stdio"], ["vtsls", "--stdio"]],
    "js": [["typescript-language-server", "--stdio"], ["vtsls", "--stdio"]],
    "jsx": [["typescript-language-server", "--stdio"], ["vtsls", "--stdio"]],
    "html": [["vscode-html-language-server", "--stdio"]],
    "css": [["vscode-css-language-server", "--stdio"]],
    "json": [["vscode-json-language-server", "--stdio"]],
}
_EXT_TO_LANG: dict[str, str] = {
    ".py": "py",
    ".pyw": "py",
    ".ts": "ts",
    ".tsx": "tsx",
    ".js": "js",
    ".mjs": "js",
    ".cjs": "js",
    ".jsx": "jsx",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".less": "css",
    ".json": "json",
    ".jsonc": "json",
}
_LANG_ID: dict[str, str] = {
    "py": "python",
    "ts": "typescript",
    "tsx": "typescriptreact",
    "js": "javascript",
    "jsx": "javascriptreact",
    "html": "html",
    "css": "css",
    "json": "json",
}
_SYMBOL_KIND_NAME: dict[int, str] = {
    1: "File",
    2: "Module",
    3: "Namespace",
    4: "Package",
    5: "Class",
    6: "Method",
    7: "Property",
    8: "Field",
    9: "Constructor",
    10: "Enum",
    11: "Interface",
    12: "Function",
    13: "Variable",
    14: "Constant",
    15: "String",
    16: "Number",
    17: "Boolean",
    18: "Array",
    19: "Object",
    20: "Key",
    21: "Null",
    22: "EnumMember",
    23: "Struct",
    24: "Event",
    25: "Operator",
    26: "TypeParameter",
}
_MAP_KINDS: frozenset[int] = frozenset({5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 26})
_SEVERITY: dict[int, str] = {1: "error", 2: "warning", 3: "info", 4: "hint"}


class LspError(Exception):
    pass


class LspClient:
    def __init__(self, proc: asyncio.subprocess.Process, root_uri: str):
        self._proc = proc
        self._root_uri = root_uri
        self._send_lock = asyncio.Lock()
        self._id_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future] = {}
        self._id = 0
        self._open_files: dict[str, int] = {}
        self._diag_store: dict[str, list] = {}
        self._diag_events: dict[str, asyncio.Event] = {}
        self._reader_task: asyncio.Task | None = None

    async def start(self, cwd: str) -> None:
        self._reader_task = asyncio.create_task(self._reader())
        await self._request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": self._root_uri,
                "workspaceFolders": [
                    {"uri": self._root_uri, "name": pathlib.Path(cwd).name}
                ],
                "capabilities": {
                    "textDocument": {
                        "synchronization": {"didOpen": True, "didChange": True},
                        "publishDiagnostics": {},
                        "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                        "definition": {},
                        "references": {},
                        "hover": {},
                        "rename": {},
                    },
                    "workspace": {"symbol": {}, "workspaceFolders": True},
                },
                "initializationOptions": {},
            },
        )
        await self._notify("initialized", {})

    async def shutdown(self) -> None:
        try:
            await asyncio.wait_for(self._request("shutdown", {}), timeout=3)
            await self._notify("exit", {})
        except Exception:
            pass
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        try:
            self._proc.terminate()
        except Exception:
            pass

    async def _sync(self, path: str) -> str:
        uri = pathlib.Path(path).as_uri()
        ext = pathlib.Path(path).suffix.lower()
        lang_id = _LANG_ID.get(_EXT_TO_LANG.get(ext, ""), "plaintext")
        text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
        if uri in self._open_files:
            ver = self._open_files[uri] + 1
            self._open_files[uri] = ver
            await self._notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": ver},
                    "contentChanges": [{"text": text}],
                },
            )
        else:
            self._open_files[uri] = 1
            await self._notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": lang_id,
                        "version": 1,
                        "text": text,
                    }
                },
            )
        return uri

    async def document_symbols(self, path: str) -> list[dict]:
        uri = await self._sync(path)
        return _as_list(
            await self._request(
                "textDocument/documentSymbol", {"textDocument": {"uri": uri}}
            )
        )

    async def hover(self, path: str, line: int, char: int) -> str:
        uri = await self._sync(path)
        result = await self._request(
            "textDocument/hover",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": char},
            },
        )
        if not result:
            return ""
        c = result.get("contents", "")
        if isinstance(c, dict):
            return c.get("value", "")
        if isinstance(c, list):
            return "\n".join((x["value"] if isinstance(x, dict) else str(x) for x in c))
        return str(c)

    async def definition(self, path: str, line: int, char: int) -> list[dict]:
        uri = await self._sync(path)
        return _as_list(
            await self._request(
                "textDocument/definition",
                {
                    "textDocument": {"uri": uri},
                    "position": {"line": line, "character": char},
                },
            )
        )

    async def references(
        self, path: str, line: int, char: int, include_declaration: bool = True
    ) -> list[dict]:
        uri = await self._sync(path)
        return _as_list(
            await self._request(
                "textDocument/references",
                {
                    "textDocument": {"uri": uri},
                    "position": {"line": line, "character": char},
                    "context": {"includeDeclaration": include_declaration},
                },
            )
        )

    async def rename(
        self, path: str, line: int, char: int, new_name: str
    ) -> dict | None:
        uri = await self._sync(path)
        return await self._request(
            "textDocument/rename",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": char},
                "newName": new_name,
            },
        )

    async def diagnostics(self, path: str, timeout: float = 8.0) -> list[dict]:
        uri = pathlib.Path(path).as_uri()
        evt = asyncio.Event()
        self._diag_events[uri] = evt
        await self._sync(path)
        try:
            await asyncio.wait_for(evt.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._diag_events.pop(uri, None)
        return self._diag_store.get(uri, [])

    async def _next_id(self) -> int:
        async with self._id_lock:
            self._id += 1
            return self._id

    async def _request(self, method: str, params: Any) -> Any:
        rid = await self._next_id()
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        await self._send(
            {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        )
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=15)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise LspError(f"LSP request '{method}' timed out")

    async def _notify(self, method: str, params: Any) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _send(self, msg: dict) -> None:
        body = json.dumps(msg).encode()
        header = f"Content-Length: {len(body)}\r\n\r\n".encode()
        async with self._send_lock:
            self._proc.stdin.write(header + body)
            await self._proc.stdin.drain()

    async def _reader(self) -> None:
        while True:
            try:
                content_length = 0
                while True:
                    raw = await self._proc.stdout.readline()
                    if not raw:
                        return
                    line = raw.decode().strip()
                    if not line:
                        break
                    if line.lower().startswith("content-length:"):
                        content_length = int(line.split(":", 1)[1].strip())
                if content_length == 0:
                    continue
                body = await self._proc.stdout.readexactly(content_length)
                msg = json.loads(body)
            except asyncio.IncompleteReadError:
                return
            except Exception as exc:
                logger.debug("LSP reader error: %s", exc)
                return
            if "id" in msg:
                fut = self._pending.pop(msg["id"], None)
                if fut and (not fut.done()):
                    if "error" in msg:
                        fut.set_exception(
                            LspError(msg["error"].get("message", str(msg["error"])))
                        )
                    else:
                        fut.set_result(msg.get("result"))
            elif msg.get("method") == "textDocument/publishDiagnostics":
                p = msg.get("params", {})
                uri = p.get("uri", "")
                self._diag_store[uri] = p.get("diagnostics", [])
                evt = self._diag_events.pop(uri, None)
                if evt:
                    evt.set()


async def _get_or_start(ctx: dict, lang: str) -> LspClient | None:
    key = f"lsp.{lang}"
    client = ctx.get(key)
    if client is not None:
        if client._proc.returncode is None:
            return client
        del ctx[key]
    user_cmds = ctx.get("lsp_cmd", {})
    candidates = user_cmds.get(lang, _DEFAULT_CMDS.get(lang, []))
    cmd = next((c for c in candidates if shutil.which(c[0])), None)
    if cmd is None:
        return None
    cwd = ctx["cwd"]
    root_uri = pathlib.Path(cwd).as_uri()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=cwd,
    )
    client = LspClient(proc, root_uri)
    await client.start(cwd)
    ctx[key] = client
    logger.debug("Started LSP server for %s: %s", lang, cmd)
    return client


def _lang_for(path: str) -> str | None:
    return _EXT_TO_LANG.get(pathlib.Path(path).suffix.lower())


def _read_lines(path: str) -> list[str]:
    try:
        return (
            pathlib.Path(path)
            .read_bytes()
            .replace(b"\r\n", b"\n")
            .replace(b"\r", b"\n")
            .decode("utf-8", errors="replace")
            .splitlines()
        )
    except OSError:
        return []


def _source_span(path: str, start_line: int, end_line: int) -> str:
    lines = _read_lines(path)
    return "\n".join(lines[max(0, start_line - 1) : min(len(lines), end_line)])


def _sym_range(sym: dict) -> tuple[int, int]:
    rng = sym.get("range") or sym.get("location", {}).get("range", {})
    if not rng:
        return (0, 0)
    return (rng.get("start", {}).get("line", 0), rng.get("end", {}).get("line", 0))


def _rebuild_hierarchy(flat: list[dict]) -> list[dict]:
    items = sorted(
        flat, key=lambda s: (_sym_range(s)[0], -(_sym_range(s)[1] - _sym_range(s)[0]))
    )
    nodes = [{**s, "children": list(s.get("children", []))} for s in items]
    roots: list[dict] = []
    stack: list[tuple[dict, int, int]] = []
    for node in nodes:
        s, e = _sym_range(node)
        while stack and stack[-1][2] < s:
            stack.pop()
        if stack:
            stack[-1][0].setdefault("children", []).append(node)
        else:
            roots.append(node)
        stack.append((node, s, e))
    return roots


def _find_symbol_by_name(syms: list[dict], name: str) -> dict | None:

    def walk(items: list[dict]) -> dict | None:
        for sym in items:
            if sym.get("name") == name:
                return sym
            found = walk(sym.get("children", []))
            if found:
                return found
        return None

    result = walk(syms)
    if result:
        return result
    name_lower = name.lower()

    def walk_ci(items: list[dict]) -> dict | None:
        for sym in items:
            if sym.get("name", "").lower() == name_lower:
                return sym
            found = walk_ci(sym.get("children", []))
            if found:
                return found
        return None

    return walk_ci(syms)


def _find_enclosing(syms: list[dict], line_0: int) -> dict | None:
    best: dict | None = None
    best_span = float("inf")

    def walk(items: list[dict]) -> None:
        nonlocal best, best_span
        for sym in items:
            s, e = _sym_range(sym)
            if s <= line_0 <= e:
                span = e - s
                if span < best_span:
                    best = sym
                    best_span = span
            walk(sym.get("children", []))

    walk(syms)
    return best


def _col_of_name(path: str, line_0: int, name: str) -> int:
    lines = _read_lines(path)
    if line_0 >= len(lines):
        return 0
    src = lines[line_0]
    idx = src.find(name)
    if idx >= 0:
        return idx
    m = re.search("[A-Za-z_][A-Za-z0-9_]*", src)
    return m.start() if m else 0


def _get_hierarchy(client: LspClient, path: str) -> Any:

    async def _inner() -> list[dict]:
        raw = await client.document_symbols(path)
        if not raw:
            return []
        if not any((s.get("children") for s in raw)):
            return _rebuild_hierarchy(raw)
        return raw

    return _inner()


_BUILTIN_HOVER_PREFIXES = (
    "dict() ->",
    "dict(mapping)",
    "dict(iterable)",
    "dict(**kwargs)",
    "list() ->",
    "list(iterable)",
    "set() ->",
    "set(iterable)",
    "frozenset(",
    "Built-in mutable sequence",
    "Built-in immutable sequence",
    "If no argument is given, the constructor",
    "str(object",
    "int(",
    "float(",
    "bool(",
    "tuple(",
)


def _is_builtin_hover(text: str) -> bool:
    t = text.strip()
    return any((t.startswith(p) for p in _BUILTIN_HOVER_PREFIXES))


def _is_import_line(src_lines: list[str], line_0: int) -> bool:
    if src_lines is None or line_0 >= len(src_lines):
        return False
    s = src_lines[line_0].lstrip()
    return s.startswith("import ") or s.startswith("from ")


def _fmt_map(
    syms: list[dict],
    depth: int = 0,
    max_depth: int = 99,
    src_lines: list[str] | None = None,
    _skip: bool = False,
) -> list[str]:
    lines = []
    for sym in syms:
        kind_id = sym.get("kind", 0)
        name = sym.get("name", "?")
        detail = sym.get("detail", "")
        children = sym.get("children", [])
        s, e = _sym_range(sym)
        skip = _skip or (src_lines is not None and _is_import_line(src_lines, s))
        if skip:
            continue
        if kind_id not in _MAP_KINDS:
            if children and depth < max_depth:
                lines.extend(_fmt_map(children, depth + 1, max_depth, src_lines))
            continue
        if s == e and kind_id in (5, 12) and (not children):
            continue
        kind_str = _SYMBOL_KIND_NAME.get(kind_id, "?")
        detail_s = f"  {detail}" if detail else ""
        span = f"line {s + 1}" if s == e else f"lines {s + 1}–{e + 1}"
        prefix = "  " * depth
        lines.append(f"{prefix}{kind_str:<12} {name}{detail_s}  ({span})")
        if children and depth < max_depth:
            lines.extend(_fmt_map(children, depth + 1, max_depth, src_lines))
    return lines


def _fmt_diagnostics(diags: list[dict], rel: str) -> str:
    if not diags:
        return f"No problems found in {rel}."
    by_sev: dict[int, list[dict]] = {}
    for d in diags:
        by_sev.setdefault(d.get("severity", 1), []).append(d)
    parts = []
    for sev in sorted(by_sev):
        label = _SEVERITY.get(sev, "info").upper()
        parts.append(f"── {label} ({len(by_sev[sev])}) " + "─" * 40)
        for d in sorted(
            by_sev[sev],
            key=lambda x: x.get("range", {}).get("start", {}).get("line", 0),
        ):
            start = d.get("range", {}).get("start", {})
            ln = start.get("line", 0) + 1
            col = start.get("character", 0) + 1
            src = f" ({d['source']})" if d.get("source") else ""
            code = f" [{d['code']}]" if d.get("code") else ""
            parts.append(f"  {rel}:{ln}:{col}{code}{src}")
            parts.append(f"    {d.get('message', '')}")
            for ri in d.get("relatedInformation") or []:
                ri_path = ri.get("location", {}).get("uri", "").removeprefix("file://")
                ri_line = (
                    ri.get("location", {})
                    .get("range", {})
                    .get("start", {})
                    .get("line", 0)
                    + 1
                )
                parts.append(
                    f"    note: {ri.get('message', '')}  ({ri_path}:{ri_line})"
                )
    return "\n".join(parts)


def _fmt_rename_edit(edit: dict, cwd: str) -> tuple[str, int]:
    lines = []

    def add(uri: str, edits: list[dict]) -> None:
        fp = _rel(uri, cwd)
        for e in edits:
            s = e.get("range", {}).get("start", {})
            en = e.get("range", {}).get("end", {})
            lines.append(
                f"  {fp}:{s.get('line', 0) + 1}:{s.get('character', 0) + 1}–{en.get('line', 0) + 1}:{en.get('character', 0) + 1}  →  {e.get('newText', '')!r}"
            )

    for uri, edits in (edit.get("changes") or {}).items():
        add(uri, edits)
    for dc in edit.get("documentChanges") or []:
        add(dc.get("textDocument", {}).get("uri", ""), dc.get("edits", []))
    return ("\n".join(lines) or "(empty edit)", len(lines))


def _rel(uri_or_path: str, cwd: str) -> str:
    p = uri_or_path.removeprefix("file://")
    try:
        return os.path.relpath(p, cwd)
    except ValueError:
        return p


def _as_list(x: Any) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


class LspParams(BaseModel):
    action: Literal[
        "map", "inspect", "callers", "definitions", "problems", "rename"
    ] = Field(
        description="map: symbol hierarchy of a file or directory — classes, functions, methods with line spans. Use this first to orient yourself, then inspect the symbols that matter. inspect: full source + type signature + docstring for a named symbol. Pass symbol='ClassName' or symbol='method_name'. No line number needed. callers: every call site of a named symbol, grouped by file with source context. definitions: where a symbol used in this file was defined (follows imports across files). problems: type errors and warnings in a file. rename: preview all sites that would change if a symbol were renamed. No writes applied."
    )
    path: str = Field(
        default=".", description="File (or directory for 'map') relative to cwd."
    )
    symbol: str | None = Field(
        default=None,
        description="Symbol name for inspect / callers / definitions / rename. Use the unqualified name: 'search_file' not 'SymbolFinder.search_file'. Not needed for map or problems.",
    )
    new_name: str | None = Field(
        default=None, description="Replacement name, required for rename."
    )
    lang: str | None = Field(
        default=None, description="Override language detection, e.g. 'py', 'ts'."
    )
    context_lines: int = Field(
        default=4,
        description="Source lines of context around each caller hit. Default 4.",
    )


class LspTool(Tool):
    name = "Lsp"
    description = "Name-based code intelligence via a language server.\n\n  map          — symbol hierarchy of a file/dir (classes, functions, methods)\n  inspect      — source body + type + docstring for a named symbol\n  callers      — every call site of a named symbol with source context\n  definitions  — where a symbol is defined (follows imports)\n  problems     — type errors and warnings\n  rename       — preview all rename sites\n\nWorks by name, not line number. Typical flow: map a file → inspect the symbols that matter → callers to understand usage."
    parameters = LspParams

    async def execute(self, params: LspParams, signal=None) -> dict:
        cwd = self.ctx["cwd"]
        abs_path = resolve_path(params.path, cwd)
        lang = params.lang or _lang_for(abs_path)
        if lang is None and (not abs_path == cwd):
            if pathlib.Path(abs_path).is_dir():
                lang = "py"
            else:
                return tout(
                    f"Unsupported file type: {pathlib.Path(params.path).suffix or '(no extension)'}",
                    True,
                )
        rel = os.path.relpath(abs_path, cwd)
        try:
            if params.action == "map":
                return await self._map(abs_path, rel, cwd, lang, params)
            client = await _get_or_start(self.ctx, lang)
            if client is None:
                names = [c[0] for c in _DEFAULT_CMDS.get(lang, [])]
                return tout(
                    f"No LSP server found for '{lang}'. Install one of: {(', '.join(names) if names else '(none configured)')}",
                    True,
                )
            if params.action == "problems":
                return await self._problems(client, abs_path, rel)
            if not params.symbol:
                return tout(
                    f"'{params.action}' requires 'symbol'. Run map first to see available symbols.",
                    True,
                )
            if params.action == "inspect":
                return await self._inspect(client, abs_path, rel, cwd, params)
            if params.action == "callers":
                return await self._callers(client, abs_path, rel, cwd, params)
            if params.action == "definitions":
                return await self._definitions(client, abs_path, rel, cwd, params)
            if params.action == "rename":
                return await self._rename(client, abs_path, rel, cwd, params)
        except LspError as exc:
            return tout(f"LSP error: {exc}", True)
        except Exception as exc:
            logger.exception("LspTool unexpected error")
            return tout(f"Unexpected error: {exc}", True)
        return tout(f"Unknown action: {params.action}", True)

    async def _map(
        self, abs_path: str, rel: str, cwd: str, lang: str | None, params: LspParams
    ) -> dict:
        p = pathlib.Path(abs_path)
        if p.is_dir():
            parts = []
            for fp in sorted(p.rglob("*")):
                if not fp.is_file():
                    continue
                file_lang = _lang_for(str(fp))
                if file_lang is None:
                    continue
                client = await _get_or_start(self.ctx, file_lang)
                if client is None:
                    continue
                syms = await _get_hierarchy(client, str(fp))
                if not syms:
                    continue
                frel = os.path.relpath(str(fp), cwd)
                src_lines = _read_lines(str(fp))
                lines = _fmt_map(syms, src_lines=src_lines)
                if lines:
                    parts.append(f"\n{frel}")
                    parts.append("─" * min(len(frel), 60))
                    parts.extend(lines)
            return tout("\n".join(parts) if parts else "No symbols found.")
        file_lang = params.lang or _lang_for(abs_path)
        if file_lang is None:
            return tout(f"Unsupported file type: {pathlib.Path(abs_path).suffix}", True)
        client = await _get_or_start(self.ctx, file_lang)
        if client is None:
            names = [c[0] for c in _DEFAULT_CMDS.get(file_lang, [])]
            return tout(
                f"No LSP server found for '{file_lang}'. Install: {', '.join(names)}",
                True,
            )
        syms = await _get_hierarchy(client, abs_path)
        if not syms:
            return tout(f"No symbols found in {rel}.", True)
        src_lines = _read_lines(abs_path)
        lines = _fmt_map(syms, src_lines=src_lines)
        return tout(f"# {rel}\n" + "\n".join(lines))

    async def _inspect(
        self, client: LspClient, abs_path: str, rel: str, cwd: str, params: LspParams
    ) -> dict:
        syms = await _get_hierarchy(client, abs_path)
        sym = _find_symbol_by_name(syms, params.symbol)
        if sym is None:
            return tout(
                f"Symbol '{params.symbol}' not found in {rel}. Run map to see available symbols.",
                True,
            )
        s, e = _sym_range(sym)
        name = sym.get("name", params.symbol)
        col = _col_of_name(abs_path, s, name)
        hover = await client.hover(abs_path, s, col)
        body = _source_span(abs_path, s + 1, e + 1)
        parts: list[str] = [f"# {name}  ({rel}:{s + 1}–{e + 1})"]
        parts.append("")
        if hover and hover.strip() and (not _is_builtin_hover(hover)):
            parts.append("## Type / Signature")
            parts.append(hover.strip())
            parts.append("")
        parts.append("## Source")
        parts.append(body)
        return tout("\n".join(parts))

    async def _callers(
        self, client: LspClient, abs_path: str, rel: str, cwd: str, params: LspParams
    ) -> dict:
        syms = await _get_hierarchy(client, abs_path)
        sym = _find_symbol_by_name(syms, params.symbol)
        if sym is None:
            return tout(
                f"Symbol '{params.symbol}' not found in {rel}. Run map to see available symbols.",
                True,
            )
        s, e = _sym_range(sym)
        name = sym.get("name", params.symbol)
        col = _col_of_name(abs_path, s, name)
        refs = await client.references(abs_path, s, col, include_declaration=False)
        if not refs:
            return tout(f"No references to '{name}' found.")
        by_file: dict[str, list[dict]] = {}
        for r in refs:
            by_file.setdefault(r.get("uri", ""), []).append(r)
        ctx_n = params.context_lines
        parts = [f"{len(refs)} reference(s) to '{name}'"]
        for uri in sorted(by_file):
            fp = uri.removeprefix("file://")
            frel = _rel(uri, cwd)
            frefs = by_file[uri]
            src_lines = _read_lines(fp)
            n = len(src_lines)
            parts.append(f"\n── {frel} ({len(frefs)}) " + "─" * 40)
            windows: list[tuple[int, int, list[dict]]] = []
            for r in sorted(
                frefs, key=lambda x: x.get("range", {}).get("start", {}).get("line", 0)
            ):
                rl = r.get("range", {}).get("start", {}).get("line", 0)
                ws = max(0, rl - ctx_n)
                we = min(n - 1, rl + ctx_n)
                if windows and ws <= windows[-1][1] + 1:
                    ps, pe, pu = windows[-1]
                    windows[-1] = (ps, max(pe, we), pu + [r])
                else:
                    windows.append((ws, we, [r]))
            for win_s, win_e, win_refs in windows:
                hit = {
                    r.get("range", {}).get("start", {}).get("line", 0) for r in win_refs
                }
                parts.append("")
                for i in range(win_s, win_e + 1):
                    src = src_lines[i] if i < n else ""
                    mark = ">" if i in hit else " "
                    parts.append(f"{mark} {i + 1:5d}  {src}")
        return tout("\n".join(parts))

    async def _definitions(
        self, client: LspClient, abs_path: str, rel: str, cwd: str, params: LspParams
    ) -> dict:
        lines = _read_lines(abs_path)
        name = params.symbol
        line_0, col_0 = (0, 0)
        for i, src in enumerate(lines):
            idx = src.find(name)
            if idx >= 0:
                line_0, col_0 = (i, idx)
                break
        defs = await client.definition(abs_path, line_0, col_0)
        if not defs:
            return tout(f"Definition of '{name}' not found.", True)
        parts = [f"Definition(s) of '{name}'"]
        for loc in defs:
            def_uri = loc.get("uri", "")
            def_path = def_uri.removeprefix("file://")
            def_rel = _rel(def_uri, cwd)
            start = loc.get("range", {}).get("start", {})
            def_line_0 = start.get("line", 0)
            try:
                def_syms = await client.document_symbols(def_path)
                if not any((s.get("children") for s in def_syms)):
                    def_syms = _rebuild_hierarchy(def_syms)
                enclosing = _find_enclosing(def_syms, def_line_0)
                if enclosing:
                    ds, de = _sym_range(enclosing)
                    body = _source_span(def_path, ds + 1, de + 1)
                    parts.append(f"\n{def_rel}:{ds + 1}–{de + 1}")
                    parts.append(body)
                else:
                    src_lines = _read_lines(def_path)
                    lo = max(0, def_line_0 - 3)
                    hi = min(len(src_lines), def_line_0 + 8)
                    snippet = "\n".join(
                        (
                            f"{('>' if i == def_line_0 else ' ')} {i + 1:5d}  {src_lines[i]}"
                            for i in range(lo, hi)
                        )
                    )
                    parts.append(f"\n{def_rel}:{def_line_0 + 1}")
                    parts.append(snippet)
            except Exception:
                parts.append(f"\n{def_rel}:{def_line_0 + 1}")
        return tout("\n".join(parts))

    async def _problems(self, client: LspClient, abs_path: str, rel: str) -> dict:
        diags = await client.diagnostics(abs_path)
        return tout(_fmt_diagnostics(diags, rel))

    async def _rename(
        self, client: LspClient, abs_path: str, rel: str, cwd: str, params: LspParams
    ) -> dict:
        if not params.new_name:
            return tout("'rename' requires 'new_name'.", True)
        syms = await _get_hierarchy(client, abs_path)
        sym = _find_symbol_by_name(syms, params.symbol)
        if sym is None:
            return tout(
                f"Symbol '{params.symbol}' not found in {rel}. Run map to see available symbols.",
                True,
            )
        s, e = _sym_range(sym)
        name = sym.get("name", params.symbol)
        col = _col_of_name(abs_path, s, name)
        edit = await client.rename(abs_path, s, col, params.new_name)
        if not edit:
            return tout(
                f"Server returned no rename edit for '{name}'. Symbol may not be renameable.",
                True,
            )
        formatted, n_sites = _fmt_rename_edit(edit, cwd)
        return tout(
            f"Rename '{name}' → '{params.new_name}' — {n_sites} site(s):\n{formatted}"
        )


async def shutdown_lsp_servers(ctx: dict) -> None:
    for key in [k for k in ctx if k.startswith("lsp.")]:
        client = ctx.pop(key, None)
        if isinstance(client, LspClient):
            try:
                await asyncio.wait_for(client.shutdown(), timeout=4)
            except Exception:
                pass

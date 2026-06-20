from __future__ import annotations
import asyncio
import contextlib
import fnmatch
import json
import os
import pathlib
import difflib
import re
import string
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import Any, Generator, Literal
from pydantic import BaseModel, Field, ValidationError, field_validator
logger = logging.getLogger(__name__)

class Tool(ABC):
    name: str
    description: str
    parameters: type[BaseModel]

    def __init__(self, ctx: dict) -> None:
        self.ctx = ctx

    @abstractmethod
    async def execute(self, params: BaseModel, signal: asyncio.Event | None=None) -> dict:
        ...

    def schema(self) -> dict[str, Any]:
        s = self.parameters.model_json_schema()
        return {'name': self.name, 'description': self.description, 'input_schema': {'type': 'object', 'properties': s.get('properties', {}), 'required': s.get('required', []), **({'$defs': s['$defs']} if '$defs' in s else {})}}

def validate_tool_call(tool: Tool, call: dict) -> BaseModel:
    try:
        return tool.parameters.model_validate(call['arguments'])
    except ValidationError as exc:
        details = '; '.join((f'{'.'.join((str(p) for p in e['loc']))}: {e['msg']}' for e in exc.errors()))
        raise ValueError(f"Invalid arguments for '{tool.name}': {details}")

def tout(text: str, iserr: bool=False) -> dict:
    return {'content': [{'type': 'text', 'text': _truncate(text)}], 'is_error': iserr}
_MAX_BYTES = 50 * 1024
_MAX_LINES = 2000

@dataclass
class _LockEntry:
    lock: asyncio.Lock = dc_field(default_factory=asyncio.Lock)
    refs: int = 0
_FILE_LOCKS: dict[str, _LockEntry] = {}
_FILE_LOCKS_GUARD: asyncio.Lock | None = None

def _get_guard() -> asyncio.Lock:
    global _FILE_LOCKS_GUARD
    if _FILE_LOCKS_GUARD is None:
        _FILE_LOCKS_GUARD = asyncio.Lock()
    return _FILE_LOCKS_GUARD

@contextlib.asynccontextmanager
async def _file_lock(path: str):
    guard = _get_guard()
    async with guard:
        entry = _FILE_LOCKS.setdefault(path, _LockEntry())
        entry.refs += 1
    try:
        async with entry.lock:
            yield
    finally:
        async with guard:
            entry.refs -= 1
            if entry.refs == 0:
                _FILE_LOCKS.pop(path, None)

def _truncate(text: str, label: str='') -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) > _MAX_LINES:
        text = ''.join(lines[:_MAX_LINES])
        text += f'\n[truncated at {_MAX_LINES} lines{(': ' + label if label else '')}]'
    encoded = text.encode()
    if len(encoded) > _MAX_BYTES:
        text = encoded[:_MAX_BYTES].decode(errors='replace')
        text += f'\n[truncated at 50 KB{(': ' + label if label else '')}]'
    return text

def resolve_path(path: str, cwd: str) -> str:
    path = path.lstrip('@')
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
        gi = d / '.gitignore'
        if gi.exists():
            try:
                for ln in gi.read_text(encoding='utf-8', errors='replace').splitlines():
                    ln = ln.strip()
                    if ln and (not ln.startswith('#')) and (not ln.startswith('!')):
                        patterns.append(ln)
            except OSError:
                pass
    return patterns

def _is_gitignored(name: str, patterns: list[str]) -> bool:
    return any((fnmatch.fnmatch(name, p) or fnmatch.fnmatch(name + '/', p) for p in patterns))

def _iter_files(search_root: str, glob: str | None) -> Generator[pathlib.Path, None, None]:
    for dirpath, dirnames, filenames in os.walk(search_root):
        gi = _gitignore_patterns(dirpath)
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and (not _is_gitignored(d, gi))]
        for fname in filenames:
            if _is_gitignored(fname, gi):
                continue
            if glob and (not fnmatch.fnmatch(fname, glob)):
                continue
            yield (pathlib.Path(dirpath) / fname)

class ReadParams(BaseModel):
    path: str = Field(description='File path to read (relative to cwd)')
    offset: int | None = Field(default=None, ge=1, description='Start line (1-based); first line is 1')
    limit: int | None = Field(default=None, description='Max lines to read')

class ReadTool(Tool):
    name = 'Read'
    description = 'Read a file. Use offset/limit for large files instead of reading the whole thing.'
    parameters = ReadParams

    async def execute(self, params: ReadParams, signal=None) -> dict:
        path = resolve_path(params.path, self.ctx['cwd'])
        try:
            lines = pathlib.Path(path).read_text(encoding='utf-8', errors='replace').splitlines(keepends=True)
        except FileNotFoundError:
            raise ValueError(f'File not found: {params.path}')
        except IsADirectoryError:
            raise ValueError(f'Path is a directory: {params.path}')
        start = max(0, params.offset - 1 if params.offset is not None else 0)
        end = start + params.limit if params.limit else len(lines)
        header = f'# {params.path}  (lines {start + 1}–{min(end, len(lines))} of {len(lines)})\n'
        return tout(_truncate(header + ''.join(lines[start:end]), params.path))

class WriteParams(BaseModel):
    path: str = Field(description='File path to create or overwrite (relative to cwd)')
    content: str = Field(description='Full file content')

class WriteTool(Tool):
    name = 'Write'
    description = "Create or overwrite a file. Prefer 'edit' for small changes to existing files."
    parameters = WriteParams

    async def execute(self, params: WriteParams, signal=None) -> dict:
        path = resolve_path(params.path, self.ctx['cwd'])
        async with _file_lock(path):
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            existed = pathlib.Path(path).exists()
            pathlib.Path(path).write_text(params.content, encoding='utf-8')
        action = 'overwrote' if existed else 'created'
        lines = params.content.count('\n') + 1
        return tout(f'{action} {params.path} ({lines} lines)')

class EditParams(BaseModel):
    path: str = Field(description='File path to edit (relative to cwd)')
    old_text: str = Field(description='Exact text to replace (must appear exactly once)')
    new_text: str = Field(description='Replacement text')

class EditTool(Tool):
    name = 'Edit'
    description = 'Replace an exact unique string in a file. Read the file first if unsure of exact text.'
    parameters = EditParams

    async def execute(self, params: EditParams, signal=None) -> dict:
        path = resolve_path(params.path, self.ctx['cwd'])
        async with _file_lock(path):
            try:
                original = pathlib.Path(path).read_text(encoding='utf-8')
            except FileNotFoundError:
                raise ValueError(f'File not found: {params.path}')
            count = original.count(params.old_text)
            if count == 0:
                raise ValueError(f'old_text not found in {params.path}')
            if count > 1:
                raise ValueError(f'old_text appears {count} times in {params.path}; make it more specific')
            modified = original.replace(params.old_text, params.new_text, 1)
            pathlib.Path(path).write_text(modified, encoding='utf-8')
            original_lines = original.splitlines(keepends=True)
            modified_lines = modified.splitlines(keepends=True)
            diff = difflib.unified_diff(original_lines, modified_lines, fromfile=f'a/{params.path}', tofile=f'b/{params.path}')
            diff_text = ''.join(diff)
            if not diff_text:
                diff_text = f'No visible changes made to {params.path}'
        return tout(diff_text)

class BashParams(BaseModel):
    command: str = Field(description='Shell command to execute')
    timeout: int = Field(default=120, description='Timeout in seconds')

class BashTool(Tool):
    name = 'Bash'
    description = 'Run a shell command, get stdout+stderr. Prefer read/grep/find/ls for file exploration.'
    parameters = BashParams

    async def execute(self, params: BashParams, signal=None) -> dict:
        proc_env = self.ctx.get('env')
        proc = await asyncio.create_subprocess_shell(params.command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=self.ctx['cwd'], env=proc_env)
        communicate_task = asyncio.create_task(proc.communicate())
        abort_task: asyncio.Task | None = None
        if signal is not None:
            abort_task = asyncio.create_task(signal.wait())
        watch: set[asyncio.Task] = {communicate_task}
        if abort_task:
            watch.add(abort_task)
        done, _ = await asyncio.wait(watch, timeout=params.timeout)
        if abort_task and (not abort_task.done()):
            abort_task.cancel()
            try:
                await abort_task
            except (asyncio.CancelledError, Exception):
                pass
        if communicate_task in done:
            stdout, _ = communicate_task.result()
            output = stdout.decode(errors='replace')
            exit_code = proc.returncode or 0
            text = _truncate(output + (f'\n[exit code {exit_code}]' if exit_code != 0 else ''), params.command)
            return tout(text)
        communicate_task.cancel()
        proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        if abort_task and abort_task.done() and (not abort_task.cancelled()):
            raise ValueError('Command aborted')
        raise ValueError(f'Command timed out after {params.timeout}s')

class GrepParams(BaseModel):
    pattern: str = Field(description='Regex or literal pattern')
    path: str | None = Field(default=None, description='Directory or file (default: cwd)')
    glob: str | None = Field(default=None, description="File glob filter e.g. '*.py'")
    ignore_case: bool = Field(default=False)
    literal: bool = Field(default=False, description='Treat pattern as literal string')
    context: int | None = Field(default=None, description='Lines of context around each match')
    limit: int = Field(default=100, description='Max matches to return')

class GrepTool(Tool):
    name = 'Grep'
    description = 'Search files for a pattern. Respects .gitignore.'
    parameters = GrepParams

    async def execute(self, params: GrepParams, signal=None) -> dict:
        cwd = self.ctx['cwd']
        search_root = resolve_path(params.path, cwd) if params.path else cwd
        flags = re.IGNORECASE if params.ignore_case else 0
        try:
            compiled = re.compile(re.escape(params.pattern) if params.literal else params.pattern, flags)
        except re.error as exc:
            raise ValueError(f'Invalid regex: {exc}')
        search_path = pathlib.Path(search_root)
        if search_path.is_file():
            file_iter = iter([search_path])
        else:
            file_iter = _iter_files(search_root, params.glob)
        matches: list[str] = []
        count = 0
        ctx_lines = params.context or 0
        for fpath in file_iter:
            if count >= params.limit:
                break
            try:
                file_lines = fpath.read_text(encoding='utf-8', errors='replace').splitlines()
            except (OSError, IsADirectoryError):
                continue
            rel = os.path.relpath(str(fpath), cwd)
            emitted: set[int] = set()
            for i, line in enumerate(file_lines):
                if count >= params.limit:
                    break
                if compiled.search(line):
                    for j in range(max(0, i - ctx_lines), min(len(file_lines), i + ctx_lines + 1)):
                        if j not in emitted:
                            matches.append(f'{rel}:{j + 1}{':-'[j != i]}{file_lines[j]}')
                            emitted.add(j)
                    if ctx_lines:
                        matches.append('--')
                    count += 1
        text = '\n'.join(matches) if matches else f"No matches for '{params.pattern}'"
        if count >= params.limit:
            text += f'\n[truncated at {params.limit} matches]'
        return tout(text)

class FindParams(BaseModel):
    path: str | None = Field(default=None, description='Directory to search (default: cwd)')
    pattern: str | None = Field(default=None, description="Filename glob pattern e.g. '*.py'")
    type: Literal['file', 'dir', 'any'] = Field(default='any')
    limit: int = Field(default=200)

class FindTool(Tool):
    name = 'Find'
    description = 'Find files or directories by name pattern. Respects .gitignore.'
    parameters = FindParams

    async def execute(self, params: FindParams, signal=None) -> dict:
        cwd = self.ctx['cwd']
        search_root = resolve_path(params.path, cwd) if params.path else cwd
        results: list[str] = []
        for dirpath, dirnames, filenames in os.walk(search_root):
            gi = _gitignore_patterns(dirpath)
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and (not _is_gitignored(d, gi))]
            if params.type in ('dir', 'any'):
                for d in dirnames:
                    if not params.pattern or fnmatch.fnmatch(d, params.pattern):
                        results.append(os.path.relpath(os.path.join(dirpath, d), cwd) + '/')
            if params.type in ('file', 'any'):
                for f in filenames:
                    if not _is_gitignored(f, gi) and (not params.pattern or fnmatch.fnmatch(f, params.pattern)):
                        results.append(os.path.relpath(os.path.join(dirpath, f), cwd))
            if len(results) >= params.limit:
                results = results[:params.limit]
                results.append(f'[truncated at {params.limit} results]')
                break
        return tout('\n'.join(results) if results else 'No results')

class LsParams(BaseModel):
    path: str | None = Field(default=None, description='Directory to list (default: cwd)')

class LsTool(Tool):
    name = 'Ls'
    description = 'List directory contents. Respects .gitignore.'
    parameters = LsParams

    async def execute(self, params: LsParams, signal=None) -> dict:
        cwd = self.ctx['cwd']
        target = resolve_path(params.path, cwd) if params.path else cwd
        p = pathlib.Path(target)
        if not p.is_dir():
            raise ValueError(f'Not a directory: {params.path or '.'}')
        gi = _gitignore_patterns(target)
        entries = [e.name + ('/' if e.is_dir() else '') for e in sorted(p.iterdir()) if not e.name.startswith('.') and (not _is_gitignored(e.name, gi))]
        return tout('\n'.join(entries) if entries else '(empty)')

class SkillParams(BaseModel):
    name: str = Field(description='Skill name to retrieve')

class SkillTool(Tool):
    name = 'Skill'
    description = 'Retrieve the full instructions for a skill by name.'
    parameters = SkillParams

    async def execute(self, params: SkillParams, signal=None) -> dict:
        skills = {s['name']: s['content'] for s in self.ctx.get('skills', [])}
        content = skills.get(params.name)
        if content is None:
            return tout(f"Skill '{params.name}' not found.", True)
        return tout(content)

class AgentParams(BaseModel):
    task: str = Field(description='Clear autonomous task for the sub-agent. Should include objective, constraints, expected output, and relevant context.')
    api: str | None = Field(default=None, description='API override (claude/gemini/codex). Defaults to parent.')
    system: str | None = Field(default=None, description='System prompt override. Defaults to parent.')
    tools: list[str] = Field(description="Tools available to the sub-agent. Must be an explicit subset of the parent's tools.")

    @field_validator('tools', mode='before')
    @classmethod
    def coerce_tools(cls, v):
        if isinstance(v, str):
            v = json.loads(v)
        if not isinstance(v, list):
            raise ValueError('tools must be a list of strings')
        return v

class AgentTool(Tool):
    name = 'Agent'
    description = 'Spawn an autonomous sub-agent to delegate complex, multi-step, or parallelizable work.\n\nGood use cases: implementing features, investigating bugs, deep codebase research, long tool chains, refactoring, parallel exploration of approaches.\n\nAvoid for: trivial single-step ops, simple reads/searches, tiny edits, tasks needing tight conversational continuity.\n\nThe sub-agent does NOT stream back. Only the final response is returned.'
    parameters = AgentParams

    async def execute(self, params: AgentParams, signal=None) -> dict:
        from gits import create_worktree, commit_worktree, merge_branch_into_worktree
        parent = self.ctx['agent']
        overrides = {}
        if params.api is not None:
            overrides['api'] = params.api
        if params.system is not None:
            overrides['system'] = params.system
        overrides['tool_names'] = params.tools
        child = parent.spawn(**overrides)
        parent_gwt = parent.ctx.get('gwt')
        parent_cwd = parent.ctx.get('cwd', os.getcwd())
        repo_dir = parent_gwt.worktree if parent_gwt else parent_cwd
        child_gwt = create_worktree(repo_dir, prefix='subagent')
        if child_gwt is None:
            return tout('Agent failed: could not create isolated worktree. Check .log.json for git errors.', True)
        child.ctx['gwt'] = child_gwt
        child.ctx['cwd'] = child_gwt.worktree
        if 'env' in child.ctx:
            child.ctx['env']['PWD'] = child_gwt.worktree
        if '_stream_log_fp' in parent.ctx:
            from .stream_log import StreamLogger
            StreamLogger.attach_to_child(child, parent.ctx, tool_name=f'sub-agent-{''.join(random.choices(string.ascii_letters + string.digits, k=2))}')
        try:
            result = await child.run(params.task)
            commit_worktree(child_gwt, child.messages)
            merge_branch_into_worktree(parent_gwt, child_gwt)
            texts = [b.get('text', '') for b in result['content'] if b.get('type') == 'text']
            combined = ''.join(texts).strip()
            if not combined:
                stop = result.get('stop_reason', 'unknown')
                err = result.get('error_message', '')
                detail = f'stop_reason={stop}' + (f', error={err}' if err else '')
                return tout(f'Agent failed: No text output ({detail}). It may have glitched during thinking or stopped without responding.', True)
            return tout(combined)
        except Exception as e:
            return tout(f'Agent failed: {e}', True)
DEFAULT_TOOLS: list[type[Tool]] = [ReadTool, GrepTool, FindTool, LsTool, WriteTool, EditTool, BashTool, SkillTool, AgentTool]
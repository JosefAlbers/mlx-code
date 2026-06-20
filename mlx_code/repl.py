from __future__ import annotations
import asyncio
import copy
import datetime
import json
import os
import pathlib
import re
import socket
import sys
import tempfile
import time
import logging
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol
from .gits import create_worktree, commit_worktree, resume_worktree, cleanup_worktree, git_new_branch, git_new_branch_at, git_switch_branch, GitError, get_commit_history_with_stats, find_rev_commit, get_diff_between_refs, get_branch_base_sha, resolve_ref_short
from .tools import Tool, validate_tool_call, DEFAULT_TOOLS
from .apis import resolve_api
logger = logging.getLogger(__name__)

def load_agent_config(path: str) -> dict:
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    with open(path, 'r') as f:
        if path.endswith(('.yaml', '.yml')):
            try:
                import yaml
            except ImportError:
                raise ValueError('PyYAML required for YAML configs: pip install pyyaml')
            return yaml.safe_load(f) or {}
        return json.load(f)

def _branch_index_title(parent_path: tuple[int, ...], existing_tabs: list) -> tuple[tuple[int, ...], str]:
    depth = len(parent_path) + 1
    child_count = sum((1 for t in existing_tabs if len(t.index_path) == depth and t.index_path[:-1] == parent_path))
    index_path = parent_path + (child_count,)
    title = 'branch-' + '-'.join((str(i) for i in index_path))
    return (index_path, title)

class Agent:

    def __init__(self, system=None, api=None, tool_names=None, extra_tool_classes=None, model=None, api_key=None, base_url=None, ctx=None):
        self.api = resolve_api(api='default' if api is None else api, model=model, api_key=api_key, base_url=base_url)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.system = '' if system is None else system
        self.messages: list[dict] = []
        self._signal: asyncio.Event | None = None
        self._listeners: set[Callable] = set()
        self._extra_tool_classes: list[type[Tool]] = extra_tool_classes or []
        self.ctx: dict = {'cwd': os.getcwd(), 'skills': [], 'gwt': None, **(ctx or {})}
        self.ctx['agent'] = self
        self._last_result_sig: str | None = None
        self._same_result_count: int = 0
        tools = [cls(self.ctx) for cls in DEFAULT_TOOLS + self._extra_tool_classes]
        if tool_names is not None:
            name_set = {n.lower() for n in tool_names}
            tools = [t for t in tools if t.name.lower() in name_set]
        self.tools = tools

    def spawn(self, **overrides) -> 'Agent':
        kwargs = {'system': self.system, 'extra_tool_classes': self._extra_tool_classes, 'model': self.model, 'api_key': self.api_key, 'base_url': self.base_url, 'ctx': {k: v for k, v in self.ctx.items() if k != 'agent'}}
        kwargs.update(overrides)
        return Agent(**kwargs)

    def branch(self) -> 'Agent':
        child = self.spawn(tool_names=[t.name for t in self.tools])
        child.messages = copy.deepcopy(self.messages)
        return child

    async def run(self, prompt: str) -> dict:
        await self._wait()
        self._signal = None
        self._last_result_sig = None
        self._same_result_count = 0
        self.messages.append({'role': 'user', 'content': prompt})
        return await self._loop()

    def abort(self) -> None:
        if self._signal is None:
            self._signal = asyncio.Event()
        self._signal.set()

    def subscribe(self, fn: Callable[[dict], Any]) -> Callable:
        self._listeners.add(fn)
        return lambda: self._listeners.discard(fn)

    async def _wait(self, timeout: float=30.0, retry_delay: float=0.2):
        parsed = urllib.parse.urlparse(self.api.base_url)
        host, port = (parsed.hostname, parsed.port)
        if port is None:
            return
        start = time.monotonic()
        loop = asyncio.get_running_loop()
        while time.monotonic() - start < timeout:
            try:
                conn = await loop.run_in_executor(None, lambda: socket.create_connection((host, port), timeout=0.2))
                conn.close()
                return
            except (OSError, ConnectionRefusedError):
                await asyncio.sleep(retry_delay)
        raise TimeoutError(f'Backend at {self.api.base_url} did not respond within {timeout}s.')

    async def _emit(self, event: dict) -> None:
        for fn in list(self._listeners):
            r = fn(event)
            if asyncio.iscoroutine(r):
                await r

    async def _loop(self) -> dict:
        await self._emit({'type': 'agent_start', 'payload': {}})
        final: dict = {'role': 'assistant', 'content': [], 'stop_reason': 'error', 'error_message': 'no turns ran', 'usage': {'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0}}
        while True:
            await self._emit({'type': 'turn_start', 'payload': {}})
            api_messages = [m for m in self.messages if m.get('role') != 'commit']
            es = await self.api.stream(api_messages, self.system, self.tools)
            async for event in es:
                if event['type'] in ('text_delta', 'thinking_delta', 'error'):
                    await self._emit({'type': event['type'], 'payload': event['payload']})
            final = await es.result()
            self.messages.append(final)
            await self._emit({'type': 'turn_end', 'payload': {'message': final}})
            if final['stop_reason'] in ('error', 'aborted'):
                break
            if self._signal and self._signal.is_set():
                final['stop_reason'] = 'aborted'
                break
            calls = [b for b in final['content'] if b['type'] == 'toolCall']
            if not calls:
                break
            results = await self._execute_tools(calls)
            result_sig = json.dumps([(r['tool_name'], r['tool_args'], r['content']) for r in results], sort_keys=True)
            if result_sig == self._last_result_sig:
                self._same_result_count += 1
            else:
                self._last_result_sig = result_sig
                self._same_result_count = 0
            if self._same_result_count >= 2:
                warn = f'\n\n[SYSTEM WARNING: Same tool result {self._same_result_count}x in a row. You are likely in a loop. Change your strategy.]\n\n'
                for r in results:
                    if r.get('content') and isinstance(r['content'], list):
                        if r['content'][0]['type'] == 'text':
                            r['content'][0]['text'] += warn
                        else:
                            r['content'].insert(0, {'type': 'text', 'text': warn})
            self.messages.extend(results)
            await self._emit({'type': 'tool_results_ready', 'payload': {}})
            if self._signal and self._signal.is_set():
                final['stop_reason'] = 'aborted'
                break
        new_gwt, diff_stat = commit_worktree(self.ctx['gwt'], self.messages)
        self.ctx['gwt'] = new_gwt
        if diff_stat:
            sha = new_gwt.commit[:8] if new_gwt else ''
            self.messages.append({'role': 'commit', 'content': f'[{sha}]\n{diff_stat}', 'sha': sha})
            await self._emit({'type': 'commit', 'payload': {'diff_stat': diff_stat, 'sha': sha}})
        await self._emit({'type': 'agent_end', 'payload': {'message': final}})
        logger.debug(json.dumps(self.messages, indent=2, ensure_ascii=False))
        return final

    async def _execute_tools(self, calls: list[dict]) -> list[dict]:
        return list(await asyncio.gather(*[self._execute_one(c) for c in calls]))

    async def _execute_one(self, call: dict) -> dict:
        await self._emit({'type': 'tool_start', 'payload': {'name': call['name'], 'args': call['arguments']}})
        tool = next((t for t in self.tools if t.name == call['name']), None)
        if tool is None:
            result = {'content': [{'type': 'text', 'text': f"Tool '{call['name']}' not found"}], 'is_error': True}
        else:
            try:
                result = await tool.execute(validate_tool_call(tool, call), self._signal)
            except Exception as exc:
                result = {'content': [{'type': 'text', 'text': str(exc)}], 'is_error': True}
        msg = {'role': 'toolResult', 'tool_call_id': call['id'], 'tool_name': call['name'], 'tool_args': call['arguments'], 'content': result['content'], 'is_error': result['is_error']}
        await self._emit({'type': 'tool_result', 'payload': {'message': msg}})
        await self._emit({'type': 'tool_end', 'payload': {'name': call['name'], 'is_error': result['is_error'], 'result': msg}})
        return msg

@dataclass
class TabModel:
    agent: Agent
    title: str
    errors: list[tuple[str, str]] = field(default_factory=list)
    index_path: tuple[int, ...] = ()
    owns_worktree: bool = False
    is_main: bool = False
    status: str = 'idle'
    running_task: asyncio.Task | None = None
    last_error: str = ''

    @property
    def is_running(self) -> bool:
        return self.running_task is not None and (not self.running_task.done())

class UIAdapter(Protocol):

    def show_error(self, text: str) -> None:
        ...

    def show_command_result(self, cmd: str, content: str | object) -> None:
        ...

    def show_diff(self, diff_text: str, ref1_label: str, ref2_label: str) -> None:
        ...

    def show_history_list(self, lines: list[str]) -> None:
        ...

    def show_history_raw(self, json_text: str) -> None:
        ...

    async def add_tab(self, tab: TabModel) -> None:
        ...

    def remove_tab(self, removed_index: int) -> None:
        ...

    def switch_to_tab(self, index: int) -> None:
        ...

    def refresh_chrome(self) -> None:
        ...

    def clear_tab_display(self, tab: TabModel) -> None:
        ...

    def on_agent_event(self, event: dict, tab: TabModel) -> None:
        ...

    async def run_captured_shell(self, command: str, cwd: str, env: dict | None) -> str:
        ...

    async def run_interactive_shell(self, command: str, cwd: str, env: dict | None) -> int:
        ...

    def exit_app(self, summary: list[dict]) -> None:
        ...
HELP_TEXT = '\nCommands:\n/help               show this message\n/clear [--config F] clear conversation; --config reconfigures agent from YAML/JSON\n/history            show full conversation transcript\n/history --raw      show raw API message log (debug)\n/diff [--all]       show side-by-side diff of changes\n/errors             show timestamped error log for this tab\n/tools              list active tools\n/branch [--rev N] [--no-worktree] [prompt]\n                    open a branch tab; optional prompt runs immediately\n/branches           list all tabs/branches\n/abort              abort the running agent\n/export [path]      export session to JSON\n/verbose            toggle verbose mode (show raw tool calls and output)\n/merge              merge this branch into its parent tab, then close\n/exit  /quit [--all] close branch tab, or exit the app\n!command            run shell command in worktree (output captured)\n$command            run interactive shell command (terminal handed to process)\n                    e.g.  !ls  !git diff  !cat file.py\n                          $vim file.py  $yazi  $less log.txt\n\nKeys (TUI only):\nEnter               submit\nCtrl-J              insert newline in editor\nCtrl-1 … Ctrl-9     jump directly to tab N\nCtrl-, / Ctrl-.     cycle through tabs\nCtrl-C              abort running agent\nCtrl-D              close branch tab (exit app if last tab)\nCtrl-R              recall last prompt into editor\n'

class CommandEngine:

    def __init__(self):
        self.ui: UIAdapter | None = None
        self.tabs: list[TabModel] = []
        self.active_index: int = 0
        self._unsubscribers: dict[int, Callable] = {}
        self.verbose: bool = False
        self.exit_summary: list[dict] | None = None

    def bind(self, ui: UIAdapter) -> None:
        self.ui = ui

    @property
    def active_tab(self) -> TabModel:
        return self.tabs[self.active_index]

    def attach_agent(self, tab: TabModel) -> None:
        key = id(tab.agent)
        if key in self._unsubscribers:
            return

        async def on_event(event: dict) -> None:
            et = event.get('type')
            payload = event.get('payload', {})
            if et == 'agent_start':
                tab.status = 'running'
                tab.last_error = ''
            elif et == 'agent_end':
                if tab.status not in ('error', 'aborting…'):
                    tab.status = 'idle'
            elif et == 'error':
                err = str(payload.get('error', payload))
                ts = datetime.datetime.now().strftime('%H:%M:%S')
                tab.errors.append((ts, err))
                tab.last_error = err
            if self.ui is not None:
                r = self.ui.on_agent_event(event, tab)
                if asyncio.iscoroutine(r):
                    await r
                if et in ('agent_start', 'agent_end', 'turn_end', 'commit'):
                    self.ui.refresh_chrome()
        self._unsubscribers[key] = tab.agent.subscribe(on_event)

    def detach_agent(self, tab: TabModel) -> None:
        key = id(tab.agent)
        if key in self._unsubscribers:
            self._unsubscribers[key]()
            del self._unsubscribers[key]

    def _find_parent_tab(self, tab: TabModel) -> TabModel | None:
        if not tab.index_path:
            return None
        parent_path = tab.index_path[:-1]
        for t in self.tabs:
            if t.index_path == parent_path:
                return t
        return None

    async def handle_input(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if text.startswith('$'):
            cmd = text[1:].strip()
            if cmd:
                await self._dispatch_interactive_shell(cmd)
        elif text.startswith('!'):
            cmd = text[1:].strip()
            if cmd:
                await self._dispatch_captured_shell(cmd)
        elif text.startswith('/'):
            await self.dispatch_command(text)
        else:
            await self._run_agent(self.active_tab, text)

    async def dispatch_command(self, text: str) -> None:
        cmd, _, arg = text.partition(' ')
        cmd = cmd.lower().strip()
        arg = arg.strip()
        handlers = {'/help': self._cmd_help, '/clear': self._cmd_clear, '/history': self._cmd_history, '/diff': self._cmd_diff, '/errors': self._cmd_errors, '/tools': self._cmd_tools, '/abort': self._cmd_abort, '/branch': self._cmd_branch, '/branches': self._cmd_branches, '/tab': self._cmd_tab, '/export': self._cmd_export, '/verbose': self._cmd_verbose, '/merge': self._cmd_merge, '/exit': self._cmd_exit, '/quit': self._cmd_exit}
        handler = handlers.get(cmd)
        if handler:
            await handler(arg)
        else:
            self.ui.show_error(f'Unknown command: {cmd!r} — try /help')

    async def _cmd_help(self, arg: str) -> None:
        self.ui.show_command_result('/help', HELP_TEXT)

    async def _cmd_clear(self, arg: str) -> None:
        cfg_match = re.match('--config\\s+(.+)', arg)
        if cfg_match:
            config_path = cfg_match.group(1).strip().strip('"').strip("'")
            try:
                cfg = load_agent_config(config_path)
            except Exception as e:
                self.ui.show_error(f'Config error: {e}')
                return
            tab = self.active_tab
            if tab.is_running:
                self.ui.show_error('Agent is running — /abort first before /clear --config.')
                return
            old = tab.agent
            new_ctx = {k: v for k, v in old.ctx.items() if k != 'agent'}
            new_agent = Agent(system=cfg.get('system'), api=cfg.get('api'), model=cfg.get('model'), api_key=cfg.get('api_key'), base_url=cfg.get('base_url'), tool_names=cfg.get('tools'), extra_tool_classes=old._extra_tool_classes, ctx=new_ctx)
            self.detach_agent(tab)
            tab.agent = new_agent
            self.attach_agent(tab)
            self.ui.clear_tab_display(tab)
            self.ui.show_command_result('/clear', f'Agent reconfigured from {config_path}\n  model: {cfg.get('model', '(default)')}\n  tools: {', '.join((t.name for t in new_agent.tools))}')
            self.ui.refresh_chrome()
        else:
            self.active_tab.agent.messages.clear()
            self.ui.clear_tab_display(self.active_tab)
            self.ui.show_command_result('/clear', 'Conversation cleared.')
            self.ui.refresh_chrome()

    async def _cmd_abort(self, arg: str) -> None:
        tab = self.active_tab
        if tab.is_running:
            tab.agent.abort()
            tab.status = 'aborting…'
            self.ui.show_command_result('/abort', 'Abort requested.')
            self.ui.refresh_chrome()
        else:
            self.ui.show_error('Nothing is running.')

    async def _cmd_export(self, arg: str) -> None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        user_cwd = self.active_tab.agent.ctx.get('user_cwd', os.getcwd())
        path = arg if os.path.isabs(arg) else os.path.join(user_cwd, arg or f'session_{ts}.json')
        data = {'version': 1, 'exported_at': ts, 'system': self.active_tab.agent.system, 'messages': self.active_tab.agent.messages}
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.active_tab.status = f'exported → {path}'
            self.ui.show_command_result('/export', f'Exported → {path}')
            self.ui.refresh_chrome()
        except OSError as exc:
            self.ui.show_error(f'Export failed: {exc}')

    async def _cmd_history(self, arg: str) -> None:
        if arg == '--raw':
            raw = json.dumps(self.active_tab.agent.messages, indent=2, default=str)
            self.ui.show_history_raw(raw)
            return
        user_msgs = [m for m in self.active_tab.agent.messages if m.get('role') == 'user']
        if not user_msgs:
            self.ui.show_command_result('/history', 'No prompts yet.')
            return
        lines: list[str] = []
        gwt = self.active_tab.agent.ctx.get('gwt')
        commit_stats: list[dict] = []
        if gwt and getattr(gwt, 'worktree', None):
            try:
                commit_stats = get_commit_history_with_stats(gwt.worktree, limit=len(user_msgs) + 5)
            except Exception as e:
                logger.warning(f'Failed to get commit stats: {e}')
        turn_commits: dict[int, list[dict]] = {}
        for c in commit_stats:
            turn = c.get('user_turns', 0)
            if turn > 0:
                turn_commits.setdefault(turn, []).append(c)
        for i, m in enumerate(user_msgs, 1):
            content = m.get('content', '')
            if isinstance(content, list):
                content = ' '.join((b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text'))
            content = re.sub('\\s+', ' ', content).strip()
            if len(content) > 100:
                content = content[:100] + '…'
            line = f'{i}. {content}'
            if i in turn_commits:
                for c in turn_commits[i]:
                    ref_str = c.get('refs', '').strip(' ()')
                    if ref_str.startswith('HEAD -> '):
                        ref_str = ref_str[8:]
                    ref_str = ref_str.replace('HEAD, ', '').strip()
                    hash_str = f'[{c['short_sha']}]'
                    if ref_str:
                        hash_str += f' on {ref_str}'
                    else:
                        hash_str += ' (detached)'
                    line += f'\n   ↪ Commit {hash_str}'
                    seen: set[str] = set()
                    unique_files = [f for f in c['files'] if not (f in seen or seen.add(f))]
                    if unique_files:
                        file_lines = '\n     '.join(unique_files[:8])
                        if len(unique_files) > 8:
                            file_lines += f'\n     ... and {len(unique_files) - 8} more'
                        line += f'\n     {file_lines}'
            lines.append(line)
        self.ui.show_history_list(lines)

    async def _cmd_diff(self, arg: str) -> None:
        gwt = self.active_tab.agent.ctx.get('gwt')
        if not gwt or not getattr(gwt, 'worktree', None):
            self.ui.show_error('No git worktree available for this tab.')
            return
        ref1, ref2 = ('HEAD~1', 'HEAD')
        is_all = '--all' in arg
        if is_all:
            base = get_branch_base_sha(gwt.worktree)
            if base:
                ref1 = base
            else:
                self.ui.show_error('Could not determine base commit for --all.')
                return
        ref1_short = resolve_ref_short(gwt.worktree, ref1)
        ref2_short = resolve_ref_short(gwt.worktree, ref2)
        if is_all:
            ref1_label = f'base ({ref1_short})' if ref1_short else 'base'
        else:
            ref1_label = f'HEAD~1 ({ref1_short})' if ref1_short else 'HEAD~1'
        ref2_label = f'HEAD ({ref2_short})' if ref2_short else 'HEAD'
        try:
            diff_text = get_diff_between_refs(gwt.worktree, ref1, ref2)
        except GitError as e:
            self.ui.show_error(f'Git diff failed: {e}')
            return
        if not diff_text.strip():
            self.ui.show_command_result('/diff', f'No differences between {ref1_label} and {ref2_label}.')
            return
        self.ui.show_diff(diff_text, ref1_label, ref2_label)

    async def _cmd_errors(self, arg: str) -> None:
        tab = self.active_tab
        if not tab.errors:
            self.ui.show_command_result('/errors', 'No errors recorded.')
        else:
            lines = [f'{ts}  {msg}' for ts, msg in tab.errors[-30:]]
            self.ui.show_command_result('/errors', 'Error log\n' + '\n'.join(lines))

    async def _cmd_tools(self, arg: str) -> None:
        tools = self.active_tab.agent.tools
        if not tools:
            self.ui.show_command_result('/tools', 'No tools enabled.')
        else:
            body = '\n'.join((f'{t.name}  {t.description}' for t in tools))
            self.ui.show_command_result('/tools', f'Active tools ({len(tools)})\n{body}')

    async def _cmd_branches(self, arg: str) -> None:
        lines: list[str] = []
        for i, t in enumerate(self.tabs):
            marker = '►' if i == self.active_index else ' '
            lines.append(f' {marker} {i + 1}. {t.title}')
        self.ui.show_command_result('/branches', '\n'.join(lines))

    async def _cmd_branch(self, arg: str) -> None:
        as_worktree = True
        rev_n: int | None = None
        prompt = arg
        if '--no-worktree' in prompt:
            as_worktree = False
            prompt = prompt.replace('--no-worktree', '').strip()
        rev_match = re.search('--rev\\s+(\\d+)', prompt)
        if rev_match:
            rev_n = int(rev_match.group(1))
            prompt = (prompt[:rev_match.start()] + prompt[rev_match.end():]).strip()
        parent = self.active_tab
        all_msgs = parent.agent.messages
        user_indices = [i for i, m in enumerate(all_msgs) if m.get('role') == 'user']
        if rev_n is not None:
            if rev_n < 1 or rev_n > len(user_indices):
                self.ui.show_error(f'--rev {rev_n}: must be between 1 and {len(user_indices)}' + (' (no user turns yet)' if not user_indices else ''))
                return
            cut_at = user_indices[rev_n - 1]
            sliced_messages = copy.deepcopy(all_msgs[:cut_at])
        else:
            sliced_messages = copy.deepcopy(all_msgs)
        child = parent.agent.branch()
        child.messages = sliced_messages
        index_path, title = _branch_index_title(parent.index_path, self.tabs)
        owns_worktree = False
        gwt = child.ctx.get('gwt')
        if as_worktree:
            repo_dir = gwt.worktree if gwt else child.ctx.get('cwd', os.getcwd())
            ref = 'HEAD'
            if rev_n is not None and gwt:
                target_sha = find_rev_commit(gwt.worktree, rev_n - 1)
                if target_sha:
                    ref = target_sha
                else:
                    self.ui.show_error(f'--rev {rev_n}: no matching commit found; file state will be HEAD')
            new_gwt = create_worktree(repo_dir, prefix=title, ref=ref)
            if new_gwt is None:
                self.ui.show_error(f'git worktree creation failed for {title!r}')
                return
            child.ctx['gwt'] = new_gwt
            child.ctx['cwd'] = new_gwt.worktree
            if 'env' in child.ctx:
                child.ctx['env']['PWD'] = new_gwt.worktree
            owns_worktree = True
        elif gwt:
            try:
                if rev_n is not None:
                    target_sha = find_rev_commit(gwt.worktree, rev_n - 1)
                    if target_sha:
                        new_gwt = git_new_branch_at(gwt.worktree, title, target_sha)
                    else:
                        self.ui.show_error(f'--rev {rev_n}: no matching commit found; file state will be HEAD')
                        new_gwt = git_new_branch(gwt.worktree, title)
                else:
                    new_gwt = git_new_branch(gwt.worktree, title)
                child.ctx['gwt'] = new_gwt
            except GitError as exc:
                logger.warning('git_new_branch failed for tab %r: %s', title, exc)
        new_tab = TabModel(agent=child, title=title, owns_worktree=owns_worktree, index_path=index_path)
        self.tabs.append(new_tab)
        self.attach_agent(new_tab)
        await self.ui.add_tab(new_tab)
        self.switch_tab(len(self.tabs) - 1)
        if prompt:
            await self.handle_input(prompt)

    async def _cmd_tab(self, arg: str) -> None:
        if arg and arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(self.tabs):
                self.switch_tab(idx)
            else:
                self.ui.show_error(f'Invalid tab: {idx + 1}. Available: 1-{len(self.tabs)}')
        else:
            self.ui.show_error('Usage: /tab <n>')

    async def _cmd_verbose(self, arg: str) -> None:
        self.verbose = not self.verbose
        state = 'on' if self.verbose else 'off'
        self.ui.show_command_result('/verbose', f'Verbose mode {state}.')

    async def _cmd_merge(self, arg: str) -> None:
        tab = self.active_tab
        if tab.is_main:
            self.ui.show_error('Cannot /merge the main tab — it has no parent.')
            return
        if tab.is_running:
            self.ui.show_error('Agent is running — /abort first.')
            return
        parent = self._find_parent_tab(tab)
        if parent is None:
            self.ui.show_error(f'Cannot find parent tab for {tab.title!r}.')
            return
        child_gwt = tab.agent.ctx.get('gwt')
        parent_gwt = parent.agent.ctx.get('gwt')
        if child_gwt is None or parent_gwt is None:
            self.ui.show_error('Both tabs need git worktrees to merge.')
            return
        commit_worktree(child_gwt, tab.agent.messages)
        from .gits import merge_branch_into_worktree
        success, msg = merge_branch_into_worktree(parent_gwt, child_gwt)
        if not success:
            self.ui.show_error(f'Merge failed: {msg}')
            return
        new_parent_gwt, diff_stat = commit_worktree(parent_gwt, parent.agent.messages)
        parent.agent.ctx['gwt'] = new_parent_gwt
        self.ui.show_command_result('/merge', f'Merged {tab.title!r} into {parent.title!r}.\n' + (f'{diff_stat}' if diff_stat else '(no changes)'))
        self._do_close_or_exit()
        parent_idx = self.tabs.index(parent)
        self.switch_tab(parent_idx)

    async def _cmd_exit(self, arg: str) -> None:
        if arg == '--all':
            summary = self._build_exit_summary()
            self.exit_summary = summary
            self.ui.exit_app(summary)
        else:
            self.close_or_exit()

    def _get_cwd_env(self, tab: TabModel) -> tuple[str, dict | None]:
        gwt = tab.agent.ctx.get('gwt')
        cwd = gwt.worktree if gwt and getattr(gwt, 'worktree', None) else tab.agent.ctx.get('cwd') or os.getcwd()
        env = tab.agent.ctx.get('env')
        return (cwd, env)

    async def _dispatch_captured_shell(self, command: str) -> None:
        if command.startswith('cd ') or command == 'cd':
            self.ui.show_error('Not allowed — use /branch or set cwd in context')
            return
        cwd, env = self._get_cwd_env(self.active_tab)
        output = await self.ui.run_captured_shell(command, cwd, env)
        self.ui.show_command_result(f'!{command}', output or '(no output)')

    async def _dispatch_interactive_shell(self, command: str) -> None:
        cwd, env = self._get_cwd_env(self.active_tab)
        returncode = await self.ui.run_interactive_shell(command, cwd, env)
        self.ui.show_command_result(f'${command}', f'[exited {returncode}]')

    async def _run_agent(self, tab: TabModel, text: str) -> None:
        if tab.is_running:
            self.ui.show_error('Agent is running — /abort first.')
            return
        tab.running_task = asyncio.create_task(self._do_run(tab, text))

    async def _do_run(self, tab: TabModel, text: str) -> None:
        try:
            await tab.agent.run(text)
        except Exception as exc:
            ts = datetime.datetime.now().strftime('%H:%M:%S')
            tab.errors.append((ts, str(exc)))
            tab.last_error = str(exc)
            tab.status = 'error'
            self.ui.show_error(str(exc))
        finally:
            tab.running_task = None
            if tab.status not in ('error', 'aborting…'):
                tab.status = 'idle'
            self.ui.refresh_chrome()

    def switch_tab(self, idx: int) -> None:
        if not 0 <= idx < len(self.tabs):
            return
        prev = self.tabs[self.active_index]
        next_t = self.tabs[idx]
        prev_gwt = prev.agent.ctx.get('gwt')
        next_gwt = next_t.agent.ctx.get('gwt')
        if prev_gwt and next_gwt and (prev_gwt.worktree == next_gwt.worktree) and (prev_gwt.branch != next_gwt.branch):
            try:
                updated = git_switch_branch(next_gwt.worktree, next_gwt.branch)
                next_t.agent.ctx['gwt'] = updated
            except GitError as exc:
                logger.warning('git switch to %r failed: %s', next_gwt.branch, exc)
                self.ui.show_error(f'git switch failed: {exc}')
                return
        self.active_index = idx
        self.ui.switch_to_tab(idx)

    def close_or_exit(self) -> None:
        tab = self.active_tab
        if tab.is_running:
            tab.agent.abort()
            if tab.running_task:
                tab.running_task.cancel()
        self._do_close_or_exit()

    def _do_close_or_exit(self) -> None:
        if len(self.tabs) > 1:
            tab = self.active_tab
            removed_index = self.active_index
            self.detach_agent(tab)
            gwt_ref = tab.agent.ctx.get('gwt')
            if gwt_ref and getattr(gwt_ref, 'worktree', None):
                try:
                    cleanup_worktree(gwt_ref, remove_branch=True)
                except Exception:
                    logger.error('Failed worktree cleanup')
            self.tabs.pop(removed_index)
            if self.active_index >= len(self.tabs):
                self.active_index = len(self.tabs) - 1
            self.ui.remove_tab(removed_index)
            self.ui.switch_to_tab(self.active_index)
        else:
            summary = self._build_exit_summary()
            self.exit_summary = summary
            self.ui.exit_app(summary)

    def _build_exit_summary(self) -> list[dict]:
        for t in self.tabs:
            if t.is_running:
                t.agent.abort()
                if t.running_task:
                    t.running_task.cancel()
        summary = []
        for t in self.tabs:
            gwt = t.agent.ctx.get('gwt')
            summary.append({'title': t.title, 'branch': gwt.branch if gwt else None, 'worktree': gwt.worktree if gwt else None, 'is_exit_tab': t is self.active_tab})
        return summary

async def _stream_to_stdout(agent: Agent, user_input: str) -> None:
    result = await agent.run(user_input)
    text = ''.join((block.get('text', '') for block in result.get('content', []) if block.get('type') == 'text'))
    if text:
        print(text)

def collect_skills(skills_dir, skills=None):
    skills = [] if skills is None else skills
    if skills_dir is not None:
        root = pathlib.Path(skills_dir)
        if root.exists():
            for md in sorted(root.rglob('SKILL.md')):
                text = md.read_text(encoding='utf-8', errors='replace')
                name, description = (md.parent.name, '')
                if text.startswith('---'):
                    end = text.find('---', 3)
                    if end != -1:
                        fm = text[3:end]
                        n = re.search('^name:\\s*(.+)$', fm, re.MULTILINE)
                        m = re.search('^description:\\s*(.+)$', fm, re.MULTILINE)
                        if not n and (not m):
                            continue
                        if n:
                            name = n.group(1).strip()
                        if m:
                            description = m.group(1).strip()
                        text = text[end + 3:].strip()
                skills.append({'name': name, 'description': description, 'content': text})
    skill_prompt = 'Available skills (use Skill to load full instructions when needed):\n' + '\n'.join((f'- {s['name']}: {s['description']}' for s in skills)) if skills else ''
    return (skills, skill_prompt)
_AGENT_ENV_ALLOWLIST: re.Pattern = re.compile('\n    ^(\n    PATH\n    | MANPATH | INFOPATH\n    | CONDA_PREFIX | CONDA_DEFAULT_ENV | CONDA_EXE | CONDA_PYTHON_EXE\n    | CONDA_SHLVL | CONDA_PROMPT_MODIFIER\n    | MAMBA_ROOT_PREFIX | MAMBA_EXE\n    | VIRTUAL_ENV | VIRTUAL_ENV_PROMPT\n    | PYTHONPATH | PYTHONHOME | PYTHONSTARTUP\n    | PYTHONDONTWRITEBYTECODE | PYTHONUNBUFFERED\n    | PYTHONFAULTHANDLER | PYTHONUTF8\n    | PIP_INDEX_URL | PIP_EXTRA_INDEX_URL\n    | PIPENV_PIPFILE | POETRY_VIRTUALENVS_IN_PROJECT\n    | LD_LIBRARY_PATH | LD_PRELOAD\n    | DYLD_LIBRARY_PATH | DYLD_FALLBACK_LIBRARY_PATH\n    | PKG_CONFIG_PATH\n    | CMAKE_PREFIX_PATH | CMAKE_BUILD_TYPE\n    | CUDA_HOME | CUDA_PATH | CUDA_VISIBLE_DEVICES\n    | NVIDIA_VISIBLE_DEVICES | NVIDIA_DRIVER_CAPABILITIES\n    | HIP_PATH | ROCR_VISIBLE_DEVICES\n    | METAL_DEVICE_WRAPPER_TYPE\n    | LANG | LANGUAGE | LC_ALL | LC_CTYPE | LC_MESSAGES\n    | LC_NUMERIC | LC_TIME | LC_COLLATE\n    | PYTHONUTF8\n    | TERM | TERM_PROGRAM | COLORTERM\n    | NO_COLOR | CLICOLOR | CLICOLOR_FORCE\n    | COLUMNS | LINES\n    | HOME\n    | SHELL\n    | TMPDIR | TEMP | TMP\n    | XDG_RUNTIME_DIR | XDG_CACHE_HOME | XDG_CONFIG_HOME | XDG_DATA_HOME\n    | CC | CXX | AR | LD | FC\n    | CFLAGS | CXXFLAGS | LDFLAGS | MAKEFLAGS\n    | JAVA_HOME | GRADLE_HOME | MAVEN_HOME\n    | GOPATH | GOROOT | GOMODCACHE\n    | CARGO_HOME | RUSTUP_HOME\n    | NODE_PATH\n    | GEM_HOME | GEM_PATH | BUNDLE_PATH\n    )$     ', re.VERBOSE)

def _make_agent_env(base: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in base.items() if _AGENT_ENV_ALLOWLIST.match(k)}

async def repl(engine: CommandEngine, init_prompt=None, bare=False):
    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    if bare and is_tty:
        from .bare import BareRepl
        r = BareRepl(engine, init_prompt=init_prompt)
        await r.run()
        return None
    if not is_tty:
        user_input = (init_prompt if init_prompt is not None else sys.stdin.read()).strip()
        if user_input:
            await _stream_to_stdout(engine.active_tab.agent, user_input)
        return None
    from .tui import ReplApp
    app = ReplApp(engine, init_prompt=init_prompt)
    await app.run_async()
    return app

def run_repl(*, base_url=None, model=None, api: Literal['claude', 'codex', 'gemini', 'deepseek', 'noapi']='noapi', system='', sdir=None, skills=None, env=None, tool_names=None, extra_tool_classes=None, api_key=None, gwt=None, ctx=None, init_prompt=None, resume_messages=None, repo=None, resume=None, stream=None, bare=False):
    repo = os.path.abspath(repo or os.getcwd())
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as _home:
        if gwt is None:
            if resume:
                result = resume_worktree(repo, resume, worktree_dir=os.path.join(_home, 'workspace'))
                if result is None or result[0] is None:
                    print(f'[error] Could not resume from commit {resume!r}. Aborting.')
                    return
                gwt, resume_messages = result
                print(f'[resumed worktree at {gwt.worktree} from commit {resume}]')
            else:
                gwt = create_worktree(repo, worktree_dir=os.path.join(_home, 'workspace'))
        cwd = gwt.worktree if gwt else repo
        if env is None:
            env = os.environ.copy()
        env.setdefault('SHELL', '/bin/bash')
        agent_env = _make_agent_env(env)
        agent_env['HOME'] = _home
        agent_env['PWD'] = cwd
        user_cwd = os.path.abspath(os.getcwd())
        sdir = os.path.abspath(sdir or cwd)
        skills, skill_prompt = collect_skills(sdir, skills)
        system = '\n\n'.join(filter(None, [system, skill_prompt]))
        merged_ctx = {'cwd': cwd, 'user_cwd': user_cwd, 'skills': skills, 'gwt': gwt, 'env': agent_env, **(ctx or {})}
        agent = Agent(system=system, api=api, model=model, tool_names=tool_names, extra_tool_classes=extra_tool_classes, api_key=api_key, base_url=base_url, ctx=merged_ctx)
        engine = CommandEngine()
        main_tab = TabModel(agent, title='main', is_main=True)
        engine.tabs = [main_tab]
        log_fp = None
        if stream is not None:
            from .stream_log import StreamLogger
            log_fp = open(stream, 'w', buffering=1)
            agent.ctx['_stream_log_fp'] = log_fp
            agent.ctx['_stream_log_depth'] = 0
            StreamLogger(agent, log_fp, depth=0, name='base')
            print(f'[streaming log: tail -f {stream}]')
        if resume_messages:
            agent.messages = list(resume_messages)
            print(f'[resumed {len(resume_messages)} messages from checkpoint]')
        try:
            asyncio.run(repl(engine, init_prompt=init_prompt, bare=bare))
        finally:
            if log_fp:
                log_fp.close()
            cleaned: set[str] = set()
            for tab in engine.tabs:
                gwt_ref = tab.agent.ctx.get('gwt')
                if gwt_ref and getattr(gwt_ref, 'worktree', None) and (gwt_ref.worktree not in cleaned):
                    cleaned.add(gwt_ref.worktree)
                    try:
                        cleanup_worktree(gwt_ref)
                    except Exception:
                        pass
            if engine.exit_summary:
                print('\n--- Session Exit Summary ---')
                for item in engine.exit_summary:
                    title = item['title']
                    branch = item['branch'] or '(no branch)'
                    marker = ' * <-- exit origin' if item['is_exit_tab'] else ''
                    print(f'  {title} ({branch}){marker}')

def main():
    import argparse
    from .util import setup_logger
    setup_logger(log_file='.log.json')
    parser = argparse.ArgumentParser(description='mlx-code REPL')
    parser.add_argument('-p', '--prompt', default=None, help='Initial prompt sent automatically when the REPL starts')
    parser.add_argument('-r', '--resume', default=None, metavar='COMMIT', help='Resume a previous session from the given git commit hash')
    parser.add_argument('-a', '--api', choices=['claude', 'codex', 'gemini', 'deepseek', 'noapi'], default='noapi', help='API backend to use')
    parser.add_argument('-m', '--model', default=None, help='Model name or ID')
    parser.add_argument('-t', '--tools', nargs='+', help='Whitelist of tool names to enable')
    parser.add_argument('--url', default='http://127.0.0.1:8000', help='Base URL of the API server')
    parser.add_argument('--system', default='', help='System prompt prepended to every conversation')
    parser.add_argument('--skill', default=None, help='Directory to scan recursively for SKILL.md files')
    parser.add_argument('--cwd', default=None, help='Working directory / git repo root')
    parser.add_argument('--key', default=None, help='API key')
    parser.add_argument('--stream', default=None, help='File to stream log into')
    parser.add_argument('--verbose', action='store_true', help='Show raw tool calls, args, and outputs')
    parser.add_argument('--bare', action='store_true', help='Use simple terminal REPL instead of TUI')
    args = parser.parse_args()
    logger.debug(args)
    url, model, tool_names, api_key = (args.url, args.model, args.tools, args.key)
    if args.api == 'deepseek':
        api_key = os.environ.get('DEEPSEEK_API_KEY') if api_key is None else api_key
        url = 'https://api.deepseek.com' if api_key else url
        model = 'deepseek-v4-flash' if model is None else model
        tool_names = [] if tool_names is None else tool_names
    elif args.api == 'gemini':
        api_key = os.environ.get('GEMINI_API_KEY') if api_key is None else api_key
        url = 'https://generativelanguage.googleapis.com' if api_key else url
        model = 'gemini-3.1-flash-lite' if model is None else model
        tool_names = [] if tool_names is None else tool_names
    run_repl(api=args.api, system=args.system, repo=args.cwd, model=model, base_url=url, tool_names=tool_names, sdir=args.skill, api_key=api_key, init_prompt=args.prompt, resume=args.resume, stream=args.stream, bare=args.bare)
if __name__ == '__main__':
    main()
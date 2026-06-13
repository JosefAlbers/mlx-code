from __future__ import annotations
import asyncio
import subprocess
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
from typing import Any, Callable, Literal
from .gits import create_worktree, commit_worktree, resume_worktree, cleanup_worktree, git_new_branch, git_new_branch_at, git_switch_branch, GitError, get_commit_history_with_stats, find_rev_commit, get_diff_between_refs, get_branch_base_sha, resolve_ref_short
from .tools import Tool, validate_tool_call, DEFAULT_TOOLS
from .apis import resolve_api
logger = logging.getLogger(__name__)
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll, Vertical
from textual import events
from textual.message import Message
from textual.widgets import ContentSwitcher, Static, TextArea
from rich.text import Text as RichText
from rich.table import Table
from rich.markup import escape
from rich.markdown import Markdown
from rich.cells import cell_len
from rich.syntax import Syntax

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
            new_gwt, diff_stat = commit_worktree(self.ctx['gwt'], self.messages)
            self.ctx['gwt'] = new_gwt
            if diff_stat:
                sha = new_gwt.commit[:8] if new_gwt else ''
                self.messages.append({'role': 'commit', 'content': f'[{sha}]\n{diff_stat}', 'sha': sha})
                await self._emit({'type': 'commit', 'payload': {'diff_stat': diff_stat, 'sha': sha}})
            if self._signal and self._signal.is_set():
                final['stop_reason'] = 'aborted'
                break
        await self._emit({'type': 'agent_end', 'payload': {'message': final}})
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

def _branch_index_title(parent_path: tuple[int, ...], existing_tabs: list) -> tuple[tuple[int, ...], str]:
    depth = len(parent_path) + 1
    child_count = sum((1 for t in existing_tabs if len(t.index_path) == depth and t.index_path[:-1] == parent_path))
    index_path = parent_path + (child_count,)
    title = 'branch-' + '-'.join((str(i) for i in index_path))
    return (index_path, title)

def _clean_block_text(text: str) -> str:
    return text.lstrip('\n').rstrip()

def _make_empty_history_table() -> Table:
    tbl = Table(show_header=False, show_lines=False, box=None, pad_edge=False, expand=True, padding=0)
    tbl.add_column(width=2)
    tbl.add_column(ratio=1)
    return tbl

def append_to_history_table(tbl: Table, messages: list[dict]) -> None:
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content')
        is_error = msg.get('is_error', False)
        if isinstance(content, (Table, RichText, Syntax)):
            blocks = [{'type': 'renderable', 'renderable': content}]
        elif isinstance(content, str):
            blocks = [{'type': 'text', 'text': content}]
        elif isinstance(content, list):
            blocks = content
        else:
            blocks = [{'type': 'text', 'text': str(content)}]
        for block in blocks:
            b_type = block.get('type', 'text')
            text = block.get('text', '') or block.get('thinking', '')
            prefix, style = ('', '')
            render_as_md = False
            if is_error:
                prefix, style = ('✗', 'bold red')
            elif role == 'commit':
                prefix, style = ('◇', 'cyan')
            elif role == 'user':
                prefix, style = ('≫', 'bold green')
            elif role == 'system':
                prefix, style = ('·', 'dim')
            elif role == 'command':
                prefix, style = ('✓', 'blue')
            elif b_type == 'thinking':
                prefix, style = ('◌', 'dim italic')
            elif role == 'toolResult':
                prefix, style = ('→', 'dim yellow')
            elif b_type == 'toolCall':
                prefix, style = ('⚙', 'yellow')
                args = block.get('arguments', {})
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                text = block.get('name', '') + ' ' + str(args)
            elif role == 'assistant':
                prefix, style = ('○', '')
                text = re.sub('\\s*<tool_call>.*?</tool_call>\\s*', '', text, flags=re.DOTALL).strip()
                render_as_md = True
            text = _clean_block_text(text)
            if b_type == 'renderable':
                prefix, style = ('✓', 'blue')
                body = block.get('renderable')
                tbl.add_row(RichText(prefix, style=style), body)
                continue
            if not text:
                continue
            if role == 'commit':
                body = Syntax(text, lexer='diff', theme='monokai')
                tbl.add_row(RichText('◇', style='cyan'), body)
                continue
            if render_as_md:
                body = Markdown(text)
            else:
                body = RichText(text, style=style)
            tbl.add_row(RichText(prefix, style=style), body)

def render_history(messages: list[dict]) -> Table:
    tbl = _make_empty_history_table()
    append_to_history_table(tbl, messages)
    return tbl
REPL_HELP = '\nCommands:\n/help               show this message\n/clear [--config F] clear conversation; --config reconfigures agent from YAML/JSON\n/history            show full conversation transcript\n/history --raw      show raw API message log (debug)\n/diff [--all]       show side-by-side diff of changes\n/errors             show timestamped error log for this tab\n/tools              list active tools\n/branch [--rev N] [--no-worktree] [prompt]\n                    open a branch tab; optional prompt runs immediately\n/abort              abort the running agent\n/export [path]      export session to JSON\n/exit /quit [--all] close branch tab, or exit the app\n!command            run shell command in worktree (output captured in TUI)\n$command            run interactive shell command (TUI suspends, terminal handed to process)\n                    e.g.  !ls  !git diff  !cat file.py\n                          $vim file.py  $yazi  $less log.txt\nKeys:\nEnter               submit\nCtrl-J              insert newline in editor\nCtrl-1 … Ctrl-9     jump directly to tab N\nCtrl-, / Ctrl-.     cycle through tabs\nCtrl-C              abort running agent\nCtrl-D              close branch tab (exit app if last tab) \nCtrl-R              recall last prompt into editor\n'

class InputBox(TextArea):
    BINDINGS = [Binding('ctrl+j', 'insert_newline', 'New line', show=False), Binding('enter', 'submit_text', 'Submit', show=False, priority=True), Binding('ctrl+r', 'recall_last', 'Recall', show=False), Binding('ctrl+d', 'request_close', 'Exit', show=False, priority=True)]

    class Submit(Message):

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class RecallLast(Message):
        pass

    class CloseRequest(Message):
        pass

    def action_submit_text(self) -> None:
        text = self.text
        if text.strip():
            self.post_message(self.Submit(text))
            self.load_text('')

    def action_insert_newline(self) -> None:
        self.insert('\n')

    def action_recall_last(self) -> None:
        self.post_message(self.RecallLast())

    def action_request_close(self) -> None:
        self.post_message(self.CloseRequest())

    def set_text_and_end(self, text: str) -> None:
        self.load_text(text)
        lines = text.split('\n')
        self.move_cursor((len(lines) - 1, len(lines[-1])))

class Tab(Vertical):
    CSS = '\n    Tab > VerticalScroll {\n        height: 1fr;\n    }\n    #cache, #stream {\n        height: auto;\n        padding: 0 1;\n    }\n    #stream { min-height: 1; }\n    '

    def __init__(self, title: str, agent: Agent, is_main: bool=False, owns_worktree: bool=False, index_path: tuple[int, ...]=()):
        super().__init__(id=f'tab-{id(agent):x}')
        self.title = title
        self.agent = agent
        self.is_main = is_main
        self.owns_worktree = owns_worktree
        self.index_path = index_path
        self.status: str = 'idle'
        self.running_task: asyncio.Task | None = None
        self.errors: list[tuple[str, str]] = []
        self.last_error: str = ''
        self._stream_blocks: list[dict] = []
        self._command_blocks: list[dict] = []
        self._cache_count: int = -1
        self._rendered_cache: Table | None = None

    @property
    def is_running(self) -> bool:
        return self.running_task is not None and (not self.running_task.done())

    def compose(self) -> ComposeResult:
        with VerticalScroll(id='scroll'):
            yield Static(id='cache')
            yield Static(id='stream')

    def apply_event(self, event: dict) -> None:
        et = event.get('type')
        payload = event.get('payload', {})
        if et == 'agent_start':
            self.status = 'running'
            self._stream_blocks = []
            self._command_blocks = []
            self.refresh_cache()
            self.refresh_stream()
        elif et == 'turn_start':
            self._stream_blocks = []
            self._command_blocks = []
            self.refresh_stream()
        elif et == 'text_delta':
            delta = payload.get('delta', '')
            if delta:
                if self._stream_blocks and self._stream_blocks[-1].get('type') == 'text' and (not self._stream_blocks[-1].get('is_error')):
                    self._stream_blocks[-1]['text'] += delta
                else:
                    self._stream_blocks.append({'type': 'text', 'text': delta})
            self.refresh_stream()
        elif et == 'thinking_delta':
            delta = payload.get('delta', '')
            if delta:
                if self._stream_blocks and self._stream_blocks[-1].get('type') == 'thinking':
                    self._stream_blocks[-1]['text'] += delta
                else:
                    self._stream_blocks.append({'type': 'thinking', 'text': delta})
            self.refresh_stream()
        elif et == 'tool_start':
            self._stream_blocks.append({'type': 'toolCall', 'name': payload.get('name', 'tool'), 'arguments': payload.get('args', {}), 'id': 'streaming'})
            self.refresh_stream()
        elif et == 'tool_end':
            if payload.get('is_error'):
                self._stream_blocks.append({'type': 'text', 'text': f'{payload.get('name', 'tool')} failed', 'is_error': True})
                self.refresh_stream()
        elif et == 'tool_results_ready':
            self.refresh_cache()
            self._stream_blocks = []
            self.refresh_stream()
        elif et == 'commit':
            self.refresh_cache()
        elif et == 'error':
            err = str(payload.get('error', payload))
            self._stream_blocks.append({'type': 'text', 'text': err, 'is_error': True})
            ts = datetime.datetime.now().strftime('%H:%M:%S')
            self.errors.append((ts, err))
            self.last_error = err
            self.refresh_stream()
        elif et == 'turn_end':
            self._stream_blocks = []
            self.refresh_stream()
            self.refresh_cache()
        elif et == 'agent_end':
            self._stream_blocks = []
            self.refresh_stream()
            self.refresh_cache()
            self.status = 'idle'

    def refresh_cache(self) -> None:
        new_count = len(self.agent.messages)
        if new_count == self._cache_count:
            return
        if self._rendered_cache is None or new_count < self._cache_count:
            self._rendered_cache = render_history(self.agent.messages)
        else:
            new_msgs = self.agent.messages[self._cache_count:]
            append_to_history_table(self._rendered_cache, new_msgs)
        self._cache_count = new_count
        self.query_one('#cache', Static).update(self._rendered_cache)
        scroll = self.query_one('#scroll', VerticalScroll)
        if scroll.max_scroll_y - scroll.scroll_y < 3:
            self.app.call_after_refresh(scroll.scroll_end, animate=False)

    def refresh_stream(self) -> None:
        msgs = []
        if self._stream_blocks:
            msgs.append({'role': 'assistant', 'content': self._stream_blocks})
        if not msgs:
            msgs.extend(self._command_blocks)
        if msgs:
            self.query_one('#stream', Static).update(render_history(msgs))
            scroll = self.query_one('#scroll', VerticalScroll)
            if scroll.max_scroll_y - scroll.scroll_y < 3:
                self.app.call_after_refresh(scroll.scroll_end, animate=False)
        else:
            self.query_one('#stream', Static).update('')

    def show_command(self, cmd, text: str) -> None:
        self._command_blocks.extend([{'role': 'user', 'content': cmd}, {'role': 'command', 'content': text}])
        self.refresh_stream()

    def clear_log(self) -> None:
        self.query_one('#cache', Static).update('')
        self.query_one('#stream', Static).update('')
        self._cache_count = 0
        self._rendered_cache = None
        self._command_blocks = []
        self._stream_blocks = []

class TabBar(Static):

    class SwitchTo(Message):

        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def __init__(self, **kwargs):
        super().__init__('', **kwargs)
        self._ranges: list[tuple[int, int, int]] = []

    def render_tabs(self, tabs: list[Tab], active_index: int) -> None:
        t = RichText(no_wrap=True, overflow='ellipsis')
        x = 0
        self._ranges = []
        for i, tab in enumerate(tabs):
            marker = '●' if tab.is_running else ' '
            label = f' {marker}{i + 1}:{tab.title} '
            width = cell_len(label)
            self._ranges.append((x, x + width, i))
            x += width
            if i == active_index:
                t.append(label, style='bold green reverse')
            elif tab.is_running:
                t.append(label, style='yellow')
            else:
                t.append(label, style='dim')
        self.update(t)

    def on_click(self, event: events.Click) -> None:
        for start, end, idx in self._ranges:
            if start <= event.x < end:
                self.post_message(self.SwitchTo(idx))
                return

class StatusBar(Static):

    def render_status(self, tab: Tab, model: str | None) -> None:
        t = RichText(no_wrap=True, overflow='ellipsis')
        t.append(f' {tab.title}', style='bold')
        t.append('  ')
        if tab.is_running:
            t.append('running …', style='yellow')
        elif tab.status not in {'idle', ''}:
            t.append(tab.status, style='dim')
        else:
            t.append('idle', style='dim')
        real_count = sum((1 for m in tab.agent.messages if m.get('role') == 'user'))
        if real_count:
            t.append(f'  turn {real_count}', style='dim')
        if model:
            t.append(f'  {model}', style='dim')
        self.update(t)

class HelpBar(Static):

    def __init__(self, **kwargs):
        self._idle_text = RichText('/help  !cmd  $interactive  Ctrl-J newline  Ctrl-,. tabs  Ctrl-C abort  Ctrl-D exit', style='dim')
        super().__init__(self._idle_text, **kwargs)

    def show_idle(self) -> None:
        self.update(self._idle_text)

    def show_error(self, msg: str) -> None:
        t = RichText()
        t.append('✗ ', style='bold red')
        t.append(escape(msg), style='red')
        self.update(t)

    def show_confirm(self, msg: str) -> None:
        t = RichText()
        t.append('⚠ ', style='bold yellow')
        t.append(escape(msg), style='yellow')
        self.update(t)

class ReplApp(App[None]):
    CSS = '\n    ReplApp { layout: vertical; background: $background; }\n    ContentSwitcher { height: 1fr; }\n    InputBox {\n        height: auto;\n        min-height: 3;\n        max-height: 8;\n        border: none;\n        border-top: tall $panel-darken-1;\n        background: $panel;\n        padding: 0 2;\n        color: $text;\n    }\n    InputBox:focus { border-top: tall $accent; }\n    TabBar { height: 1; background: $panel; padding: 0 1; }\n    StatusBar { height: 1; background: $panel; color: $text-muted; padding: 0 1; }\n    HelpBar { height: 1; color: $text-muted; padding: 0 1; }\n    '
    BINDINGS = [Binding('ctrl+c', 'abort_agent', 'Abort', priority=True, show=False), Binding('ctrl+d', 'close_or_exit', 'Exit', show=False), Binding('ctrl+full_stop', 'next_tab', 'Next tab', show=False, priority=True), Binding('ctrl+comma', 'prev_tab', 'Prev tab', show=False, priority=True), Binding('ctrl+1', 'switch_tab(1)', 'Tab 1', show=False), Binding('ctrl+2', 'switch_tab(2)', 'Tab 2', show=False), Binding('ctrl+3', 'switch_tab(3)', 'Tab 3', show=False), Binding('ctrl+4', 'switch_tab(4)', 'Tab 4', show=False), Binding('ctrl+5', 'switch_tab(5)', 'Tab 5', show=False), Binding('ctrl+6', 'switch_tab(6)', 'Tab 6', show=False), Binding('ctrl+7', 'switch_tab(7)', 'Tab 7', show=False), Binding('ctrl+8', 'switch_tab(8)', 'Tab 8', show=False), Binding('ctrl+9', 'switch_tab(9)', 'Tab 9', show=False), Binding('ctrl+z', 'suspend_app', 'Suspend', show=False, priority=True)]

    def __init__(self, agent: Agent, init_prompt: str | None=None) -> None:
        super().__init__()
        self.tabs: list[Tab] = [Tab('main', agent, is_main=True)]
        self.active_index = 0
        self._unsubscribers: dict[int, Callable] = {}
        self._pending_init = init_prompt.strip() if init_prompt else None
        self._confirm_close = False
        self._exit_summary: list[dict] | None = None
        self._attach_agent(self.tabs[0])

    def compose(self) -> ComposeResult:
        yield TabBar(id='tabbar')
        yield ContentSwitcher(self.tabs[0], initial=self.tabs[0].id)
        yield InputBox()
        yield StatusBar(id='statusbar')
        yield HelpBar(id='helpbar')

    def on_mount(self) -> None:
        self._refresh_chrome()
        self._switch_to(0)
        if self._pending_init:
            self.set_timer(0.05, self._run_pending_init)

    def _run_pending_init(self) -> None:
        if self._pending_init:
            prompt, self._pending_init = (self._pending_init, None)
            asyncio.create_task(self._run_submit(prompt))

    def _refresh_chrome(self) -> None:
        try:
            self.query_one('#tabbar', TabBar).render_tabs(self.tabs, self.active_index)
            self.query_one('#statusbar', StatusBar).render_status(self.active_tab, self.active_tab.agent.model)
        except Exception:
            pass

    def action_suspend_app(self) -> None:
        import signal
        with self.suspend():
            os.kill(os.getpid(), signal.SIGTSTP)

    @property
    def active_tab(self) -> Tab:
        return self.tabs[self.active_index]

    def _attach_agent(self, tab: Tab) -> None:
        key = id(tab.agent)
        if key in self._unsubscribers:
            return

        async def on_event(event: dict) -> None:
            tab.apply_event(event)
            if event['type'] in ('agent_start', 'agent_end', 'turn_end', 'commit'):
                self._refresh_chrome()
        self._unsubscribers[key] = tab.agent.subscribe(on_event)

    def on_input_box_submit(self, message: InputBox.Submit) -> None:
        asyncio.create_task(self._run_submit(message.text))

    def on_input_box_recall_last(self, _msg: InputBox.RecallLast) -> None:
        tab = self.active_tab
        real = [m for m in tab.agent.messages if m.get('role') == 'user']
        if real:
            self.query_one(InputBox).set_text_and_end(real[-1].get('content', ''))

    def on_input_box_close_request(self, _msg: InputBox.CloseRequest) -> None:
        self._do_close_or_exit()

    async def _run_submit(self, raw: str) -> None:
        text = raw.strip()
        if not text:
            return
        self._confirm_close = False
        tab = self.active_tab
        tab.errors.clear()
        tab.last_error = ''
        self.query_one('#helpbar', HelpBar).show_idle()
        if text.startswith('$'):
            command = text[1:].strip()
            if command:
                await self._run_interactive_command(tab, command)
            return
        if text.startswith('!'):
            command = text[1:].strip()
            if not command:
                return
            await self._run_shell_command(tab, command)
            return
        if text.startswith('/'):
            await self._handle_command(tab, text)
            return
        if tab.is_running:
            self.query_one('#helpbar', HelpBar).show_error('Agent is running — /abort first.')
            return
        tab.running_task = asyncio.create_task(self._run_agent(tab, text))

    async def _run_agent(self, tab: Tab, text: str) -> None:
        try:
            await tab.agent.run(text)
        except Exception as exc:
            tab.errors.append((datetime.datetime.now().strftime('%H:%M:%S'), str(exc)))
            tab.last_error = str(exc)
            tab.status = 'error'
            self.query_one('#helpbar', HelpBar).show_error(str(exc))
        finally:
            tab.running_task = None
            if not tab.last_error:
                tab.status = 'idle'
            self._refresh_chrome()

    async def _run_shell_command(self, tab: Tab, command: str) -> None:
        if not command:
            return
        if command.startswith('cd ') or command == 'cd':
            tab.show_command(f'!{command}', 'Not allowed — use /branch or set cwd in context')
            return
        gwt = tab.agent.ctx.get('gwt')
        cwd = gwt.worktree if gwt and getattr(gwt, 'worktree', None) else tab.agent.ctx.get('cwd') or os.getcwd()
        env = tab.agent.ctx.get('env')
        try:
            proc = await asyncio.create_subprocess_shell(command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env if env else None)
            stdout, stderr = await proc.communicate()
            out = stdout.decode(errors='replace').rstrip('\n')
            err = stderr.decode(errors='replace').rstrip('\n')
            body = out
            if err:
                body = (body + '\n' if body else '') + f'[stderr]\n{err}'
            if proc.returncode and proc.returncode != 0:
                body += f'\n[exit {proc.returncode}]'
            tab.show_command(f'!{command}', body or '(no output)')
        except Exception as e:
            tab.show_command(f'!{command}', f'Error: {e}')

    async def _run_interactive_command(self, tab: Tab, command: str) -> None:
        gwt = tab.agent.ctx.get('gwt')
        cwd = gwt.worktree if gwt and getattr(gwt, 'worktree', None) else tab.agent.ctx.get('cwd') or os.getcwd()
        env = tab.agent.ctx.get('env')

        def _blocking_run() -> int:
            return subprocess.run(command, shell=True, cwd=cwd, env={**os.environ, **env}).returncode
        with self.suspend():
            loop = asyncio.get_running_loop()
            returncode = await loop.run_in_executor(None, _blocking_run)
        tab.show_command(f'${command}', f'[exited {returncode}]')

    async def _handle_command(self, tab: Tab, text: str) -> None:
        cmd, _, arg = text.partition(' ')
        cmd = cmd.lower().strip()
        arg = arg.strip()
        if cmd == '/help':
            tab.show_command(text, REPL_HELP)
        elif cmd == '/clear':
            cfg_match = re.match('--config\\s+(.+)', arg)
            if cfg_match:
                config_path = cfg_match.group(1).strip().strip('"').strip("'")
                try:
                    cfg = load_agent_config(config_path)
                except Exception as e:
                    self.query_one('#helpbar', HelpBar).show_error(f'Config error: {e}')
                    return
                old = tab.agent
                new_ctx = {k: v for k, v in old.ctx.items() if k != 'agent'}
                new_agent = Agent(system=cfg.get('system'), api=cfg.get('api'), model=cfg.get('model'), api_key=cfg.get('api_key'), base_url=cfg.get('base_url'), tool_names=cfg.get('tools'), extra_tool_classes=old._extra_tool_classes, ctx=new_ctx)
                old_key = id(old)
                if old_key in self._unsubscribers:
                    self._unsubscribers[old_key]()
                    del self._unsubscribers[old_key]
                tab.agent = new_agent
                self._attach_agent(tab)
                tab.clear_log()
                self._refresh_chrome()
            else:
                tab.agent.messages.clear()
                tab.clear_log()
                self._refresh_chrome()
        elif cmd == '/history':
            if arg == '--raw':
                raw_json = json.dumps(tab.agent.messages, indent=2, default=str)
                tab.show_command(text, raw_json)
            else:
                user_msgs = [m for m in tab.agent.messages if m.get('role') == 'user']
                if not user_msgs:
                    tab.show_command(text, 'No prompts yet.')
                else:
                    lines = []
                    gwt = tab.agent.ctx.get('gwt')
                    commit_stats = []
                    if gwt and getattr(gwt, 'worktree', None):
                        try:
                            commit_stats = get_commit_history_with_stats(gwt.worktree, limit=len(user_msgs) + 5)
                        except Exception as e:
                            logger.warning(f'Failed to get commit stats: {e}')
                    turn_commits = {}
                    for c in commit_stats:
                        turn = c.get('user_turns', 0)
                        if turn > 0:
                            if turn not in turn_commits:
                                turn_commits[turn] = []
                            turn_commits[turn].append(c)
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
                                seen = set()
                                unique_files = [f for f in c['files'] if not (f in seen or seen.add(f))]
                                if unique_files:
                                    file_lines = '\n     '.join(unique_files[:8])
                                    if len(unique_files) > 8:
                                        file_lines += f'\n     ... and {len(unique_files) - 8} more'
                                    line += f'\n     {file_lines}'
                        lines.append(line)
                    tab.show_command(text, '\n'.join(lines))
        elif cmd == '/diff':
            gwt = tab.agent.ctx.get('gwt')
            if not gwt or not getattr(gwt, 'worktree', None):
                self.query_one('#helpbar', HelpBar).show_error('No git worktree available for this tab.')
                return
            ref1 = 'HEAD~1'
            ref2 = 'HEAD'
            is_all = '--all' in arg
            if is_all:
                base = get_branch_base_sha(gwt.worktree)
                if base:
                    ref1 = base
                else:
                    self.query_one('#helpbar', HelpBar).show_error('Could not determine base commit for --all.')
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
                self.query_one('#helpbar', HelpBar).show_error(f'Git diff failed: {e}')
                return
            if not diff_text.strip():
                tab.show_command(text, f'No differences between {ref1_label} and {ref2_label}.')
                return
            renderable = Syntax(diff_text, lexer='diff', theme='monokai')
            tab._command_blocks.extend([{'role': 'user', 'content': text}, {'role': 'command', 'content': renderable}])
            tab.refresh_stream()
        elif cmd == '/errors':
            if not tab.errors:
                tab.show_command(text, 'No errors recorded.')
            else:
                lines = [f'{ts}  {msg}' for ts, msg in tab.errors[-30:]]
                tab.show_command(text, 'Error log\n' + '\n'.join(lines))
        elif cmd == '/tools':
            tools = tab.agent.tools
            if not tools:
                tab.show_command(text, 'No tools enabled.')
            else:
                body = '\n'.join((f'{t.name}  {t.description}' for t in tools))
                tab.show_command(text, f'Active tools ({len(tools)})\n{body}')
        elif cmd == '/abort':
            if tab.is_running:
                tab.agent.abort()
                tab.status = 'aborting…'
                self._refresh_chrome()
            else:
                self.query_one('#helpbar', HelpBar).show_error('Nothing is running.')
        elif cmd == '/branch':
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
                    self.query_one('#helpbar', HelpBar).show_error(f'--rev {rev_n}: must be between 1 and {len(user_indices)}' + (' (no user turns yet)' if not user_indices else ''))
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
                        self.query_one('#helpbar', HelpBar).show_error(f'--rev {rev_n}: no matching commit found; file state will be HEAD')
                new_gwt = create_worktree(repo_dir, prefix=title, ref=ref)
                if new_gwt is None:
                    self.query_one('#helpbar', HelpBar).show_error(f'git worktree creation failed for {title!r}')
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
                            self.query_one('#helpbar', HelpBar).show_error(f'--rev {rev_n}: no matching commit found; file state will be HEAD')
                            new_gwt = git_new_branch(gwt.worktree, title)
                    else:
                        new_gwt = git_new_branch(gwt.worktree, title)
                    child.ctx['gwt'] = new_gwt
                except GitError as exc:
                    logger.warning('git_new_branch failed for tab %r: %s', title, exc)
            new_tab = Tab(title, child, owns_worktree=owns_worktree, index_path=index_path)
            self.tabs.append(new_tab)
            self._attach_agent(new_tab)
            switcher = self.query_one(ContentSwitcher)
            await switcher.mount(new_tab)
            self._switch_to(len(self.tabs) - 1)
            if prompt:
                await self._run_submit(prompt)
        elif cmd == '/tab':
            if arg and arg.isdigit():
                self.action_switch_tab(int(arg))
            else:
                self.query_one('#helpbar', HelpBar).show_error('Usage: /tab <n>')
        elif cmd == '/export':
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            user_cwd = tab.agent.ctx.get('user_cwd', os.getcwd())
            if arg:
                path = arg if os.path.isabs(arg) else os.path.join(user_cwd, arg)
            else:
                path = os.path.join(user_cwd, f'session_{ts}.json')
            data = {'version': 1, 'exported_at': ts, 'system': tab.agent.system, 'messages': tab.agent.messages}
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                tab.status = f'exported → {path}'
                self._refresh_chrome()
            except OSError as exc:
                self.query_one('#helpbar', HelpBar).show_error(f'Export failed: {exc}')
        elif cmd in {'/exit', '/quit'}:
            if arg == '--all':
                self._exit_with_summary(tab)
            else:
                self._do_close_or_exit()
        else:
            self.query_one('#helpbar', HelpBar).show_error(f'Unknown command: {cmd!r}  — try /help')

    def _exit_with_summary(self, exit_tab: Tab) -> None:
        for t in self.tabs:
            if t.is_running:
                t.agent.abort()
                if t.running_task:
                    t.running_task.cancel()
        summary = []
        for t in self.tabs:
            gwt = t.agent.ctx.get('gwt')
            summary.append({'title': t.title, 'branch': gwt.branch if gwt else None, 'worktree': gwt.worktree if gwt else None, 'is_exit_tab': t is exit_tab})
        self._exit_summary = summary
        self.exit()

    def _switch_to(self, idx: int) -> None:
        if not 0 <= idx < len(self.tabs):
            return
        prev_tab = self.tabs[self.active_index] if self.tabs else None
        next_tab = self.tabs[idx]
        if prev_tab is not None and prev_tab is not next_tab:
            prev_gwt = prev_tab.agent.ctx.get('gwt')
            next_gwt = next_tab.agent.ctx.get('gwt')
            if prev_gwt and next_gwt and (prev_gwt.worktree == next_gwt.worktree) and (prev_gwt.branch != next_gwt.branch):
                try:
                    updated = git_switch_branch(next_gwt.worktree, next_gwt.branch)
                    next_tab.agent.ctx['gwt'] = updated
                except GitError as exc:
                    logger.warning('git switch to %r failed: %s', next_gwt.branch, exc)
                    self.query_one('#helpbar', HelpBar).show_error(f'git switch failed: {exc}')
        self.active_index = idx
        self._confirm_close = False
        try:
            self.query_one(ContentSwitcher).current = self.tabs[idx].id
        except Exception:
            pass
        tab = self.tabs[idx]
        tab.refresh_cache()
        self.query_one(InputBox).focus()
        self._refresh_chrome()
        self.query_one('#helpbar', HelpBar).show_idle()

    def on_tab_bar_switch_to(self, msg: TabBar.SwitchTo) -> None:
        self._switch_to(msg.index)

    def action_next_tab(self) -> None:
        self._switch_to((self.active_index + 1) % len(self.tabs))

    def action_prev_tab(self) -> None:
        self._switch_to((self.active_index - 1) % len(self.tabs))

    def action_switch_tab(self, n: int) -> None:
        self._switch_to(n - 1)

    def action_close_or_exit(self) -> None:
        self._do_close_or_exit()

    def _do_close_or_exit(self) -> None:
        tab = self.active_tab
        if tab.is_running:
            if not self._confirm_close:
                self._confirm_close = True
                self.query_one('#helpbar', HelpBar).show_confirm('Agent is running — press Ctrl-D again to force close.')
                return
            tab.agent.abort()
            if tab.running_task:
                tab.running_task.cancel()
        self._confirm_close = False
        if len(self.tabs) > 1:
            key = id(tab.agent)
            if key in self._unsubscribers:
                self._unsubscribers[key]()
                del self._unsubscribers[key]
            tab.remove()
            self.tabs.pop(self.active_index)
            if self.active_index >= len(self.tabs):
                self.active_index = len(self.tabs) - 1
            self._switch_to(self.active_index)
        else:
            self._exit_with_summary(tab)

    def action_abort_agent(self) -> None:
        input_box = self.query_one(InputBox)
        if input_box.text:
            input_box.load_text('')
            input_box.focus()
            return
        tab = self.active_tab
        if tab.is_running:
            tab.agent.abort()
            tab.status = 'aborting…'
            self._refresh_chrome()

async def _stream_to_stdout(agent: Agent, user_input: str) -> None:
    result = await agent.run(user_input)
    text = ''.join((block.get('text', '') for block in result.get('content', []) if block.get('type') == 'text'))
    if text:
        print(text)

async def repl(agent, init_prompt=None, notui=False):
    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    if notui and is_tty:
        from .ntui import SimpleRepl
        sr = SimpleRepl(agent, init_prompt=init_prompt)
        await sr.run()
        return None
    if not is_tty:
        user_input = (init_prompt if init_prompt is not None else sys.stdin.read()).strip()
        if user_input:
            await _stream_to_stdout(agent, user_input)
        return None
    app = ReplApp(agent, init_prompt=init_prompt)
    await app.run_async()
    return app

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
_AGENT_ENV_ALLOWLIST: re.Pattern = re.compile('\n    ^(\n    # ── Execution paths ────────────────────────────────────────────────────\n    PATH\n    | MANPATH | INFOPATH\n\n    # ── Python / conda / virtualenv ────────────────────────────────────────\n    | CONDA_PREFIX | CONDA_DEFAULT_ENV | CONDA_EXE | CONDA_PYTHON_EXE\n    | CONDA_SHLVL | CONDA_PROMPT_MODIFIER\n    | MAMBA_ROOT_PREFIX | MAMBA_EXE\n    | VIRTUAL_ENV | VIRTUAL_ENV_PROMPT\n    | PYTHONPATH | PYTHONHOME | PYTHONSTARTUP\n    | PYTHONDONTWRITEBYTECODE | PYTHONUNBUFFERED\n    | PYTHONFAULTHANDLER | PYTHONUTF8\n    | PIP_INDEX_URL | PIP_EXTRA_INDEX_URL   # private PyPI mirrors (no auth tokens)\n    | PIPENV_PIPFILE | POETRY_VIRTUALENVS_IN_PROJECT\n\n    # ── Native / compiled libs ─────────────────────────────────────────────\n    | LD_LIBRARY_PATH | LD_PRELOAD\n    | DYLD_LIBRARY_PATH | DYLD_FALLBACK_LIBRARY_PATH   # macOS\n    | PKG_CONFIG_PATH\n    | CMAKE_PREFIX_PATH | CMAKE_BUILD_TYPE\n\n    # ── CUDA / GPU ─────────────────────────────────────────────────────────\n    | CUDA_HOME | CUDA_PATH | CUDA_VISIBLE_DEVICES\n    | NVIDIA_VISIBLE_DEVICES | NVIDIA_DRIVER_CAPABILITIES\n    | HIP_PATH | ROCR_VISIBLE_DEVICES           # AMD\n    | METAL_DEVICE_WRAPPER_TYPE                 # Apple\n\n    # ── Locale / encoding ──────────────────────────────────────────────────\n    | LANG | LANGUAGE | LC_ALL | LC_CTYPE | LC_MESSAGES\n    | LC_NUMERIC | LC_TIME | LC_COLLATE\n    | PYTHONUTF8\n\n    # ── Terminal (needed by tools that check if output is a tty) ───────────\n    | TERM | TERM_PROGRAM | COLORTERM\n    | NO_COLOR | CLICOLOR | CLICOLOR_FORCE\n    | COLUMNS | LINES\n\n    # ── Process identity (non-secret) ──────────────────────────────────────\n    | HOME          # overridden to temp dir by run_repl — safe\n    | SHELL\n    | TMPDIR | TEMP | TMP\n    | XDG_RUNTIME_DIR | XDG_CACHE_HOME | XDG_CONFIG_HOME | XDG_DATA_HOME\n\n    # ── Toolchain / build (no secrets) ─────────────────────────────────────\n    | CC | CXX | AR | LD | FC\n    | CFLAGS | CXXFLAGS | LDFLAGS | MAKEFLAGS\n    | JAVA_HOME | GRADLE_HOME | MAVEN_HOME\n    | GOPATH | GOROOT | GOMODCACHE\n    | CARGO_HOME | RUSTUP_HOME\n    | NODE_PATH                                 # not NODE_AUTH_TOKEN\n    | GEM_HOME | GEM_PATH | BUNDLE_PATH\n    )$\n', re.VERBOSE)

def _make_agent_env(base: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in base.items() if _AGENT_ENV_ALLOWLIST.match(k)}

def run_repl(*, base_url=None, model=None, api: Literal['claude', 'codex', 'gemini', 'deepseek', 'noapi']='noapi', system='', sdir=None, skills=None, env=None, tool_names=None, extra_tool_classes=None, api_key=None, gwt=None, ctx=None, init_prompt=None, resume_messages=None, repo=None, resume=None, stream=None, verbose_transcript=False, notui=False):
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
        app_instance = None
        try:
            app_instance = asyncio.run(repl(agent, init_prompt=init_prompt, notui=notui))
        finally:
            if log_fp:
                log_fp.close()
            if app_instance:
                cleaned = set()
                for t in app_instance.tabs:
                    gwt = t.agent.ctx.get('gwt')
                    if gwt and getattr(gwt, 'worktree', None) and (gwt.worktree not in cleaned):
                        cleaned.add(gwt.worktree)
                        try:
                            cleanup_worktree(gwt)
                        except Exception:
                            pass
            if app_instance and hasattr(app_instance, '_exit_summary') and app_instance._exit_summary:
                print('\n--- Session Exit Summary ---')
                for item in app_instance._exit_summary:
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
    parser.add_argument('--verbose-transcript', action='store_true', help='Reserved; not yet implemented')
    parser.add_argument('--notui', action='store_true', help='Use simple terminal REPL instead of TUI')
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
    run_repl(api=args.api, system=args.system, repo=args.cwd, model=model, base_url=url, tool_names=tool_names, sdir=args.skill, api_key=api_key, init_prompt=args.prompt, resume=args.resume, stream=args.stream, notui=args.notui)
if __name__ == '__main__':
    main()

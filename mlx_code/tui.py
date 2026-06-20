from __future__ import annotations
import asyncio
import datetime
import json
import os
import re
import subprocess
import logging
from typing import Callable
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
from .repl import Agent, TabModel, CommandEngine, UIAdapter, HELP_TEXT
logger = logging.getLogger(__name__)

def _clean_block_text(text: str) -> str:
    return text.lstrip('\n').rstrip()

def _make_empty_history_table() -> Table:
    tbl = Table(show_header=False, show_lines=False, box=None, pad_edge=False, expand=True, padding=0)
    tbl.add_column(width=2)
    tbl.add_column(ratio=1)
    return tbl

def append_to_history_table(tbl: Table, messages: list[dict], verbose: bool=False) -> None:
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
                if not verbose:
                    text = ''
            elif b_type == 'toolCall':
                prefix, style = ('⚙', 'yellow')
                text = block.get('name', '')
                if verbose:
                    args = block.get('arguments', {})
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    text += ' ' + str(args)
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

def render_history(messages: list[dict], verbose: bool=False) -> Table:
    tbl = _make_empty_history_table()
    append_to_history_table(tbl, messages, verbose=verbose)
    return tbl

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

    def __init__(self, model: TabModel, engine: CommandEngine):
        super().__init__(id=f'tab-{id(model.agent):x}')
        self.model = model
        self.engine = engine
        self._stream_blocks: list[dict] = []
        self._command_blocks: list[dict] = []
        self._cache_count: int = -1
        self._rendered_cache: Table | None = None
        self._tool_call_buf: str = ''

    @property
    def agent(self) -> Agent:
        return self.model.agent

    @agent.setter
    def agent(self, value: Agent) -> None:
        self.model.agent = value

    @property
    def errors(self):
        return self.model.errors

    @errors.setter
    def errors(self, value):
        self.model.errors = value

    @property
    def last_error(self):
        return self.model.last_error

    @last_error.setter
    def last_error(self, value):
        self.model.last_error = value

    @property
    def status(self):
        return self.model.status

    @status.setter
    def status(self, value):
        self.model.status = value

    @property
    def is_running(self) -> bool:
        return self.model.is_running

    def compose(self) -> ComposeResult:
        with VerticalScroll(id='scroll'):
            yield Static(id='cache')
            yield Static(id='stream')

    def apply_event(self, event: dict) -> None:
        et = event.get('type')
        payload = event.get('payload', {})
        verbose = self.engine.verbose
        if et == 'agent_start':
            self._tool_call_buf = ''
            self._stream_blocks = []
            self._command_blocks = []
            self.refresh_cache()
            self.refresh_stream()
        elif et == 'turn_start':
            self._tool_call_buf = ''
            self._stream_blocks = []
            self._command_blocks = []
            self.refresh_stream()
        elif et == 'text_delta':
            delta = payload.get('delta', '')
            if delta:
                if not verbose:
                    self._tool_call_buf = getattr(self, '_tool_call_buf', '') + delta
                    cleaned = re.sub('<tool_call>.*?</tool_call>', '', self._tool_call_buf, flags=re.DOTALL)
                    if '<tool_call>' in cleaned:
                        cut = cleaned.index('<tool_call>')
                        emit = cleaned[:cut]
                        self._tool_call_buf = cleaned[cut:]
                    else:
                        emit = cleaned
                        self._tool_call_buf = ''
                else:
                    emit = delta
                if emit:
                    if self._stream_blocks and self._stream_blocks[-1].get('type') == 'text' and (not self._stream_blocks[-1].get('is_error')):
                        self._stream_blocks[-1]['text'] += emit
                    else:
                        self._stream_blocks.append({'type': 'text', 'text': emit})
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
            self.refresh_stream()
        elif et == 'turn_end':
            self._stream_blocks = []
            self.refresh_stream()
            self.refresh_cache()
        elif et == 'agent_end':
            self._stream_blocks = []
            self.refresh_stream()
            self.refresh_cache()

    def refresh_cache(self) -> None:
        new_count = len(self.agent.messages)
        if new_count == self._cache_count:
            return
        if self._rendered_cache is None or new_count < self._cache_count:
            self._rendered_cache = render_history(self.agent.messages, verbose=self.engine.verbose)
        else:
            new_msgs = self.agent.messages[self._cache_count:]
            append_to_history_table(self._rendered_cache, new_msgs, verbose=self.engine.verbose)
        self._cache_count = new_count
        self.query_one('#cache', Static).update(self._rendered_cache)
        scroll = self.query_one('#scroll', VerticalScroll)
        if scroll.max_scroll_y - scroll.scroll_y < 3:
            self.app.call_after_refresh(scroll.scroll_end, animate=False)

    def refresh_stream(self) -> None:
        msgs: list[dict] = []
        if self._stream_blocks:
            msgs.append({'role': 'assistant', 'content': self._stream_blocks})
        if not msgs:
            msgs.extend(self._command_blocks)
        if msgs:
            self.query_one('#stream', Static).update(render_history(msgs, verbose=self.engine.verbose))
            scroll = self.query_one('#scroll', VerticalScroll)
            if scroll.max_scroll_y - scroll.scroll_y < 3:
                self.app.call_after_refresh(scroll.scroll_end, animate=False)
        else:
            self.query_one('#stream', Static).update('')

    def show_command(self, cmd: str, text: str) -> None:
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
            label = f' {marker}{i + 1}:{tab.model.title} '
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
        t.append(f' {tab.model.title}', style='bold')
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

class TuiAdapter:

    def __init__(self, app: 'ReplApp'):
        self.app = app

    def _widget_for(self, tab_model: TabModel) -> Tab:
        return self.app._widget_by_model_id[id(tab_model)]

    def show_error(self, text: str) -> None:
        self.app.query_one('#helpbar', HelpBar).show_error(text)

    def show_command_result(self, cmd: str, content: str | object) -> None:
        tab = self._widget_for(self.app.engine.active_tab)
        if isinstance(content, str):
            tab.show_command(cmd, content)
        else:
            tab._command_blocks.extend([{'role': 'user', 'content': cmd}, {'role': 'command', 'content': content}])
            tab.refresh_stream()

    def show_diff(self, diff_text: str, ref1_label: str, ref2_label: str) -> None:
        renderable = Syntax(diff_text, lexer='diff', theme='monokai')
        tab = self._widget_for(self.app.engine.active_tab)
        tab._command_blocks.extend([{'role': 'user', 'content': f'/diff {ref1_label}..{ref2_label}'}, {'role': 'command', 'content': renderable}])
        tab.refresh_stream()

    def show_history_list(self, lines: list[str]) -> None:
        tab = self._widget_for(self.app.engine.active_tab)
        tab.show_command('/history', '\n'.join(lines))

    def show_history_raw(self, json_text: str) -> None:
        tab = self._widget_for(self.app.engine.active_tab)
        tab.show_command('/history --raw', json_text)

    async def add_tab(self, tab_model: TabModel) -> None:
        widget = Tab(tab_model, self.app.engine)
        self.app.tab_widgets.append(widget)
        self.app._widget_by_model_id[id(tab_model)] = widget
        switcher = self.app.query_one(ContentSwitcher)
        await switcher.mount(widget)

    def remove_tab(self, removed_index: int) -> None:
        if 0 <= removed_index < len(self.app.tab_widgets):
            widget = self.app.tab_widgets.pop(removed_index)
            model_id = id(widget.model)
            self.app._widget_by_model_id.pop(model_id, None)
            widget.remove()

    def switch_to_tab(self, index: int) -> None:
        self.app._do_switch_to(index)

    def refresh_chrome(self) -> None:
        self.app._refresh_chrome()

    def clear_tab_display(self, tab_model: TabModel) -> None:
        widget = self._widget_for(tab_model)
        widget.clear_log()

    def on_agent_event(self, event: dict, tab: TabModel) -> None:
        widget = self._widget_for(tab)
        widget.apply_event(event)

    async def run_captured_shell(self, command: str, cwd: str, env: dict | None) -> str:
        proc = await asyncio.create_subprocess_shell(command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
        stdout, stderr = await proc.communicate()
        out = stdout.decode(errors='replace').rstrip('\n')
        err = stderr.decode(errors='replace').rstrip('\n')
        body = out
        if err:
            body = (body + '\n' if body else '') + f'[stderr]\n{err}'
        if proc.returncode and proc.returncode != 0:
            body += f'\n[exit {proc.returncode}]'
        return body

    async def run_interactive_shell(self, command: str, cwd: str, env: dict | None) -> int:

        def _blocking() -> int:
            return subprocess.run(command, shell=True, cwd=cwd, env={**os.environ, **(env or {})}).returncode
        with self.app.suspend():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _blocking)

    def exit_app(self, summary: list[dict]) -> None:
        self.app.exit()

class ReplApp(App[None]):
    CSS = '\n    ReplApp { layout: vertical; background: $background; }\n    ContentSwitcher { height: 1fr; }\n    InputBox {\n        height: auto;\n        min-height: 3;\n        max-height: 8;\n        border: none;\n        border-top: tall $panel-darken-1;\n        background: $panel;\n        padding: 0 2;\n        color: $text;\n    }\n    InputBox:focus { border-top: tall $accent; }\n    TabBar { height: 1; background: $panel; padding: 0 1; }\n    StatusBar { height: 1; background: $panel; color: $text-muted; padding: 0 1; }\n    HelpBar { height: 1; color: $text-muted; padding: 0 1; }\n    '
    BINDINGS = [Binding('ctrl+c', 'abort_agent', 'Abort', priority=True, show=False), Binding('ctrl+d', 'close_or_exit', 'Exit', show=False), Binding('ctrl+full_stop', 'next_tab', 'Next tab', show=False, priority=True), Binding('ctrl+comma', 'prev_tab', 'Prev tab', show=False, priority=True), Binding('ctrl+1', 'switch_tab(1)', 'Tab 1', show=False), Binding('ctrl+2', 'switch_tab(2)', 'Tab 2', show=False), Binding('ctrl+3', 'switch_tab(3)', 'Tab 3', show=False), Binding('ctrl+4', 'switch_tab(4)', 'Tab 4', show=False), Binding('ctrl+5', 'switch_tab(5)', 'Tab 5', show=False), Binding('ctrl+6', 'switch_tab(6)', 'Tab 6', show=False), Binding('ctrl+7', 'switch_tab(7)', 'Tab 7', show=False), Binding('ctrl+8', 'switch_tab(8)', 'Tab 8', show=False), Binding('ctrl+9', 'switch_tab(9)', 'Tab 9', show=False), Binding('ctrl+z', 'suspend_app', 'Suspend', show=False, priority=True)]

    def __init__(self, engine: CommandEngine, init_prompt: str | None=None) -> None:
        super().__init__()
        self.engine = engine
        self.adapter = TuiAdapter(self)
        self.engine.bind(self.adapter)
        self.tab_widgets: list[Tab] = []
        self._widget_by_model_id: dict[int, Tab] = {}
        self._confirm_close = False
        self._pending_init = init_prompt.strip() if init_prompt else None
        main_model = engine.tabs[0]
        main_widget = Tab(main_model, engine)
        self.tab_widgets = [main_widget]
        self._widget_by_model_id[id(main_model)] = main_widget
        self.engine.attach_agent(main_model)

    def compose(self) -> ComposeResult:
        yield TabBar(id='tabbar')
        yield ContentSwitcher(self.tab_widgets[0], initial=self.tab_widgets[0].id)
        yield InputBox()
        yield StatusBar(id='statusbar')
        yield HelpBar(id='helpbar')

    def on_mount(self) -> None:
        self._refresh_chrome()
        self._do_switch_to(0)
        if self._pending_init:
            self.set_timer(0.05, self._run_pending_init)

    def _run_pending_init(self) -> None:
        if self._pending_init:
            prompt, self._pending_init = (self._pending_init, None)
            asyncio.create_task(self.engine.handle_input(prompt))

    def _refresh_chrome(self) -> None:
        try:
            self.query_one('#tabbar', TabBar).render_tabs(self.tab_widgets, self.engine.active_index)
            active_widget = self.tab_widgets[self.engine.active_index]
            self.query_one('#statusbar', StatusBar).render_status(active_widget, active_widget.agent.model)
        except Exception:
            pass

    def _do_switch_to(self, idx: int) -> None:
        if not 0 <= idx < len(self.tab_widgets):
            return
        self._confirm_close = False
        try:
            self.query_one(ContentSwitcher).current = self.tab_widgets[idx].id
        except Exception:
            pass
        self.tab_widgets[idx].refresh_cache()
        self.query_one(InputBox).focus()
        self._refresh_chrome()
        self.query_one('#helpbar', HelpBar).show_idle()

    @property
    def active_tab(self) -> Tab:
        return self.tab_widgets[self.engine.active_index]

    def action_suspend_app(self) -> None:
        import signal
        with self.suspend():
            os.kill(os.getpid(), signal.SIGTSTP)

    def on_input_box_submit(self, message: InputBox.Submit) -> None:
        asyncio.create_task(self.engine.handle_input(message.text))

    def on_input_box_recall_last(self, _msg: InputBox.RecallLast) -> None:
        real = [m for m in self.active_tab.agent.messages if m.get('role') == 'user']
        if real:
            self.query_one(InputBox).set_text_and_end(real[-1].get('content', ''))

    def on_input_box_close_request(self, _msg: InputBox.CloseRequest) -> None:
        tab = self.active_tab
        if tab.is_running:
            if not self._confirm_close:
                self._confirm_close = True
                self.query_one('#helpbar', HelpBar).show_confirm('Agent is running — press Ctrl-D again to force close.')
                return
        self._confirm_close = False
        self.engine.close_or_exit()

    def on_tab_bar_switch_to(self, msg: TabBar.SwitchTo) -> None:
        self.engine.switch_tab(msg.index)

    def action_next_tab(self) -> None:
        self.engine.switch_tab((self.engine.active_index + 1) % len(self.engine.tabs))

    def action_prev_tab(self) -> None:
        self.engine.switch_tab((self.engine.active_index - 1) % len(self.engine.tabs))

    def action_switch_tab(self, n: int) -> None:
        self.engine.switch_tab(n - 1)

    def action_close_or_exit(self) -> None:
        tab = self.active_tab
        if tab.is_running:
            if not self._confirm_close:
                self._confirm_close = True
                self.query_one('#helpbar', HelpBar).show_confirm('Agent is running — press Ctrl-D again to force close.')
                return
        self._confirm_close = False
        self.engine.close_or_exit()

    def action_abort_agent(self) -> None:
        input_box = self.query_one(InputBox)
        if input_box.text:
            input_box.load_text('')
            input_box.focus()
            return
        self.engine._cmd_abort('')
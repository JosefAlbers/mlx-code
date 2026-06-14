from __future__ import annotations
import asyncio
import copy
import datetime
import json
import os
import re
import sys
from .gits import create_worktree, find_rev_commit, git_new_branch, git_new_branch_at, GitError, get_branch_base_sha, resolve_ref_short, get_diff_between_refs

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
    child_count = sum((1 for t in existing_tabs if len(t.get('index_path', ())) == depth and t.get('index_path', ())[:-1] == parent_path))
    index_path = parent_path + (child_count,)
    title = 'branch-' + '-'.join((str(i) for i in index_path))
    return (index_path, title)

class SimpleRepl:

    def __init__(self, agent, init_prompt=None):
        self.tabs = [{'agent': agent, 'title': 'main', 'errors': [], 'index_path': (), 'owns_worktree': False}]
        self.active_index = 0
        self._unsubscribers = {}
        self.init_prompt = init_prompt
        self._pending_nls: int = 0
        self._awaiting_content: bool = False
        self._has_output: bool = False
        self._last_stream_type: str | None = None
        self._attach_agent(0)

    @property
    def active_tab(self):
        return self.tabs[self.active_index]

    def _write_delta(self, text: str, delta_type: str):
        if delta_type != self._last_stream_type:
            self._pending_nls = 0
            self._awaiting_content = True
            self._last_stream_type = delta_type
        if self._awaiting_content:
            text = text.lstrip('\n')
            if not text:
                return
        if self._awaiting_content:
            if self._has_output:
                print()
            self._awaiting_content = False
        if not self._awaiting_content and self._pending_nls > 0:
            print('\n' * self._pending_nls, end='', flush=True)
            self._pending_nls = 0
        rstripped = text.rstrip('\n')
        if rstripped:
            if delta_type == 'thinking_delta':
                print(f'\x1b[2m{rstripped}\x1b[0m', end='', flush=True)
            else:
                print(rstripped, end='', flush=True)
            self._has_output = True
        self._pending_nls = len(text) - len(rstripped)

    def _attach_agent(self, index):
        tab = self.tabs[index]
        key = id(tab['agent'])
        if key in self._unsubscribers:
            return

        async def on_event(ev):
            t, p = (ev['type'], ev.get('payload', {}))
            if t in ('text_delta', 'thinking_delta'):
                delta = p.get('delta', '')
                if delta:
                    self._write_delta(delta, t)
            elif t == 'tool_start':
                self._pending_nls = 0
                self._awaiting_content = False
                self._has_output = True
                self._last_stream_type = t
            elif t == 'tool_end':
                result_msg = p.get('result', {})
                content = result_msg.get('content')
                is_err = p.get('is_error', False)
                out_text = ''
                if content:
                    parts = []
                    if isinstance(content, str):
                        parts.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                parts.append(block.get('text', ''))
                    out_text = '\n'.join(parts).strip('\n')
                if is_err:
                    prefix = '✗ '
                    if not out_text:
                        out_text = f'{p.get('name', '?')} failed'
                else:
                    prefix = '→ ' if out_text else ''
                if out_text:
                    self._write_delta(prefix + out_text, 'tool_result')
                self._last_stream_type = t
                print()
            elif t == 'commit':
                self._pending_nls = 0
                self._awaiting_content = False
                self._has_output = True
                print(f'\n◇ [{p.get('sha', '')}] committed', flush=True)
                self._last_stream_type = t
            elif t == 'error':
                self._pending_nls = 0
                self._awaiting_content = False
                self._has_output = True
                err = str(p.get('error', p))
                print(f'\n✗ {err}', flush=True)
                tab['errors'].append((datetime.datetime.now().strftime('%H:%M:%S'), err))
                self._last_stream_type = t
            elif t in ('agent_start', 'turn_start'):
                self._pending_nls = 0
                self._awaiting_content = False
                self._has_output = False
                self._last_stream_type = None
            elif t == 'agent_end':
                self._pending_nls = 0
                if self._has_output:
                    print()
                self._last_stream_type = None
                self._has_output = False
                self._awaiting_content = False
        self._unsubscribers[key] = tab['agent'].subscribe(on_event)

    async def run(self):
        loop = asyncio.get_running_loop()
        if self.init_prompt:
            p, self.init_prompt = (self.init_prompt, None)
            await self.active_tab['agent'].run(p)
        while True:
            try:
                line = await loop.run_in_executor(None, self._read_input)
            except KeyboardInterrupt:
                print('\n(Use /exit or Ctrl-D to quit)')
                continue
            except EOFError:
                print()
                break
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith('/'):
                await self._handle_command(line)
            elif line.startswith('!!'):
                cmd = line[2:].strip()
                if cmd:
                    await self._run_interactive(cmd)
            elif line.startswith('!'):
                cmd = line[1:].strip()
                if cmd:
                    await self._run_shell(cmd)
            elif line.lower() in {'exit', 'quit'}:
                break
            else:
                try:
                    await self.active_tab['agent'].run(line)
                except Exception as e:
                    print(f'\n✗ Error: {e}')

    def _read_input(self):
        tab = self.active_tab
        prompt = f'[{tab['title']}] ≫ '
        lines = []
        while True:
            try:
                line = input(prompt)
            except EOFError:
                return None
            lines.append(line)
            if line.endswith('\\'):
                lines[-1] = line[:-1]
                prompt = '... '
            else:
                break
        return '\n'.join(lines)

    async def _run_shell(self, command):
        gwt = self.active_tab['agent'].ctx.get('gwt')
        cwd = gwt.worktree if gwt and getattr(gwt, 'worktree', None) else self.active_tab['agent'].ctx.get('cwd', os.getcwd())
        env = self.active_tab['agent'].ctx.get('env')
        proc = await asyncio.create_subprocess_shell(command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env if env else None)
        stdout, stderr = await proc.communicate()
        out = stdout.decode(errors='replace').rstrip('\n')
        err = stderr.decode(errors='replace').rstrip('\n')
        if out:
            print(out)
        if err:
            print(f'[stderr]\n{err}')
        if proc.returncode:
            print(f'[exit {proc.returncode}]')

    async def _run_interactive(self, command):
        gwt = self.active_tab['agent'].ctx.get('gwt')
        cwd = gwt.worktree if gwt and getattr(gwt, 'worktree', None) else self.active_tab['agent'].ctx.get('cwd', os.getcwd())
        env = self.active_tab['agent'].ctx.get('env')
        proc = await asyncio.create_subprocess_shell(command, cwd=cwd, stdin=None, stdout=None, stderr=None, env=env if env else None)
        await proc.wait()

    async def _handle_command(self, text):
        cmd, _, arg = text.partition(' ')
        cmd = cmd.lower().strip()
        arg = arg.strip()
        if cmd == '/help':
            print('\nCommands:\n/help               show this message\n/clear [--config F] clear conversation; --config reconfigures agent from YAML/JSON\n/history            show full conversation transcript\n/history --raw      show raw API message log (debug)\n/diff [--all]       show side-by-side diff of changes\n/errors             show timestamped error log for this tab\n/tools              list active tools\n/branch [--rev N] [--no-worktree] [prompt]\n                    open a branch tab; optional prompt runs immediately\n/abort              abort the running agent\n/export [path]      export session to JSON\n/exit  /quit        close branch tab, or exit the app\n!command            run shell command in worktree (output captured in TUI)\n!!command           run interactive shell command (TUI suspends, terminal handed to process)\n                    e.g.  !ls  !git diff  !cat file.py\n                          !!vim file.py  !!yazi  !!less log.txt\nKeys:\nEnter               submit\nCtrl-J              insert newline in editor\nAlt-1 … Alt-9       jump directly to tab N\nTab / Shift-Tab     cycle through tabs\nCtrl-C              abort running agent\nCtrl-D              close branch tab, or exit app\nCtrl-R              recall last prompt into editor\n')
        elif cmd == '/tab':
            await self._cmd_tab(arg)
        elif cmd == '/branches':
            self._cmd_branches()
        elif cmd == '/branch':
            await self._cmd_branch(arg)
        elif cmd == '/clear':
            cfg_match = re.match('--config\\s+(.+)', arg)
            if cfg_match:
                config_path = cfg_match.group(1).strip().strip('"').strip("'")
                try:
                    cfg = load_agent_config(config_path)
                except Exception as e:
                    print(f'✗ Config error: {e}')
                    return
                old = self.active_tab['agent']
                new_ctx = {k: v for k, v in old.ctx.items() if k != 'agent'}
                new_agent = Agent(system=cfg.get('system'), api=cfg.get('api'), model=cfg.get('model'), api_key=cfg.get('api_key'), base_url=cfg.get('base_url'), tool_names=cfg.get('tools'), extra_tool_classes=old._extra_tool_classes, ctx=new_ctx)
                old_key = id(old)
                if old_key in self._unsubscribers:
                    self._unsubscribers[old_key]()
                    del self._unsubscribers[old_key]
                self.active_tab['agent'] = new_agent
                self._attach_agent(self.active_index)
                print(f'Agent reconfigured from {config_path}')
                print(f'  model: {cfg.get('model', '(default)')}')
                print(f'  tools: {', '.join((t.name for t in new_agent.tools))}')
            else:
                self.active_tab['agent'].messages.clear()
                print('Conversation cleared.')
        elif cmd == '/history':
            self._cmd_history(arg)
        elif cmd == '/diff':
            self._cmd_diff(arg)
        elif cmd == '/errors':
            self._cmd_errors()
        elif cmd == '/tools':
            self._cmd_tools()
        elif cmd == '/abort':
            self.active_tab['agent'].abort()
            print('Abort requested.')
        elif cmd == '/export':
            self._cmd_export(arg)
        elif cmd in ('/exit', '/quit'):
            raise SystemExit
        else:
            print(f'Unknown command: {cmd!r} — try /help')

    def _cmd_branches(self):
        for i, t in enumerate(self.tabs):
            marker = '►' if i == self.active_index else ' '
            print(f' {marker} {i + 1}. {t['title']}')

    async def _cmd_tab(self, arg):
        if not arg or not arg.isdigit():
            print('Usage: /tab <n>')
            return
        n = int(arg) - 1
        if not 0 <= n < len(self.tabs):
            print(f'Invalid tab: {n + 1}. Available: 1-{len(self.tabs)}')
            return
        self.active_index = n
        self._attach_agent(n)
        self._render_tab_delimiter()
        self._print_history_for_tab(self.active_tab)

    async def _cmd_branch(self, arg):
        as_worktree = False
        rev_n = None
        prompt = arg
        if '--as-worktree' in prompt:
            as_worktree = True
            prompt = prompt.replace('--as-worktree', '').strip()
        rev_match = re.search('--rev\\s+(\\d+)', prompt)
        if rev_match:
            rev_n = int(rev_match.group(1))
            prompt = (prompt[:rev_match.start()] + prompt[rev_match.end():]).strip()
        parent = self.active_tab
        parent_index_path = parent.get('index_path', ())
        all_msgs = parent['agent'].messages
        user_indices = [i for i, m in enumerate(all_msgs) if m.get('role') == 'user']
        if rev_n is not None:
            if rev_n < 1 or rev_n > len(user_indices):
                print(f'--rev {rev_n}: must be between 1 and {len(user_indices)}')
                return
            cut_at = user_indices[rev_n - 1]
            sliced_messages = copy.deepcopy(all_msgs[:cut_at])
        else:
            sliced_messages = copy.deepcopy(all_msgs)
        child = parent['agent'].branch()
        child.messages = sliced_messages
        index_path, title = _branch_index_title(parent_index_path, self.tabs)
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
                    print(f'--rev {rev_n}: no matching commit found; file state will be HEAD')
            new_gwt = create_worktree(repo_dir, prefix=title, ref=ref)
            if new_gwt is None:
                print(f'git worktree creation failed for {title!r}')
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
                        print(f'--rev {rev_n}: no matching commit found; file state will be HEAD')
                        new_gwt = git_new_branch(gwt.worktree, title)
                else:
                    new_gwt = git_new_branch(gwt.worktree, title)
                child.ctx['gwt'] = new_gwt
            except GitError as exc:
                print(f'git_new_branch failed for {title!r}: {exc}')
        new_tab = {'agent': child, 'title': title, 'errors': [], 'owns_worktree': owns_worktree, 'index_path': index_path}
        self.tabs.append(new_tab)
        self._attach_agent(len(self.tabs) - 1)
        self.active_index = len(self.tabs) - 1
        self._render_tab_delimiter()
        self._print_history_for_tab(self.active_tab)
        if prompt:
            if prompt.startswith('/'):
                await self._handle_command(prompt)
            else:
                await self.active_tab['agent'].run(prompt)

    def _render_tab_delimiter(self):
        tab_strs = []
        for i, t in enumerate(self.tabs):
            if i == self.active_index:
                tab_strs.append(f'\x1b[1m▶ {i + 1}. {t['title']}\x1b[0m')
            else:
                tab_strs.append(f'\x1b[2m▷ {i + 1}. {t['title']}\x1b[0m')
        print('\n' + '┗━━┫' + ' ' + ' ┃ '.join(tab_strs) + ' ┃')

    def _print_history_for_tab(self, tab):
        for msg in tab['agent'].messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            is_error = msg.get('is_error', False)
            if isinstance(content, list):
                blocks = content
            elif isinstance(content, str):
                blocks = [{'type': 'text', 'text': content}]
            else:
                continue
            if role == 'toolResult':
                parts = []
                for block in blocks:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        t = block.get('text', '').strip('\n')
                        if t:
                            parts.append(t)
                if parts:
                    prefix = '✗ ' if is_error else '→ '
                    print(prefix + '\n'.join(parts))
                continue
            for block in blocks:
                btype = block.get('type', 'text')
                if btype == 'toolCall':
                    args = block.get('arguments', {})
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    print(f'⚙ {block.get('name', '')} {args}')
                    continue
                text = block.get('text', '') or block.get('thinking', '') or ''
                text = text.strip('\n')
                if not text:
                    continue
                if btype == 'thinking':
                    print(f'\x1b[2m{text}\x1b[0m')
                elif is_error:
                    print(f'✗ {text}')
                elif role == 'user':
                    print(f'≫ {text}')
                elif role == 'commit':
                    print(f'◇ {text}')
                elif role == 'toolResult':
                    print(f'→ {text}')
                else:
                    print(text)

    def _cmd_history(self, arg):
        if arg == '--raw':
            print(json.dumps(self.active_tab['agent'].messages, indent=2, default=str))
            return
        user_msgs = [m for m in self.active_tab['agent'].messages if m.get('role') == 'user']
        if not user_msgs:
            print('No prompts yet.')
            return
        for i, m in enumerate(user_msgs, 1):
            content = m.get('content', '')
            if isinstance(content, list):
                content = ' '.join((b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text'))
            content = re.sub('\\s+', ' ', content).strip()
            if len(content) > 100:
                content = content[:100] + '…'
            print(f'{i}. {content}')

    def _cmd_diff(self, arg):
        gwt = self.active_tab['agent'].ctx.get('gwt')
        if not gwt or not getattr(gwt, 'worktree', None):
            print('No git worktree available for this tab.')
            return
        ref1, ref2 = ('HEAD~1', 'HEAD')
        is_all = '--all' in arg
        if is_all:
            base = get_branch_base_sha(gwt.worktree)
            if base:
                ref1 = base
            else:
                print('Could not determine base commit for --all.')
                return
        try:
            diff_text = get_diff_between_refs(gwt.worktree, ref1, ref2)
        except GitError as e:
            print(f'Git diff failed: {e}')
            return
        if not diff_text.strip():
            print('No differences.')
            return
        print(diff_text)

    def _cmd_errors(self):
        if not self.active_tab['errors']:
            print('No errors recorded.')
            return
        for ts, msg in self.active_tab['errors'][-30:]:
            print(f'{ts}  {msg}')

    def _cmd_tools(self):
        tools = self.active_tab['agent'].tools
        if not tools:
            print('No tools enabled.')
            return
        for t in tools:
            print(f'{t.name}  {t.description}')

    def _cmd_export(self, arg):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        user_cwd = self.active_tab['agent'].ctx.get('user_cwd', os.getcwd())
        path = arg if arg and os.path.isabs(arg) else os.path.join(user_cwd, arg or f'session_{ts}.json')
        data = {'version': 1, 'exported_at': ts, 'system': self.active_tab['agent'].system, 'messages': self.active_tab['agent'].messages}
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f'Exported → {path}')
        except OSError as e:
            print(f'Export failed: {e}')
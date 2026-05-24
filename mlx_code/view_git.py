from __future__ import annotations
import curses
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

try:
    import git as _git
except ImportError:
    print("gitpython not installed. Run: pip install gitpython")
    sys.exit(1)


@dataclass
class CommitNode:
    hexsha: str
    short: str
    summary: str
    committed_dt: str
    author: str
    kind: str
    branch_name: str | None = None
    parent_hexsha: str | None = None
    depth: int = 0
    expanded: bool = False
    children: list["CommitNode"] = field(default_factory=list)

    @property
    def is_expandable(self) -> bool:
        return bool(self.children)


def _short(h: str) -> str:
    return h[:8]


def _fmt_dt(commit) -> str:
    try:
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(commit.committed_date, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


def build_tree(repo_dir: str, repo: _git.Repo | None = None) -> list[CommitNode]:
    if repo is None:
        try:
            repo = _git.Repo(repo_dir, search_parent_directories=True)
        except _git.InvalidGitRepositoryError:
            raise ValueError(f"Not a git repo: {repo_dir}")
    agent_branches = [
        b
        for b in repo.branches
        if b.name.startswith("agent/") or b.name.startswith("resume/")
    ]
    if not agent_branches:
        return []
    base_map: dict[str, list] = {}
    for b in agent_branches:
        seg = b.name.split("/", 1)[1]
        dash = seg.find("-")
        base_hex = seg if dash == -1 else seg[:dash]
        try:
            full = repo.commit(base_hex).hexsha
        except Exception:
            full = base_hex
        base_map.setdefault(full, []).append(b)
    base_dates: dict[str, int] = {}
    for hexsha in base_map:
        try:
            base_dates[hexsha] = repo.commit(hexsha).committed_date
        except Exception:
            base_dates[hexsha] = 0
    sorted_bases = sorted(
        base_map.keys(), key=lambda h: base_dates.get(h, 0), reverse=True
    )
    flat: list[CommitNode] = []
    for base_hex in sorted_bases:
        try:
            bc = repo.commit(base_hex)
        except Exception:
            continue
        base_node = CommitNode(
            hexsha=bc.hexsha,
            short=_short(bc.hexsha),
            summary=bc.summary[:60],
            committed_dt=_fmt_dt(bc),
            author=bc.author.name or "?",
            kind="base",
            depth=0,
        )
        branches = sorted(
            base_map[base_hex], key=lambda b: b.commit.committed_date, reverse=True
        )
        for br in branches:
            try:
                branch_commits = list(repo.iter_commits(f"{base_hex}..{br.name}"))
            except Exception:
                branch_commits = []
            branch_commits.reverse()
            tip = br.commit
            label = CommitNode(
                hexsha=tip.hexsha,
                short=_short(tip.hexsha),
                summary=br.name,
                committed_dt=_fmt_dt(tip),
                author=tip.author.name or "?",
                kind="branch_label",
                branch_name=br.name,
                parent_hexsha=base_hex,
                depth=1,
            )
            for i, c in enumerate(branch_commits):
                parent_hex = branch_commits[i - 1].hexsha if i > 0 else base_hex
                label.children.append(
                    CommitNode(
                        hexsha=c.hexsha,
                        short=_short(c.hexsha),
                        summary=c.summary[:60],
                        committed_dt=_fmt_dt(c),
                        author=c.author.name or "?",
                        kind="commit",
                        branch_name=br.name,
                        parent_hexsha=parent_hex,
                        depth=2,
                    )
                )
            base_node.children.append(label)
        flat.append(base_node)
    return flat


def flatten_visible(roots: list[CommitNode]) -> list[CommitNode]:

    def _walk(nodes: list[CommitNode]) -> list[CommitNode]:
        result = []
        for node in nodes:
            result.append(node)
            if node.expanded and node.children:
                result.extend(_walk(node.children))
        return result

    return _walk(roots)


def expand_all(node: CommitNode) -> int:
    count = 0
    if node.children:
        if not node.expanded:
            node.expanded = True
            count += 1
        for child in node.children:
            count += expand_all(child)
    return count


def get_diff(
    repo: _git.Repo, node: CommitNode, mark_node: CommitNode | None = None
) -> list[str]:
    try:
        if mark_node and mark_node.hexsha != node.hexsha:
            diff = repo.git.diff(mark_node.hexsha, node.hexsha)
        else:
            parent_hex = node.parent_hexsha
            if parent_hex:
                diff = repo.git.diff(parent_hex, node.hexsha)
            elif node.kind == "base":
                commit = repo.commit(node.hexsha)
                if commit.parents:
                    diff = repo.git.diff(commit.parents[0].hexsha, node.hexsha)
                else:
                    diff = repo.git.show(node.hexsha, "--format=")
            else:
                diff = repo.git.show(node.hexsha, "--format=")
        return diff.splitlines() if diff else ["(no changes)"]
    except Exception as e:
        return [f"(diff error: {e})"]


def get_message(repo: _git.Repo, node: CommitNode) -> list[str]:
    try:
        commit = repo.commit(node.hexsha)
        msg = commit.message.rstrip()
        return msg.splitlines() if msg else []
    except Exception:
        return []


def get_stat(
    repo: _git.Repo, node: CommitNode, mark_node: CommitNode | None = None
) -> str:
    try:
        if mark_node and mark_node.hexsha != node.hexsha:
            return repo.git.diff(mark_node.hexsha, node.hexsha, "--stat")
        else:
            parent_hex = node.parent_hexsha
            if parent_hex:
                return repo.git.diff(parent_hex, node.hexsha, "--stat")
            elif node.kind == "base":
                commit = repo.commit(node.hexsha)
                if commit.parents:
                    return repo.git.diff(
                        commit.parents[0].hexsha, node.hexsha, "--stat"
                    )
                else:
                    return repo.git.show(node.hexsha, "--stat", "--format=")
            else:
                return repo.git.show(node.hexsha, "--stat", "--format=")
    except Exception:
        return ""


def restore_snapshot(
    repo: _git.Repo, node: CommitNode, base_dir: str
) -> tuple[bool, str]:
    dest = os.path.join(base_dir, f"_restored_{node.short}")
    try:
        repo.git.worktree("prune")
    except Exception:
        pass
    if os.path.exists(dest):
        return (False, f"already exists: {dest}")
    try:
        repo.git.worktree("add", "--detach", dest, node.hexsha)
        return (True, dest)
    except Exception as e:
        return (False, str(e))


C_NORMAL = 0
C_BASE = 1
C_BRANCH = 2
C_COMMIT = 3
C_SELECTED = 4
C_MARKED = 5
C_DIFF_ADD = 6
C_DIFF_REM = 7
C_DIFF_HUNK = 8
C_DIM = 9
C_HEADER = 10
C_STATUS_OK = 11
C_STATUS_ER = 12


def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_BASE, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_BRANCH, curses.COLOR_CYAN, -1)
    curses.init_pair(C_COMMIT, curses.COLOR_WHITE, -1)
    curses.init_pair(C_SELECTED, curses.COLOR_BLACK, curses.COLOR_GREEN)
    curses.init_pair(C_MARKED, curses.COLOR_BLACK, curses.COLOR_YELLOW)
    curses.init_pair(C_DIFF_ADD, curses.COLOR_GREEN, -1)
    curses.init_pair(C_DIFF_REM, curses.COLOR_RED, -1)
    curses.init_pair(C_DIFF_HUNK, curses.COLOR_CYAN, -1)
    curses.init_pair(C_DIM, 8, -1)
    curses.init_pair(C_HEADER, curses.COLOR_WHITE, -1)
    curses.init_pair(C_STATUS_OK, curses.COLOR_GREEN, -1)
    curses.init_pair(C_STATUS_ER, curses.COLOR_RED, -1)


@dataclass
class State:
    roots: list[CommitNode]
    repo: _git.Repo
    repo_dir: str
    cursor: int = 0
    diff_scroll: int = 0
    mark_node: CommitNode | None = None
    status_msg: str = ""
    status_ok: bool = True
    show_help: bool = False
    quit_snapshot_path: str | None = None
    _visible_cache: list[CommitNode] | None = field(default=None, repr=False)
    _diff_cache: tuple[str, str | None, list[str], str, list[str]] | None = field(
        default=None, repr=False
    )

    def invalidate_visible(self):
        self._visible_cache = None

    @property
    def visible(self) -> list[CommitNode]:
        if self._visible_cache is None:
            self._visible_cache = flatten_visible(self.roots)
        return self._visible_cache

    def current_node(self) -> CommitNode | None:
        v = self.visible
        if not v:
            return None
        return v[min(self.cursor, len(v) - 1)]

    def clamp_cursor(self):
        v = self.visible
        if not v:
            self.cursor = 0
        else:
            self.cursor = max(0, min(self.cursor, len(v) - 1))

    def get_diff_cached(self) -> list[str]:
        node = self.current_node()
        if not node:
            return []
        mark_sha = self.mark_node.hexsha if self.mark_node else None
        if (
            self._diff_cache is not None
            and self._diff_cache[0] == node.hexsha
            and (self._diff_cache[1] == mark_sha)
        ):
            return self._diff_cache[2]
        lines = get_diff(self.repo, node, self.mark_node)
        stat = get_stat(self.repo, node, self.mark_node)
        msg = get_message(self.repo, node)
        self._diff_cache = (node.hexsha, mark_sha, lines, stat, msg)
        return lines

    def get_stat_cached(self) -> str:
        node = self.current_node()
        if not node:
            return ""
        mark_sha = self.mark_node.hexsha if self.mark_node else None
        if (
            self._diff_cache is not None
            and self._diff_cache[0] == node.hexsha
            and (self._diff_cache[1] == mark_sha)
        ):
            return self._diff_cache[3]
        self.get_diff_cached()
        return self._diff_cache[3] if self._diff_cache else ""

    def get_message_cached(self) -> list[str]:
        node = self.current_node()
        if not node:
            return []
        mark_sha = self.mark_node.hexsha if self.mark_node else None
        if (
            self._diff_cache is not None
            and self._diff_cache[0] == node.hexsha
            and (self._diff_cache[1] == mark_sha)
        ):
            return self._diff_cache[4]
        self.get_diff_cached()
        return self._diff_cache[4] if self._diff_cache else []

    def invalidate_diff(self):
        self._diff_cache = None


TREE_W_RATIO = 0.4


def draw_tree(win, state: State, h: int, w: int):
    win.erase()
    visible = state.visible
    rows = h - 1
    start = max(0, state.cursor - rows // 2)
    start = min(start, max(0, len(visible) - rows))
    for i, node in enumerate(visible[start : start + rows]):
        abs_i = i + start
        is_cursor = abs_i == state.cursor
        is_marked = bool(state.mark_node and state.mark_node.hexsha == node.hexsha)
        if is_cursor and is_marked:
            attr = curses.color_pair(C_MARKED) | curses.A_BOLD
        elif is_cursor:
            attr = curses.color_pair(C_SELECTED) | curses.A_BOLD
        elif is_marked:
            attr = curses.color_pair(C_MARKED)
        elif node.kind == "base":
            attr = curses.color_pair(C_BASE)
        elif node.kind == "branch_label":
            attr = curses.color_pair(C_BRANCH)
        else:
            attr = curses.color_pair(C_COMMIT)
        if node.is_expandable:
            toggle = "▾ " if node.expanded else "▸ "
        else:
            toggle = "  "
        indent = "  " * node.depth
        if node.kind == "base":
            content = f"● {node.committed_dt}  {node.summary}"
        elif node.kind == "branch_label":
            inner = (
                node.branch_name.split("/", 1)[1]
                if node.branch_name and "/" in node.branch_name
                else node.branch_name or ""
            )
            seg = inner.split("-", 1)
            if len(seg) == 2:
                label_name = f"{seg[0][:6]}…{seg[1][:8]}"
            else:
                label_name = inner[:18]
            n = len(node.children)
            commit_count = f"  ({n} commit{('s' if n != 1 else '')})"
            content = f"⎇ {label_name}  {node.committed_dt}{commit_count}"
        else:
            siblings = []
            for vi in range(abs_i - 1, -1, -1):
                if (
                    visible[vi].kind == "branch_label"
                    and visible[vi].branch_name == node.branch_name
                ):
                    siblings = visible[vi].children
                    break
            is_last = not siblings or node.hexsha == siblings[-1].hexsha
            tree_char = "╰─" if is_last else "├─"
            content = f"{tree_char} {node.committed_dt}  {node.summary}"
        mark_flag = " ◆" if is_marked else ""
        line = f"{indent}{toggle}{content}{mark_flag}"
        line = line[: w - 1].ljust(w - 1)
        try:
            win.addstr(i, 0, line, attr)
        except curses.error:
            pass
    win.noutrefresh()


def draw_diff(win, state: State, h: int, w: int):
    win.erase()
    node = state.current_node()
    if not node:
        win.noutrefresh()
        return
    lines: list[tuple[str, int]] = []
    mark = state.mark_node
    if mark and mark.hexsha != node.hexsha:
        hdr = f" RANGE  {mark.hexsha} ──▶ {node.hexsha}"
    else:
        hdr = f" {node.hexsha}  {node.committed_dt}  {node.author}"
    lines.append(
        (
            hdr.ljust(w - 1),
            curses.color_pair(C_HEADER) | curses.A_BOLD | curses.A_REVERSE,
        )
    )
    if node.kind == "branch_label" and node.branch_name:
        lines.append((f"  branch: {node.branch_name}", curses.color_pair(C_DIM)))
    stat = state.get_stat_cached()
    if stat:
        lines.append(("", C_NORMAL))
        summary_hdr = (" ── summary " + "─" * (w - 13))[: w - 1]
        lines.append((summary_hdr, curses.color_pair(C_DIM) | curses.A_BOLD))
        for line in stat.splitlines():
            lines.append((line.lstrip()[: w - 1], curses.color_pair(C_NORMAL)))
        lines.append(("", C_NORMAL))
    diff_lines = state.get_diff_cached()
    if diff_lines:
        changes_hdr = (" ── changes " + "─" * (w - 13))[: w - 1]
        lines.append((changes_hdr, curses.color_pair(C_DIM) | curses.A_BOLD))
        for dl in diff_lines:
            if dl.startswith("+++") or dl.startswith("---"):
                lines.append((dl[: w - 1], curses.color_pair(C_DIM) | curses.A_BOLD))
            elif dl.startswith("+"):
                lines.append((dl[: w - 1], curses.color_pair(C_DIFF_ADD)))
            elif dl.startswith("-"):
                lines.append((dl[: w - 1], curses.color_pair(C_DIFF_REM)))
            elif dl.startswith("@@"):
                lines.append((dl[: w - 1], curses.color_pair(C_DIFF_HUNK)))
            elif dl.startswith("diff ") or dl.startswith("index "):
                lines.append((dl[: w - 1], curses.color_pair(C_DIM) | curses.A_BOLD))
            else:
                lines.append((dl[: w - 1], curses.color_pair(C_NORMAL)))
    msg_lines = state.get_message_cached()
    if msg_lines:
        lines.append(("", C_NORMAL))
        msg_hdr = (" ── commit message " + "─" * (w - 20))[: w - 1]
        lines.append((msg_hdr, curses.color_pair(C_DIM) | curses.A_BOLD))
        for ml in msg_lines:
            lines.append((ml[: w - 1], curses.color_pair(C_NORMAL)))
    max_scroll = max(0, len(lines) - h + 1)
    state.diff_scroll = max(0, min(state.diff_scroll, max_scroll))
    visible_lines = lines[state.diff_scroll : state.diff_scroll + h]
    for i, (text, attr) in enumerate(visible_lines):
        try:
            win.addstr(i, 0, text.ljust(w - 1)[: w - 1], attr)
        except curses.error:
            pass
    if max_scroll > 0:
        pct = int(100 * state.diff_scroll / max_scroll)
        indicator = f" {pct}% "
        try:
            win.addstr(
                h - 1,
                max(0, w - len(indicator) - 1),
                indicator,
                curses.color_pair(C_DIM),
            )
        except curses.error:
            pass
    win.noutrefresh()


def diff_pager(stdscr, state: State):
    node = state.current_node()
    if not node:
        return
    curses.curs_set(0)
    scroll = state.diff_scroll
    lines: list[tuple[str, int]] = []
    h, w = stdscr.getmaxyx()
    mark = state.mark_node
    if mark and mark.hexsha != node.hexsha:
        hdr = f" RANGE  {mark.hexsha} ──▶ {node.hexsha}"
    else:
        hdr = f" {node.hexsha}  {node.committed_dt}  {node.author}"
    lines.append((hdr, curses.color_pair(C_HEADER) | curses.A_BOLD | curses.A_REVERSE))
    if node.kind == "branch_label" and node.branch_name:
        lines.append((f"  branch: {node.branch_name}", curses.color_pair(C_DIM)))
    stat = state.get_stat_cached()
    if stat:
        lines.append(("", C_NORMAL))
        summary_hdr = (" ── summary " + "─" * (w - 13))[: w - 1]
        lines.append((summary_hdr, curses.color_pair(C_DIM) | curses.A_BOLD))
        for line in stat.splitlines():
            lines.append((line.lstrip(), curses.color_pair(C_NORMAL)))
        lines.append(("", C_NORMAL))
    diff_lines = state.get_diff_cached()
    if diff_lines:
        changes_hdr = (" ── changes " + "─" * (w - 13))[: w - 1]
        for dl in diff_lines:
            if dl.startswith("+++") or dl.startswith("---"):
                lines.append((dl, curses.color_pair(C_DIM) | curses.A_BOLD))
            elif dl.startswith("+"):
                lines.append((dl, curses.color_pair(C_DIFF_ADD)))
            elif dl.startswith("-"):
                lines.append((dl, curses.color_pair(C_DIFF_REM)))
            elif dl.startswith("@@"):
                lines.append((dl, curses.color_pair(C_DIFF_HUNK)))
            elif dl.startswith("diff ") or dl.startswith("index "):
                lines.append((dl, curses.color_pair(C_DIM) | curses.A_BOLD))
            else:
                lines.append((dl, curses.color_pair(C_NORMAL)))
    msg_lines = state.get_message_cached()
    if msg_lines:
        lines.append(("", C_NORMAL))
        lines.append(
            (" ── commit message " + "─" * 40, curses.color_pair(C_DIM) | curses.A_BOLD)
        )
        for ml in msg_lines:
            lines.append((ml, curses.color_pair(C_NORMAL)))
    total = len(lines)
    while True:
        h, w = stdscr.getmaxyx()
        visible = h - 2
        scroll = max(0, min(scroll, max(0, total - visible)))
        stdscr.erase()
        header = " diff pager  [o/q/Esc] back · j/k/↑/↓ scroll · PgUp/PgDn/[/] page · g/G top/bottom "
        try:
            stdscr.addstr(
                0,
                0,
                header[: w - 1].ljust(w - 1),
                curses.color_pair(C_SELECTED) | curses.A_BOLD,
            )
        except curses.error:
            pass
        for screen_row in range(visible):
            line_idx = scroll + screen_row
            if line_idx >= total:
                break
            text, col = lines[line_idx]
            try:
                stdscr.addstr(screen_row + 1, 0, text[: w - 1].ljust(w - 1), col)
            except curses.error:
                pass
        if total > visible:
            pct = int(100 * scroll / max(1, total - visible))
            hint = f" {scroll + 1}-{min(scroll + visible, total)}/{total}  {pct}% "
            try:
                stdscr.addstr(
                    h - 1, 0, hint[: w - 1].ljust(w - 1), curses.color_pair(C_SELECTED)
                )
            except curses.error:
                pass
        stdscr.noutrefresh()
        curses.doupdate()
        key = stdscr.getch()
        if key in (ord("o"), ord("q"), 27):
            state.diff_scroll = scroll
            break
        elif key in (curses.KEY_DOWN, ord("j")):
            scroll = min(scroll + 1, max(0, total - visible))
        elif key in (curses.KEY_UP, ord("k")):
            scroll = max(0, scroll - 1)
        elif key in (curses.KEY_NPAGE, ord("]")):
            scroll = min(scroll + visible, max(0, total - visible))
        elif key in (curses.KEY_PPAGE, ord("[")):
            scroll = max(0, scroll - visible)
        elif key == ord("g"):
            scroll = 0
        elif key == ord("G"):
            scroll = max(0, total - visible)
        elif key == curses.KEY_RESIZE:
            pass
        elif key == curses.KEY_MOUSE:
            try:
                _, _mx, _my, _, _bstate = curses.getmouse()
                if _bstate & curses.BUTTON4_PRESSED:
                    scroll = max(0, scroll - 3)
                elif _bstate & curses.BUTTON5_PRESSED:
                    scroll = min(scroll + 3, max(0, total - visible))
            except curses.error:
                pass


def draw_statusbar(stdscr, state: State, h: int, w: int):
    node = state.current_node()
    mark = state.mark_node
    if state.status_msg:
        msg = state.status_msg
        attr = (
            curses.color_pair(C_STATUS_OK)
            if state.status_ok
            else curses.color_pair(C_STATUS_ER)
        )
    else:
        parts = []
        if node:
            parts.append(f"{node.short}")
        if mark:
            parts.append(f"mark:{mark.short}")
        parts.append(
            "j/k:move  l/h:expand  E:expand-all  v:mark  Esc:clr/help  r:restore  o:fullscreen  q:quit  ?:help"
        )
        msg = "  ".join(parts)
        attr = curses.color_pair(C_DIM)
    try:
        stdscr.addstr(h - 1, 0, msg[: w - 1].ljust(w - 1), attr)
    except curses.error:
        pass


HELP_TEXT = "\n  view_git  –  worktree snapshot navigator\n  ─────────────────────────────────────────\n  j / k / ↑ / ↓    move cursor up/down\n  l / → / Space     expand node (base commit or branch)\n  h / ←             collapse node / jump to parent\n  E                 expand all nested nodes under cursor\n  o                 open diff pane in full screen pager\n  v                 set mark at cursor; if mark exists, move it here\n  Esc               close help; or clear mark\n  r                 restore cursor commit to ./_restored_<hash>/\n  q                 quit  (if mark alive → also restores cursor commit)\n  ? / F1            toggle this help\n\n  Diff pane:\n    PgUp / PgDn / [ / ]   scroll diff\n\n  Tree structure:\n    ●  base commit on main/master  (yellow)\n    ⎇  agent branch label         (cyan)  — no duplicate commit row\n    ├─ / ╰─  commit inside branch (white)\n    ◆  marked commit\n\n  Right pane shows full commit SHA for copy-paste.\n".strip().splitlines()


def draw_help(stdscr, h: int, w: int):
    bh = min(len(HELP_TEXT) + 4, h - 4)
    bw = min(64, w - 4)
    by = (h - bh) // 2
    bx = (w - bw) // 2
    try:
        box = curses.newwin(bh, bw, by, bx)
        box.bkgd(" ", curses.color_pair(C_HEADER) | curses.A_REVERSE)
        box.box()
        box.addstr(0, 2, " help ", curses.A_BOLD)
        for i, line in enumerate(HELP_TEXT[: bh - 2]):
            box.addstr(i + 1, 2, line[: bw - 4])
        box.refresh()
    except curses.error:
        pass


def _find_git_dir(repo_dir: str) -> str:
    p = Path(repo_dir).resolve()
    for candidate in [p, *p.parents]:
        git_dir = candidate / ".git"
        if git_dir.is_dir():
            return str(git_dir)
    return os.path.join(repo_dir, ".git")


def _git_mtime(repo_dir: str) -> float:
    git_dir = _find_git_dir(repo_dir)
    total = 0.0
    refs_heads = os.path.join(git_dir, "refs", "heads")
    try:
        total += os.path.getmtime(refs_heads)
    except OSError:
        pass
    try:
        for entry in os.scandir(refs_heads):
            if entry.is_file():
                total += entry.stat().st_mtime
            elif entry.is_dir():
                for sub in os.scandir(entry.path):
                    if sub.is_file():
                        total += sub.stat().st_mtime
    except OSError:
        pass
    try:
        total += os.path.getmtime(os.path.join(git_dir, "worktrees"))
    except OSError:
        pass
    try:
        total += os.path.getmtime(os.path.join(git_dir, "packed-refs"))
    except OSError:
        pass
    return total


def run(stdscr, repo_dir: str, refresh_secs: float = 0.0):
    curses.curs_set(0)
    stdscr.timeout(50)
    init_colors()
    curses.mousemask(curses.ALL_MOUSE_EVENTS)
    try:
        repo = _git.Repo(repo_dir, search_parent_directories=True)
    except _git.InvalidGitRepositoryError:
        stdscr.addstr(0, 0, f"Not a git repo: {repo_dir}")
        stdscr.getch()
        return (None, None)
    roots = build_tree(repo_dir, repo=repo)
    if not roots:
        stdscr.addstr(0, 0, "No agent/* branches found in this repo.")
        stdscr.addstr(1, 0, "Press any key to exit.")
        stdscr.getch()
        return (None, None)
    if roots:
        roots[0].expanded = True
        if roots[0].children:
            roots[0].children[0].expanded = True
    state = State(roots=roots, repo=repo, repo_dir=repo_dir)
    state.clamp_cursor()
    status_clear_counter = 0
    _tick_interval_ms = 50
    _refresh_ticks = (
        max(1, int(refresh_secs * 1000 / _tick_interval_ms)) if refresh_secs > 0 else 0
    )
    _poll_counter = 0
    _last_mtime = _git_mtime(repo_dir) if _refresh_ticks else 0.0
    while True:
        h, w = stdscr.getmaxyx()
        if h < 10 or w < 40:
            stdscr.erase()
            stdscr.addstr(0, 0, "Terminal too small.")
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                break
            continue
        tree_w = max(30, int(w * TREE_W_RATIO))
        diff_w = w - tree_w - 1
        if state.status_msg:
            status_clear_counter += 1
            if status_clear_counter > 40:
                state.status_msg = ""
                status_clear_counter = 0
        else:
            status_clear_counter = 0
        stdscr.erase()
        for row in range(h - 1):
            try:
                stdscr.addch(row, tree_w, "│", curses.color_pair(C_DIM))
            except curses.error:
                pass
        tree_win = stdscr.derwin(h - 1, tree_w, 0, 0)
        diff_win = stdscr.derwin(h - 1, diff_w, 0, tree_w + 1)
        draw_tree(tree_win, state, h - 1, tree_w)
        draw_diff(diff_win, state, h - 1, diff_w)
        draw_statusbar(stdscr, state, h, w)
        if state.show_help:
            draw_help(stdscr, h, w)
        curses.doupdate()
        stdscr.refresh()
        key = stdscr.getch()
        if key == -1:
            if _refresh_ticks:
                _poll_counter += 1
                if _poll_counter >= _refresh_ticks:
                    _poll_counter = 0
                    mtime = _git_mtime(repo_dir)
                    if mtime != _last_mtime:
                        _last_mtime = mtime
                        cur_sha = (
                            state.current_node().hexsha
                            if state.current_node()
                            else None
                        )
                        try:
                            fresh_repo = _git.Repo(
                                repo_dir, search_parent_directories=True
                            )
                            new_roots = build_tree(repo_dir, repo=fresh_repo)
                        except Exception:
                            fresh_repo = None
                            new_roots = None
                        if new_roots is not None and fresh_repo is not None:

                            def _collect_expanded(nodes):
                                shas, branches = (set(), set())
                                for n in nodes:
                                    if n.expanded:
                                        if n.kind == "branch_label" and n.branch_name:
                                            branches.add(n.branch_name)
                                        else:
                                            shas.add(n.hexsha)
                                    s2, b2 = _collect_expanded(n.children)
                                    shas |= s2
                                    branches |= b2
                                return (shas, branches)

                            expanded_shas, expanded_branches = _collect_expanded(
                                state.roots
                            )

                            def _restore_expanded(nodes):
                                for n in nodes:
                                    if n.kind == "branch_label" and n.branch_name:
                                        if n.branch_name in expanded_branches:
                                            n.expanded = True
                                    elif n.hexsha in expanded_shas:
                                        n.expanded = True
                                    _restore_expanded(n.children)

                            _restore_expanded(new_roots)
                            repo = fresh_repo
                            state.repo = fresh_repo
                            state.roots = new_roots
                            state.invalidate_visible()
                            new_vis = flatten_visible(state.roots)
                            if cur_sha:
                                for idx, n in enumerate(new_vis):
                                    if n.hexsha == cur_sha:
                                        state.cursor = idx
                                        break
                            state.clamp_cursor()
                            state.invalidate_diff()
            continue
        visible = state.visible
        node = state.current_node()
        if key in (ord("j"), curses.KEY_DOWN):
            state.cursor = min(state.cursor + 1, len(visible) - 1)
            state.diff_scroll = 0
            state.invalidate_diff()
        elif key in (ord("k"), curses.KEY_UP):
            state.cursor = max(state.cursor - 1, 0)
            state.diff_scroll = 0
            state.invalidate_diff()
        elif key in (ord("l"), curses.KEY_RIGHT, ord(" ")):
            if node and node.is_expandable and (not node.expanded):
                node.expanded = True
                state.invalidate_visible()
            elif node and node.is_expandable and node.expanded:
                state.cursor += 1
            state.clamp_cursor()
            state.invalidate_diff()
        elif key in (ord("h"), curses.KEY_LEFT):
            if node:
                if node.expanded:
                    node.expanded = False
                    state.invalidate_visible()
                else:
                    last_parent_idx = None
                    for pi in range(state.cursor - 1, -1, -1):
                        if visible[pi].depth < node.depth:
                            last_parent_idx = pi
                            break
                    if last_parent_idx is not None:
                        state.cursor = last_parent_idx
            state.clamp_cursor()
            state.invalidate_diff()
        elif key == ord("E"):
            if node:
                n = expand_all(node)
                state.invalidate_visible()
                state.clamp_cursor()
                state.invalidate_diff()
                state.status_msg = (
                    f"expanded {n} node{('s' if n != 1 else '')} under cursor"
                )
                state.status_ok = True
        elif key in (curses.KEY_PPAGE, ord("[")):
            state.diff_scroll = max(0, state.diff_scroll - h // 2)
        elif key in (curses.KEY_NPAGE, ord("]")):
            state.diff_scroll += h // 2
        elif key == curses.KEY_MOUSE:
            try:
                _, _mx, _my, _, _bstate = curses.getmouse()
                if _bstate & curses.BUTTON4_PRESSED:
                    state.diff_scroll = max(0, state.diff_scroll - 3)
                elif _bstate & curses.BUTTON5_PRESSED:
                    state.diff_scroll += 3
            except curses.error:
                pass
        elif key == ord("v"):
            if node:
                state.mark_node = node
                state.status_msg = f"mark set: {node.short}"
                state.status_ok = True
                state.diff_scroll = 0
                state.invalidate_diff()
        elif key == 27:
            if state.show_help:
                state.show_help = False
            elif state.mark_node:
                state.mark_node = None
                state.status_msg = "mark cleared"
                state.status_ok = True
                state.diff_scroll = 0
                state.invalidate_diff()
        elif key == ord("o"):
            if node:
                stdscr.clearok(True)
                diff_pager(stdscr, state)
                stdscr.clearok(True)
        elif key == ord("r"):
            if node:
                ok, result = restore_snapshot(repo, node, os.getcwd())
                if ok:
                    state.status_msg = f"restored → {result}"
                    state.status_ok = True
                else:
                    state.status_msg = f"restore failed: {result}"
                    state.status_ok = False
        elif key in (ord("?"), curses.KEY_F1):
            state.show_help = not state.show_help
        elif key in (ord("q"), ord("Q")):
            if state.mark_node:
                ok, result = restore_snapshot(repo, state.mark_node, os.getcwd())
                state.quit_snapshot_path = result if ok else None
            break
    return (
        state.quit_snapshot_path,
        state.mark_node.hexsha if state.mark_node else None,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="view_git – worktree snapshot navigator"
    )
    parser.add_argument("repo_dir", nargs="?", default=os.getcwd())
    parser.add_argument(
        "--refresh",
        "-R",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="poll .git for changes every N seconds and rebuild tree (0 = disabled)",
    )
    args = parser.parse_args()
    repo_dir = os.path.abspath(args.repo_dir)
    os.environ.setdefault("ESCDELAY", "25")
    result = curses.wrapper(run, repo_dir, args.refresh)
    if result:
        snapshot_path, mark_sha = result
        if snapshot_path:
            print(f"✓ snapshot restored to: {snapshot_path}")
        if mark_sha:
            print(f"\n✓ marked commit: {mark_sha}")
    print("bye.")


if __name__ == "__main__":
    main()

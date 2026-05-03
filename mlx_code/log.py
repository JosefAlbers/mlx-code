# Copyright 2026 J Joe

import curses
import json
import sys
from pathlib import Path
import argparse

LEVEL_COLORS = {
    "DEBUG":    1,
    "INFO":     2,
    "WARNING":  3,
    "WARN":     3,
    "ERROR":    4,
    "CRITICAL": 5,
}

LEVEL_NUMERIC = {
    "DEBUG":    10,
    "INFO":     20,
    "WARNING":  30,
    "WARN":     30,
    "ERROR":    40,
    "CRITICAL": 50,
}

def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN,    -1)  
    curses.init_pair(2, curses.COLOR_GREEN,   -1)  
    curses.init_pair(3, curses.COLOR_YELLOW,  -1)  
    curses.init_pair(4, curses.COLOR_RED,     -1)  
    curses.init_pair(5, curses.COLOR_MAGENTA, -1)  
    curses.init_pair(6, curses.COLOR_BLACK,   curses.COLOR_WHITE)   
    curses.init_pair(7, curses.COLOR_WHITE,   -1)  
    curses.init_pair(8, curses.COLOR_BLACK,   curses.COLOR_CYAN)    
    curses.init_pair(9, curses.COLOR_BLACK,   curses.COLOR_YELLOW)  
    curses.init_pair(10, curses.COLOR_WHITE,  -1)    


def load_logs(path: str) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                entries.append({
                    "timestamp": "?",
                    "level":     "ERROR",
                    "message":   f"[parse error on line {lineno}] {line}",
                    "logger":    "",
                    "file":      "",
                    "function":  "",
                    "line":      "",
                })
    return entries


def parse_filter(raw: str) -> dict:
    clauses = {}
    for part in raw.split(";"):
        part = part.strip()
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if not val:
            continue
        if key == "lvl":
            try:
                clauses["lvl"] = int(val)
            except ValueError:
                pass
        elif key in ("level",):
            vals = [v.strip().lower() for v in val.split(",") if v.strip()]
            clauses.setdefault("level", []).extend(vals)
        elif key in ("fn", "func", "function"):
            vals = [v.strip().lower() for v in val.split(",") if v.strip()]
            clauses.setdefault("fn", []).extend(vals)
        elif key == "file":
            vals = [v.strip().lower() for v in val.split(",") if v.strip()]
            clauses.setdefault("file", []).extend(vals)
        elif key == "msg":
            clauses["msg"] = val.lower()
    return clauses


def entry_matches(entry: dict, clauses: dict) -> bool:
    level_str = entry.get("level", "").upper()

    if "lvl" in clauses:
        entry_numeric = LEVEL_NUMERIC.get(level_str, 0)
        if entry_numeric < clauses["lvl"]:
            return False

    if "level" in clauses:
        level_lower = level_str.lower()
        if not any(pat in level_lower for pat in clauses["level"]):
            return False

    if "file" in clauses:
        file_lower = Path(entry.get("file", "")).name.lower()
        if not any(pat in file_lower for pat in clauses["file"]):
            return False

    if "fn" in clauses:
        fn_lower = entry.get("function", "").lower()
        if not any(pat in fn_lower for pat in clauses["fn"]):
            return False

    if "msg" in clauses:
        msg_lower = entry.get("message", "").lower()
        if clauses["msg"] not in msg_lower:
            return False

    return True


def apply_filter(entries: list[dict], raw: str) -> list[dict]:
    raw = raw.strip()
    if not raw:
        return entries
    clauses = parse_filter(raw)
    if not clauses:
        return entries
    return [e for e in entries if entry_matches(e, clauses)]


def short_ts(ts: str) -> str:
    if "T" in ts:
        time_part = ts.split("T")[1]
        return time_part[:8]
    return ts[:8]

def short_file(path: str) -> str:
    return Path(path).name if path else ""

def truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s

def draw_border_box(win, title: str = ""):
    win.box()
    if title:
        label = f" {title} "
        win.addstr(0, 2, label, curses.color_pair(7) | curses.A_BOLD)

def wrap_text(text: str, width: int) -> list[str]:
    lines = []
    for paragraph in text.splitlines():
        if not paragraph:
            lines.append("")
            continue
        while len(paragraph) > width:
            lines.append(paragraph[:width])
            paragraph = paragraph[width:]
        lines.append(paragraph)
    return lines


def render_detail(panel, entry: dict):
    panel.erase()
    draw_border_box(panel, "detail")
    h, w = panel.getmaxyx()
    inner_w = w - 4
    row = 1

    def put(label, value, color=7):
        nonlocal row
        if row >= h - 1:
            return
        line = f"{label:<11}: {value}"
        try:
            panel.addstr(row, 2, truncate(line, inner_w),
                         curses.color_pair(color))
        except curses.error:
            pass
        row += 1

    level = entry.get("level", "?")
    color = LEVEL_COLORS.get(level, 7)

    put("time",     entry.get("timestamp", ""),    7)
    put("level",    level,                          color)
    put("logger",   entry.get("logger", ""),        7)
    put("file",     entry.get("file", ""),          7)
    put("function", entry.get("function", ""),      7)
    put("line",     str(entry.get("line", "")),     7)

    extra = entry.get("extra", {})
    if extra:
        row += 1
        if row < h - 1:
            try:
                panel.addstr(row, 2, "── extra ──", curses.color_pair(7))
            except curses.error:
                pass
            row += 1
        for k, v in extra.items():
            put(k[:11], str(v), 7)

    row += 1
    if row < h - 1:
        try:
            panel.addstr(row, 2, "── message ──", curses.color_pair(7))
        except curses.error:
            pass
        row += 1

    msg = entry.get("message", "")
    for line in wrap_text(msg, inner_w):
        if row >= h - 1:
            break
        try:
            panel.addstr(row, 2, line, curses.color_pair(2))
        except curses.error:
            pass
        row += 1

    exc = entry.get("exception")
    if exc:
        row += 1
        if row < h - 1:
            try:
                panel.addstr(row, 2, "── exception ──", curses.color_pair(4))
            except curses.error:
                pass
            row += 1
        for line in wrap_text(exc, inner_w):
            if row >= h - 1:
                break
            try:
                panel.addstr(row, 2, line, curses.color_pair(4))
            except curses.error:
                pass
            row += 1

    panel.noutrefresh()


def pager(stdscr, entry: dict):
    curses.curs_set(0)

    def build_lines(w: int) -> list[tuple[str, int]]:
        lines = []
        level = entry.get("level", "?").upper()
        color = LEVEL_COLORS.get(level, 7)

        def section(title):
            lines.append((f"  {title}", 7))

        def field(label, value, col=7):
            lines.append((f"  {label:<11}: {value}", col))

        field("time",      entry.get("timestamp", ""))
        field("level",     level, color)
        field("logger",    entry.get("logger", ""))
        field("file",      entry.get("file", ""))
        field("function",  entry.get("function", ""))
        field("line",      str(entry.get("line", "")))

        extra = entry.get("extra", {})
        if extra:
            lines.append(("", 7))
            section("── extra ──")
            for k, v in extra.items():
                for ln in wrap_text(f"  {k:<11}: {v}", w - 4):
                    lines.append((ln, 7))

        lines.append(("", 7))
        section("── message ──")
        for ln in wrap_text(entry.get("message", ""), w - 4):
            lines.append(("  " + ln, 2))

        exc = entry.get("exception")
        if exc:
            lines.append(("", 7))
            section("── exception ──")
            for ln in wrap_text(exc, w - 4):
                lines.append(("  " + ln, 4))

        return lines

    scroll = 0

    while True:
        h, w = stdscr.getmaxyx()
        lines = build_lines(w)
        total = len(lines)
        visible = h - 2  

        scroll = max(0, min(scroll, max(0, total - visible)))

        stdscr.erase()

        header = " detail  [o/q/Esc] back · j/k scroll · g/G top/bottom "
        try:
            stdscr.addstr(0, 0, header[:w - 1].ljust(w - 1),
                          curses.color_pair(8) | curses.A_BOLD)
        except curses.error:
            pass

        for screen_row in range(visible):
            line_idx = scroll + screen_row
            if line_idx >= total:
                break
            text, col = lines[line_idx]
            try:
                stdscr.addstr(screen_row + 1, 0,
                              text[:w - 1].ljust(w - 1),
                              curses.color_pair(col))
            except curses.error:
                pass

        if total > visible:
            pct = int(100 * scroll / max(1, total - visible))
            hint = f" {scroll + 1}-{min(scroll + visible, total)}/{total}  {pct}% "
            try:
                stdscr.addstr(h - 1, 0, hint[:w - 1].ljust(w - 1),
                              curses.color_pair(9))
            except curses.error:
                pass

        stdscr.noutrefresh()
        curses.doupdate()

        key = stdscr.getch()

        if key in (ord("o"), ord("q"), 27):
            break
        elif key in (curses.KEY_DOWN, ord("j")):
            scroll = min(scroll + 1, max(0, total - visible))
        elif key in (curses.KEY_UP, ord("k")):
            scroll = max(0, scroll - 1)
        elif key == curses.KEY_NPAGE:
            scroll = min(scroll + visible, max(0, total - visible))
        elif key == curses.KEY_PPAGE:
            scroll = max(0, scroll - visible)
        elif key == ord("g"):
            scroll = 0
        elif key == ord("G"):
            scroll = max(0, total - visible)
        elif key == curses.KEY_RESIZE:
            pass  


def _related_key(entry: dict):
    return (
        entry.get("level", "").upper(),
        Path(entry.get("file", "")).name,
        entry.get("function", ""),
    )


def render_list(win, entries: list[dict], cursor: int, scroll: int,
                col_widths: dict, highlight_on: bool = False,
                related_keys: list | None = None):
    win.erase()
    draw_border_box(win, "logs")
    h, w = win.getmaxyx()
    visible = h - 2  

    cursor_key = related_keys[cursor] if (related_keys and entries) else None

    ts_w   = col_widths["ts"]
    lvl_w  = col_widths["lvl"]
    file_w = col_widths["file"]
    fn_w   = col_widths["fn"]
    msg_w  = max(w - ts_w - lvl_w - file_w - fn_w - 10, 12)

    header = (
        f"{'TIME':<{ts_w}}  "
        f"{'LVL':<{lvl_w}}  "
        f"{'FILE':<{file_w}}  "
        f"{'FUNC':<{fn_w}}  "
        f"{'MESSAGE':<{msg_w}}"
    )
    try:
        win.addstr(1, 2, truncate(header, w - 4),
                   curses.color_pair(8) | curses.A_BOLD)
    except curses.error:
        pass

    for idx in range(visible - 1):
        entry_idx = scroll + idx
        if entry_idx >= len(entries):
            break
        entry   = entries[entry_idx]
        is_sel  = entry_idx == cursor
        row     = idx + 2

        level  = entry.get("level", "?").upper()
        color  = LEVEL_COLORS.get(level, 7)
        ts     = short_ts(entry.get("timestamp", ""))
        f_name = short_file(entry.get("file", ""))
        fn     = entry.get("function", "")
        msg    = entry.get("message", "").replace("\n", " ")

        line = (
            f"{ts:<{ts_w}}  "
            f"{level:<{lvl_w}}  "
            f"{f_name:<{file_w}}  "
            f"{fn:<{fn_w}}  "
            f"{truncate(msg, msg_w)}"
        )
        line = truncate(line, w - 4)

        is_related = (
            highlight_on
            and not is_sel
            and cursor_key is not None
            and (related_keys[entry_idx] if related_keys else _related_key(entry)) == cursor_key
        )

        if is_sel:
            try:
                win.addstr(row, 2, f"{line:<{w-4}}",
                           curses.color_pair(6) | curses.A_BOLD)
            except curses.error:
                pass
        elif is_related:
            try:
                win.addstr(row, 2, f"{line:<{w-4}}",
                           curses.color_pair(10))
            except curses.error:
                pass
        else:
            try:
                win.addstr(row, 2, line, curses.color_pair(color))
            except curses.error:
                pass

    win.noutrefresh()

def render_status(stdscr, cursor: int, total: int, all_total: int,
                  log_file: str, active_filter: str):
    h, w = stdscr.getmaxyx()

    filter_indicator = f"  filter: {active_filter}" if active_filter else ""
    count_str = f"{cursor + 1}/{total}" if total else "0/0"
    if active_filter and total != all_total:
        count_str += f" (of {all_total})"

    status = (
        f"  {log_file}  │  {count_str}{filter_indicator}  │  "
        "↑/k ↓/j · PgUp/PgDn · g/G · n/N related · * highlight · o open · ; filter · q quit  "
    )
    try:
        stdscr.addstr(h - 1, 0, truncate(status, w - 1),
                      curses.color_pair(8))
    except curses.error:
        pass

def render_filter_bar(stdscr, filter_buf: str):
    h, w = stdscr.getmaxyx()
    prompt = " filter> "
    bar = f"{prompt}{filter_buf}"
    bar = bar[:w - 1].ljust(w - 1)
    try:
        stdscr.addstr(h - 1, 0, bar, curses.color_pair(9) | curses.A_BOLD)
        cursor_x = min(len(prompt) + len(filter_buf), w - 2)
        stdscr.move(h - 1, cursor_x)
    except curses.error:
        pass


def read_filter(stdscr, initial: str = "") -> str | None:
    curses.curs_set(1)
    buf = initial

    while True:
        render_filter_bar(stdscr, buf)
        stdscr.noutrefresh()
        curses.doupdate()

        key = stdscr.getch()

        if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            curses.curs_set(0)
            return buf
        elif key == 27: 
            curses.curs_set(0)
            return None
        elif key in (curses.KEY_BACKSPACE, 127, ord("\b")):
            buf = buf[:-1]
        elif 32 <= key < 127: 
            buf += chr(key)

def tui(stdscr, entries, log_file, initial_filter="", initial_visible=None):
    init_colors()
    curses.curs_set(0)
    stdscr.keypad(True)
    all_entries     = entries
    active_filter   = initial_filter
    visible_entries = initial_visible if initial_visible is not None else entries

    cursor = 0
    scroll = 0
    highlight_on = False

    stdscr.clear()
    stdscr.noutrefresh()
    curses.doupdate()

    def _make_windows(h, w):
        list_w   = max(int(w * 0.60), 40)
        detail_w = max(w - list_w, 30)
        pane_h   = h - 1
        col_w = {
            "ts":   8,
            "lvl":  8,
            "file": min(16, max(8, list_w // 6)),
            "fn":   min(16, max(8, list_w // 6)),
        }
        lw = curses.newwin(pane_h, list_w,   0, 0)
        dw = curses.newwin(pane_h, detail_w, 0, list_w)
        return lw, dw, list_w, detail_w, pane_h, col_w

    h, w = stdscr.getmaxyx()
    list_win, detail_win, list_w, detail_w, pane_h, col_widths = _make_windows(h, w)

    related_keys = [_related_key(e) for e in visible_entries]

    while True:
        visible_rows = pane_h - 3 

        if not visible_entries:
            cursor = 0
            scroll = 0
            current = {}
        else:
            cursor  = max(0, min(cursor, len(visible_entries) - 1))
            current = visible_entries[cursor]

        if cursor < scroll:
            scroll = cursor
        elif cursor >= scroll + visible_rows:
            scroll = cursor - visible_rows + 1

        current_key = related_keys[cursor] if visible_entries else None

        render_list(list_win, visible_entries, cursor, scroll, col_widths,
                    highlight_on, related_keys)
        render_detail(detail_win, current)
        render_status(stdscr, cursor, len(visible_entries),
                      len(all_entries), log_file, active_filter)
        stdscr.noutrefresh()
        curses.doupdate()

        key = stdscr.getch()

        if key in (ord("q"), 27):
            break
        elif key == ord("o") and visible_entries:
            stdscr.clearok(True)
            pager(stdscr, current)
            stdscr.clearok(True)
        elif key == ord(";"):
            result = read_filter(stdscr, active_filter)
            if result is not None:
                active_filter   = result.strip()
                visible_entries = apply_filter(all_entries, active_filter)
                related_keys    = [_related_key(e) for e in visible_entries]
                cursor = 0
                scroll = 0
            stdscr.clearok(True)
        elif key == ord("*"): 
            highlight_on = not highlight_on
        elif key == ord("n") and visible_entries:
            for i in range(cursor + 1, len(visible_entries)):
                if related_keys[i] == current_key:
                    cursor = i
                    break
        elif key == ord("N") and visible_entries:
            for i in range(cursor - 1, -1, -1):
                if related_keys[i] == current_key:
                    cursor = i
                    break
        elif key in (curses.KEY_UP, ord("k")):
            cursor = max(0, cursor - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            cursor = min(max(len(visible_entries) - 1, 0), cursor + 1)
        elif key in (curses.KEY_PPAGE,):
            cursor = max(0, cursor - 10)
        elif key in (curses.KEY_NPAGE,): 
            cursor = min(max(len(visible_entries) - 1, 0), cursor + 10)
        elif key == ord("g"):
            cursor = 0
        elif key == ord("G"):
            cursor = max(len(visible_entries) - 1, 0)
        elif key == curses.KEY_RESIZE:
            h, w = stdscr.getmaxyx()
            list_win, detail_win, list_w, detail_w, pane_h, col_widths = _make_windows(h, w)
            stdscr.clearok(True)

def main():
    parser = argparse.ArgumentParser(
        description="TUI viewer for JSON log files"
    )

    parser.add_argument(
        "logfile",
        nargs="?",
        default="log.json",
        help="Path to log file (default: log.json)"
    )

    parser.add_argument(
        "-f", "--filter",
        default=f"lvl:10;file:main,pie",
        help="Initial filter string (same syntax as in UI)"
    )

    args = parser.parse_args()

    log_path = args.logfile
    cli_filter = args.filter

    try:
        logs = load_logs(log_path)
    except FileNotFoundError:
        print(f"Error: file not found: {log_path}")
        sys.exit(1)

    if not logs:
        print("No log entries found.")
        sys.exit(0)

    initial_entries = apply_filter(logs, cli_filter) if cli_filter else logs

    curses.wrapper(tui, logs, log_path, cli_filter, initial_entries)

if __name__ == "__main__":
    main()

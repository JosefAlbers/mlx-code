import sys
import io
import unittest
from unittest.mock import patch
from mlx_code.repl import _read_input_unix


def make_stdin(chars: str):
    raw = chars.replace("\n", "\r")
    return iter(raw)


class FakeStdin:
    def __init__(self, chars):
        self._iter = iter(chars.replace("\n", "\r"))

    def fileno(self):
        return 0

    def read(self, n=1):
        try:
            return next(self._iter)
        except StopIteration:
            return "\x04"


class TestReadInputUnix(unittest.TestCase):
    def _run(self, input_chars):
        fake_stdin = FakeStdin(input_chars)
        fake_stdout = io.StringIO()
        with (
            patch("sys.stdin", fake_stdin),
            patch("sys.stdout", fake_stdout),
            patch("mlx_code.repl.tty.setraw", lambda fd: None),
            patch("mlx_code.repl.termios.tcgetattr", lambda fd: None),
            patch("mlx_code.repl.termios.tcsetattr", lambda fd, when, old: None),
        ):
            return _read_input_unix("")

    def test_simple(self):
        self.assertEqual(self._run("hello\n"), "hello")

    def test_continuation(self):
        self.assertEqual(self._run("hello\\\nworld\n"), "hello\nworld")

    def test_backspace(self):
        self.assertEqual(self._run("helo\x7flo\n"), "hello")

    def test_backspace_empty(self):
        self.assertEqual(self._run("\x7fhi\n"), "hi")

    def test_paste_with_newlines(self):
        paste = "\x1b[200~line1\nline2\x1b[201~\n"
        self.assertEqual(self._run(paste), "line1\nline2")

    def test_ctrl_c_empty_raises(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run("\x03")

    def test_ctrl_c_mid_input_resets(self):
        self.assertEqual(self._run("hello\x03world\n"), "world")

    def test_ctrl_d_raises(self):
        with self.assertRaises(EOFError):
            self._run("\x04")

    def test_strip(self):
        self.assertEqual(self._run("  hi  \n"), "hi")

    def test_empty_submit(self):
        self.assertEqual(self._run("\n"), "")


if __name__ == "__main__":
    unittest.main()

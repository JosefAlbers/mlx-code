import asyncio
import contextlib
import io
import json
import re
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from mlx_code.repl import Agent, Tab, _stream_to_stdout, _truncate

class EventStream:

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._result: dict | None = None

    def push(self, event: dict) -> None:
        self._queue.put_nowait(event)

    def finish(self, result: dict) -> None:
        self._result = result
        self._queue.put_nowait(None)

    def _attach(self, task: asyncio.Task) -> None:
        self._task = task

    async def result(self) -> dict:
        if self._result is None:
            async for _ in self:
                pass
        assert self._result is not None
        return self._result

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

def _message_text(message: dict) -> str:
    content = message.get('content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join((str(block.get('text', '')) for block in content if isinstance(block, dict) and block.get('type') == 'text'))
    return str(content)

class EchoChat:
    model = 'echo'
    base_url = 'echo://local'

    async def stream(self, messages: list[dict], system: str, tools: list):
        es = EventStream()
        prompt = ''
        for message in reversed(messages):
            if message.get('role') == 'user':
                prompt = _message_text(message)
                break

        async def _run() -> None:
            msg = {'role': 'assistant', 'content': [], 'stop_reason': 'stop', 'error_message': None, 'usage': {'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0}, 'timestamp': int(time.time() * 1000)}
            text = f'echo: {prompt}'
            msg['content'].append({'type': 'text', 'text': text})
            es.push({'type': 'thinking_delta', 'payload': {'delta': 'hidden thinking'}})
            for chunk in re.findall('.{1,24}', text, flags=re.DOTALL) or ['']:
                await asyncio.sleep(0.01)
                es.push({'type': 'text_delta', 'payload': {'delta': chunk}})
            es.finish(msg)
        es._attach(asyncio.create_task(_run()))
        return es

def start_mock_llm_server(host: str='127.0.0.1', port: int=0):

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, format, *args):
            pass

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get('Content-Length') or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b'{}')
            except json.JSONDecodeError:
                payload = {}
            prompt = ''
            for message in reversed(payload.get('messages') or []):
                if message.get('role') == 'user':
                    prompt = _message_text(message)
                    break
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for chunk in re.findall('.{1,24}', f'echo: {prompt}', flags=re.DOTALL) or ['']:
                data = {'choices': [{'delta': {'content': chunk}}]}
                self.wfile.write(f'data: {json.dumps(data)}\n\n'.encode())
                self.wfile.flush()
            self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return (server, f'http://{server.server_address[0]}:{server.server_address[1]}')

class TestReplHelpers(unittest.TestCase):

    def test_truncate_collapses_whitespace_and_adds_ellipsis(self):
        self.assertEqual(_truncate('hello\nworld', 50), 'hello world')
        self.assertEqual(_truncate('abcdefghijklmnopqrstuvwxyz', 10), 'abcdefghi…')

    def test_tab_apply_event_text_delta_appends_to_stream_msg(self):
        tab = Tab('main', Agent(api=EchoChat(), tool_names=[]))
        tab.apply_event({'type': 'agent_start', 'payload': {}})
        tab.apply_event({'type': 'text_delta', 'payload': {'delta': 'Hello '}})
        tab.apply_event({'type': 'text_delta', 'payload': {'delta': 'World'}})
        self.assertIsNotNone(tab._stream_msg)
        text_blocks = [b for b in tab._stream_msg['content'] if b.get('type') == 'text']
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(text_blocks[0]['text'], 'Hello World')

    def test_tab_apply_event_thinking_delta_creates_thinking_block(self):
        tab = Tab('main', Agent(api=EchoChat(), tool_names=[]))
        tab.apply_event({'type': 'agent_start', 'payload': {}})
        tab.apply_event({'type': 'thinking_delta', 'payload': {'delta': 'hmm'}})
        thinking_blocks = [b for b in tab._stream_msg['content'] if b.get('type') == 'thinking']
        self.assertEqual(len(thinking_blocks), 1)
        self.assertEqual(thinking_blocks[0]['text'], 'hmm')

    def test_tab_apply_event_agent_end_clears_stream_msg(self):
        tab = Tab('main', Agent(api=EchoChat(), tool_names=[]))
        tab.apply_event({'type': 'agent_start', 'payload': {}})
        self.assertIsNotNone(tab._stream_msg)
        tab.apply_event({'type': 'agent_end', 'payload': {}})
        self.assertIsNone(tab._stream_msg)
        self.assertEqual(tab.status, 'idle')

    def test_tab_error_tracking(self):
        tab = Tab('main', Agent(api=EchoChat(), tool_names=[]))
        tab.add_error('oops')
        self.assertEqual(tab.last_error, 'oops')
        self.assertEqual(len(tab.errors), 1)
        tab.clear_errors()
        self.assertEqual(tab.last_error, '')
        self.assertEqual(len(tab.errors), 0)

    def test_tab_filter_xml_suppresses_anreCommand_tags(self):
        tab = Tab('main', Agent(api=EchoChat(), tool_names=[]))
        result = tab._filter_xml('before <anreCommand hidden>after</anreCommand> end')
        self.assertEqual(result, 'before  end')

    def test_tab_filter_xml_handles_partial_start_tag_across_calls(self):
        tab = Tab('main', Agent(api=EchoChat(), tool_names=[]))
        result1 = tab._filter_xml('before <anre')
        self.assertEqual(result1, 'before ')
        self.assertEqual(tab._xml_tail, '<anre')
        result2 = tab._filter_xml('Command hidden>after</anreCommand> end')
        self.assertEqual(result2, ' end')

    def test_tab_apply_event_error_adds_to_errors_and_stream(self):
        tab = Tab('main', Agent(api=EchoChat(), tool_names=[]))
        tab.apply_event({'type': 'agent_start', 'payload': {}})
        tab.apply_event({'type': 'error', 'payload': {'error': 'something broke'}})
        self.assertEqual(tab.last_error, 'something broke')
        error_blocks = [b for b in tab._stream_msg['content'] if b.get('is_error')]
        self.assertTrue(any(('something broke' in b.get('text', '') for b in error_blocks)))

    def test_tab_apply_event_tool_start_and_end(self):
        tab = Tab('main', Agent(api=EchoChat(), tool_names=[]))
        tab.apply_event({'type': 'agent_start', 'payload': {}})
        tab.apply_event({'type': 'tool_start', 'payload': {'name': 'Read', 'args': {'path': '/tmp/f.txt'}}})
        tool_calls = [b for b in tab._stream_msg['content'] if b.get('type') == 'toolCall']
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]['name'], 'Read')
        tab.apply_event({'type': 'tool_end', 'payload': {'name': 'Read', 'is_error': False, 'result': {'content': [{'type': 'text', 'text': 'file contents'}]}}})
        text_blocks = [b for b in tab._stream_msg['content'] if b.get('type') == 'text' and (not b.get('is_error'))]
        self.assertTrue(any(('file contents' in b.get('text', '') for b in text_blocks)))

    def test_tab_apply_event_tool_end_error_appends_failure_text(self):
        tab = Tab('main', Agent(api=EchoChat(), tool_names=[]))
        tab.apply_event({'type': 'agent_start', 'payload': {}})
        tab.apply_event({'type': 'tool_end', 'payload': {'name': 'Write', 'is_error': True, 'result': None}})
        error_blocks = [b for b in tab._stream_msg['content'] if b.get('is_error')]
        self.assertTrue(any(('Write failed' in b.get('text', '') for b in error_blocks)))

    def test_echo_chat_outputs_final_text_only_to_stdout(self):

        async def run():
            agent = Agent(api=EchoChat(), tool_names=[])
            fake_stdout = io.StringIO()
            with contextlib.redirect_stdout(fake_stdout):
                await _stream_to_stdout(agent, 'hello tui')
            return fake_stdout.getvalue()
        output = asyncio.run(run())
        self.assertIn('echo: hello tui', output)
        self.assertNotIn('hidden thinking', output)

    def test_noninteractive_repl_reads_stdin_once(self):

        async def run():
            from mlx_code.repl import repl
            agent = Agent(api=EchoChat(), tool_names=[])
            fake_stdin = io.StringIO('piped input')
            fake_stdout = io.StringIO()
            with patch('sys.stdin', fake_stdin), patch('sys.stdout', fake_stdout):
                await repl(agent)
            return fake_stdout.getvalue()
        self.assertIn('echo: piped input', asyncio.run(run()))
if __name__ == '__main__':
    unittest.main()
import os
import sys

class StreamLogger:

    def __init__(self, agent, fp, depth=0, name='master-agent'):
        self.depth = depth
        self.name = name
        self.turn = 0
        self.fp = fp
        self._pending_header = True
        self._at_line_start = True
        agent.subscribe(self._on_event)

    async def _on_event(self, event):
        et, p = (event['type'], event['payload'])
        if et == 'turn_start':
            self._pending_header = True
        elif et == 'turn_end':
            self.turn += 1
        elif et in ('text_delta', 'thinking_delta'):
            delta = p.get('delta', '')
            self._write(delta)

    def _line_prefix(self):
        if self.depth == 0:
            return '  ' * self.depth
        return '  ' * self.depth + ' | '

    def _write(self, text):
        if not text:
            return
        if self._pending_header:
            indent = '  ' * self.depth
            connector = '└ ' if self.depth > 0 else ''
            self.fp.write(f'\n\n{indent}{connector}({self.name}:{self.turn}) ')
            self._pending_header = False
            self._at_line_start = False
        parts = text.split('\n')
        for i, part in enumerate(parts):
            if i > 0:
                self.fp.write('\n')
                self._at_line_start = True
            if self._at_line_start and part:
                self.fp.write(self._line_prefix())
                self._at_line_start = False
            self.fp.write(part)
        try:
            self.fp.flush()
        except:
            pass

    @classmethod
    def attach_to_child(cls, child_agent, parent_ctx, tool_name='sub'):
        fp = parent_ctx.get('_stream_log_fp')
        if fp is None:
            return
        depth = parent_ctx.get('_stream_log_depth', 0) + 1
        logger = cls(child_agent, fp, depth=depth, name=tool_name)
        child_agent.ctx['_stream_log_fp'] = fp
        child_agent.ctx['_stream_log_depth'] = depth
        return logger
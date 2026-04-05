# mlx-code: Local Coding Agent

[![Link](https://raw.githubusercontent.com/JosefAlbers/mlx-code/main/assets/mlx-code.gif)](https://youtu.be/Rba-uTsYuXg)

<details><summary>Click to expand</summary><pre>

    pie REPL  •  type /help for commands, Ctrl-C or 'exit' to quit.

    You: hi
    Assistant: The user just said "hi" which is a simple greeting. I should respond with a friendly greeting and offer to help them with something.

    Hello! How can I help you today?

    You: wassup
    Assistant: The user is just saying "wassup" which is a casual greeting. I should respond in a friendly, casual way since they're using informal language.

    Not much, just here and ready to help! What's up with you?

    You: explain how output is parsed in mlx-code.py
    Assistant: The user wants me to explain how output is parsed in a file called mlx-code.py. I should first read this file to understand its structure and how it handles output parsing.

    <tool_call>
    <function=Read>
    <parameter=path>
    mlx-code.py
    </parameter>
    </function>
    </tool_call>

    [Read] {"path": "mlx-code.py"}

    Now I have a good understanding of the file. Let me analyze how output is parsed in this file. The key function for parsing output appears to be the `decode` function, which is called in the `do_POST` method of the HTTP handler.

    Let me trace through the output parsing flow:

    1. The `do_POST` method receives a POST request
    2. It calls `encode()` to prepare the prompt
    3. It calls `generate()` to generate tokens
    4. It calls `decode()` to parse the generated tokens into blocks
    5. It calls `blocks_to_sse()` to convert blocks to SSE format
    6. The SSE data is sent back to the client

    The key parsing happens in the `decode()` function. Let me explain this in detail.

    Based on my analysis of `mlx-code.py`, here's how output is parsed:

    ## Output Parsing Flow

    The output parsing happens primarily in the **`decode()`** function, which is called after model generation in the HTTP handler's `do_POST` method.

    ### Key Function: `decode()`

    ```python
    def decode(raw_text, tokenizer, parse_think, single_think=False):
    ```

    This function parses the raw model output into structured blocks. Here's how it works:

    ### 1. **Escape Special Characters**
    ```python
    def escape(text):
        def repl(match):
            inner = match.group(1)
            inner = inner.replace('<', '‹').replace('>', '›')
            return f'`{inner}`'
        return re.sub(r'`([^\n`]*)`', repl, text)
    ```
    Escapes backticks to prevent regex conflicts.

    ### 2. **Handle "Thinking" vs "Response" Order**
    ```python
    raw_text = '' + raw_text if (c := raw_text.find('')) != -1 and ((o := raw_text.find('')) == -1 or c < o) else raw_text
    ```
    Ensures thinking blocks come before response blocks.

    ### 3. **Split into Parts**
    ```python
    if parse_think:
        parts = re.split(r'(<think>.*?)', raw_text, flags=re.DOTALL, maxsplit=1 if single_think else 0)
    else:
        parts = [raw_text]
    ```
    Splits text into thinking and non-thinking parts.

    ### 4. **Parse Blocks**
    For each part, it creates blocks with different types:

    | Block Type | Pattern | Description |
    |------------|---------|-------------|
    | `thinking` | Starts with `‹think›` and ends with `‹/think›` | Model's internal reasoning |
    | `text` | Regular text content | Final output text |
    | `tool_use` | XML tags `‹tool_call›...‹/tool_call›` | Tool/function calls |

    ### 5. **Tool Call Extraction**
    ```python
    tool_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    for match in tool_pattern.finditer(part):
        content = match.group(1).strip()
        fn_match = re.search(r"<function=([^\s>]+)>", content)
        params = re.findall(r"<parameter=([^\s>]+)>\s*(.*?)\s*</parameter>", content, re.DOTALL)
    ```
    Extracts function name and parameters from XML-wrapped tool calls.

    ### 6. **Determine Stop Reason**
    ```python
    stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    ```

    ## Output Format: SSE (Server-Sent Events)

    The parsed blocks are converted to SSE format via `blocks_to_sse()`:

    ```python
    def blocks_to_sse(blocks, msg_id, in_tokens, out_tokens, stop_reason='end_turn'):
        def event(name: str, data: dict) -> bytes:
            return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
    ```

    ### SSE Events Generated:
    - `message_start` - Message header
    - `content_block_start` - Start of a content block
    - `content_block_delta` - Content delta (text/thinking/tool)
    - `content_block_stop` - End of a content block
    - `message_delta` - Message delta with stop reason
    - `message_stop` - Message stop marker

    ## HTTP Response

    The SSE data is sent with headers:
    ```python
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Cache-Control", "no-cache")
    ```

    ## Summary

    The parsing flow is:
    1. **Generate** raw tokens from the model
    2. **Decode** tokens into structured blocks (text, thinking, tool_use)
    3. **Convert** blocks to SSE format
    4. **Stream** SSE events to the client

    This allows streaming output with support for thinking traces and tool calls.

</pre></details><br>

### Quick Start

Install via pip and launch the standalone agent immediately:

```bash
pip install mlx-code
mlxcode # Runs standalone mode by default
```

### Options

You can customize the model, server, and behavior using command-line flags:

```bash
mlxcode [options] [-- harness options]
```

- **Standalone (Default):** Uses the built-in `pie` agent (python port of [pi](https://github.com/badlogic/pi-mono))
- **Harness Mode:** Use `--harness <command>` to proxy an external agent (e.g., `claude`).

### Credits

- `main`: Built on [MLX](https://github.com/ml-explore/mlx) and [MLX LM](https://github.com/ml-explore/mlx-lm) by Apple.
- `pie`: Adapted from [pi](https://github.com/badlogic/pi-mono) by Mario Zechner (MIT License).

### Licence

Apache License 2.0 — see LICENSE for details.


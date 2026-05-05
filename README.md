# mlx-code

[![Link](https://raw.githubusercontent.com/JosefAlbers/mlx-code/main/assets/mlx-code.gif)](https://youtu.be/bizPhrHL1_w)

A lightweight, powerful Coding Agent for Mac. Built on top of Apple's MLX framework, `mlx-code` provides fast, local inference with built-in prompt caching and robust tool-calling capabilities. 

### Features

*   **Multi-Provider Compatibility**: Seamlessly translates and handles requests formatted for **Claude**, **Gemini**, **Codex**, and **DeepSeek**.
*   **Built-in REPL & Tools**: Comes with `pie`, a fully-featured REPL with tool execution and reasoning token support.
*   **Server Mode**: Easily spin up a local LLM server.

### Quick Start

Install via pip and launch the agent immediately:

```bash
pip install mlx-code
mc
```

### Command Line Interfaces

The package installs three primary command-line tools:
- mc (Main/Server): Launches the local LLM API server (defaults to 127.0.0.1:8000).
- me (TUI/Harness): Runs the interactive pie chat REPL.

### Options

You can customize the model, server, and behavior using command-line flags.

```bash
# Use Gemini CLI as the harness
mc --harness gemini 

# Use a custom local LLM backend
mc --model mlx-community/Qwen3.5-4B-OptiQ-4bit

# Use DeepSeek V4 Flash API
me --deepseek

# Run the server only
mc --nocc

# General shell piping and chaining works too
echo "explain symgraph.py" | mc -d | cat - PLAN.md | mc
```
*(For a full list of mc server arguments, run mc --help)*

### Credits

- `main.py`: Built on [MLX](https://github.com/ml-explore/mlx) and [MLX LM](https://github.com/ml-explore/mlx-lm) by Apple.
- `pie.py`: Adapted from [pi](https://github.com/badlogic/pi-mono) by Mario Zechner (MIT License).

### Licence

Apache License 2.0 — see LICENSE for details.

<details><summary>...</summary><pre>

~/D/tmp> mc
    pie REPL  •  type /help for commands, Ctrl-D or 'exit' to quit.

    ≫ explain symgraph.py
    π The user wants me to explain symgraph.py. I should first look at the file to understand its contents. Let me use the ReadAST function to inspect this file.

    ReadAST {"path": "symgraph.py"} This looks like a Python file that defines a graph-based system for analyzing code. Let me get more details about the main components by looking up the key symbols.

    ReadAST {"path": "symgraph.py", "symbols": "[\"Edge\", \"Graph\", \"DefCollector\", \"TypeInfo\", \"TypeInferrer\", \"EdgeEmitte This is a comprehensive Python file that builds a symbolic graph representation of Python code. Let me provide a clear explanation of what this file does.

    ## Overview

    `symgraph.py` is a **symbolic graph builder** for Python code that constructs a directed graph representation of Python modules, classes, functions, and their relationships. It uses Python's `ast` (Abstract Syntax Tree) module to analyze code and build a graph where nodes are symbols and edges represent relationships like calls and references.

    ## Key Components

    ### 1. **Edge** (47-60)
    Represents relationships between symbols:
    - `src`, `dst`: source and destination symbols
    - `kind`: "call" or "ref" (call or reference)
    - `via`: optional alias/variable name for indirect references

    ### 2. **Graph** (67-167)
    The main graph structure that stores:
    - `symbols`: list of all discovered symbols
    - `edges`: list of all relationships
    - Internal indexes for efficient lookup (`_sym_by_qname`, `_uses`, `_used_by`)

    ### 3. **DefCollector** (186-238)
    First pass visitor that collects all definitions:
    - Classes, functions, methods, module-level variables
    - Tracks scope (class/function nesting)
    - No type inference, just definition collection

    ### 4. **TypeInferrer** (253-464)
    Second pass that infers types using **fixpoint iteration**:
    - Tracks local variable types per scope
    - Resolves return types from annotations
    - Handles class instances and method calls
    - Uses `ast.NodeVisitor` with proper scope tracking

    ### 5. **EdgeEmitter** (471-700)
    Third pass that emits edges based on type information:
    - Emits "call" edges for function/method calls
    - Emits "ref" edges for variable references
    - Handles `__call__` for class instances
    - Tracks live local variable environments

    ### 6. **build()** (730-755)
    Main entry point that orchestrates the three passes:
    1. Collect definitions
    2. Infer types (fixpoint)
    3. Emit edges

    ### 7. **main()** (762-776)
    CLI interface:
    - `symgraph.py <path>` → outputs all edges as JSON
    - `symgraph.py <path> <query>` → queries for symbols matching a name

    ## How It Works

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │  Pass 1: DefCollector                                       │
    │  └─ Collects all symbols (classes, functions, variables)    │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Pass 2: TypeInferrer (fixpoint)                            │
    │  └─ Infers types, return types, class instances             │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Pass 3: EdgeEmitter                                        │
    │  └─ Emits call/ref edges based on type info                 │
    └─────────────────────────────────────────────────────────────┘
    ```

    ## Use Cases

    This tool is useful for:
    - **Dependency analysis** - understand call graphs and dependencies
    - **Code navigation** - find where symbols are used
    - **Refactoring** - understand impact of changes
    - **Static analysis** - detect unused code, circular dependencies

    The graph can be queried to find all uses of a symbol, all callers, or all references within a module.
    ≫ Add a method to Graph called clear_edges() that resets edges, _edge_keys, _uses, and _used_by to empty.
    π The user wants me to add a `clear_edges()` method to the `Graph` class that resets the edges and related internal structures. I need to:1. Add the method to the Graph class2. Reset `self.edges`, `self._edge_keys`, `self._uses`, and `self._used_by` to emptyLet me first read the Graph class to see its current structure, then add the method.

    Edit {"path": "symgraph.py", "old_text": "# \u2500\u2500 query \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u The edit was successful. Let me verify the change by reading the updated section of the file to confirm the method was added correctly.

    ReadAST {"path": "symgraph.py", "symbols": "[\"Graph\"]"} The edit was successful. The `clear_edges()` method has been added to the `Graph` class. It resets:- `self.edges` - the list of edges- `self._edge_keys` - the set of edge keys (to prevent duplicate edges)- `self._uses` - the dictionary tracking which symbols use which edges- `self._used_by` - the dictionary tracking which symbols are used by which edgesThe symbols list is NOT cleared, which makes sense since we want to keep the symbol definitions but just clear the edges.Let me provide a summary of what was added.

    Done! I've added the `clear_edges()` method to the `Graph` class. Here's what it does:

    ```python
    def clear_edges(self) -> None:
        """Reset all edges and related internal structures."""
        self.edges.clear()
        self._edge_keys.clear()
        self._uses.clear()
        self._used_by.clear()
    ```

    This method:
    - Clears all edges from the graph
    - Resets the edge key set (prevents duplicate edges)
    - Clears the `_uses` mapping (which symbols use which edges)
    - Clears the `_used_by` mapping (which symbols are used by which edges)

    **Note:** It does NOT clear `self.symbols` or `self._sym_by_qname`, so symbol definitions are preserved while only the edge relationships are reset. This is useful for:
    - Re-running edge emission on the same symbols
    - Building multiple graphs from the same codebase
    - Resetting the graph state without losing collected definitions
    ≫ exit
    Bye!
    ~/D/tmp> gg *.py
    staging: log.py
      exists
    staging: main.py
      exists
    staging: pie.py
      exists
    staging: symgraph.py
      exists

    Diff preview:
     Users/jjms/Downloads/tmp/symgraph.py | 7 +++++++
     1 file changed, 7 insertions(+)

    Show per-file diffs? [s=skip]: s

    read> symgraph edit by mlx-code
    Saved snapshot 09529bc9bcbc6d1ff761618c982ece9a290f50fb

~/D/tmp> md

    ┌─ logs ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐┌─ detail ──────────────────────────────────────────────────────────────────────────┐
    │ TIME      LVL       FILE              FUNC              MESSAGE                                                            …││ time       : 2026-05-05T20:55:26.707364+00:00                                     │
    │ 20:55:21  INFO      main.py           main              args=Namespace(model='mlx-community/Qwen3.5-4B-OptiQ-4bit', api='de…││ level      : INFO                                                                 │
    │ 20:55:26  INFO      pie.py            _repl             explain symgraph.py                                                 ││ logger     : root                                                                 │
    │ 20:55:26  INFO      main.py           do_POST           self.path='/v1/chat/completions' { "model": "jj", "max_tokens": 819…││ file       : tmp/main.py                                                          │
    │ 20:55:26  DEBUG     main.py           translate         messages=[Message(role='system', content='You are a helpful assista…││ function   : do_POST                                                              │
    │ 20:55:26  INFO      main.py           encode            ckpts=[742] <|im_start|>system # Tools You have access to the follo…││ line       : 1432                                                                 │
    │ 20:55:26  INFO      main.py           __call__          ckpts=[742] len(prompt)=753 len(self.hx)=0 cl=0                     ││ id         : bdedc2c5-00a5-49od-9bj9-890a6e2a89b1                                 │
    │ 20:55:26  DEBUG     main.py           load              cache/mlxcommunityQwen354BOptiQ4bit_742_3bb19df63c78dde8.safetensor…││                                                                                   │
    │ 20:55:27  DEBUG     main.py           generate          pmt Processed 753 input tokens in 0 seconds (4684 tokens per second…││ ── message ──                                                                     │
    │ 20:55:27  INFO      main.py           generate          gen The user wants me to explain symgraph.py. I should first look a…││ self.path='/v1/chat/completions'                                                  │
    │ 20:55:27  DEBUG     pie.py            _loop             AssistantMessage(content=[ThinkingContent(thinking='The user wants …││ {                                                                                 │
    │ 20:55:27  DEBUG     pie.py            on_event           symgraph.py ─────────────── [C] Symbol (32-43) [C] Edge (47-60) [C…││   "model": "jj",                                                                  │
    │ 20:55:28  INFO      main.py           do_POST           self.path='/v1/chat/completions' { "model": "jj", "max_tokens": 819…││   "max_tokens": 8192,                                                             │
    │ 20:55:28  DEBUG     main.py           translate         messages=[Message(role='system', content='You are a helpful assista…││   "messages": [                                                                   │
    │ 20:55:28  INFO      main.py           encode            ckpts=[742] <|im_start|>system # Tools You have access to the follo…││     {                                                                             │
    │ 20:55:28  INFO      main.py           __call__          ckpts=[742] len(prompt)=1015 len(self.hx)=818 cl=818                ││       "role": "system",                                                           │
    │ 20:55:28  DEBUG     main.py           __call__          cont                                                                ││       "content": "You are a helpful assistant."                                   │
    │ 20:55:28  DEBUG     main.py           generate          pmt Processed 1015 input tokens in 1 seconds (1761 tokens per secon…││     },                                                                            │
    │ 20:55:30  INFO      main.py           generate          gen This looks like a Python file that defines a graph-based system…││     {                                                                             │
    │ 20:55:30  DEBUG     pie.py            _loop             AssistantMessage(content=[ThinkingContent(thinking='This looks like…││       "role": "user",                                                             │
    │ 20:55:30  DEBUG     pie.py            on_event          # DEF class symgraph.Edge /private/var/folders/_5/vz_p3mls23l5rtlhj…││       "content": "explain symgraph.py"                                            │
    │ 20:55:30  INFO      main.py           do_POST           self.path='/v1/chat/completions' { "model": "jj", "max_tokens": 819…││     }                                                                             │
    │ 20:55:30  DEBUG     main.py           translate         messages=[Message(role='system', content='You are a helpful assista…││   ],                                                                              │
    │ 20:55:30  INFO      main.py           encode            ckpts=[742] <|im_start|>system # Tools You have access to the follo…││   "stream": true,                                                                 │
    │ 20:55:30  INFO      main.py           __call__          ckpts=[742] len(prompt)=12781 len(self.hx)=1117 cl=1117             ││   "tools": [                                                                      │
    │ 20:55:30  DEBUG     main.py           __call__          cont                                                                ││     {                                                                             │
    │ 20:55:58  DEBUG     main.py           generate          pmt Processed 12781 input tokens in 28 seconds (454 tokens per seco…││       "type": "function",                                                         │
    │ 20:56:11  INFO      main.py           generate          gen This is a comprehensive Python file that builds a symbolic grap…││       "function": {                                                               │
    │ 20:56:11  DEBUG     pie.py            _loop             AssistantMessage(content=[ThinkingContent(thinking='This is a compr…││         "name": "Read",                                                           │
    │ 20:56:25  INFO      pie.py            _repl             Add a method to Graph called clear_edges() that resets edges, _edge…││         "description": "Read a file. Use offset/limit for large files instead of  │
    │ 20:56:25  INFO      main.py           do_POST           self.path='/v1/chat/completions' { "model": "jj", "max_tokens": 819…││ reading the whole thing.",                                                        │
    │ 20:56:25  DEBUG     main.py           translate         messages=[Message(role='system', content='You are a helpful assista…││         "parameters": {                                                           │
    │ 20:56:25  INFO      main.py           encode            ckpts=[742, 13481] <|im_start|>system # Tools You have access to th…││           "type": "object",                                                       │
    │ 20:56:25  INFO      main.py           __call__          ckpts=[13481, 742] len(prompt)=13515 len(self.hx)=13586 cl=751      ││           "properties": {                                                         │
    │ 20:56:25  DEBUG     main.py           load              cache/mlxcommunityQwen354BOptiQ4bit_742_3bb19df63c78dde8.safetensor…││             "path": {                                                             │
    │ 20:56:25  DEBUG     main.py           generate          save_fn 13481                                                       ││               "description": "File path to read (relative to cwd)",               │
    │ 20:56:56  DEBUG     main.py           generate          pmt Processed 13515 input tokens in 31 seconds (438 tokens per seco…││               "title": "Path",                                                    │
    │ 20:57:00  INFO      main.py           generate          gen The user wants me to add a `clear_edges()` method to the `Graph…││               "type": "string"                                                    │
    │ 20:57:00  DEBUG     pie.py            _loop             AssistantMessage(content=[ThinkingContent(thinking='The user wants …││             },                                                                    │
    │ 20:57:00  DEBUG     pie.py            on_event          edited symgraph.py: replaced 3 line(s) with 10 line(s)              ││             "offset": {                                                           │
    │ 20:57:00  INFO      main.py           do_POST           self.path='/v1/chat/completions' { "model": "jj", "max_tokens": 819…││               "anyOf": [                                                          │
    │ 20:57:00  DEBUG     main.py           translate         messages=[Message(role='system', content='You are a helpful assista…││                 {                                                                 │
    │ 20:57:00  INFO      main.py           encode            ckpts=[742, 13481] <|im_start|>system # Tools You have access to th…││                   "type": "integer"                                               │
    │ 20:57:00  INFO      main.py           __call__          ckpts=[13481, 742] len(prompt)=13801 len(self.hx)=13768 cl=13768    ││                 },                                                                │
    │ 20:57:00  DEBUG     main.py           __call__          cont                                                                ││                 {                                                                 │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘└───────────────────────────────────────────────────────────────────────────────────┘
     All  bdedc2c5-0  8j7j3o1e-6
      log.json  │  3/58 (of 172)  filter: lvl:10;file:main,pie  │  ↑/k ↓/j · PgUp/PgDn · g/G · n/N · * highlight · o open · ; filter · h/l tabs · q quit
</pre></details><br>

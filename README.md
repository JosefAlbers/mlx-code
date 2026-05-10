# mlx-code

A lightweight coding agent for Mac, built on Apple's MLX framework. Fast local inference, built-in prompt caching, robust tool-calling.

[![Link](https://raw.githubusercontent.com/JosefAlbers/mlx-code/main/assets/mlx-code.gif)](https://youtu.be/bizPhrHL1_w)

Modern coding agents are like luxury apartments: impressive and shiny, but you don't hold the deed. The company behind the tool can raise the rent, change the features, or change the locks whenever they please.

I wanted a [backyard shed](https://poyo.co/note/20260202T150723/) for myself. Something I understand end to end, can break on purpose, and fix without filing a support ticket.

`mlx-code` is that shed. It’s deliberately minimal, extremely transparent, and designed around one core idea: [feedback loop](https://www.robert-glaser.de/what-if-iteration-is-all-we-need/) is the thing that matters, not the interface around it. The tighter and faster you can close the loop between intent and output, the better the work gets. Everything else is ceremony. 

So the terminal stays the interface. Text in, text out. No full-screen TUI fighting your terminal emulator, no proprietary context format, no behavior that shifts between versions. 

**Just a loop you control, composed from Unix primitives you already know.**

## How It Works

`mlx-code` has two lightweight, loosely coupled pieces:

- [**main.py**](https://github.com/JosefAlbers/mlx-code/blob/main/mlx_code/main.py): LLM server for Apple Silicon. It loads quantized models and exposes a standard OpenAI-compatible completions endpoint.
- [**pie.py**](https://github.com/JosefAlbers/mlx-code/blob/main/mlx_code/pie.py): Agentic harness based on Mario Zechner's awesome [pi](https://github.com/badlogic/pi-mono)).
- [**ledger.py**](https://github.com/JosefAlbers/mlx-code/blob/main/mlx_code/ledger.py): Git worktree manager that creates isolated branches and working directories for every agent (and sub‑agen) runs.

The CLI is intentionally boring and familiar:

- `mc`: Local agent (LLM server ± harness)
- `me`: Harness. Connects to any compatible API (Claude, DeepSeek, Gemini, OpenAI, or local `mc`)
- `md`: Log viewer

Agentic work lives on a spectrum from tight, synchronous co-driving to loose, asynchronous delegation. The right tool for both ends is a loop that closes quickly, not a UI that abstracts it away. 

Text streams compose. They pipe. They chain. They work the same way they did thirty years ago and will work thirty years from now. 

That's the [constraint](https://jordanlord.co.uk/blog/3-constraints/) that shapes the whole tool.

## Features

- **Local or remote execution**: Run models locally via MLX or connect to Claude, Gemini, Codex, DeepSeek, or any OpenAI‑compatible endpoint.
- **Git worktree isolation**: Every agent run (and every sub‑agent) lives in its own git worktree and branch. Changes are automatically snapshotted (`commit_worktree`), and you can `cleanup_worktree` when done. This makes experimentation safe and fully reversible.
- **Autonomous sub‑agents**: Spawn isolated agents in fresh git worktrees. Each sub‑agent has its own conversation, tool set, and working directory. They can be used for parallel exploration, refactoring, or deep research without polluting the main context.
- **Symbol‑aware source inspection (`ReadTree`)**: Uses tree‑sitter to outline code or fetch exact definitions/calls for a symbol. Drastically reduces token usage compared to reading full files.
- **Built‑in tools**: Read, Write, Edit, Bash, Grep, Find, Ls, ReadTree, GetSkill, and Agent.
- **Prompt caching**: KV cache is saved to disk and reused across requests automatically.
- **REPL with `/commands`**: `/clear`, `/history`, `/tools`, `/branch`, `/abort`, `/help`.
- **TUI log viewer (`md`)**: Explore structured JSON logs with filtering by level, file, function, etc. Mark entries to export.

## Quick Start

Install via pip and launch the agent immediately:

```bash
pip install mlx-code
mc
```

## Command Line Interfaces

### `mc`: Local agent (LLM server ± harness)

```bash
# Start local MLX server and launch the default pie harness (Default)
mc

# Choose a different harness (claude, gemini, codex, deepseek, pie)
mc --leash gemini
mc --leash codex
mc --leash claude

# Server only, no harness
mc --leash none

# Limit allowed tools
mc --tools Ls ReadTree Edit

# Use a custom system prompt
mc --system "You are a helpful Python expert."

# Load skills from a directory
mc --skill ./my-skills

# Shell piping and chaining
echo "explain symgraph.py" | mc -d | cat - PLAN.md | mc
```

### `me`: Harness (connects to API)

```bash
# Connect to local server at 127.0.0.1:8000 (Default)
me

# Use a remote provider
me --api deepseek
me --api claude
me --api gemini
me --api codex

# Specify model
me --api deepseek --model deepseek-v4-pro

# Point to a custom LLM API endpoint
me --url http://localhost:9000

# Load skills from a directory
me --skill ./my-skills
```

### `md`: TUI log viewer

View and filter structured JSON logs from any session.

```bash
md
md --filter "lvl:10;file:main,pie"    # show DEBUG logs from main.py and pie.py
md --out picks.json                   # write marked entries to a file
```

Inside the TUI:
- `j/k` or ↑/↓ – navigate
- `o` – open detailed view for current entry
- `*` – highlight related entries (same file/function)
- `;` – set a live filter
- `h/l` – switch between request‑grouped tabs
- `v` – mark/unmark entry for export
- `q` – quit (marked entries printed or saved with `--out`)

<details><summary>Click to expand</summary><pre>

~/tmp> mc

    pie REPL  •  type /help for commands, Ctrl-D or 'exit' to quit.

    ≫ hi
    π The user is just saying "hi" - this is a simple greeting. I should respond in a friendly way and offer to help them with something.
    Hello! How can I help you today?
    ≫ how many files in current dir?
    π The user wants to know how many files are in the current directory. I can use the Ls function to list the directory contents and count the files.
    Ls
    Looking at the directory listing, I can see:
    - LICENSE (file)
    - README.md (file)
    - cache/ (directory)
    - log.json (file)
    - mlx_code/ (directory)
    - mlx_code.egg-info/ (directory)
    - setup.py (file)

    So there are 4 files and 3 directories in the current directory.
    There are **4 files** in the current directory:
    - LICENSE
    - README.md
    - log.json
    - setup.py

    And **3 directories**:
    - cache/
    - mlx_code/
    - mlx_code.egg-info/
    ≫
    Bye!

~/tmp> mc

    pie REPL  •  type /help for commands, Ctrl-D or 'exit' to quit.

    ≫ explain sittree.py
    π The user is asking me to explain a file called "sittree.py". I should first read this file to understand what it does. Let me use the Read function to read the file.
    Read {"path": "sittree.py"}
    This is a comprehensive file. Let me provide a clear explanation of what this file does.
    # sittree.py — Symbol Usage Finder & Codebase Outliner

    This is a **tree-sitter-based code analysis tool** for coding agents. It provides two main capabilities:

    ## 🎯 Core Features

    ### 1. **Codebase Outlining**
    Shows all top-level symbols (functions, classes, variables, types, etc.) in a file or directory.

    **Usage:**
    ```bash
    python sittree.py <path>              # Outline symbols
    python sittree.py <path> --depth 2    # Include nested symbols
    python sittree.py <path> --context    # Show source context
    ```

    ### 2. **Symbol Search**
    Finds all usages of specific symbols across a codebase.

    **Usage:**
    ```bash
    python sittree.py <path> myFunc myClass  # Search for symbols
    python sittree.py <path> myFunc --kinds definition  # Filter by kind
    ```

    ## 📋 Supported Languages

    | Extension | Language |
    |-----------|----------|
    | `.py` | Python |
    | `.js`, `.jsx`, `.ts`, `.tsx` | JavaScript/TypeScript |
    | `.go` | Go |
    | `.rs` | Rust |
    | `.java` | Java |
    | `.c`, `.cpp`, `.cc`, `.cs` | C/C++/C# |
    | `.rb` | Ruby |
    | `.php` | PHP |
    | `.swift`, `.kt` | Swift/Kotlin |
    | `.lua`, `.hs`, `.ex`, `.exs` | Lua/Haskell/Elixir |
    | `.r`, `.sh` | R/Bash |
    | `.toml`, `.yaml`, `.yml` | Config files |

    ## 🔍 Key Capabilities

    ### Symbol Kinds Detected
    - **definition** — Function/class/type definitions
    - **assignment** — Variable declarations
    - **call** — Function/method calls
    - **import** — Import statements
    - **reference** — Bare identifier references

    ### Language-Specific Features
    - **Python**: Handles `__main__` entry points, decorated functions, augmented assignments
    - **JavaScript/TypeScript**: Handles ES6+ features (arrow functions, destructuring, type aliases)
    - **Go**: Handles methods, type specs, package-level declarations
    - **Rust**: Handles traits, structs, enums, visibility modifiers
    - **Java**: Handles classes, interfaces, enums, method invocations

    ## 🛠️ Technical Details

    ### Tree-Sitter Integration
    - Uses **bespoke queries** for languages with rich grammar support
    - Falls back to **generic identifier scanning** for simpler languages
    - Caches parsers and queries for performance

    ### Output Formats
    - **Human-readable**: Color-coded with line numbers and context
    - **JSON**: Machine-parseable output via `--json` flag
    - **Raw**: Paste-safe source bodies for editing

    ### Advanced Features
    - **Context snippets**: Show surrounding source lines
    - **Entry point detection**: Identify `main()`, `__main__`, etc.
    - **Export detection**: Find public/exported symbols
    - **Destructuring binding extraction**: Handle complex patterns like `[a, b]` or `{x, y}`

    ## 💡 Use Cases

    1. **Codebase exploration**: Get an overview of a project's structure
    2. **Symbol search**: Find all usages of a function/class
    3. **Refactoring support**: Locate all references before renaming
    4. **Documentation generation**: Extract API surface information
    5. **Agent assistance**: Provide context for code review/editing tasks

    ## ⚙️ Configuration Options

    | Option | Description |
    |--------|-------------|
    | `--lang` | Override language detection |
    | `--depth` | Max nesting depth for outline |
    | `--context` | Include source context |
    | `--no-references` | Skip bare identifier hits |
    | `--kinds` | Filter by symbol kind |
    | `--warn-skipped` | Show grammar load errors |

    ---

    This tool is designed to work seamlessly with coding agents, providing structured code analysis that's both human-readable and machine-parseable.
    ≫
    Bye!

~/tmp> mc --tools ReadTree

    pie REPL  •  type /help for commands, Ctrl-D or 'exit' to quit.

    ≫ explain sittree.py
    π The user is asking me to explain sittree.py. I should first read this file to understand what it does. Let me use the ReadTree function to inspect this file.
    ReadTree {"path": "sittree.py"}
    This is a Python file that appears to be a tool for inspecting source code using tree-sitter. It has various functions for handling different programming languages (Python, JS, Go, Rust, Java), extracting bindings, finding symbols, and printing outlines. Let me read the full file to provide a comprehensive explanation.
    ReadTree {"path": "sittree.py", "depth": "2"}
    I need to read the actual file content to understand what it does. Let me read the file directly.
    ReadTree {"path": "sittree.py", "symbols": "[\"main\", \"SymbolFinder\", \"Capability\", \"outline_path\", \"search_symbols\", \"print_outline\"]"}
    Now I have a good understanding of the file. Let me provide a comprehensive explanation of what sittree.py does.
    # sittree.py - Source Code Tree Inspector

    This is a **tree-sitter-based source code analysis tool** that can inspect, search, and outline code across multiple programming languages. It's designed to work with the tree-sitter parser library for accurate AST (Abstract Syntax Tree) parsing.

    ## 🎯 Main Features

    ### 1. **Symbol Search**
    - Find specific symbols (functions, classes, variables, etc.) in source files
    - Supports multiple programming languages: Python, JavaScript, TypeScript, Go, Rust, Java
    - Can search directories recursively
    - Includes reference tracking and context lines

    ### 2. **Code Outlining**
    - Generate hierarchical outlines of files/directories
    - Shows entry points, exported symbols, and line ranges
    - Supports depth limiting for large projects

    ### 3. **Multi-Language Support**
    - Uses bespoke queries for Python, JS, Go, Rust, Java
    - Falls back to generic identifier scanning when no bespoke query exists
    - Handles UTF-8 encoding issues with proper character decoding

    ## 📦 Key Components

    ### **Capability Enum**
    ```python
    class Capability(str, Enum):
        FULL    = "full"     # Bespoke queries with correct semantics
        PARTIAL = "partial"  # Generic identifier scan (kind is always 'reference')
        NONE    = "none"     # Grammar has no identifier-like nodes
    ```

    ### **SymbolFinder Class**
    - Core class for searching symbols in files
    - Handles symbol matching with proper UTF-8 decoding
    - Groups results by original symbol name
    - Supports references and context inclusion

    ### **Main Functions**
    - `outline_path()`: Recursively outline files/directories
    - `search_symbols()`: Search for multiple symbols in a file/directory
    - `print_outline()`: Display formatted code outlines
    - `print_usages()`: Display symbol usage information

    ## 🚀 Command-Line Usage

    ```bash
    # Outline mode - show code structure
    python sittree.py /path/to/file --lang py

    # Search mode - find symbols
    python sittree.py /path/to/file --symbols "my_function" --lang py

    # Search in directory
    python sittree.py /path/to/dir --symbols "class" --lang py

    # JSON output
    python sittree.py /path/to/file --symbols "my_function" --json
    ```

    ## 🔧 Technical Details

    ### **Query System**
    - Uses tree-sitter queries for precise symbol matching
    - Supports `definition`, `assignment`, `call`, and `import` kinds
    - Has priority ordering for different node kinds

    ### **Language Extensions**
    - Maps file extensions to language packages
    - Supports both extension-based and explicit language specification
    - Handles files with unexpected extensions when `--lang` is specified

    ### **Error Handling**
    - Returns exit code 2 for grammar failures (distinct from path errors)
    - Warns when grammars are unavailable for certain file types
    - Handles non-UTF-8 source files gracefully

    ## 💡 Use Cases

    1. **IDE Integration**: Provides the foundation for a code search/outline feature
    2. **Static Analysis**: Find symbols across large codebases
    3. **Documentation Generation**: Create code structure maps
    4. **Refactoring Support**: Locate all usages of a symbol before renaming

    The tool is particularly useful for developers who need precise, language-aware code navigation and search capabilities without relying on IDE-specific features.

~/tmp> md -o test

    ┌─ logs ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐┌─ detail ──────────────────────────────────────────────────────────────────────────┐
    │ TIME      LVL       FILE              FUNC              MESSAGE                                                            …││ time       : 2026-05-09T17:35:17.021887+00:00                                     │
    │ 17:35:01  DEBUG     main.py           main              args=Namespace(model='mlx-community/Qwen3.5-4B-OptiQ-4bit', leash='…││ level      : DEBUG                                                                │
    │ 17:35:03  DEBUG     main.py           serve             Server bound to http://127.0.0.1:8000                               ││ logger     : root                                                                 │
    │ 17:35:03  WARNING   pie.py            __init__          No api-key                                                          ││ file       : tmp/main.py                                                          │
    │ 17:35:15  DEBUG     pie.py            repl              explain sittree.py                                                  ││ function   : encode                                                               │
    │ 17:35:15  DEBUG     pie.py            stream            { "model": "jj", "max_tokens": 8192, "messages": [ { "role": "syste…││ line       : 684                                                                  │
    │ 17:35:15  DEBUG     main.py           do_POST           self.path='/v1/chat/completions' { "model": "jj", "max_tokens": 819…││ id         : f8e15ee6-aaa5-4532-8735-172d6a7b9707                                 │
    │ 17:35:15  DEBUG     main.py           translate         messages=[Message(role='system', content='Available skills (use Get…││                                                                                   │
    │ 17:35:15  DEBUG     main.py           encode            ckpts=[1898] <|im_start|>system # Tools You have access to the foll…││ ── message ──                                                                     │
    │ 17:35:15  DEBUG     main.py           __call__          ckpts=[1898] len(prompt)=1909 len(self.hx)=0 cl=0                   ││ ckpts=[1898]                                                                      │
    │ 17:35:15  DEBUG     main.py           load              cache/mlxcommunityQwen354BOptiQ4bit_1898_21d53f79d6f145d5.safetenso…││ <|im_start|>system                                                                │
    │ 17:35:15  DEBUG     main.py           generate          Processed 1909 input tokens in 0 seconds (11222 tokens per second)  ││ # Tools                                                                           │
    │ 17:35:16  INFO      main.py           generate          The user is asking me to explain a file called "sittree.py". I shou…││                                                                                   │
    │ 17:35:16  DEBUG     pie.py            _loop             AssistantMessage(content=[ThinkingContent(thinking='The user is ask…││ You have access to the following functions:                                       │
    │ 17:35:16  INFO      pie.py            _execute_one      # sittree.py (lines 1–1266 of 1266) """ sittree.py — Symbol usage f…││                                                                                   │
    │ 17:35:16  DEBUG     pie.py            stream            { "model": "jj", "max_tokens": 8192, "messages": [ { "role": "syste…││ <tools>                                                                           │
    │ 17:35:16  DEBUG     main.py           do_POST           self.path='/v1/chat/completions' { "model": "jj", "max_tokens": 819…││ {"type": "function", "function": {"name": "Read", "description": "Read a file. Us │
    │ 17:35:16  DEBUG     main.py           translate         messages=[Message(role='system', content='Available skills (use Get…││ e offset/limit for large files instead of reading the whole thing.", "parameters" │
    │>17:35:17  DEBUG     main.py           encode            ckpts=[1898] <|im_start|>system # Tools You have access to the foll…││ : {"type": "object", "properties": {"path": {"description": "File path to read (r │
    │ 17:35:17  DEBUG     main.py           __call__          ckpts=[1898] len(prompt)=14602 len(self.hx)=1979 cl=1979            ││ elative to cwd)", "title": "Path", "type": "string"}, "offset": {"anyOf": [{"type │
    │ 17:35:17  DEBUG     main.py           __call__          cont                                                                ││ ": "integer"}, {"type": "null"}], "default": null, "description": "Start line (1- │
    │ 17:35:47  DEBUG     main.py           generate          Processed 14602 input tokens in 31 seconds (476 tokens per second)  ││ based)", "title": "Offset"}, "limit": {"anyOf": [{"type": "integer"}, {"type": "n │
    │ 17:36:02  INFO      main.py           generate          This is a comprehensive file. Let me provide a clear explanation of…││ ull"}], "default": null, "description": "Max lines to read", "title": "Limit"}},  │
    │ 17:36:02  DEBUG     pie.py            _loop             AssistantMessage(content=[ThinkingContent(thinking='This is a compr…││ "required": ["path"]}}}                                                           │
    │                                                                                                                             ││ {"type": "function", "function": {"name": "Write", "description": "Create or over │
    │                                                                                                                             ││ write a file. Prefer 'edit' for small changes to existing files.", "parameters":  │
    │                                                                                                                             ││ {"type": "object", "properties": {"path": {"description": "File path to create or │
    │                                                                                                                             ││  overwrite (relative to cwd)", "title": "Path", "type": "string"}, "content": {"d │
    │                                                                                                                             ││ escription": "Full file content", "title": "Content", "type": "string"}}, "requir │
    │                                                                                                                             ││ ed": ["path", "content"]}}}                                                       │
    │                                                                                                                             ││ {"type": "function", "function": {"name": "Edit", "description": "Replace an exac │
    │                                                                                                                             ││ t unique string in a file. Read the file first if unsure of exact text.", "parame │
    │                                                                                                                             ││ ters": {"type": "object", "properties": {"path": {"description": "File path to ed │
    │                                                                                                                             ││ it (relative to cwd)", "title": "Path", "type": "string"}, "old_text": {"descript │
    │                                                                                                                             ││ ion": "Exact text to replace (must appear exactly once)", "title": "Old Text", "t │
    │                                                                                                                             ││ ype": "string"}, "new_text": {"description": "Replacement text", "title": "New Te │
    │                                                                                                                             ││ xt", "type": "string"}}, "required": ["path", "old_text", "new_text"]}}}          │
    │                                                                                                                             ││ {"type": "function", "function": {"name": "Bash", "description": "Run a shell com │
    │                                                                                                                             ││ mand, get stdout+stderr. Prefer read/grep/find/ls for file exploration.", "parame │
    │                                                                                                                             ││ ters": {"type": "object", "properties": {"command": {"description": "Shell comman │
    │                                                                                                                             ││ d to execute", "title": "Command", "type": "string"}, "timeout": {"default": 120, │
    │                                                                                                                             ││  "description": "Timeout in seconds", "title": "Timeout", "type": "integer"}}, "r │
    │                                                                                                                             ││ equired": ["command"]}}}                                                          │
    │                                                                                                                             ││ {"type": "function", "function": {"name": "Grep", "description": "Search files fo │
    │                                                                                                                             ││ r a pattern. Respects .gitignore.", "parameters": {"type": "object", "properties" │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘└───────────────────────────────────────────────────────────────────────────────────┘
     All  cef534bf-5  50e0af6d-1  b578ec4d-f  f8e15ee6-a  d25f55fd-2  9a03247f-e  5160a1d3-1
      log.json  │  18/23 (of 592)  filter: lvl:10;file:main,pie  │  ↑/k ↓/j · PgUp/PgDn · g/G · n/N · * highlight · o open · ; filter · h/l tabs · v mark · q quit


    ┌─ logs ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐┌─ detail ──────────────────────────────────────────────────────────────────────────┐
    │ TIME      LVL       FILE              FUNC              MESSAGE                                                            …││ time       : 2026-05-09T17:51:22.117961+00:00                                     │
    │ 17:50:55  DEBUG     main.py           main              args=Namespace(model='mlx-community/Qwen3.5-4B-OptiQ-4bit', leash='…││ level      : DEBUG                                                                │
    │ 17:50:57  DEBUG     main.py           serve             Server bound to http://127.0.0.1:8000                               ││ logger     : root                                                                 │
    │ 17:50:57  WARNING   pie.py            __init__          No api-key                                                          ││ file       : tmp/main.py                                                          │
    │ 17:51:13  DEBUG     pie.py            repl              explain sittree.py                                                  ││ function   : encode                                                               │
    │ 17:51:13  DEBUG     pie.py            stream            { "model": "jj", "max_tokens": 8192, "messages": [ { "role": "syste…││ line       : 684                                                                  │
    │ 17:51:13  DEBUG     main.py           do_POST           self.path='/v1/chat/completions' { "model": "jj", "max_tokens": 819…││ id         : 5160a1d3-1552-40a7-a82e-ebaf3d0995f8                                 │
    │ 17:51:13  DEBUG     main.py           translate         messages=[Message(role='system', content='Available skills (use Get…││                                                                                   │
    │ 17:51:13  DEBUG     main.py           encode            ckpts=[698] <|im_start|>system # Tools You have access to the follo…││ ── message ──                                                                     │
    │ 17:51:13  DEBUG     main.py           __call__          ckpts=[698] len(prompt)=709 len(self.hx)=0 cl=0                     ││ ckpts=[698]                                                                       │
    │ 17:51:13  DEBUG     main.py           __call__          anew                                                                ││ <|im_start|>system                                                                │
    │ 17:51:13  DEBUG     main.py           generate          save_fn 698                                                         ││ # Tools                                                                           │
    │ 17:51:15  DEBUG     main.py           save              cache/mlxcommunityQwen354BOptiQ4bit_698_6f0ceeb516faea93.safetensor…││                                                                                   │
    │ 17:51:15  DEBUG     main.py           generate          Processed 709 input tokens in 2 seconds (394 tokens per second)     ││ You have access to the following functions:                                       │
    │ 17:51:16  INFO      main.py           generate          The user is asking me to explain a file called "sittree.py". I shou…││                                                                                   │
    │ 17:51:16  DEBUG     pie.py            _loop             AssistantMessage(content=[ThinkingContent(thinking='The user is ask…││ <tools>                                                                           │
    │ 17:51:16  INFO      pie.py            _execute_one       /private/var/folders/_5/vz_p3mls23l5rtlhj3nzvwjw0000gn/T/tmp8sx8z7…││ {"type": "function", "function": {"name": "ReadTree", "description": "Inspect sou │
    │ 17:51:16  DEBUG     pie.py            stream            { "model": "jj", "max_tokens": 8192, "messages": [ { "role": "syste…││ rce code using tree-sitter. Works for any supported language (Python, JS/TS, Go,  │
    │ 17:51:16  DEBUG     main.py           do_POST           self.path='/v1/chat/completions' { "model": "jj", "max_tokens": 819…││ Rust, Java, C/C++, Ruby, and more). Two modes:\n  OUTLINE (no symbols): returns t │
    │ 17:51:16  DEBUG     main.py           translate         messages=[Message(role='system', content='Available skills (use Get…││ he symbol tree of a file or directory — class/function/method/var names with line │
    │ 17:51:16  DEBUG     main.py           encode            ckpts=[698] <|im_start|>system # Tools You have access to the follo…││  ranges. Use this first to orient yourself before reading or editing code.\n  SYM │
    │ 17:51:16  DEBUG     main.py           __call__          ckpts=[698] len(prompt)=2211 len(self.hx)=792 cl=792                ││ BOL LOOKUP (symbols=[...]): returns the full source body of every definition of t │
    │ 17:51:16  DEBUG     main.py           __call__          cont                                                                ││ hose names, plus every call/assignment site with context. Output is paste-safe fo │
    │ 17:51:19  DEBUG     main.py           generate          Processed 2211 input tokens in 3 seconds (660 tokens per second)    ││ r use as old_str in an Edit call. Accepts dotted names like 'ClassName.method' to │
    │ 17:51:22  INFO      main.py           generate          This is a Python file that appears to be a tool for inspecting sour…││  narrow results.", "parameters": {"type": "object", "properties": {"path": {"desc │
    │ 17:51:22  DEBUG     pie.py            _loop             AssistantMessage(content=[ThinkingContent(thinking='This is a Pytho…││ ription": "File or directory to inspect (relative to cwd)", "title": "Path", "typ │
    │ 17:51:22  INFO      pie.py            _execute_one      # DEF outline_path /private/var/folders/_5/vz_p3mls23l5rtlhj3nzvwjw…││ e": "string"}, "symbols": {"description": "Symbol names to look up, e.g. [\"MyCla │
    │ 17:51:22  DEBUG     pie.py            stream            { "model": "jj", "max_tokens": 8192, "messages": [ { "role": "syste…││ ss\", \"my_fn\", \"ClassName.method\"]. Omit (or pass []) for outline mode. Dotte │
    │ 17:51:22  DEBUG     main.py           do_POST           self.path='/v1/chat/completions' { "model": "jj", "max_tokens": 819…││ d names are accepted — the last component is matched and results are labelled wit │
    │ 17:51:22  DEBUG     main.py           translate         messages=[Message(role='system', content='Available skills (use Get…││ h the full qualified name.", "items": {"type": "string"}, "title": "Symbols", "ty │
    │>17:51:22  DEBUG     main.py           encode            ckpts=[698] <|im_start|>system # Tools You have access to the follo…││ pe": "array"}, "depth": {"default": 1, "description": "Outline mode: nesting dept │
    │ 17:51:22  DEBUG     main.py           __call__          ckpts=[698] len(prompt)=6534 len(self.hx)=2354 cl=2354              ││ h (1=top-level only, 2=classes+methods).", "title": "Depth", "type": "integer"},  │
    │ 17:51:22  DEBUG     main.py           __call__          cont                                                                ││ "kinds": {"description": "Filter symbol-lookup results by kind. Valid values: def │
    │ 17:51:32  DEBUG     main.py           generate          Processed 6534 input tokens in 10 seconds (660 tokens per second)   ││ inition, assignment, call, import, reference. Default: definition + assignment +  │
    │ 17:51:43  INFO      main.py           generate          Now I have a good understanding of the sittree.py file. Let me prov…││ call + import (no bare references).", "items": {"type": "string"}, "title": "Kind │
    │ 17:51:43  DEBUG     pie.py            _loop             AssistantMessage(content=[ThinkingContent(thinking='Now I have a go…││ s", "type": "array"}, "lang": {"anyOf": [{"type": "string"}, {"type": "null"}], " │
    │                                                                                                                             ││ default": null, "description": "Override language detection, e.g. 'py', 'ts', 'go │
    │                                                                                                                             ││ '.", "title": "Lang"}}, "required": ["path"]}}}                                   │
    │                                                                                                                             ││ </tools>                                                                          │
    │                                                                                                                             ││                                                                                   │
    │                                                                                                                             ││ If you choose to call a function ONLY reply in the following format with NO suffi │
    │                                                                                                                             ││ x:                                                                                │
    │                                                                                                                             ││                                                                                   │
    │                                                                                                                             ││ <tool_call>                                                                       │
    │                                                                                                                             ││ <function=example_function_name>                                                  │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘└───────────────────────────────────────────────────────────────────────────────────┘
     All  cef534bf-5  50e0af6d-1  b578ec4d-f  f8e15ee6-a  d25f55fd-2  9a03247f-e  5160a1d3-1
      log.json  │  30/35 (of 592)  filter: lvl:10;file:main,pie  │  ↑/k ↓/j · PgUp/PgDn · g/G · n/N · * highlight · o open · ; filter · h/l tabs · v mark · q quit

~/tmp[1 jobs]> cat prompt test | me -a deepseek

    Yes, it did — and by a **substantial margin**. Here’s the comparison from the logs you provided:

    |                      | `Read` session (full file) | `ReadTree` session (outline + symbols) |
    |----------------------|----------------------------|----------------------------------------|
    | Tool calls           | 1 (Read whole file)        | 3 (outline, depth 2, symbol bodies)    |
    | Token count up to<br>final assistant response | **14,602** tokens | **7,827** tokens |
    | **Savings**          | —                    | **6,775 tokens fewer (≈46% reduction)** |

    (That token count comes from the `---14602` and `---7827` markers at the end of the two conversation excerpts; both are measured at the same point — after the tool responses and assistant `think`, just before the final explanations are written.)

    ## What made the difference

    - **`Read`** dumped the entire 1,266‑line file into the context (≈35 KB of text, including all comments, blank lines, and internal helper logic).
    - **`ReadTree`** only supplied:
      1. A one‑line‑per‑symbol outline (≈70 lines).
      2. The full source for *only* the 5 requested symbols (`Capability`, `outline_path`, `SymbolFinder`, `search_symbols`, `print_outline`, `main`) plus their call sites — a fraction of the file.

    Everything the agent actually needed for a high‑level explanation was in that reduced set, so the rest of the file was just noise.

    ## But was it “as good”?

    As we covered earlier, the `ReadTree`‑based explanation was slightly less detailed — it missed some configuration flags, language lists, and advanced features that only appeared in the comments and constants of the full file. But for **understanding the tool’s structure and main capabilities**, it was perfectly sufficient. So you saved nearly half the context window while still getting a perfectly usable answer.

    **Bottom line:** `ReadTree` is a sharp tool for context efficiency — it can slash input size dramatically for tasks like code explanation, and the smaller context window often allows the model to think faster and more focused.

</pre></details><br>

## Tools Overview

| Tool        | Description |
|-------------|-------------|
| `Read`      | Read a file (supports offset/limit for large files). |
| `Write`     | Create or overwrite a file. |
| `Edit`      | Replace a unique string in a file. |
| `Bash`      | Run a shell command (timeout, stdout+stderr). |
| `Grep`      | Search file contents with regex, respects `.gitignore`. |
| `Find`      | Find files/directories by name pattern. |
| `Ls`        | List directory contents (respects `.gitignore`). |
| `ReadTree`  | Outline a file/directory or fetch exact bodies of definitions/calls. Uses tree‑sitter and works for many languages (Python, JS/TS, Go, Rust, Java, C/C++, C#, Ruby, etc.). |
| `GetSkill`  | Retrieve full instructions for a named skill (from `--skill` directory). |
| `Agent`     | Spawn an autonomous sub‑agent in an isolated git worktree. For parallel tasks, deep research, or safe experimentation. |

## Skills

Skills are directories containing a `SKILL.md` file with frontmatter (`name:` and `description:`). Place them in a directory and pass `--skill <dir>`. The agent will list available skills and can load their full instructions via `GetSkill`. This keeps the system prompt lean while allowing on‑demand detailed guidance.


## Credits

- `main.py`: Built on [mlx](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple.
- `pie.py`: Adapted from [pi](https://github.com/badlogic/pi-mono) by Mario Zechner (MIT License).

## Licence

Apache License 2.0 — see LICENSE for details.

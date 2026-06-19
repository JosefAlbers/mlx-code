# mlx-code

A Git-native coding agent that can run entirely on your Mac. No API keys, no cloud, and no data leaving your machine. Powered by Apple MLX, it turns commits, branches, and worktrees into the agent’s state, history, and execution model

<a href="https://asciinema.org/a/1257590" target="_blank"><img src="https://asciinema.org/a/1257590.svg" /></a>

---

## Architecture

```
Worktrees:

  main ──●──●──●──●──●──●──●──●──●──●──●──●──●──●───────────► Node = git commit + chat history
            │        │
            │        └── branch-1 ──●──●──●
            │                          │ ┌────────────┐
            │                          └─┤ branch-1-0 ├──●──●
            │                            └─────┬──────┘
            └── branch-0 ──●──●──●             │
                                               │
                                               │
Tabs:                                          ├────────────► Tab = git branch + Agent
                                               │
                                               │
┌──────────────────────────────────────────────┼─────────┐
│  TUI tabs                                    │         │
│  ┌──────┐  ┌──────────┐  ┌──────────┐  ┌─────┴──────┐  │
│  │ main │  │ branch-0 │  │ branch-1 │  │ branch-1-0 │  │
│  └──────┘  └────┬─────┘  └──────────┘  └────────────┘  │
└─────────────────┼──────────────────────────────────────┘
                  │
Agents:           ├─────────────────────────────────────────► Each tab runs its own Agent
                  │
             ┌────┴─────────────────────────────────────┐
             │  Agent                                   │
             │  ┌────────────────┐  ┌────────────────┐  │
             │  │ API:           │  │ Tools:         │  │
             │  │ Local (mlx-lm) │  │ Read    Write  │  │
             │  │ Gemini         │  │ Edit    Bash   │  │
             │  │ Claude         │  │ Grep    Find   │  │
             │  │ Codex          │  │ Ls      Skill  │  │
             │  │ DeepSeek       │  │ Agent ─────────┼──┼───► Recursively spawns sub-Agents
             │  └────────────────┘  └────────────────┘  │     
             │  Git worktree                            │
             │  (isolation + session state)             │
             └──────────────────────────────────────────┘
```

Each layer is importable and composable on its own. A commit records state, a branch records an alternative path, and a tab is just a live view over an `Agent`.

```python
from mlx_code.repl import Agent
from mlx_code.tools import ReadTool, WriteTool, EditTool

agent = Agent(api='claude', tool_names=['Read', 'Write', 'Edit'])
result = await agent.run('refactor utils.py to use dataclasses')
```

---

## Core ideas

- **Git is the state machine.** Every file-changing agent step is committed with the conversation that produced it, so you can inspect, resume, and branch from any checkpoint.
- **Branches are alternative futures.** A branch is not just a Git branch; it is a different reasoning path with its own worktree and session state.
- **Agents are the primitive.** Tabs, branches, and delegated subtasks are all instances of the same `Agent` abstraction.
- **Worktrees provide isolation.** The agent edits in a separate worktree, so your main checkout stays clean and recoverable.

---

## Quick start

```bash
# ephemeral run (no installation)
uvx --from mlx-code mlc

# or install into the current environment
pip install mlx-code

# launch
mlc                              # with a local MLX model
mlc-run --api gemini             # or use a remote provider
```

That's it. The first run starts a local inference server and drops you into the REPL.

---

## Why mlx-code

**Agents as reusable workflow atoms.** Tabs, branches, and tasks are all managed within instances of `Agent`. Each one gets its own conversation, its own tools, and its own worktree. Agents can spawn sub-agents to delegate subtasks, and each child is a full agent with its own scoped tool set.

**Git is the database.** When the agent makes file changes, they’re committed to a git worktree with the full conversation embedded in the commit message. Resume any past session by hash, branch from any checkpoint, and inspect the agent timeline with `git log`. No proprietary state files, just Git. 

**Built-in safety nets.** Your working directory is never at risk. The agent operates inside a `git worktree`, not your checkout. It can make a mess, and you can inspect or discard it without ever touching `main`. Subprocess environment variables go through an explicit allowlist, so secrets in your shell are never leaked to agent-spawned processes.

**Batteries included.** Everything ships in one pip install: the MLX inference engine, the multi-protocol API server, the agent loop, the tools, and the TUI. No llama.cpp, no ollama, no vLLM bridge to find and configure. And the server natively speaks OpenAI, Anthropic, Gemini, and Codex wire formats simultaneously, so `claude`, `codex`, and `gemini` CLIs can all work against your local model without a translation layer.

**Continuous batching.** The local inference server runs a continuous batching engine that processes multiple sequences concurrently. When you spawn parallel agents (eg, multiple tabs, `asyncio.gather` pipelines, or delegated sub-tasks) they all share the same GPU context and are stepped together each tick. A prefix cache persists KV snapshots to disk, so repeated system prompts and conversation prefixes are prefilled once and reused across sessions. No request queueing, no waiting for the previous agent to finish.

---

## Agent primitive

Every surface in mlx-code composes the same abstraction:

| Surface | What it does | How it creates an Agent |
|---------|-------------|------------------------|
| Python API | Programmatic pipeline | `Agent(...)` |
| Agent tool | Model delegates a subtask | `agent.spawn()` |
| `/branch` | Fork from any checkpoint | `agent.branch()` |
| TUI tab | Parallel conversation thread | `Tab('title', Agent(...))` |

They're all the same object. A branch is an agent. A sub-task is an agent. A tab is an agent in a tab container. This means patterns compose: a branched agent can spawn sub-agents, a sub-agent can be branched, and any agent can be inspected, diffed, or resumed because it's backed by git.

---

## Sessions as git history

When the agent makes file changes, they're committed to the worktree with the full conversation embedded in the commit message. This means you can always pick up where you left off:

```bash
# Resume a session from any commit hash
mlc --resume abc1234
```

```python
# Or from Python
from mlx_code.gits import resume_worktree
from mlx_code.repl import Agent

gwt, messages = resume_worktree(".", "abc1234")
agent = Agent(ctx={"gwt": gwt})
agent.messages = messages
await agent.run("now add unit tests")
```

Branch from any point in the conversation. Each branch gets its own worktree:

```
/branch                      # branch from current state
/branch --rev 2              # branch from the 2nd user turn
/branch --rev 3 make it use httpx instead
```

Since it's just git, you can inspect the timeline outside the REPL:

```bash
git log --oneline        # commits = agent turns that changed files
git diff HEAD~1          # what the agent changed
git worktree list        # all active worktrees
```

---

## Swapping agents for different phases

Coding is a pipeline: spec, draft, review, verify. Each phase benefits from a different mind: a planner needs broad context and no write access; a coder needs tight constraints and fast iteration; a reviewer needs read-only access and a critical eye.

mlx-code lets you hot-swap the agent configuration mid-session. The conversation continues, but the system prompt, model, and tool set change:

```
/clear --config reviewer.yaml
```

```yaml
# reviewer.yaml
model: gemini-3.1-flash-lite
api: gemini
system: |
  Do NOT make edits. Only comment.
  Challenge incorrect assumptions.
  Identify edge cases and failure modes.
  Prefer robust solutions over quick hacks.
  Keep implementations simple.
  Avoid introducing dependencies unless they provide clear value.
  Ask when unsure.
tools:
  - Read
  - Grep
  - Bash
```

```yaml
# coder.yaml
model: deepseek-v4-flash
api: deepseek
system: |
  Read first. Change second.
  Understand the relevant files before editing.
  Make focused, minimal changes.
  Follow existing code style and architecture.
  Do not rewrite working code unnecessarily.
  Fix root causes rather than symptoms.
  After coding, verify that the solution is consistent with the requirements and surrounding code.
tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
```

Reliability comes from specialization plus constraint. A read-only reviewer can't silently break your code. A scoped implementer can't wander off into architecture discussions. Each agent does one thing well.

---

## Continuous batching

The local server can run multiple inference sequences concurrently inside a single batch step. Instead of a global lock that serialises one request at a time, the batching engine maintains a live set of active sequences and yields tokens for all of them on every step.

```bash
mlc --engine batch            # continuous batching + built-in REPL
```

This unlocks true parallelism for multi-agent workloads:

```python
import asyncio
from mlx_code.repl import Agent

async def main():
    agents = [Agent() for _ in range(4)]
    await asyncio.gather(*[
        a.run(f"Research topic: {t}")
        for a, t in zip(agents, ["consensus", "cryptography", "networking", "storage"])
    ])

asyncio.run(main())
```

All four agents generate simultaneously inside the same batch. No sequential blocking.

### Health endpoint

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","model":"mlx-community/Qwen3.5-4B-OptiQ-4bit","active_sequences":2,"prefix_cache_files":5}
```

`active_sequences` shows how many agents are generating right now; `prefix_cache_files` shows how many prefix KV snapshots are stored on disk.

---

## Command Line

### `mlc`: local server + harness

Starts the MLX inference server and launches the built-in TUI harness against it.

```bash
# Default: local server + default harness
mlc

# Continuous batching mode (default is sequential caching mode)
mlc --engine batch

# Server only, no harness
mlc --leash none

# Use a different harness (routes traffic through the local server)
mlc --leash claude
mlc --leash gemini
mlc --leash codex

# Specify a model
mlc --model mlx-community/Qwen3.5-4B-OptiQ-4bit

# Restrict the tools available to the agent
mlc --tools Read Write Bash

# Custom system prompt
mlc --system "You are a helpful assistant."

# Load skills from a directory (scans recursively for SKILL.md files)
mlc --skill ./my-skills

# Resume a previous session from a git commit hash
mlc --resume <commit-hash>

# Send an initial prompt automatically when the REPL starts
mlc --prompt "summarise all changes in the last 7 days"

# Stream the agent log to a file (useful for monitoring long runs)
mlc --stream agent.log

# Bind to a specific host/port (default: 127.0.0.1:8000, auto-increments if busy)
mlc --host 0.0.0.0 --port 9000
```

Because `mlc` reads from stdin when it isn’t a TTY, it composes naturally with shell pipes:

```bash
echo "Here's the solution you proposed: <excerpt>$(mlc -p "write code for a chrome extension to play youtube x5 speed")</excerpt> Now argue against it." | mlc
```

### `mlc-run`: harness only

Runs the agent harness against an already-running server or a remote provider.

```bash
# Connect to a local server at 127.0.0.1:8000 (default)
mlc-run

# Remote providers
mlc-run --api claude
mlc-run --api gemini
mlc-run --api deepseek
mlc-run --api deepseek --model deepseek-v4-pro
mlc-run --api codex

# Custom endpoint
echo "explain lsp.py" | mlc-run -a deepseek | cat - PLAN.md | mlc-run --url http://localhost:9000

# Simple terminal REPL (no TUI)
mlc-run --bare
```

---

## Using as a Library

Import the pieces you need to build background workers, scheduled jobs, or event-triggered handlers.

### Spawn an agent from Python

```python
import asyncio
from mlx_code.repl import Agent

async def main():
    agent = Agent(system="You are a concise technical writer.")
    await agent.run("Summarise all *.py files changed in the last 7 days. Save to digest.md.")

asyncio.run(main())
```

### Multi-agent pipeline

```python
import asyncio
from mlx_code.repl import Agent

async def main():
    researcher = Agent(system="You are a research assistant.")
    await researcher.run("Research PBFT consensus. Save a structured summary to kb/draft.md.")

    reviewer = Agent(system="You are a critical reviewer.")
    await reviewer.run(
        "Read kb/draft.md. Write a one-paragraph critique to kb/critique.md. "
        "Use only information in that file."
    )

asyncio.run(main())
```

### Parallel workers with `asyncio.gather`

```python
import asyncio
from mlx_code.repl import Agent

async def main():
    topics = ["history", "algorithms", "industry_usage"]
    agents = [Agent() for _ in topics]
    await asyncio.gather(*[
        a.run(f"Research the {t} of Byzantine Fault Tolerance. Save to kb/{t}.md.")
        for a, t in zip(agents, topics)
    ])
    reducer = Agent()
    await reducer.run("Read all files in kb/. Synthesise into final_report.md.")

asyncio.run(main())
```

### Branch an agent

```python
child = agent.branch()   # deep-copies messages; independent worktree
await child.run("Try an alternative approach and save to alt.py.")
```

### Resume a session from a git commit

mlx-code stores the full conversation as JSON in each commit message, so you can restore both the workspace state and the agent's memory from any checkpoint.

```python
import asyncio
from mlx_code.gits import resume_worktree
from mlx_code.repl import Agent, repl

async def main():
    gwt, messages = resume_worktree(".", "abc1234")
    agent = Agent(ctx={"gwt": gwt})
    agent.messages = messages
    await repl(agent)

asyncio.run(main())
```

### Load agent config from a file

Pass a JSON or YAML file to reconfigure the agent at runtime (also available as `/clear --config F` in the TUI).

```python
from mlx_code.repl import load_agent_config
config = load_agent_config("agent.yaml")
agent = Agent(**config)
```

### Custom tools

Subclass `Tool`, define a Pydantic schema, and pass the class at instantiation.

```python
from mlx_code.tools import Tool
from mlx_code.repl import Agent
from pydantic import BaseModel, Field

class QueryParams(BaseModel):
    query: str = Field(description="SQL query to run")

class LiveDBTool(Tool):
    name = "QueryDB"
    description = "Execute a query against the dev database"
    parameters = QueryParams

    async def execute(self, params: QueryParams, signal=None) -> dict:
        result = run_query(params.query)   # your logic here
        return {"content": [{"type": "text", "text": result}], "is_error": False}

agent = Agent(extra_tool_classes=[LiveDBTool], tool_names=["QueryDB"])
```

## Reference

### Commands

| Command | Description |
|---|---|
| `/branch [--rev N] [prompt]` | Open a new branch tab from the current (or earlier) checkpoint |
| `/diff [--all]` | Show a side-by-side diff of changes in the worktree |
| `/clear [--config F]` | Clear conversation; `--config` reloads agent from a JSON/YAML file |
| `/tab [N]` | Jump to tab N |
| `/history [--raw]` | Show conversation transcript; `--raw` shows the raw API message log |
| `/tools` | List active tools |
| `/abort` | Abort the running agent |
| `/errors` | Show timestamped error log for the current tab |
| `/export [path]` | Export session to JSON |
| `/exit [--all]` | Close branch tab, or exit the app |
| `/help` | Show command reference |
| `!command` | Run a shell command; output captured in the TUI (eg, `ls`, `cat hello.c`) |
| `$command` | Run an interactive command (eg, `vim`, `yazi`, `less hello.c`) |

### Key bindings

| Key | Action |
|---|---|
| `Enter` | Submit |
| `Ctrl-J` | Insert newline |
| `Ctrl-1` … `Ctrl-9` | Jump to tab N |
| `Ctrl-,` / `Ctrl-.` | Cycle through tabs |
| `Ctrl-C` | Clear input, or abort running agent |
| `Ctrl-D` | Close branch tab, or exit app |
| `Ctrl-R` | Recall last prompt into editor |

### Tools

| Tool | What it does |
|------|-------------|
| `Read` | Read file with optional offset/limit for large files |
| `Write` | Create or overwrite a file |
| `Edit` | Replace an exact unique string in a file |
| `Bash` | Run a shell command with timeout and abort support |
| `Grep` | Search files by pattern, respects .gitignore |
| `Find` | Find files/dirs by name glob, respects .gitignore |
| `Ls` | List directory contents, respects .gitignore |
| `Skill` | Retrieve named skill instructions from config |
| `Agent` | Spawn an autonomous sub-agent for delegated work |

All file tools enforce path sandboxing. The agent cannot read or write outside the worktree.

### Backends

| Backend | Flag | Notes |
|---------|------|-------|
| MLX (local) | `--api noapi` | Default. Runs on-device, no API key needed |
| Claude | `--api claude` | Requires `ANTHROPIC_API_KEY` |
| Gemini | `--api gemini` | Requires `GOOGLE_API_KEY` |
| DeepSeek | `--api deepseek` | DeepSeek API or compatible endpoint |
| Codex | `--api codex` | OpenAI Codex CLI integration |
| OpenAI | `--api openai` | Any OpenAI-compatible endpoint |

### Frontends

The local MLX server speaks OpenAI, Anthropic, and Gemini wire formats simultaneously, so you can use any compatible CLI as the frontend:

```bash
mlc --leash claude       # claude CLI routes through local model
mlc --leash codex        # codex CLI routes through local model
mlc --leash gemini       # gemini CLI routes through local model
mlc --leash none         # server only
```

---

## Credits

Built on [mlx](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm). Inspired by Mario Zechner's [pi](https://github.com/badlogic/pi-mono).

## License

Apache License 2.0: see [LICENSE](LICENSE) for details.

# mlx-code

[![Link](https://raw.githubusercontent.com/JosefAlbers/mlx-code/main/assets/mlx-code.gif)](https://youtu.be/Rba-uTsYuXg)

### Quick Start

```bash
brew install --cask claude-code
pip install mlx-code
mlx-code
```

### Options

```bash
mlx-code [options] [-- claude options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `mlx-community/Qwen3.5-4B-OptiQ-4bit` | MLX model to load |
| `--port` | `8000` | Server port |
| `--host` | `127.0.0.1` | Server host |
| `--cache` | `cache/cache.safetensors` | Prompt cache file (saved/loaded across runs) |
| `--system` | None | System prompt override. Use `{env}` to inject working directory info |
| `--names` | `Read Edit Write Grep Glob Bash Agent Skill` | Tool names to pass through to the model |
| `--skips` | suggestion-mode block | Regex patterns to strip from message content |
| `--work` | `$CWD` | Working directory mirrored into the Claude session |
| `--home` | temp dir | Home directory for the Claude process |

Any extra arguments after `--` are forwarded to the `claude` CLI

### Licence

Apache License 2.0 — see LICENSE for details.



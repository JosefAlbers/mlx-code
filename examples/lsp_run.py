from __future__ import annotations
import argparse
import asyncio
import os
from mlx_code.repl import run_repl
from mlx_code.lsp_tool import LspTool, shutdown_lsp_servers


def main() -> None:
    parser = argparse.ArgumentParser(description="mlx-code REPL with Ls + Lsp")
    parser.add_argument("--cwd", default=None)
    parser.add_argument("-a", "--api", default="noapi")
    parser.add_argument("-m", "--model", default=None)
    args = parser.parse_args()
    cwd = os.path.abspath(args.cwd or os.getcwd())
    ctx = {"cwd": cwd}
    try:
        run_repl(
            api=args.api,
            model=args.model,
            repo=cwd,
            ctx=ctx,
            tool_names=["Ls", "Lsp"],
            extra_tool_classes=[LspTool],
            init_prompt="explain tree.py",
        )
    finally:
        asyncio.run(shutdown_lsp_servers(ctx))


if __name__ == "__main__":
    main()

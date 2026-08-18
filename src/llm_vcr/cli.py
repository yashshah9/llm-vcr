"""CLI utilities for llm-vcr."""

from __future__ import annotations

import argparse
import sys

from llm_vcr import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llm-vcr", description="Record/replay LLM HTTP traffic.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("health", help="Verify installation")
    args = parser.parse_args(argv)
    if args.command == "health":
        print(f"llm-vcr {__version__} OK")
        return
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()

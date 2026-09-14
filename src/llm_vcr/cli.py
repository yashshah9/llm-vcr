"""CLI utilities for llm-vcr."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_vcr import __version__
from llm_vcr.cassette import Cassette
from llm_vcr.matching import diff_bodies, normalize_body


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llm-vcr", description="Record/replay LLM HTTP traffic.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("health", help="Verify installation")

    diff_p = sub.add_parser(
        "diff",
        help="Show normalized differences between two JSON bodies or cassette interactions",
    )
    diff_p.add_argument("left", help="JSON file, cassette YAML, or '-' for stdin")
    diff_p.add_argument("right", help="JSON file or cassette YAML")
    diff_p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Cassette interaction index when comparing YAML cassettes (default 0)",
    )

    args = parser.parse_args(argv)
    if args.command in {None, "health"}:
        print(f"llm-vcr {__version__} OK")
        print("  pytest plugin: @llm_vcr / llm_vcr_client / --llm-vcr-record")
        print("  commands: health | diff")
        return
    if args.command == "diff":
        left = _load_body(args.left, args.index)
        right = _load_body(args.right, args.index)
        lines = diff_bodies(left, right)
        if not lines:
            print("match (semantically equal after normalization)")
            return
        print("differences (normalized):")
        for line in lines:
            print(f"  {line}")
        sys.exit(1)
    parser.print_help()
    sys.exit(1)


def _load_body(path: str, index: int) -> dict:
    if path == "-":
        return json.load(sys.stdin)
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        cassette = Cassette.load(p)
        if index < 0 or index >= len(cassette.interactions):
            raise SystemExit(f"cassette index {index} out of range")
        body = cassette.interactions[index].request_body
        return body if isinstance(body, dict) else {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise SystemExit("JSON body must be an object")
    return normalize_body(parsed)


if __name__ == "__main__":
    main()

"""CLI utilities for llm-vcr."""

import click


@click.group()
def main() -> None:
    """Record/replay utilities for LLM HTTP traffic."""


@main.command("health")
def health() -> None:
    """Verify installation."""
    click.echo("llm-vcr OK")


if __name__ == "__main__":
    main()

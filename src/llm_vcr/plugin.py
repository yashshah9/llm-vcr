"""pytest plugin and @llm_vcr decorator."""

from __future__ import annotations

import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

import httpx
import pytest

from llm_vcr.cassette import Cassette
from llm_vcr.transport import VCRTransport

F = TypeVar("F", bound=Callable[..., Any])

CASSETTE_DIR = Path(os.environ.get("LLM_VCR_CASSETTE_DIR", "tests/cassettes"))
RECORD_MODE = os.environ.get("LLM_VCR_RECORD", "false").lower() in {"1", "true", "yes"}


def llm_vcr(name: str | None = None) -> Callable[[F], F]:
    """Decorator to enable cassette record/replay for a test function."""

    def decorator(func: F) -> F:
        cassette_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cassette_path = CASSETTE_DIR / f"{cassette_name}.yaml"
            record = RECORD_MODE or not cassette_path.exists()

            if record:
                cassette = Cassette(name=cassette_name)
            else:
                cassette = Cassette.load(cassette_path)

            transport = VCRTransport(cassette, record_mode=record)
            client = httpx.Client(transport=transport)

            try:
                if "http_client" in func.__code__.co_varnames:
                    kwargs["http_client"] = client
                    result = func(*args, **kwargs)
                else:
                    result = func(*args, client=client, **kwargs)
            finally:
                client.close()
                if record:
                    cassette.save(cassette_path)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--llm-vcr-record", action="store_true", help="Record new cassettes")


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--llm-vcr-record"):
        os.environ["LLM_VCR_RECORD"] = "true"

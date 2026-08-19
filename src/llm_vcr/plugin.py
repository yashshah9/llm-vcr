"""pytest plugin and @llm_vcr decorator."""

from __future__ import annotations

import inspect
import os
import warnings
from collections.abc import AsyncGenerator, Callable, Generator
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import httpx
import pytest

from llm_vcr.cassette import Cassette
from llm_vcr.transport import AsyncVCRTransport, VCRTransport

F = TypeVar("F", bound=Callable[..., Any])


def _cassette_dir() -> Path:
    return Path(os.environ.get("LLM_VCR_CASSETTE_DIR", "tests/cassettes"))


def _recording() -> bool:
    return os.environ.get("LLM_VCR_RECORD", "false").lower() in {"1", "true", "yes"}


def _bind_client(func: Callable[..., Any], kwargs: dict[str, Any], client: httpx.Client) -> None:
    params = inspect.signature(func).parameters
    if "http_client" in params and "http_client" not in kwargs:
        kwargs["http_client"] = client
    elif "client" in params and "client" not in kwargs:
        kwargs["client"] = client


def llm_vcr(name: str | None = None, sequential: bool = False) -> Callable[[F], F]:
    """Decorator to enable cassette record/replay for a test function."""

    def decorator(func: F) -> F:
        cassette_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cassette_path = _cassette_dir() / f"{cassette_name}.yaml"
            record = _recording() or not cassette_path.exists()

            if record:
                cassette = Cassette(name=cassette_name)
            else:
                cassette = Cassette.load(cassette_path)

            transport = VCRTransport(cassette, record_mode=record, sequential=sequential)
            client = httpx.Client(transport=transport)

            try:
                _bind_client(func, kwargs, client)
                result = func(*args, **kwargs)
            finally:
                client.close()
                if sequential and not record and transport.unused():
                    warnings.warn(
                        f"cassette {cassette_name} has {transport.unused()} unused interaction(s)",
                        stacklevel=3,
                    )
                if record:
                    cassette.save(cassette_path)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


@pytest.fixture
def llm_vcr_client(request: pytest.FixtureRequest) -> Generator[httpx.Client, None, None]:
    """pytest fixture that yields an httpx client bound to a named cassette."""
    cassette_name = getattr(request, "param", request.node.name)
    cassette_path = _cassette_dir() / f"{cassette_name}.yaml"
    record = _recording() or not cassette_path.exists()
    cassette = Cassette(name=str(cassette_name)) if record else Cassette.load(cassette_path)
    client = httpx.Client(transport=VCRTransport(cassette, record_mode=record))
    yield client
    client.close()
    if record:
        cassette.save(cassette_path)


@pytest.fixture
async def llm_vcr_async_client(
    request: pytest.FixtureRequest,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    cassette_name = getattr(request, "param", request.node.name)
    cassette_path = _cassette_dir() / f"{cassette_name}.yaml"
    record = _recording() or not cassette_path.exists()
    cassette = Cassette(name=str(cassette_name)) if record else Cassette.load(cassette_path)
    client = httpx.AsyncClient(transport=AsyncVCRTransport(cassette, record_mode=record))
    yield client
    await client.aclose()
    if record:
        cassette.save(cassette_path)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--llm-vcr-record", action="store_true", help="Record new cassettes")


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--llm-vcr-record"):
        os.environ["LLM_VCR_RECORD"] = "true"

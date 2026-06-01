"""Shared pytest fixtures.

Provides an :class:`EdgarClient` wired to an ``httpx.MockTransport`` so tests
exercise the real client logic without ever touching the SEC network.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from finsage.data.edgar_client import EdgarClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Default mocked vLLM chat-completion payload used by API route tests.
MOCK_CHAT_PAYLOAD: dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "finsage-7b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The main risks are supply chain disruption and competition.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 42, "completion_tokens": 12, "total_tokens": 54},
}

MOCK_MODELS_PAYLOAD: dict[str, Any] = {
    "object": "list",
    "data": [{"id": "finsage-7b", "object": "model"}],
}


def _load_fixture(name: str) -> str:
    """Read a fixture file's text.

    Args:
        name: File name within the fixtures directory.

    Returns:
        The file contents as a string.
    """
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


@pytest.fixture
def sample_10k_html() -> str:
    """Return the sample 10-K HTML fixture."""
    return _load_fixture("sample_10k.html")


@pytest.fixture
def processed_manifest_path() -> Path:
    """Return the path to the sample processed manifest fixture."""
    return FIXTURES_DIR / "processed_manifest_sample.jsonl"


@pytest.fixture
def eval_test_file() -> Path:
    """Return the path to the sample evaluation test set fixture."""
    return FIXTURES_DIR / "eval_test_sample.jsonl"


@pytest.fixture
def train_sample_file() -> Path:
    """Return the path to the sample training set fixture."""
    return FIXTURES_DIR / "train_sample.jsonl"


@pytest.fixture
def validation_sample_file() -> Path:
    """Return the path to the sample validation set fixture."""
    return FIXTURES_DIR / "validation_sample.jsonl"


@pytest.fixture
def baseline_results_file() -> Path:
    """Return the path to the sample baseline results fixture."""
    return FIXTURES_DIR / "baseline_results_sample.json"


@pytest.fixture
def baseline_predictions_file() -> Path:
    """Return the path to the sample baseline predictions fixture."""
    return FIXTURES_DIR / "baseline_predictions_sample.jsonl"


@pytest.fixture
def finetuned_results_file() -> Path:
    """Return the path to the sample fine-tuned results fixture."""
    return FIXTURES_DIR / "finetuned_results_sample.json"


@pytest.fixture
def finetuned_predictions_file() -> Path:
    """Return the path to the sample fine-tuned predictions fixture."""
    return FIXTURES_DIR / "finetuned_predictions_sample.jsonl"


@pytest.fixture
def mock_transport() -> httpx.MockTransport:
    """Return a MockTransport routing SEC URLs to local fixtures."""
    tickers = _load_fixture("company_tickers_sample.json")
    submissions = _load_fixture("submissions_sample.json")
    sample_html = _load_fixture("sample_10k.html")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "company_tickers.json" in url:
            return httpx.Response(200, text=tickers)
        if "/submissions/CIK" in url:
            return httpx.Response(200, text=submissions)
        if "/Archives/edgar/data/" in url:
            return httpx.Response(200, text=sample_html)
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


class FakeVLLMClient:
    """A stand-in for :class:`VLLMClient` that never touches the network.

    Args:
        chat_payload: Payload returned by chat calls.
        models_payload: Payload returned by health/model calls.
        available: When ``False``, all calls raise ``VLLMClientError``.
    """

    def __init__(
        self,
        chat_payload: dict[str, Any] | None = None,
        models_payload: dict[str, Any] | None = None,
        available: bool = True,
    ) -> None:
        self.chat_payload = chat_payload if chat_payload is not None else MOCK_CHAT_PAYLOAD
        self.models_payload = models_payload if models_payload is not None else MOCK_MODELS_PAYLOAD
        self.available = available
        self.calls: list[dict[str, Any]] = []

    def _guard(self) -> None:
        from finsage.serving.vllm_client import VLLMClientError

        if not self.available:
            raise VLLMClientError("connection refused")

    async def async_health(self) -> dict[str, Any]:
        self._guard()
        return self.models_payload

    async def async_chat(
        self, messages: list[dict[str, str]], max_tokens: int = 256, temperature: float = 0.0
    ) -> dict[str, Any]:
        self._guard()
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        )
        return self.chat_payload

    async def async_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._guard()
        self.calls.append(payload)
        return self.chat_payload


@pytest.fixture
def make_api_app(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[..., Any]]:
    """Return a factory that builds a configured FinSage API app for tests.

    The factory configures auth/rate-limit/disclaimer via env vars, clears the
    cached settings, builds the app, and injects a fake vLLM client.
    """
    from finsage.config import get_settings

    def _factory(
        *,
        secret: str = "test-secret",
        environment: str = "development",
        rate_limit: int = 60,
        disclaimer: bool = True,
        vllm: Any | None = None,
    ) -> Any:
        monkeypatch.setenv("API_SECRET_KEY", secret)
        monkeypatch.setenv("ENVIRONMENT", environment)
        monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", str(rate_limit))
        monkeypatch.setenv("DISCLAIMER_ENABLED", "true" if disclaimer else "false")
        monkeypatch.setenv("SERVED_MODEL_NAME", "finsage-7b")
        get_settings.cache_clear()

        from finsage.serving.app import create_app
        from finsage.serving.routes import get_vllm_client

        app = create_app()
        client = vllm if vllm is not None else FakeVLLMClient()
        app.dependency_overrides[get_vllm_client] = lambda: client
        app.state.test_vllm = client
        return app

    yield _factory
    get_settings.cache_clear()


@pytest.fixture
def edgar_client(tmp_path: Path, mock_transport: httpx.MockTransport) -> EdgarClient:
    """Return an EdgarClient backed by the mock transport (no network)."""
    client = EdgarClient(
        user_agent="FinSage Test test@example.com",
        rate_limit_per_second=0,  # disable throttling in tests
        cache_dir=tmp_path / "cache",
    )
    client._client = httpx.Client(
        transport=mock_transport,
        headers={"User-Agent": client.user_agent},
    )
    return client

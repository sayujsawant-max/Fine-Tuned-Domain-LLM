"""Lightweight client for an OpenAI-compatible vLLM server.

Uses ``httpx`` directly — no ``openai`` package dependency. Talks to the
``/models`` and ``/chat/completions`` endpoints exposed by ``vllm serve``.
"""

from __future__ import annotations

from typing import Any

import httpx

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are FinSage-7B, a financial filing analysis assistant. Answer using only "
    "the provided filing excerpt. Do not provide investment advice."
)


class VLLMClientError(RuntimeError):
    """Raised when a vLLM request fails or returns a malformed response."""


class VLLMClient:
    """Minimal client for an OpenAI-compatible vLLM endpoint.

    Args:
        base_url: Base URL of the vLLM server (e.g. ``http://localhost:8000/v1``).
        model: Served model name.
        api_key: Optional API key; sent as a bearer token when provided.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "finsage-7b",
        api_key: str | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self._async_transport = async_transport

    def _headers(self) -> dict[str, str]:
        """Build request headers, adding bearer auth only if a key is set."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health(self) -> dict[str, Any]:
        """Query the ``/models`` endpoint.

        Returns:
            The parsed JSON payload listing served models.

        Raises:
            VLLMClientError: On a network error or non-2xx response.
        """
        url = f"{self.base_url}/models"
        try:
            response = self._client.get(url, headers=self._headers())
            response.raise_for_status()
            return dict(response.json())
        except httpx.HTTPStatusError as exc:
            raise VLLMClientError(
                f"Health check failed: HTTP {exc.response.status_code} from {url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise VLLMClientError(f"Health check failed: {exc}") from exc

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Send a chat-completion request.

        Args:
            prompt: The user message content.
            system_prompt: Optional system message; a finance-assistant default
                is used when ``None``.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The parsed chat-completion response.

        Raises:
            VLLMClientError: On a network error, non-2xx response, or a payload
                missing the ``choices`` field.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            response = self._client.post(url, json=payload, headers=self._headers())
            response.raise_for_status()
            data = dict(response.json())
        except httpx.HTTPStatusError as exc:
            raise VLLMClientError(
                f"Chat request failed: HTTP {exc.response.status_code} from {url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise VLLMClientError(f"Chat request failed: {exc}") from exc

        if not data.get("choices"):
            raise VLLMClientError(f"Malformed chat response (no 'choices'): {data}")
        return data

    def chat_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Return only the assistant message content from a chat completion.

        Args:
            prompt: The user message content.
            system_prompt: Optional system message.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The assistant message text.

        Raises:
            VLLMClientError: If the response is malformed or missing content.
        """
        data = self.chat(
            prompt, system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature
        )
        return extract_message_content(data)

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Forward a raw OpenAI-compatible chat-completion payload (sync).

        The payload's ``model`` defaults to this client's model when absent.

        Args:
            payload: An OpenAI ``/chat/completions`` request body.

        Returns:
            The parsed chat-completion response, including any ``usage`` field.

        Raises:
            VLLMClientError: On a network error, non-2xx response, or a payload
                missing the ``choices`` field.
        """
        body = {**payload}
        body.setdefault("model", self.model)
        url = f"{self.base_url}/chat/completions"
        try:
            response = self._client.post(url, json=body, headers=self._headers())
            response.raise_for_status()
            data = dict(response.json())
        except httpx.HTTPStatusError as exc:
            raise VLLMClientError(
                f"Chat request failed: HTTP {exc.response.status_code} from {url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise VLLMClientError(f"Chat request failed: {exc}") from exc
        if not data.get("choices"):
            raise VLLMClientError(f"Malformed chat response (no 'choices'): {data}")
        return data

    async def async_health(self) -> dict[str, Any]:
        """Async variant of :meth:`health`.

        Returns:
            The parsed ``/models`` payload.

        Raises:
            VLLMClientError: On a network error or non-2xx response.
        """
        url = f"{self.base_url}/models"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=self._async_transport
            ) as client:
                response = await client.get(url, headers=self._headers())
                response.raise_for_status()
                return dict(response.json())
        except httpx.HTTPStatusError as exc:
            raise VLLMClientError(
                f"Health check failed: HTTP {exc.response.status_code} from {url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise VLLMClientError(f"Health check failed: {exc}") from exc

    async def async_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Async variant of :meth:`chat_completion`.

        Args:
            payload: An OpenAI ``/chat/completions`` request body.

        Returns:
            The parsed chat-completion response, including any ``usage`` field.

        Raises:
            VLLMClientError: On a network error, non-2xx response, or a payload
                missing the ``choices`` field.
        """
        body = {**payload}
        body.setdefault("model", self.model)
        url = f"{self.base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=self._async_transport
            ) as client:
                response = await client.post(url, json=body, headers=self._headers())
                response.raise_for_status()
                data = dict(response.json())
        except httpx.HTTPStatusError as exc:
            raise VLLMClientError(
                f"Chat request failed: HTTP {exc.response.status_code} from {url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise VLLMClientError(f"Chat request failed: {exc}") from exc
        if not data.get("choices"):
            raise VLLMClientError(f"Malformed chat response (no 'choices'): {data}")
        return data

    async def async_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Send an async chat completion from prebuilt messages.

        Args:
            messages: OpenAI-style messages list.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The parsed chat-completion response.

        Raises:
            VLLMClientError: On a network error, non-2xx response, or malformed
                payload.
        """
        return await self.async_chat_completion(
            {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        )


def extract_message_content(data: dict[str, Any]) -> str:
    """Extract the assistant message content from a chat-completion payload.

    Args:
        data: A parsed chat-completion response.

    Returns:
        The assistant message text.

    Raises:
        VLLMClientError: If the content cannot be located.
    """
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VLLMClientError(f"Could not extract message content: {data}") from exc
    return str(content)

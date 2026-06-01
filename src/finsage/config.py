"""Application settings loaded from the environment / ``.env`` file.

The :class:`Settings` object is the single source of truth for runtime
configuration. It is intentionally lightweight so it can be imported on a
plain developer machine without any GPU or ML dependencies installed.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Task types the FinSage-7B API accepts as a ``task_type`` hint.
SUPPORTED_TASK_TYPES: tuple[str, ...] = (
    "risk_summary",
    "mda_explanation",
    "metric_extraction",
    "yoy_comparison",
    "business_risk_identification",
    "revenue_driver_explanation",
    "filing_qa",
    "analyst_summary",
    "outlook_classification",
    "hallucination_detection",
)


class Settings(BaseSettings):
    """Runtime configuration for FinSage-7B.

    Values are read from environment variables (case-insensitive) and from a
    ``.env`` file if present. See ``.env.example`` for the full list.

    Attributes:
        hf_token: Hugging Face access token, used for model/dataset access.
        wandb_api_key: Weights & Biases API key for training runs.
        openai_api_key: OpenAI key for the optional LLM judge during evaluation.
        edgar_user_agent: Descriptive User-Agent required by SEC EDGAR.
        model_id: Base model identifier on the Hugging Face Hub.
        adapter_path: Local path to the trained LoRA adapter.
        merged_model_path: Local path to the merged (adapter + base) model.
        vllm_host: Host of the internal vLLM inference server.
        vllm_port: Port of the internal vLLM inference server.
        vllm_base_url: OpenAI-compatible base URL of the vLLM server. When unset
            it is derived from ``vllm_host`` / ``vllm_port``.
        served_model_name: Model name advertised by the vLLM server.
        request_timeout_seconds: Per-request timeout for vLLM calls.
        api_host: Host the public FastAPI service binds to.
        api_port: Port the public FastAPI service binds to.
        api_secret_key: Shared secret used to authenticate API requests.
        environment: Deployment environment (``"development"`` or
            ``"production"``); controls strictness of auth defaults.
        cors_allowed_origins: Comma-separated list of allowed CORS origins.
        rate_limit_requests_per_minute: Per-client request budget per minute.
        disclaimer_enabled: Whether the financial disclaimer is injected.
        log_request_body: Whether to log raw request bodies (off by default to
            avoid logging filing text).
        log_level: Root logging level (e.g. ``"INFO"``, ``"DEBUG"``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Secrets / credentials
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    wandb_api_key: str | None = Field(default=None, alias="WANDB_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    edgar_user_agent: str | None = Field(default=None, alias="EDGAR_USER_AGENT")

    # Model + checkpoint paths
    model_id: str = Field(default="mistralai/Mistral-7B-Instruct-v0.3", alias="MODEL_ID")
    adapter_path: str = Field(default="./checkpoints/finsage-7b-adapter", alias="ADAPTER_PATH")
    merged_model_path: str = Field(
        default="./checkpoints/finsage-7b-merged", alias="MERGED_MODEL_PATH"
    )

    # vLLM internal server
    vllm_host: str = Field(default="localhost", alias="VLLM_HOST")
    vllm_port: int = Field(default=8000, alias="VLLM_PORT")
    vllm_base_url: str | None = Field(default=None, alias="VLLM_BASE_URL")
    served_model_name: str = Field(default="finsage-7b", alias="SERVED_MODEL_NAME")
    request_timeout_seconds: float = Field(default=60.0, alias="REQUEST_TIMEOUT_SECONDS")

    # Public FastAPI service
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8080, alias="API_PORT")
    api_secret_key: str = Field(default="change-me", alias="API_SECRET_KEY")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    cors_allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:8501",
        alias="CORS_ALLOWED_ORIGINS",
    )
    rate_limit_requests_per_minute: int = Field(default=60, alias="RATE_LIMIT_REQUESTS_PER_MINUTE")
    disclaimer_enabled: bool = Field(default=True, alias="DISCLAIMER_ENABLED")
    log_request_body: bool = Field(default=False, alias="LOG_REQUEST_BODY")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @model_validator(mode="after")
    def _default_vllm_base_url(self) -> Settings:
        """Derive ``vllm_base_url`` from host/port when not explicitly set."""
        if not self.vllm_base_url:
            self.vllm_base_url = f"http://{self.vllm_host}:{self.vllm_port}/v1"
        return self

    @property
    def cors_origins(self) -> list[str]:
        """Return CORS origins as a list (empty when unset).

        Returns:
            The configured allowed origins, comma-separated values trimmed.
        """
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        """Return whether the service is configured for production."""
        return self.environment.strip().lower() == "production"

    @property
    def effective_vllm_base_url(self) -> str:
        """Return the resolved vLLM base URL as a non-null string."""
        return self.vllm_base_url or f"http://{self.vllm_host}:{self.vllm_port}/v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Using an LRU cache means the environment is parsed once per process, which
    keeps imports cheap and makes the settings object easy to inject in tests.

    Returns:
        The process-wide :class:`Settings` singleton.
    """
    return Settings()

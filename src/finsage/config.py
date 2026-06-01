"""Application settings loaded from the environment / ``.env`` file.

The :class:`Settings` object is the single source of truth for runtime
configuration. It is intentionally lightweight so it can be imported on a
plain developer machine without any GPU or ML dependencies installed.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        api_host: Host the public FastAPI service binds to.
        api_port: Port the public FastAPI service binds to.
        api_secret_key: Shared secret used to authenticate API requests.
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

    # Public FastAPI service
    api_host: str = Field(default="localhost", alias="API_HOST")
    api_port: int = Field(default=8080, alias="API_PORT")
    api_secret_key: str = Field(default="change-me", alias="API_SECRET_KEY")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def vllm_base_url(self) -> str:
        """Return the OpenAI-compatible base URL of the vLLM server."""
        return f"http://{self.vllm_host}:{self.vllm_port}/v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Using an LRU cache means the environment is parsed once per process, which
    keeps imports cheap and makes the settings object easy to inject in tests.

    Returns:
        The process-wide :class:`Settings` singleton.
    """
    return Settings()

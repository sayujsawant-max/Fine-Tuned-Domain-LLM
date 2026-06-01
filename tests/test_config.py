"""Tests for finsage.config.Settings."""

from __future__ import annotations

from finsage.config import Settings, get_settings


def test_settings_loads_defaults(monkeypatch):
    """Settings should fall back to documented defaults when env is empty."""
    for var in ("MODEL_ID", "VLLM_PORT", "API_PORT", "API_SECRET_KEY", "LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.model_id == "mistralai/Mistral-7B-Instruct-v0.3"
    assert settings.vllm_host == "localhost"
    assert settings.vllm_port == 8000
    assert settings.api_port == 8080
    assert settings.api_secret_key == "change-me"
    assert settings.log_level == "INFO"


def test_settings_reads_env(monkeypatch):
    """Settings should read overrides from the environment via aliases."""
    monkeypatch.setenv("MODEL_ID", "some/other-model")
    monkeypatch.setenv("VLLM_PORT", "9001")

    settings = Settings(_env_file=None)

    assert settings.model_id == "some/other-model"
    assert settings.vllm_port == 9001
    assert settings.vllm_base_url == "http://localhost:9001/v1"


def test_get_settings_is_cached():
    """get_settings should return the same cached instance."""
    assert get_settings() is get_settings()

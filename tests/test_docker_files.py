"""Tests validating the Docker / Compose deployment files (no Docker required)."""

from __future__ import annotations

from pathlib import Path

import yaml

DOCKER = Path("docker")
COMPOSE = DOCKER / "docker-compose.yml"
COMPOSE_DEMO = DOCKER / "docker-compose.demo.yml"
COMPOSE_GPU = DOCKER / "docker-compose.gpu.yml"
DOCKERFILES = [
    DOCKER / "Dockerfile.api",
    DOCKER / "Dockerfile.serving",
    DOCKER / "Dockerfile.frontend",
]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_compose_files_exist():
    """All three compose files exist."""
    assert COMPOSE.exists()
    assert COMPOSE_DEMO.exists()
    assert COMPOSE_GPU.exists()


def test_dockerfiles_exist():
    """All three Dockerfiles exist."""
    for df in DOCKERFILES:
        assert df.exists(), df


def test_compose_defines_all_services():
    """The base compose defines vllm, api, and frontend services."""
    services = _load(COMPOSE)["services"]
    assert {"vllm", "api", "frontend"} <= set(services)


def test_api_depends_on_vllm_health():
    """The api service waits for vLLM to be healthy."""
    api = _load(COMPOSE)["services"]["api"]
    assert api["depends_on"]["vllm"]["condition"] == "service_healthy"


def test_frontend_depends_on_api_health():
    """The frontend service waits for the API to be healthy."""
    frontend = _load(COMPOSE)["services"]["frontend"]
    assert frontend["depends_on"]["api"]["condition"] == "service_healthy"


def test_compose_files_are_valid_yaml():
    """Every compose file parses as valid YAML with a services map."""
    for path in (COMPOSE, COMPOSE_DEMO, COMPOSE_GPU):
        data = _load(path)
        assert "services" in data, path


def test_demo_compose_has_no_vllm():
    """Demo mode runs without a vLLM service and enables frontend demo mode."""
    services = _load(COMPOSE_DEMO)["services"]
    assert "vllm" not in services
    frontend = services["frontend-demo"]
    assert frontend["environment"]["NEXT_PUBLIC_DEMO_MODE"] == "true"


def test_gpu_override_adds_gpu_reservation():
    """The GPU override reserves an NVIDIA device for vLLM."""
    vllm = _load(COMPOSE_GPU)["services"]["vllm"]
    devices = vllm["deploy"]["resources"]["reservations"]["devices"]
    assert any(d.get("driver") == "nvidia" for d in devices)
    assert vllm["environment"]["NVIDIA_VISIBLE_DEVICES"] == "all"


def test_api_secret_not_hardcoded():
    """API_SECRET_KEY uses env interpolation, never a real literal."""
    for path in (COMPOSE, COMPOSE_DEMO):
        services = _load(path)["services"]
        for svc in services.values():
            env = svc.get("environment", {})
            if "API_SECRET_KEY" in env:
                assert "${" in str(env["API_SECRET_KEY"]), path


def test_api_dockerfile_does_not_install_vllm():
    """The API image must not install vLLM / GPU dependencies."""
    text = (DOCKER / "Dockerfile.api").read_text(encoding="utf-8")
    # The serving extra (which pulls vLLM) must never be installed here.
    assert "[serving]" not in text
    assert "pip install vllm" not in text.lower()
    # Base install only — no extras bracket on the local package install.
    assert "pip install .[" not in text


def test_serving_dockerfile_installs_serving_extra():
    """The serving image installs the vLLM serving extra."""
    text = (DOCKER / "Dockerfile.serving").read_text(encoding="utf-8")
    assert "[serving]" in text


def test_serving_dockerfile_cuda_base_is_parameterised():
    """The CUDA base image is overridable via a build arg."""
    text = (DOCKER / "Dockerfile.serving").read_text(encoding="utf-8")
    assert "ARG CUDA_IMAGE=" in text
    assert "FROM ${CUDA_IMAGE}" in text


def test_vllm_not_published_to_all_interfaces():
    """vLLM's host port (if any) must bind to loopback only, never 0.0.0.0."""
    vllm = _load(COMPOSE)["services"]["vllm"]
    for mapping in vllm.get("ports", []):
        # Mapping is "127.0.0.1:8000:8000"; reject bare "8000:8000" / "0.0.0.0:".
        assert str(mapping).startswith("127.0.0.1:"), mapping

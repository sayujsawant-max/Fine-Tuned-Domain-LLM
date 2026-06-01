"""Tests for the Phase 10 deployment scripts (no Docker/GPU required)."""

from __future__ import annotations

import os
import re
from pathlib import Path

from scripts.check_full_stack import app as check_app
from typer.testing import CliRunner

runner = CliRunner()

DEPLOY_LOCAL = Path("scripts/deploy_local.sh")
DEPLOY_GPU_VM = Path("scripts/deploy_gpu_vm.sh")
EXPORT_BUNDLE = Path("scripts/export_deployment_bundle.sh")

# Allowed non-secret placeholder tokens.
PLACEHOLDERS = {"change-me", "change-this-before-deployment"}


def test_deploy_local_exists_and_executable():
    """deploy_local.sh exists and is marked executable."""
    assert DEPLOY_LOCAL.exists()
    assert os.access(DEPLOY_LOCAL, os.X_OK)


def test_deploy_local_supports_all_modes():
    """deploy_local.sh handles --demo, --full, and --gpu."""
    content = DEPLOY_LOCAL.read_text(encoding="utf-8")
    assert "--demo" in content
    assert "--full" in content
    assert "--gpu" in content
    assert "docker-compose.demo.yml" in content
    assert "docker-compose.gpu.yml" in content
    assert "set -euo pipefail" in content


def test_deploy_gpu_vm_exists_and_executable():
    """deploy_gpu_vm.sh exists and is marked executable."""
    assert DEPLOY_GPU_VM.exists()
    assert os.access(DEPLOY_GPU_VM, os.X_OK)
    content = DEPLOY_GPU_VM.read_text(encoding="utf-8")
    assert "NVIDIA Container Toolkit" in content


def test_export_bundle_exists_and_executable():
    """export_deployment_bundle.sh exists and is marked executable."""
    assert EXPORT_BUNDLE.exists()
    assert os.access(EXPORT_BUNDLE, os.X_OK)
    content = EXPORT_BUNDLE.read_text(encoding="utf-8")
    # The bundle must never include weights/checkpoints/secrets.
    assert "finsage-deployment-bundle" in content


def test_check_full_stack_help():
    """check_full_stack.py --help renders and documents the URLs."""
    result = runner.invoke(check_app, ["--help"])
    assert result.exit_code == 0
    assert "frontend-url" in result.output
    assert "api-url" in result.output
    assert "vllm-url" in result.output


# Matches a real shell assignment at the start of a line: `API_SECRET_KEY=...`
# or `export API_SECRET_KEY=...` (ignores mentions inside echo/comments).
_ASSIGN = re.compile(r"^\s*(?:export\s+)?API_SECRET_KEY=(\S+)")


def test_no_hardcoded_real_secrets():
    """Scripts contain no secret literals beyond known placeholders."""
    for script in (DEPLOY_LOCAL, DEPLOY_GPU_VM, EXPORT_BUNDLE):
        for line in script.read_text(encoding="utf-8").splitlines():
            match = _ASSIGN.match(line)
            if not match:
                continue
            value = match.group(1)
            if "${" in value:  # env interpolation is safe
                continue
            cleaned = value.strip("\"'")
            assert cleaned in PLACEHOLDERS or cleaned == "", f"suspicious secret: {line!r}"

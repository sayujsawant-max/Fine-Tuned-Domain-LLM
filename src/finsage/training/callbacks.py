"""Training callbacks for QLoRA fine-tuning (Phase 5).

The concrete callbacks subclass ``transformers.TrainerCallback``, which is
imported lazily inside builder functions so this module loads on a CPU-only
machine without transformers installed. Call a builder to instantiate a
callback; a helpful :class:`ImportError` is raised if transformers is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

_MISSING_DEPS_MSG = (
    "Training callbacks require transformers. Install training dependencies with "
    "pip install -e '.[training,ml]'"
)


def _require_trainer_callback() -> Any:
    """Return ``transformers.TrainerCallback`` or raise a helpful error.

    Returns:
        The ``TrainerCallback`` base class.

    Raises:
        ImportError: If transformers is not installed.
    """
    try:
        from transformers import TrainerCallback
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise ImportError(_MISSING_DEPS_MSG) from exc
    return TrainerCallback


def build_loss_logging_callback() -> Any:
    """Build a callback that logs train/eval loss as they are reported.

    Returns:
        A ``TrainerCallback`` instance.

    Raises:
        ImportError: If transformers is not installed.
    """
    base = _require_trainer_callback()

    class LossLoggingCallback(base):  # type: ignore[misc, valid-type]
        """Logs train and eval loss from each ``on_log`` event."""

        def on_log(
            self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any
        ) -> None:
            """Log loss values when present in ``logs``."""
            logs = logs or {}
            if "loss" in logs:
                logger.info("step %s | train_loss=%.4f", state.global_step, logs["loss"])
            if "eval_loss" in logs:
                logger.info("step %s | eval_loss=%.4f", state.global_step, logs["eval_loss"])

    return LossLoggingCallback()


def build_nan_loss_callback() -> Any:
    """Build a callback that aborts training on a NaN/Inf loss.

    Returns:
        A ``TrainerCallback`` instance.

    Raises:
        ImportError: If transformers is not installed.
    """
    base = _require_trainer_callback()
    import math

    class NaNLossCallback(base):  # type: ignore[misc, valid-type]
        """Stops training if a non-finite loss is observed."""

        def on_log(
            self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any
        ) -> Any:
            """Set ``control.should_training_stop`` if loss is NaN/Inf."""
            logs = logs or {}
            loss = logs.get("loss")
            if loss is not None and not math.isfinite(float(loss)):
                logger.error("Non-finite loss (%s) at step %s; stopping.", loss, state.global_step)
                control.should_training_stop = True
            return control

    return NaNLossCallback()


def build_save_config_callback(configs: dict[str, Any], output_dir: Path | str) -> Any:
    """Build a callback that snapshots config dicts at training start.

    Args:
        configs: A mapping of name -> config dict to snapshot.
        output_dir: Directory to write the snapshot into.

    Returns:
        A ``TrainerCallback`` instance.

    Raises:
        ImportError: If transformers is not installed.
    """
    base = _require_trainer_callback()
    out_dir = Path(output_dir)

    class SaveConfigCallback(base):  # type: ignore[misc, valid-type]
        """Writes config snapshots to ``output_dir`` on training start."""

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            """Write the config snapshot."""
            out_dir.mkdir(parents=True, exist_ok=True)
            snapshot = out_dir / "config_snapshot.json"
            snapshot.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Saved config snapshot to %s", snapshot)

    return SaveConfigCallback()


def build_default_callbacks(
    configs: dict[str, Any] | None = None,
    output_dir: Path | str = "checkpoints/finsage-7b",
) -> list[Any]:
    """Build the default set of training callbacks.

    Args:
        configs: Optional config snapshot to save at training start.
        output_dir: Directory for the config snapshot.

    Returns:
        A list of ``TrainerCallback`` instances.

    Raises:
        ImportError: If transformers is not installed.
    """
    callbacks = [build_loss_logging_callback(), build_nan_loss_callback()]
    if configs is not None:
        callbacks.append(build_save_config_callback(configs, output_dir))
    return callbacks

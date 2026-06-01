"""Training callbacks (Phase 6 stub).

Returns lightweight, framework-agnostic callback descriptors in Phase 1. The
Phase 6 implementation maps these to ``transformers.TrainerCallback`` instances
(e.g. early stopping, W&B logging) once the ``training`` extras are installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from finsage.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CallbackSpec:
    """A serialisable description of a training callback.

    Attributes:
        name: Identifier of the callback (e.g. ``"early_stopping"``).
        params: Keyword parameters used to construct the concrete callback.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


def build_default_callbacks(
    early_stopping_patience: int = 3,
    log_to_wandb: bool = True,
) -> list[CallbackSpec]:
    """Return the default set of callback specs for a training run.

    Args:
        early_stopping_patience: Eval steps without improvement before stopping.
        log_to_wandb: Whether to include a Weights & Biases logging callback.

    Returns:
        A list of :class:`CallbackSpec` describing the callbacks to attach.
    """
    specs = [
        CallbackSpec(
            name="early_stopping",
            params={"patience": early_stopping_patience},
        )
    ]
    if log_to_wandb:
        specs.append(CallbackSpec(name="wandb", params={}))
    logger.debug("Built %d default callback specs", len(specs))
    return specs

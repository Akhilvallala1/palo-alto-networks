"""Classifier tuning knobs, read from the optional `classifier:` block of
`config/routing.yaml`.

`conduit.config.RoutingConfig` models the two blocks the epic froze — `tiers`
and `budgets` — and ignores anything else in the file, so the router keeps its
own tuning parameters here rather than widening a schema that four components
share. Overrides follow the same convention as the rest of the config layer::

    CONDUIT_ROUTING__CLASSIFIER__CONFIDENCE_THRESHOLD=0.7
    CONDUIT_ROUTING__CLASSIFIER__CACHE_SIZE=2048
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from conduit.config import ConfigError, default_config_dir
from conduit.contracts import Complexity

__all__ = ["CLASSIFIER_ENV_PREFIX", "ClassifierSettings", "load_classifier_settings"]

CLASSIFIER_ENV_PREFIX = "CONDUIT_ROUTING__CLASSIFIER__"

# L2C workflow tags (see the epic's graph: intake=trivial, analyst=complex).
DEFAULT_WORKFLOW_HINTS: dict[str, Complexity] = {
    "intake": Complexity.TRIVIAL,
    "extraction": Complexity.TRIVIAL,
    "classification": Complexity.TRIVIAL,
    "lead_scoring": Complexity.TRIVIAL,
    "policy_rag": Complexity.STANDARD,
    "explainer": Complexity.STANDARD,
    "marketing_copy": Complexity.STANDARD,
    "summarization": Complexity.STANDARD,
    "discount_analyst": Complexity.COMPLEX,
    "approval_router": Complexity.COMPLEX,
    "quote_reasoning": Complexity.COMPLEX,
    "comp_planning": Complexity.COMPLEX,
}


class ClassifierSettings(BaseModel):
    """Thresholds and cache sizing for the two-stage classifier."""

    confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    cache_size: int = Field(default=512, ge=0)
    tiebreak_enabled: bool = True
    tiebreak_model: str | None = None  # None => cheapest model in the trivial chain
    workflow_hints: dict[str, Complexity] = Field(
        default_factory=lambda: dict(DEFAULT_WORKFLOW_HINTS)
    )


def _coerce_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def load_classifier_settings(
    config_dir: Path | None = None, env: Mapping[str, str] | None = None
) -> ClassifierSettings:
    """Load the `classifier:` block of `routing.yaml`, applying env overrides.

    The block is optional: a `routing.yaml` without it yields defaults.
    """
    directory = default_config_dir() if config_dir is None else Path(config_dir)
    path = directory / "routing.yaml"
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            parsed: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read classifier settings from {path}: {exc}") from exc
        if isinstance(parsed, dict):
            block = parsed.get("classifier")
            if block is not None:
                if not isinstance(block, dict):
                    raise ConfigError(f"{path}: `classifier` must be a mapping")
                data = {str(key): value for key, value in block.items()}

    environ = os.environ if env is None else env
    for env_key in sorted(environ):
        if not env_key.upper().startswith(CLASSIFIER_ENV_PREFIX):
            continue
        field = env_key[len(CLASSIFIER_ENV_PREFIX) :].lower()
        if not field:
            raise ConfigError(f"{env_key} has no key after the {CLASSIFIER_ENV_PREFIX!r} prefix")
        data[field] = _coerce_scalar(environ[env_key])

    try:
        return ClassifierSettings.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid classifier settings in {path}:\n{exc}") from exc

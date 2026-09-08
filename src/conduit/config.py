"""Typed loader for `config/*.yaml` with environment-variable overrides.

Configuration is plumbing, not policy: this module parses and validates YAML
into pydantic models. Interpreting those values — picking a model, pricing a
request, enforcing a budget — belongs to the router, registry and guards.

Overrides use the form `CONDUIT_<SECTION>__<KEY>[__<KEY>...]`, where `SECTION`
is the config file stem and `__` separates nesting levels. Values are parsed as
YAML scalars, so numbers, booleans and inline lists survive the round trip::

    CONDUIT_ROUTING__BUDGETS__DEFAULT_DAILY_USD=25.0
    CONDUIT_GUARDS__INJECTION__RISK_THRESHOLD=0.9
    CONDUIT_ROUTING__TIERS__TRIVIAL='["mock:echo"]'
"""

import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from conduit.contracts import Complexity

__all__ = [
    "BudgetSettings",
    "ConduitConfig",
    "ConfigError",
    "GuardsConfig",
    "InjectionSettings",
    "ModelSpec",
    "ModelsConfig",
    "PIISettings",
    "RoutingConfig",
    "default_config_dir",
    "load_config",
    "load_guards",
    "load_models",
    "load_routing",
]

ENV_PREFIX = "CONDUIT_"
CONFIG_DIR_ENV = "CONDUIT_CONFIG_DIR"


class ConfigError(Exception):
    """Raised when configuration is missing, unparseable, or fails validation."""


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class BudgetSettings(BaseModel):
    default_daily_usd: float = Field(ge=0.0)
    on_exceed: Literal["downgrade_tier", "reject"]


class RoutingConfig(BaseModel):
    """`config/routing.yaml` — tier to model-chain mapping and budget policy."""

    tiers: dict[Complexity, list[str]]
    budgets: BudgetSettings

    @field_validator("tiers")
    @classmethod
    def _every_tier_has_a_chain(
        cls, value: dict[Complexity, list[str]]
    ) -> dict[Complexity, list[str]]:
        missing = sorted(tier.value for tier in Complexity if tier not in value)
        if missing:
            raise ValueError(f"no model chain configured for tier(s): {', '.join(missing)}")
        empty = sorted(tier.value for tier, chain in value.items() if not chain)
        if empty:
            raise ValueError(f"empty model chain for tier(s): {', '.join(empty)}")
        return value


class ModelSpec(BaseModel):
    """One row of the model registry: who serves it, what it costs, where it fits."""

    provider: str
    input_per_1m_usd: float = Field(ge=0.0)
    output_per_1m_usd: float = Field(ge=0.0)
    context_window: int = Field(gt=0)
    tiers: list[Complexity]


class ModelsConfig(BaseModel):
    """`config/models.yaml` — the model registry. Prices live here, never in logic."""

    models: dict[str, ModelSpec]


class PIISettings(BaseModel):
    enabled: bool = True
    entities: list[str] = Field(default_factory=list)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    redact: bool = True


class InjectionSettings(BaseModel):
    enabled: bool = True
    risk_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    llm_classifier: bool = False
    llm_classifier_above: float = Field(default=0.5, ge=0.0, le=1.0)


class GuardsConfig(BaseModel):
    """`config/guards.yaml` — PII and prompt-injection guard settings."""

    fail_closed: bool = True
    pii: PIISettings = Field(default_factory=PIISettings)
    injection: InjectionSettings = Field(default_factory=InjectionSettings)


class ConduitConfig(BaseModel):
    """Every config file, loaded and validated together."""

    routing: RoutingConfig
    models: ModelsConfig
    guards: GuardsConfig


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def default_config_dir() -> Path:
    """The `config/` directory: `$CONDUIT_CONFIG_DIR`, else the repo-root one."""
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "config"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    try:
        parsed: object = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ConfigError(f"{path} must contain a YAML mapping, got {type(parsed).__name__}")
    return {str(key): value for key, value in parsed.items()}


def _coerce_scalar(raw: str) -> Any:
    """Parse an env value as a YAML scalar, falling back to the literal string."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _assign(data: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = data
    for key in path[:-1]:
        nested = cursor.get(key)
        if not isinstance(nested, dict):
            nested = {}
            cursor[key] = nested
        cursor = nested
    cursor[path[-1]] = value


def _apply_env_overrides(
    section: str, data: dict[str, Any], env: Mapping[str, str]
) -> dict[str, Any]:
    prefix = f"{ENV_PREFIX}{section.upper()}__"
    merged = deepcopy(data)
    for env_key in sorted(env):
        if not env_key.upper().startswith(prefix):
            continue
        path = [part.lower() for part in env_key[len(prefix) :].split("__") if part]
        if not path:
            raise ConfigError(f"{env_key} has no key path after the {prefix!r} prefix")
        _assign(merged, path, _coerce_scalar(env[env_key]))
    return merged


ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_section(
    schema: type[ModelT],
    section: str,
    config_dir: Path | None,
    env: Mapping[str, str] | None,
) -> ModelT:
    directory = default_config_dir() if config_dir is None else Path(config_dir)
    path = directory / f"{section}.yaml"
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    data = _apply_env_overrides(section, _read_yaml(path), os.environ if env is None else env)
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {path}:\n{exc}") from exc


def load_routing(
    config_dir: Path | None = None, env: Mapping[str, str] | None = None
) -> RoutingConfig:
    """Load `routing.yaml`, applying `CONDUIT_ROUTING__*` overrides."""
    return _load_section(RoutingConfig, "routing", config_dir, env)


def load_models(
    config_dir: Path | None = None, env: Mapping[str, str] | None = None
) -> ModelsConfig:
    """Load `models.yaml`, applying `CONDUIT_MODELS__*` overrides."""
    return _load_section(ModelsConfig, "models", config_dir, env)


def load_guards(
    config_dir: Path | None = None, env: Mapping[str, str] | None = None
) -> GuardsConfig:
    """Load `guards.yaml`, applying `CONDUIT_GUARDS__*` overrides."""
    return _load_section(GuardsConfig, "guards", config_dir, env)


def load_config(
    config_dir: Path | None = None, env: Mapping[str, str] | None = None
) -> ConduitConfig:
    """Load and validate every config file as one `ConduitConfig`."""
    return ConduitConfig(
        routing=load_routing(config_dir, env),
        models=load_models(config_dir, env),
        guards=load_guards(config_dir, env),
    )

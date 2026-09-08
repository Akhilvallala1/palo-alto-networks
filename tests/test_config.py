"""Tests for the typed config loader and its env-var overrides."""

from pathlib import Path

import pytest

from conduit.config import (
    CONFIG_DIR_ENV,
    ConfigError,
    default_config_dir,
    load_config,
    load_guards,
    load_models,
    load_routing,
)
from conduit.contracts import Complexity

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"


def test_repo_config_loads() -> None:
    cfg = load_config(REPO_CONFIG, env={})
    assert set(cfg.routing.tiers) == set(Complexity)
    assert cfg.routing.budgets.default_daily_usd == 5.00
    assert cfg.routing.budgets.on_exceed == "downgrade_tier"
    assert cfg.guards.fail_closed is True


def test_every_routed_model_exists_in_the_registry() -> None:
    """A chain naming an unknown model would break the router at request time."""
    cfg = load_config(REPO_CONFIG, env={})
    routed = {model for chain in cfg.routing.tiers.values() for model in chain}
    assert routed <= set(cfg.models.models)


def test_registry_prices_and_tier_eligibility() -> None:
    models = load_models(REPO_CONFIG, env={}).models
    haiku = models["claude-haiku-4-5-20251001"]
    assert haiku.provider == "anthropic"
    assert haiku.output_per_1m_usd > haiku.input_per_1m_usd
    assert Complexity.TRIVIAL in haiku.tiers
    assert models["mock:echo"].input_per_1m_usd == 0.0


def test_default_config_dir_honours_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG_DIR_ENV, str(REPO_CONFIG))
    assert default_config_dir() == REPO_CONFIG


def test_env_override_replaces_a_scalar() -> None:
    cfg = load_routing(REPO_CONFIG, env={"CONDUIT_ROUTING__BUDGETS__DEFAULT_DAILY_USD": "25.5"})
    assert cfg.budgets.default_daily_usd == 25.5


def test_env_override_coerces_types_not_just_strings() -> None:
    guards = load_guards(
        REPO_CONFIG,
        env={
            "CONDUIT_GUARDS__INJECTION__RISK_THRESHOLD": "0.9",
            "CONDUIT_GUARDS__PII__ENABLED": "false",
            "CONDUIT_GUARDS__FAIL_CLOSED": "false",
        },
    )
    assert guards.injection.risk_threshold == 0.9
    assert guards.pii.enabled is False
    assert guards.fail_closed is False


def test_env_override_parses_an_inline_list() -> None:
    cfg = load_routing(REPO_CONFIG, env={"CONDUIT_ROUTING__TIERS__TRIVIAL": '["mock:echo"]'})
    assert cfg.tiers[Complexity.TRIVIAL] == ["mock:echo"]


def test_env_override_leaves_untouched_keys_alone() -> None:
    baseline = load_routing(REPO_CONFIG, env={})
    overridden = load_routing(
        REPO_CONFIG, env={"CONDUIT_ROUTING__BUDGETS__DEFAULT_DAILY_USD": "1.0"}
    )
    assert overridden.tiers == baseline.tiers
    assert overridden.budgets.on_exceed == baseline.budgets.on_exceed


def test_env_vars_for_other_sections_are_ignored() -> None:
    cfg = load_routing(REPO_CONFIG, env={"CONDUIT_GUARDS__FAIL_CLOSED": "false", "PATH": "/x"})
    assert cfg.budgets.default_daily_usd == 5.00


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="missing config file"):
        load_routing(tmp_path, env={})


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    (tmp_path / "routing.yaml").write_text("tiers: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_routing(tmp_path, env={})


def test_non_mapping_yaml_raises_config_error(tmp_path: Path) -> None:
    (tmp_path / "guards.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_guards(tmp_path, env={})


def test_missing_tier_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "routing.yaml").write_text(
        "tiers:\n  trivial: [mock:echo]\nbudgets:\n  default_daily_usd: 1.0\n  on_exceed: reject\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="no model chain configured"):
        load_routing(tmp_path, env={})


def test_empty_tier_chain_is_rejected() -> None:
    with pytest.raises(ConfigError, match="empty model chain"):
        load_routing(REPO_CONFIG, env={"CONDUIT_ROUTING__TIERS__COMPLEX": "[]"})


def test_unknown_on_exceed_value_is_rejected() -> None:
    with pytest.raises(ConfigError, match="invalid config"):
        load_routing(REPO_CONFIG, env={"CONDUIT_ROUTING__BUDGETS__ON_EXCEED": "explode"})


def test_env_override_without_a_key_path_is_rejected() -> None:
    with pytest.raises(ConfigError, match="no key path"):
        load_routing(REPO_CONFIG, env={"CONDUIT_ROUTING__": "1"})

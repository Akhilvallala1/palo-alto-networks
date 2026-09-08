"""Unit tests for the classifier's slice of `config/routing.yaml`."""

from pathlib import Path

import pytest

from conduit.config import ConfigError
from conduit.contracts import Complexity
from conduit.router.settings import ClassifierSettings, load_classifier_settings

ROUTING_STUB = """
tiers:
  trivial: [cheap]
  standard: [mid]
  complex: [dear]
budgets:
  default_daily_usd: 5.0
  on_exceed: reject
"""


def write_routing(tmp_path: Path, extra: str = "") -> Path:
    (tmp_path / "routing.yaml").write_text(ROUTING_STUB + extra, encoding="utf-8")
    return tmp_path


def test_missing_classifier_block_yields_defaults(tmp_path: Path) -> None:
    settings = load_classifier_settings(write_routing(tmp_path), env={})
    assert settings == ClassifierSettings()
    assert settings.workflow_hints["intake"] is Complexity.TRIVIAL


def test_classifier_block_is_read_from_the_file(tmp_path: Path) -> None:
    block = """
classifier:
  confidence_threshold: 0.9
  cache_size: 4
  workflow_hints:
    intake: complex
"""
    settings = load_classifier_settings(write_routing(tmp_path, block), env={})
    assert settings.confidence_threshold == 0.9
    assert settings.cache_size == 4
    assert settings.workflow_hints == {"intake": Complexity.COMPLEX}


def test_env_overrides_beat_the_file(tmp_path: Path) -> None:
    block = "\nclassifier:\n  cache_size: 4\n"
    settings = load_classifier_settings(
        write_routing(tmp_path, block),
        env={
            "CONDUIT_ROUTING__CLASSIFIER__CACHE_SIZE": "64",
            "CONDUIT_ROUTING__CLASSIFIER__TIEBREAK_ENABLED": "false",
        },
    )
    assert settings.cache_size == 64
    assert settings.tiebreak_enabled is False


def test_invalid_values_raise_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid classifier settings"):
        load_classifier_settings(
            write_routing(tmp_path),
            env={"CONDUIT_ROUTING__CLASSIFIER__CONFIDENCE_THRESHOLD": "7"},
        )


def test_non_mapping_classifier_block_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_classifier_settings(write_routing(tmp_path, "\nclassifier: 3\n"), env={})


def test_shipped_config_loads() -> None:
    settings = load_classifier_settings(env={})
    assert 0.0 < settings.confidence_threshold <= 1.0
    assert settings.cache_size > 0

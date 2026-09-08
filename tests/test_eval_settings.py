"""Eval config validation, chiefly the rule that a model never grades itself."""

from pathlib import Path

import pytest

from conduit.config import ConfigError
from conduit.eval.settings import EvalSettings, SelfGradingError, SuiteSpec, load_eval

SUITES = {"demo": SuiteSpec(path=Path("evals/golden/routing.jsonl"), scorer="tier")}


def test_a_model_may_not_grade_its_own_output() -> None:
    """Epic AC-14, enforced at construction rather than inside the runner."""
    with pytest.raises(SelfGradingError, match="may not grade its own output"):
        EvalSettings(model_under_test="claude-opus-5", judge_model="claude-opus-5", suites=SUITES)


def test_the_self_grading_error_names_the_offending_model_and_the_fix() -> None:
    with pytest.raises(SelfGradingError) as caught:
        EvalSettings(model_under_test="mock:echo", judge_model="mock:echo", suites=SUITES)
    message = str(caught.value)
    assert "'mock:echo'" in message
    assert "config/eval.yaml" in message
    assert "--judge-model" in message


def test_surrounding_whitespace_does_not_smuggle_a_model_past_the_check() -> None:
    with pytest.raises(SelfGradingError):
        EvalSettings(model_under_test="claude-opus-5 ", judge_model=" claude-opus-5", suites=SUITES)


def test_distinct_models_are_accepted() -> None:
    settings = EvalSettings(
        model_under_test="mock:echo", judge_model="offline-judge:rubric", suites=SUITES
    )
    assert settings.judge_model != settings.model_under_test


def test_a_settings_object_with_no_suites_is_rejected() -> None:
    with pytest.raises(ValueError, match="declares no suites"):
        EvalSettings(suites={})


def test_an_unknown_suite_name_lists_the_configured_ones() -> None:
    settings = EvalSettings(suites=SUITES)
    with pytest.raises(KeyError, match="configured suites: demo"):
        settings.suite("nope")


def test_the_shipped_config_is_valid_and_offline_by_default() -> None:
    """AC-5: a CI run must not need a key, so both defaults are free."""
    settings = load_eval()
    assert set(settings.suites) == {"routing", "guards", "l2c_quality"}
    assert settings.model_under_test == "mock:echo"
    assert settings.judge_model == "offline-judge:rubric"
    assert settings.regression_threshold == 0.05


def test_env_overrides_are_re_validated_for_self_grading() -> None:
    """The override path is the one an operator would reach for, so it is checked."""
    with pytest.raises(ConfigError, match="may not grade its own output"):
        load_eval(env={"CONDUIT_EVAL__JUDGE_MODEL": "mock:echo"})


def test_relative_dataset_and_baseline_paths_resolve_against_a_given_root(
    tmp_path: Path,
) -> None:
    settings = EvalSettings(suites=SUITES)
    spec = settings.suite("demo")
    assert settings.resolve_path(spec, tmp_path) == tmp_path / "evals/golden/routing.jsonl"
    assert settings.baseline_path("demo", tmp_path) == tmp_path / "evals/baselines/demo.json"

"""End-to-end: every suite against the mock provider, and both gate directions.

These are the tests that would catch the eval plane silently not running. They
drive the real CLI entry point, the real config, and the real datasets — the
only substitution is that nothing leaves the process, which is the point of
epic AC-5.
"""

import json
from pathlib import Path

import pytest

from conduit.eval.cli import main
from conduit.eval.gate import Baseline, write_baseline
from conduit.eval.runner import EvalHarness
from conduit.eval.settings import load_eval

SETTINGS = load_eval()


@pytest.fixture
def harness() -> EvalHarness:
    return EvalHarness(SETTINGS)


# --------------------------------------------------------------------------- #
# Full suites against mock
# --------------------------------------------------------------------------- #


async def test_the_routing_suite_reports_per_case_pass_fail(harness: EvalHarness) -> None:
    """AC-1 of the issue: a report with a verdict for every case."""
    result = await harness.run("routing")
    assert result.total == 105
    assert result.errors == 0
    assert len(result.cases) == result.total
    assert all(case.actual in {"trivial", "standard", "complex"} for case in result.cases)
    # The honest measured number, and where the loss is. See docs/EVAL.md.
    assert result.pass_rate == pytest.approx(0.829, abs=0.005)
    by_slice = {stat.name: stat.pass_rate for stat in result.slices()}
    assert by_slice["canonical"] == pytest.approx(0.983, abs=0.005)
    assert by_slice["hard"] == pytest.approx(0.622, abs=0.005)


async def test_the_guard_suite_measures_recall_and_false_positives(
    harness: EvalHarness,
) -> None:
    result = await harness.run("guards")
    assert result.total == 60
    assert result.errors == 0
    by_slice = {stat.name: stat for stat in result.slices()}
    # Attack recall. The corpus was authored independently of the detector under
    # issue #8, but issue #13 then extended the pattern set in response to the
    # nine cases it missed, so this slice is no longer a held-out measurement —
    # it is the set the rules were fitted to. See docs/EVAL.md.
    assert by_slice["attack"].pass_rate == pytest.approx(1.0)
    # No benign business text is blocked: the false-positive rate is zero.
    assert by_slice["benign"].pass_rate == pytest.approx(1.0)
    assert by_slice["pii"].pass_rate == pytest.approx(0.90, abs=0.005)


async def test_the_quality_suite_runs_the_judge_end_to_end(harness: EvalHarness) -> None:
    result = await harness.run("l2c_quality")
    assert result.total == 20
    assert result.errors == 0
    assert all(case.reasoning for case in result.cases)
    assert 0.0 < result.mean_score < 1.0


async def test_a_full_run_spends_nothing(harness: EvalHarness) -> None:
    """AC-5: CI runs every suite at zero API spend, and the report says so."""
    results = await harness.run_all()
    assert {result.suite for result in results} == set(SETTINGS.suites)
    assert sum(result.total_cost_usd for result in results) == 0.0


# --------------------------------------------------------------------------- #
# The gate, both directions
# --------------------------------------------------------------------------- #


def write_synthetic_baselines(directory: Path, pass_rate: float) -> None:
    for name in SETTINGS.suites:
        write_baseline(
            directory / f"{name}.json",
            Baseline(suite=name, pass_rate=pass_rate, mean_score=pass_rate, cases=1),
        )


def test_the_gate_exits_zero_when_no_suite_regressed(tmp_path: Path) -> None:
    """A baseline below the current run is not a regression."""
    write_synthetic_baselines(tmp_path, 0.10)
    code = main(
        [
            "run",
            "--gate",
            "--format",
            "json",
            "--report-dir",
            str(tmp_path / "report"),
            "--config-dir",
            str(_config_dir_with(tmp_path)),
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "report" / "report.json").read_text(encoding="utf-8"))
    assert payload["gate"]["passed"] is True
    assert payload["gate"]["exit_code"] == 0


def test_the_gate_exits_non_zero_on_a_regression_past_five_percent(tmp_path: Path) -> None:
    """A baseline of 1.0 against a run below 0.95 must fail the build.

    The flagged set is checked against the gate's own arithmetic rather than
    against a hardcoded list of suites. Hardcoding it meant the test quietly
    depended on the guard suite scoring badly, and it broke the moment issue #13
    improved injection recall — a passing test that fails on an improvement is
    measuring the wrong thing.
    """
    write_synthetic_baselines(tmp_path, 1.0)
    code = main(
        [
            "run",
            "--gate",
            "--format",
            "json",
            "--report-dir",
            str(tmp_path / "report"),
            "--config-dir",
            str(_config_dir_with(tmp_path)),
        ]
    )
    assert code == 1
    payload = json.loads((tmp_path / "report" / "report.json").read_text(encoding="utf-8"))
    assert payload["gate"]["passed"] is False
    outcomes = payload["gate"]["outcomes"]
    assert {o["suite"] for o in outcomes} == set(SETTINGS.suites)
    regressed = {o["suite"] for o in outcomes if o["regressed"]}
    expected = {o["suite"] for o in outcomes if o["relative_drop"] > o["threshold"]}
    assert regressed == expected
    # …and the failing exit code has to come from somewhere.
    assert regressed


def test_the_committed_baselines_hold_against_a_real_run(tmp_path: Path) -> None:
    """The baselines in evals/baselines/ are current, not aspirational."""
    code = main(["run", "--gate", "--format", "json", "--report-dir", str(tmp_path)])
    assert code == 0, (tmp_path / "report.md").read_text(encoding="utf-8")


def test_a_baseline_is_only_written_when_asked(tmp_path: Path) -> None:
    """AC-7: a gate that records its own baseline cannot fail twice."""
    config_dir = _config_dir_with(tmp_path)
    baseline = tmp_path / "routing.json"
    write_baseline(baseline, Baseline(suite="routing", pass_rate=0.1, mean_score=0.1, cases=1))
    before = baseline.read_text(encoding="utf-8")
    assert (
        main(["run", "--suite", "routing", "--format", "json", "--config-dir", str(config_dir)])
        == 0
    )
    assert baseline.read_text(encoding="utf-8") == before

    assert (
        main(
            [
                "run",
                "--suite",
                "routing",
                "--update-baseline",
                "--format",
                "json",
                "--config-dir",
                str(config_dir),
            ]
        )
        == 0
    )
    assert json.loads(baseline.read_text(encoding="utf-8"))["pass_rate"] > 0.5


def test_gating_and_updating_a_baseline_in_one_run_is_refused() -> None:
    """Recording from the run being gated would make the gate unfailable."""
    with pytest.raises(SystemExit) as caught:
        main(["run", "--gate", "--update-baseline"])
    assert caught.value.code == 2


def test_the_cli_rejects_a_judge_that_grades_itself() -> None:
    """AC-3: the check holds on the flag path, not only in the config file."""
    assert main(["run", "--model", "mock:echo", "--judge-model", "mock:echo"]) == 2


def test_the_report_is_the_only_thing_on_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`conduit-eval run > report.md` must not capture 120 guard verdicts."""
    assert main(["run", "--suite", "guards", "--format", "markdown"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("# Conduit eval report")
    assert "guard verdict" not in captured.out


def test_suites_subcommand_lists_what_is_configured(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["suites"]) == 0
    listed = {line.split("\t")[0] for line in capsys.readouterr().out.splitlines() if line}
    assert listed == set(SETTINGS.suites)


def _config_dir_with(tmp_path: Path) -> Path:
    """A copy of `config/` whose eval.yaml points baselines at `tmp_path`.

    Copied rather than mutated so a gate test can never rewrite the committed
    baselines it is supposed to be checking.
    """
    from conduit.config import default_config_dir
    from conduit.eval.settings import repo_root

    source = default_config_dir()
    target = tmp_path / "config"
    target.mkdir(exist_ok=True)
    for path in source.glob("*.yaml"):
        (target / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    eval_yaml = target / "eval.yaml"
    text = eval_yaml.read_text(encoding="utf-8")
    text = text.replace("baseline_dir: evals/baselines", f"baseline_dir: {tmp_path.as_posix()}")
    text = text.replace("path: evals/", f"path: {(repo_root() / 'evals').as_posix()}/")
    eval_yaml.write_text(text, encoding="utf-8")
    return target

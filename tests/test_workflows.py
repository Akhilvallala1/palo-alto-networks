"""Guards on the delivery pipeline itself.

The deploy workflow's safety properties are one careless edit away from being
untrue, and the edit that breaks them looks harmless in review (`on: push`
added "just for staging"). Every property the workflow comments claim is
asserted here, because a comment does not fail a build.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
DEPLOY = WORKFLOWS / "deploy.yml"
CI = WORKFLOWS / "ci.yml"

#: Env-var names that carry a credential. `--set-env-vars` puts its argument in
#: a command line and in `gcloud run services describe` output; only
#: `--set-secrets`, which passes a Secret Manager resource name, may carry one.
SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_CREDENTIALS")

#: Providers whose adapter the registry builds purely because the var is set.
PROVIDER_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")

#: `${{ secrets.NAME }}`, however it is spaced.
SECRET_REF = re.compile(r"\$\{\{\s*secrets\.[A-Za-z0-9_]+\s*\}\}")


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML resolves an unquoted `on:` key to the boolean True (the YAML 1.1
    # "norway problem"), so a naive workflow["on"] raises KeyError and a naive
    # workflow.get("on", {}) makes every assertion below vacuously pass.
    raw = workflow[True] if True in workflow else workflow["on"]
    if isinstance(raw, str):
        return {raw: None}
    if isinstance(raw, list):
        return dict.fromkeys(raw)
    assert isinstance(raw, dict)
    return raw


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """Lines with comments dropped: a comment describing `on: push` is fine."""
    return [
        (n, line.strip())
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.strip() and not line.strip().startswith("#")
    ]


def test_deploy_is_manual_trigger_only() -> None:
    """AC-6: the deploy workflow can never run as a side effect of merging.

    Not "does not currently deploy on push" — cannot. A `workflow_run` chain on
    CI, a `push` filter, or a `release` trigger all fail here.
    """
    assert list(_triggers(_load(DEPLOY))) == ["workflow_dispatch"]


def test_ci_never_deploys() -> None:
    """The other half of AC-6: CI must not grow a deploy step of its own.

    A deploy that lives in two files is a deploy whose trigger nobody can state
    from one place.
    """
    ci = _load(CI)
    commands = " ".join(
        str(step.get("run", "")) for job in ci["jobs"].values() for step in job["steps"]
    )
    for forbidden in ("gcloud run deploy", "docker push", "gcloud auth"):
        assert forbidden not in commands, (
            f"ci.yml runs {forbidden!r}; deploying is deploy.yml's job"
        )


def test_ci_reads_no_secret_at_all() -> None:
    """AC-7, stated at its strongest: CI has nothing to leak because it holds nothing.

    Broader than `test_ci_sets_no_provider_key`, which only names the three
    provider variables — this catches any repository secret reaching the runner.
    """
    found = SECRET_REF.findall(CI.read_text(encoding="utf-8"))
    assert not found, f"ci.yml references {found}; it is meant to need no credential"


def test_deploy_is_gated_by_a_github_environment() -> None:
    """Manual is not the same as approved.

    `environment:` is what puts a reviewer in front of production; without it
    `workflow_dispatch` is still one click from anyone with write access.
    """
    jobs = _load(DEPLOY)["jobs"]
    for name, job in jobs.items():
        assert job.get("environment"), f"deploy job {name!r} has no environment gate"


def test_deploy_authenticates_without_a_stored_key() -> None:
    """WIF, not a service-account JSON blob in a repo secret."""
    deploy = _load(DEPLOY)
    assert deploy["permissions"]["id-token"] == "write"


def test_no_secret_is_passed_as_a_cloud_run_env_var() -> None:
    """AC-7: credentials arrive by reference (`--set-secrets`), never by value.

    `--set-env-vars` values are readable forever afterwards via
    `gcloud run services describe`, so a key placed there leaks long after the
    run log expires.
    """
    for number, line in _code_lines(DEPLOY):
        if "--set-env-vars" not in line:
            continue
        for assignment in re.findall(r"([A-Z][A-Z0-9_]*)=", line):
            assert not assignment.endswith(SECRET_SUFFIXES), (
                f"{DEPLOY.name}:{number} passes {assignment} by value; use --set-secrets"
            )


def test_the_runtime_credentials_are_passed_by_reference() -> None:
    """The positive half: the key does reach the service, just not by value."""
    text = DEPLOY.read_text(encoding="utf-8")
    assert "--set-secrets" in text
    assert "ANTHROPIC_API_KEY=conduit-anthropic-api-key:latest" in text


def test_no_workflow_echoes_a_secret() -> None:
    """AC-7: nothing prints a value that GitHub would have to mask."""
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for number, line in _code_lines(path):
            if not line.startswith(("echo", "printf")):
                continue
            assert "secrets." not in line, f"{path.name}:{number} echoes a secret: {line}"
            assert "print-identity-token" not in line, (
                f"{path.name}:{number} echoes an identity token: {line}"
            )


def test_ci_runs_the_checks_the_epic_requires() -> None:
    """AC-18: lint, types, the vendor guard, both test stages, and the eval gate."""
    ci = _load(CI)
    assert "pull_request" in _triggers(ci)
    steps = "\n".join(
        str(step.get("run", "")) for job in ci["jobs"].values() for step in job["steps"]
    )
    for required in (
        "ruff check",
        "ruff format --check",
        "mypy --strict",
        "test_no_vendor_imports.py",
        'pytest -m "not integration"',
        "pytest -m integration",
        "coverage report",
        "conduit-eval run --gate",
    ):
        assert required in steps, f"CI never runs: {required}"


def test_the_coverage_gate_is_a_committed_threshold() -> None:
    """AC-19. In pyproject, not a CI flag, so a laptop fails the same way."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "fail_under = 80" in pyproject


@pytest.mark.parametrize("var", PROVIDER_KEYS)
def test_ci_sets_no_provider_key(var: str) -> None:
    """A fake key is not a neutral default here.

    The registry instantiates an adapter because the env var *exists*, so
    `ANTHROPIC_API_KEY: test-key` in CI would exercise a code path that no
    deployment runs and would stop testing the zero-key path AC-17 claims.
    """
    for number, line in _code_lines(CI):
        assert not line.startswith(f"{var}:"), f"{CI.name}:{number} sets {var}"
        assert f"{var}=" not in line, f"{CI.name}:{number} sets {var}"

"""`config/eval.yaml` — which suites exist, what runs them, and who judges.

The load-bearing rule here is epic AC-14: a model never grades its own output.
That is enforced as a validator on the settings object rather than as a check
inside the runner, so it fires at startup on every construction path — file,
environment override, or CLI flag — and cannot be reached by a suite that
forgets to call it.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from conduit.config import ConfigError, _load_section

__all__ = [
    "EvalSettings",
    "SelfGradingError",
    "SuiteSpec",
    "load_eval",
]

#: How a suite turns a case plus a produced answer into a score.
ScorerName = Literal["tier", "guard", "judge"]


class SelfGradingError(ConfigError):
    """Raised when the judge model is also the model under test.

    Deliberately not a `ValueError`: pydantic swallows those raised inside a
    validator and re-emits a generic `ValidationError`, which would leave a
    caller unable to tell self-grading apart from a typo in a suite path.
    """


class SuiteSpec(BaseModel):
    """One suite: where its golden data lives and how it is scored."""

    path: Path
    scorer: ScorerName
    #: A case passes when its score reaches this. Exact-match scorers score 1.0
    #: or 0.0, so the threshold only bites on the rubric-judged suite.
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    description: str = ""


class EvalSettings(BaseModel):
    """`config/eval.yaml`, validated."""

    # `model_under_test` collides with pydantic's protected `model_` prefix.
    # The epic's wording is "judge model == model-under-test", so the field
    # keeps that name and the namespace guard is switched off deliberately.
    model_config = ConfigDict(protected_namespaces=())

    model_under_test: str = "mock:echo"
    judge_model: str = "offline-judge:rubric"
    #: Cap on in-flight cases. Evals fan out over a shared gateway, so this is
    #: the difference between a suite and an accidental load test.
    concurrency: int = Field(default=8, ge=1, le=128)
    #: One retry on a malformed judge response, then the case is an error.
    judge_retries: int = Field(default=1, ge=0, le=3)
    #: A suite regressing by more than this fraction of its baseline fails the
    #: gate. 0.05 is the epic's ">5% regression".
    regression_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    baseline_dir: Path = Path("evals/baselines")
    suites: dict[str, SuiteSpec]

    @model_validator(mode="after")
    def _judge_is_not_the_model_under_test(self) -> "EvalSettings":
        if self.judge_model.strip() == self.model_under_test.strip():
            raise SelfGradingError(
                f"judge model and model under test are both {self.judge_model!r}; "
                "a model may not grade its own output (epic AC-14). Set a different "
                "`judge_model` in config/eval.yaml or pass --judge-model."
            )
        if not self.suites:
            raise ValueError("config/eval.yaml declares no suites")
        return self

    def suite(self, name: str) -> SuiteSpec:
        try:
            return self.suites[name]
        except KeyError:
            known = ", ".join(sorted(self.suites))
            raise KeyError(f"unknown suite {name!r}; configured suites: {known}") from None

    def resolve_path(self, spec: SuiteSpec, root: Path | None = None) -> Path:
        """Absolute path to a suite's dataset, relative paths taken from root."""
        if spec.path.is_absolute():
            return spec.path
        return (repo_root() if root is None else root) / spec.path

    def baseline_path(self, name: str, root: Path | None = None) -> Path:
        base = self.baseline_dir
        if not base.is_absolute():
            base = (repo_root() if root is None else root) / base
        return base / f"{name}.json"


def repo_root() -> Path:
    """The repository root, i.e. the parent of `src/`."""
    return Path(__file__).resolve().parents[3]


def load_eval(config_dir: Path | None = None, env: dict[str, str] | None = None) -> EvalSettings:
    """Load `eval.yaml`, applying `CONDUIT_EVAL__*` overrides."""
    return _load_section(EvalSettings, "eval", config_dir, env)

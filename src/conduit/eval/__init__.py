"""Eval plane: golden datasets, LLM-as-a-Judge rubrics, and the CI gate.

    from conduit.eval import EvalHarness, load_eval

    settings = load_eval()
    results = await EvalHarness(settings).run_all()

Or from a shell, which is where it earns its keep::

    conduit-eval run --gate        # exit 1 on a >5% regression against baseline

The datasets in `evals/golden/` were authored under issue #8, independently of
the components they measure. `docs/EVAL.md` records why that matters and what
changed when the routing labels were re-authored.
"""

from .dataset import DatasetError, EvalCase, load_cases
from .gate import Baseline, GateOutcome, GateReport, compare, gate, load_baseline, write_baseline
from .judge import Judge, JudgeError, JudgeParseError, JudgeResult, parse_judge_response
from .offline import OFFLINE_JUDGE_MODEL, OfflineJudgeProvider
from .report import render_json, render_markdown
from .rubric import Criterion, Rubric, RubricError, evaluate_rubric, parse_rubric
from .runner import CaseResult, EvalHarness, SuiteResult, run_suite
from .settings import EvalSettings, SelfGradingError, SuiteSpec, load_eval

__all__ = [
    "OFFLINE_JUDGE_MODEL",
    "Baseline",
    "CaseResult",
    "Criterion",
    "DatasetError",
    "EvalCase",
    "EvalHarness",
    "EvalSettings",
    "GateOutcome",
    "GateReport",
    "Judge",
    "JudgeError",
    "JudgeParseError",
    "JudgeResult",
    "OfflineJudgeProvider",
    "Rubric",
    "RubricError",
    "SelfGradingError",
    "SuiteResult",
    "SuiteSpec",
    "compare",
    "evaluate_rubric",
    "gate",
    "load_baseline",
    "load_cases",
    "load_eval",
    "parse_judge_response",
    "parse_rubric",
    "render_json",
    "render_markdown",
    "run_suite",
    "write_baseline",
]

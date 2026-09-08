"""`conduit-eval` — run suites, gate a pull request, record a baseline.

    conduit-eval run                              # every suite, report to stdout
    conduit-eval run --suite routing              # one suite
    conduit-eval run --gate                       # exit 1 on a >5% regression
    conduit-eval run --suite routing --update-baseline
    conduit-eval suites                           # what is configured

`--gate` and `--update-baseline` are mutually exclusive: recording a baseline
from the same run that is being gated would make the gate unfailable.
"""

import argparse
import asyncio
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import structlog

from conduit.config import ConfigError

from .gate import GateReport, baseline_from, gate, write_baseline
from .report import render_json, render_markdown
from .runner import EvalHarness, SuiteResult
from .settings import EvalSettings, SelfGradingError, load_eval

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conduit-eval", description="Conduit eval plane: golden suites, judge, CI gate."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one or more suites")
    run.add_argument(
        "--suite",
        action="append",
        dest="suites",
        metavar="NAME",
        help="suite to run; repeatable. Default: every configured suite.",
    )
    run.add_argument("--model", help="override the model under test")
    run.add_argument("--judge-model", help="override the judge model")
    run.add_argument("--concurrency", type=int, help="cap on in-flight cases")
    run.add_argument(
        "--gate",
        action="store_true",
        help="compare against evals/baselines and exit non-zero on a regression",
    )
    run.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite the committed baselines from this run (never automatic)",
    )
    run.add_argument(
        "--format",
        choices=("markdown", "json", "both"),
        default="markdown",
        help="report format written to stdout",
    )
    run.add_argument(
        "--report-dir",
        type=Path,
        help="also write report.md and report.json into this directory",
    )
    run.add_argument("--config-dir", type=Path, help="directory holding eval.yaml")

    suites = sub.add_parser("suites", help="list configured suites")
    suites.add_argument("--config-dir", type=Path, help="directory holding eval.yaml")
    return parser


def _apply_overrides(settings: EvalSettings, args: argparse.Namespace) -> EvalSettings:
    """Re-validate on every override path, so AC-14 cannot be flagged around."""
    updates: dict[str, object] = {}
    if getattr(args, "model", None):
        updates["model_under_test"] = args.model
    if getattr(args, "judge_model", None):
        updates["judge_model"] = args.judge_model
    if getattr(args, "concurrency", None):
        updates["concurrency"] = args.concurrency
    if not updates:
        return settings
    return EvalSettings.model_validate({**settings.model_dump(), **updates})


def _write_reports(directory: Path, markdown: str, payload: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.md").write_text(markdown, encoding="utf-8")
    (directory / "report.json").write_text(payload, encoding="utf-8")


async def _run(args: argparse.Namespace, settings: EvalSettings) -> int:
    harness = EvalHarness(settings)
    names = args.suites or sorted(settings.suites)
    for name in names:
        settings.suite(name)  # fail fast on a typo before spending anything
    results: list[SuiteResult] = await harness.run_all(names)

    gate_report: GateReport | None = None
    if args.gate:
        gate_report = gate(
            results,
            {name: settings.baseline_path(name) for name in names},
            threshold=settings.regression_threshold,
        )

    markdown = render_markdown(results, gate_report)
    payload = render_json(results, gate_report)
    if args.format in ("markdown", "both"):
        sys.stdout.write(markdown)
    if args.format in ("json", "both"):
        sys.stdout.write(payload)
    if args.report_dir:
        _write_reports(args.report_dir, markdown, payload)

    if args.update_baseline:
        for result in results:
            path = write_baseline(settings.baseline_path(result.suite), baseline_from(result))
            print(f"baseline updated: {path}", file=sys.stderr)

    if gate_report is not None:
        for outcome in gate_report.outcomes:
            print(outcome.describe(), file=sys.stderr)
        return gate_report.exit_code
    return 0


@contextmanager
def _logs_to_stderr() -> Iterator[None]:
    """Keep stdout clean enough to redirect, then put structlog back.

    The report is this command's product, and the guards under test log a line
    per verdict through structlog, whose default logger writes to stdout. Left
    alone, `conduit-eval run > report.md` produces a file with 120 guard verdicts
    interleaved through the tables.

    structlog's configuration is process-global, so it is restored on the way
    out: an in-process caller — the gateway's own test suite, for one — must not
    find its logging rewired by having run an eval.
    """
    previous = structlog.get_config()
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
    try:
        yield
    finally:
        structlog.configure(**previous)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "gate", False) and getattr(args, "update_baseline", False):
        parser.error("--gate and --update-baseline are mutually exclusive")
    try:
        settings = _apply_overrides(load_eval(args.config_dir), args)
    except SelfGradingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "suites":
        for name in sorted(settings.suites):
            spec = settings.suites[name]
            print(f"{name}\t{spec.scorer}\t{spec.path}")
        return 0
    with _logs_to_stderr():
        return asyncio.run(_run(args, settings))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

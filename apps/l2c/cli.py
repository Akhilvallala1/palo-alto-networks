"""`l2c quote --request examples/quote_request.json`.

Two ways to reach a gateway, and the default is the interesting one:

* **Embedded (default).** The CLI builds the real Conduit app in-process and
  drives it over ASGI. No server to start, no keys to set, no network — the
  request still goes through auth, the guards, the router, failover and
  telemetry. This is the path that makes acceptance criterion 7 true.
* **Remote.** `--gateway-url http://localhost:8000` talks to a running gateway,
  which is what the container path uses.

`--json` prints the whole decision as JSON for a test or a pipe; the default
output is the human summary, and it leads with the citation because that is the
thing the workflow exists to produce.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TextIO

from .client import DEFAULT_API_KEY, ConduitClient
from .graph import run_quote
from .models import Decision, QuoteRequest
from .rag import build_index

__all__ = ["main"]


def _load_request(path: Path) -> QuoteRequest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"l2c: cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"l2c: {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"l2c: {path} must contain a JSON object, not {type(payload).__name__}")
    return QuoteRequest.model_validate(payload)


def _render(decision: Decision, stream: TextIO) -> None:
    """The human summary. Citation first, then the reasons, then routing."""
    analysis = decision.analysis
    verdict = "ESCALATE" if decision.escalated else "APPROVE"
    print(f"\n  Decision   {verdict}", file=stream)
    print(f"  Approver   {decision.approver.replace('_', ' ')}", file=stream)
    print(f"  Citation   [{decision.citation}] {decision.citation_title}", file=stream)
    print(f'             "{decision.citation_quote}"', file=stream)

    print("\n  Numbers", file=stream)
    print(f"    list            ${analysis.list_amount_usd:>14,.2f}", file=stream)
    print(
        f"    discount        ${analysis.discount_value_usd:>14,.2f} ({analysis.discount_pct:g}%)",
        file=stream,
    )
    print(f"    net             ${analysis.net_amount_usd:>14,.2f}", file=stream)
    if analysis.partner_fee_usd:
        print(f"    partner fee     ${analysis.partner_fee_usd:>14,.2f}", file=stream)
        print(f"    net after fee   ${analysis.net_after_partner_fee_usd:>14,.2f}", file=stream)

    print("\n  Reasons", file=stream)
    for reason in decision.reasons:
        print(f"    - {reason}", file=stream)

    print("\n  Retrieved policy", file=stream)
    for item in decision.evidence:
        print(f"    {item.section_id:12s} {item.score:>7.3f}  {item.title}", file=stream)

    print("\n  Routing per node (this is the point: one tier per node)", file=stream)
    print(f"    {'node':<18} {'workflow':<18} {'tier':<9} {'model':<22} cost", file=stream)
    for call in decision.calls:
        print(
            f"    {call.node:<18} {call.workflow:<18} {call.tier:<9} "
            f"{call.model:<22} ${call.cost_usd:.6f}",
            file=stream,
        )
    total = sum(call.cost_usd for call in decision.calls)
    print(f"    {'':<18} {'':<18} {'':<9} {'total':<22} ${total:.6f}", file=stream)

    if decision.explanation:
        print(f"\n  Explanation\n    {decision.explanation.strip()}", file=stream)
    print(file=stream)


async def _run(args: argparse.Namespace, stream: TextIO) -> int:
    request = _load_request(args.request)
    index = build_index()
    try:
        if args.gateway_url:
            client = ConduitClient(base_url=args.gateway_url, api_key=args.api_key)
        else:
            # Imported here so `--gateway-url` runs never build a gateway, and
            # so `l2c --help` does not pay for importing FastAPI.
            from conduit.gateway.app import create_app

            app = create_app(setup_logging=False)
            client = ConduitClient(app=app, api_key=args.api_key)

        async with client:
            decision = await run_quote(request, client=client, index=index)
    finally:
        index.close()

    if args.json:
        print(decision.model_dump_json(indent=2), file=stream)
    else:
        _render(decision, stream)
    # A non-zero exit for an escalation would be wrong: escalating is a
    # successful run of the workflow, not a failure of it.
    return 0


def main(argv: list[str] | None = None, stream: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="l2c", description="Lead-to-Cash quote-approval workflow on Conduit."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    quote = subcommands.add_parser("quote", help="run the quote-approval workflow")
    quote.add_argument(
        "--request", type=Path, required=True, help="path to a quote request JSON file"
    )
    quote.add_argument(
        "--gateway-url",
        default=None,
        help="URL of a running Conduit gateway (default: run one in-process)",
    )
    quote.add_argument("--api-key", default=DEFAULT_API_KEY, help="Conduit API key")
    quote.add_argument("--json", action="store_true", help="print the decision as JSON")

    args = parser.parse_args(argv)
    out = stream if stream is not None else sys.stdout
    return asyncio.run(_run(args, out))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

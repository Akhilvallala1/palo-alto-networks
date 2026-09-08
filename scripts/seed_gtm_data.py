"""Generate the synthetic GTM dataset the L2C workflow runs against.

    python scripts/seed_gtm_data.py --out data/gtm --seed 20260908

**Synthetic means synthetic.** Every account, contact, opportunity, quote and
comp plan below is generated from word lists in this file. Nothing is sampled
from, derived from, or anonymised out of a real CRM, and the email domains are
all under `.example`, which RFC 2606 reserves precisely so that generated
contact details cannot reach a real mailbox. The epic forbids real customer or
GTM data anywhere in this repo; this file is where that rule is kept.

**Determinism.** Everything is drawn from a single `random.Random(seed)` in a
fixed order, and dates are offsets from `EPOCH_DATE` rather than from today. So
the same seed produces byte-identical JSON on any machine, on any day, which is
what lets tests assert on specific records (AC-6). Adding a draw in the middle
of a generator renumbers everything after it — append new fields at the end of
a record instead.

The policy corpus is deliberately *not* generated: it is authored in
`apps/l2c/policies.py` and copied out here, because a decision that cites a
clause needs the clause to be coherent prose. See that module for why.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# Running as a script puts `scripts/` on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apps.l2c.policies import DISCOUNT_BANDS, POLICY_SECTIONS

__all__ = [
    "DEFAULT_SEED",
    "GtmDataset",
    "generate",
    "main",
    "write_dataset",
]

DEFAULT_SEED = 20260908

#: All generated dates are offsets from here, never from `date.today()`, so a
#: dataset generated in March and one generated in November are identical.
EPOCH_DATE = date(2026, 1, 1)

_COMPANY_HEADS = (
    "Northwind",
    "Contoso",
    "Fabrikam",
    "Litware",
    "Proseware",
    "Adventure",
    "Tailspin",
    "Wingtip",
    "Lucerne",
    "Woodgrove",
    "Blue Yonder",
    "Graphic Design",
    "Trey",
    "Relecloud",
    "Fourth Coffee",
    "Alpine Ski",
)
_COMPANY_TAILS = (
    "Traders",
    "Industries",
    "Logistics",
    "Financial",
    "Health",
    "Robotics",
    "Analytics",
    "Systems",
    "Networks",
    "Manufacturing",
)
_INDUSTRIES = (
    "financial services",
    "healthcare",
    "manufacturing",
    "public sector",
    "retail",
    "technology",
    "energy",
    "education",
)
_REGIONS = ("AMER-East", "AMER-West", "EMEA-North", "EMEA-South", "APAC-North", "APAC-South")
_SEGMENTS = ("enterprise", "mid-market", "commercial")
_STAGES = ("qualification", "discovery", "proposal", "negotiation", "closed-won", "closed-lost")
_SUPPORT_TIERS = ("standard", "premium")
_PRODUCTS = (
    ("SEC-1100", "Perimeter Gateway 1100", 4200.0),
    ("SEC-3200", "Perimeter Gateway 3200", 11800.0),
    ("SEC-5220", "Perimeter Gateway 5220", 24500.0),
    ("CLD-200", "Cloud Posture Manager", 3100.0),
    ("CLD-400", "Cloud Workload Defender", 6400.0),
    ("SOC-900", "Threat Analytics Suite", 15200.0),
)
_FIRST_NAMES = (
    "Jordan",
    "Priya",
    "Mateo",
    "Ingrid",
    "Kwame",
    "Sora",
    "Nadia",
    "Declan",
    "Yuki",
    "Amara",
    "Tomas",
    "Freya",
)
_LAST_NAMES = (
    "Ellis",
    "Raman",
    "Alvarez",
    "Lindqvist",
    "Mensah",
    "Tanaka",
    "Haddad",
    "O'Rourke",
    "Sato",
    "Okafor",
    "Novak",
    "Berg",
)
_TITLES = (
    "VP Security Operations",
    "Director of Infrastructure",
    "Head of IT",
    "CISO",
    "Security Architect",
    "Procurement Lead",
)
_JUSTIFICATIONS = (
    "competitive displacement",
    "multi-year prepayment",
    "strategic logo acquisition",
    "",  # a request with no justification, so the DISC-4.6 path is exercised
)


def _slug(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


class GtmDataset:
    """The generated corpus, in the shape it is written to disk."""

    def __init__(
        self,
        seed: int,
        accounts: list[dict[str, Any]],
        contacts: list[dict[str, Any]],
        opportunities: list[dict[str, Any]],
        quotes: list[dict[str, Any]],
        comp_plans: list[dict[str, Any]],
        quote_requests: list[dict[str, Any]],
    ) -> None:
        self.seed = seed
        self.accounts = accounts
        self.contacts = contacts
        self.opportunities = opportunities
        self.quotes = quotes
        self.comp_plans = comp_plans
        self.quote_requests = quote_requests

    @property
    def policies(self) -> list[dict[str, Any]]:
        """The authored policy corpus, as citable chunks."""
        return [section.model_dump() for section in POLICY_SECTIONS]

    @property
    def discount_bands(self) -> list[dict[str, Any]]:
        return [band.model_dump() for band in DISCOUNT_BANDS]

    def files(self) -> dict[str, Any]:
        """Filename -> payload, which is exactly what lands in the output dir."""
        return {
            "accounts.json": self.accounts,
            "contacts.json": self.contacts,
            "opportunities.json": self.opportunities,
            "quotes.json": self.quotes,
            "comp_plans.json": self.comp_plans,
            "quote_requests.json": self.quote_requests,
            "policies.json": self.policies,
            "discount_bands.json": self.discount_bands,
            "manifest.json": {
                "seed": self.seed,
                "synthetic": True,
                "source": "scripts/seed_gtm_data.py",
                "note": (
                    "Fully synthetic GTM data. No real customer, account or policy "
                    "content. Email domains are under .example (RFC 2606)."
                ),
                "counts": {
                    "accounts": len(self.accounts),
                    "contacts": len(self.contacts),
                    "opportunities": len(self.opportunities),
                    "quotes": len(self.quotes),
                    "comp_plans": len(self.comp_plans),
                    "quote_requests": len(self.quote_requests),
                    "policies": len(self.policies),
                },
            },
        }


def generate(
    seed: int = DEFAULT_SEED, *, accounts: int = 12, quotes_per_account: int = 2
) -> GtmDataset:
    """Build the dataset. Same seed and counts in, same bytes out."""
    rng = random.Random(seed)

    account_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    opportunity_rows: list[dict[str, Any]] = []
    quote_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []

    for account_index in range(accounts):
        head = rng.choice(_COMPANY_HEADS)
        tail = rng.choice(_COMPANY_TAILS)
        name = f"{head} {tail}"
        account_id = f"ACC-{10000 + account_index:05d}"
        domain = f"{_slug(head)}.example"
        account_rows.append(
            {
                "account_id": account_id,
                "name": name,
                "domain": domain,
                "industry": rng.choice(_INDUSTRIES),
                "region": rng.choice(_REGIONS),
                "segment": rng.choice(_SEGMENTS),
                "employees": rng.choice((250, 900, 2400, 7500, 21000)),
                "current_arr_usd": round(rng.uniform(40_000, 900_000), 2),
            }
        )

        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        contact_rows.append(
            {
                "contact_id": f"CON-{20000 + account_index:05d}",
                "account_id": account_id,
                "name": f"{first} {last}",
                "title": rng.choice(_TITLES),
                # .example is reserved by RFC 2606: these addresses cannot route.
                "email": f"{first[0].lower()}{_slug(last)}@{domain}",
            }
        )

        for quote_index in range(quotes_per_account):
            ordinal = account_index * quotes_per_account + quote_index
            opportunity_id = f"OPP-{80000 + ordinal:05d}"
            sku, product_name, unit_price = rng.choice(_PRODUCTS)
            units = rng.choice((4, 8, 12, 20, 40, 75))
            term_months = rng.choice((12, 24, 36))
            list_amount = round(unit_price * units, 2)
            discount_pct = round(rng.uniform(0.0, 45.0), 1)
            support_tier = rng.choice(_SUPPORT_TIERS)
            partner = rng.random() < 0.35
            justification = rng.choice(_JUSTIFICATIONS)
            close_offset = rng.randint(30, 300)

            opportunity_rows.append(
                {
                    "opportunity_id": opportunity_id,
                    "account_id": account_id,
                    "name": f"{name} - {product_name} ({term_months}mo)",
                    "stage": rng.choice(_STAGES),
                    "amount_usd": list_amount,
                    "close_date": (EPOCH_DATE + timedelta(days=close_offset)).isoformat(),
                    "owner": f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}",
                }
            )

            quote_id = f"Q-2026-{ordinal + 100:05d}"
            quote_rows.append(
                {
                    "quote_id": quote_id,
                    "opportunity_id": opportunity_id,
                    "account_id": account_id,
                    "sku": sku,
                    "product": product_name,
                    "units": units,
                    "unit_list_price_usd": unit_price,
                    "list_amount_usd": list_amount,
                    "discount_pct": discount_pct,
                    "net_amount_usd": round(list_amount * (1 - discount_pct / 100), 2),
                    "term_months": term_months,
                    "support_tier": support_tier,
                    "partner_involved": partner,
                    "status": "pending_approval",
                    "created_date": (EPOCH_DATE + timedelta(days=close_offset - 21)).isoformat(),
                }
            )

            contact = contact_rows[-1]
            request_rows.append(
                {
                    "request_id": f"REQ-{30000 + ordinal:05d}",
                    "account_id": account_id,
                    "account_name": name,
                    "opportunity_id": opportunity_id,
                    "quote_id": quote_id,
                    "list_amount_usd": list_amount,
                    "discount_pct": discount_pct,
                    "term_months": term_months,
                    "support_tier": support_tier,
                    "partner_involved": partner,
                    "justification": justification,
                    "requested_by": contact["name"],
                    "requested_by_email": contact["email"],
                }
            )

    comp_rows: list[dict[str, Any]] = []
    for rep_index in range(6):
        quota = rng.choice((900_000, 1_200_000, 1_500_000, 2_000_000))
        comp_rows.append(
            {
                "plan_id": f"COMP-{40000 + rep_index:05d}",
                "rep": f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}",
                "region": rng.choice(_REGIONS),
                "annual_quota_usd": quota,
                "attainment_pct": round(rng.uniform(45.0, 135.0), 1),
                "base_commission_rate": 0.08,
                "accelerator_rate_above_quota": 0.12,
                # Mirrors COMP-6.1: quota retires on net, not list.
                "quota_retires_on": "net_booked_amount",
            }
        )

    return GtmDataset(
        seed=seed,
        accounts=account_rows,
        contacts=contact_rows,
        opportunities=opportunity_rows,
        quotes=quote_rows,
        comp_plans=comp_rows,
        quote_requests=request_rows,
    )


def write_dataset(dataset: GtmDataset, out_dir: Path) -> list[Path]:
    """Write the dataset as formatted JSON. Returns the files written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, payload in dataset.files().items():
        path = out_dir / filename
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="seed_gtm_data",
        description="Generate the synthetic GTM dataset for the L2C workflow.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/gtm"), help="output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument("--accounts", type=int, default=12, help="number of accounts")
    parser.add_argument(
        "--quotes-per-account", type=int, default=2, help="quotes generated per account"
    )
    args = parser.parse_args(argv)

    dataset = generate(
        args.seed, accounts=args.accounts, quotes_per_account=args.quotes_per_account
    )
    written = write_dataset(dataset, args.out)
    counts = dataset.files()["manifest.json"]["counts"]
    print(f"seed {args.seed} -> {args.out}")
    for name, count in sorted(counts.items()):
        print(f"  {name:16s} {count}")
    print(f"  {len(written)} files written")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

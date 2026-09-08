"""One-shot authoring script for evals/golden/routing.jsonl (issue #8).

Labels were re-authored directly from the tier definitions in docs/SPEC.md,
reading only the inputs (dumped without their #4 labels) and without reading
src/conduit/router/classifier.py or the classifier block of config/routing.yaml.

Kept in the tree as a provenance record: the label for every inherited case is
listed here beside the SPEC category it was judged against, so the claim in
docs/EVAL.md can be audited rather than taken on trust.
"""

import json
from collections import Counter
from pathlib import Path

DATASET = Path(__file__).resolve().parents[2] / "evals" / "golden" / "routing.jsonl"

src = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
by_id = {r["id"]: r["input"] for r in src}
original = {r["id"]: r["expected"] for r in src}

relabelled = {
    "route-001": "trivial",   # classification: segment the lead
    "route-002": "trivial",   # extraction: two fields out of an email
    "route-003": "trivial",   # formatting: date normalisation
    "route-004": "trivial",   # classification: sentiment
    "route-005": "trivial",   # formatting: canonical account name
    "route-006": "trivial",   # extraction: SKU list
    "route-007": "trivial",   # classification: inbound vs outbound
    "route-008": "trivial",   # classification: vertical tagging
    "route-009": "trivial",   # formatting: contact block to JSON
    "route-010": "trivial",   # one-sentence translation, a mechanical transform
    "route-011": "trivial",   # extraction: renewal date
    "route-012": "trivial",   # classification: request type
    "route-013": "trivial",   # extraction: name and role
    "route-014": "trivial",   # classification: bucketed ICP score
    "route-015": "trivial",   # extraction: quote number
    "route-016": "trivial",   # classification: discount band
    "route-017": "trivial",   # classification: boolean over a supplied field
    "route-018": "trivial",   # formatting: address to one line
    "route-019": "trivial",   # extraction: ARR figure
    "route-020": "trivial",   # classification: urgency
    "route-021": "standard",  # summarisation of a transcript
    "route-022": "standard",  # drafting an email
    "route-023": "standard",  # summarisation into an executive paragraph
    "route-024": "standard",  # drafting: audience rewrite
    "route-025": "standard",  # single-hop RAG over payment terms plus explanation
    "route-026": "standard",  # drafting: subject lines
    "route-027": "standard",  # single-hop RAG plus summarisation of one section
    "route-028": "standard",  # summarisation of a supplied delta
    "route-029": "standard",  # drafting a talk track from one account fact set
    "route-030": "standard",  # drafting: paraphrase of one clause
    "route-031": "standard",  # drafting an internal note
    "route-032": "standard",  # single-hop RAG over standard terms plus drafting
    "route-033": "standard",  # summarisation of pipeline notes
    "route-034": "standard",  # drafting plus one volume-tier lookup
    "route-035": "standard",  # drafting release notes
    "route-036": "standard",  # summarisation of one comp-plan section
    "route-037": "standard",  # single-hop RAG over one entitlement table
    "route-038": "standard",  # drafting a reply
    "route-039": "complex",   # multi-hop: policy hierarchy to decision to next step
    "route-040": "complex",   # math: layered margin arithmetic against a floor
    "route-041": "complex",   # planning: workflow design with trade-offs
    "route-042": "complex",   # code: debugging
    "route-043": "complex",   # multi-hop reconciliation with arithmetic
    "route-044": "complex",   # multi-hop: policy, margin and precedent
    "route-045": "complex",   # math plus multi-hop comparison and recommendation
    "route-046": "complex",   # code: refactor with explanation
    "route-047": "complex",   # multi-hop root cause across four factors
    "route-048": "complex",   # multi-hop: risk, precedent and policy per path
    "route-049": "complex",   # math: derivation with intermediate steps
    "route-050": "complex",   # planning: approval sequencing
    "route-051": "complex",   # code: implement a scoring function
    "route-052": "complex",   # multi-hop troubleshooting across three systems
    "route-053": "complex",   # multi-hop prioritisation against three dimensions
    "route-054": "complex",   # math: trend extrapolation with signal separation
    "route-055": "complex",   # multi-hop: analyse, decide, draft, in order
    "route-056": "complex",   # multi-hop across three comp rule sets
    "route-057": "complex",   # planning: data-flow and idempotency design
    "route-058": "complex",   # planning plus math: threshold optimisation
    "route-059": "complex",   # multi-hop: weigh four attributes, cite, decide
    "route-060": "complex",   # math: proration plus a customer explanation
}

assert set(relabelled) == set(by_id), set(relabelled) ^ set(by_id)

# Independently authored slice. Same three tier definitions, but the surface
# form is deliberately uncorrelated with the tier: trivial work phrased at
# length, standard work phrased as a bare question, complex work phrased
# casually. A classifier that keys on an opening imperative verb will miss here.
hard = [
    ("hard-001", "trivial", "quick one before my next call — out of the blurb below, which bucket does this account fall into: enterprise, mid-market, or SMB? just the bucket.\n\nRiverbend Freight, 310 employees, two sites, no dedicated security staff."),
    ("hard-002", "trivial", "Who is the economic buyer here? Name only.\n\nThanks again for the time yesterday. Looping in Marta Vance, our CFO, who signs off on anything over half a million. Dev Patel from my team will run the technical evaluation and Sofia Reyes in procurement will handle the paperwork once we get there. I'll keep driving this from the infrastructure side."),
    ("hard-003", "trivial", "I need this turned into a table. Columns: SKU, qty, unit price. Nothing else.\n\nPA-3410 x4 at $18,500; PA-5220 x2 at $61,000; PAN-SVC-PREM x6 at $2,900"),
    ("hard-004", "trivial", "Does the note below mention a competitor? yes/no.\n\n\"They're also talking to the incumbent, decision expected in six weeks.\""),
    ("hard-005", "trivial", "Strip the signature block and give me back just the message body.\n\nSounds good, let's target Thursday.\n\n--\nJordan Ellis\nVP Security Operations | Northwind Hospitals Group\njellis@northwind.example | +1 512 555 0142"),
    ("hard-006", "trivial", "Is the close date in this quarter or next? One word.\n\nClose date 2026-10-14. The current quarter ends 2026-09-30."),
    ("hard-007", "trivial", "Title case these account names, one per line.\n\nnorthwind hospitals group\nacme robotics inc\nriverbend freight llc"),
    ("hard-008", "trivial", "From the transcript below, pull every date mentioned. ISO format, comma separated, nothing else.\n\nWe kicked the evaluation off on the third of March 2026 and the proof of concept wrapped on April 18th 2026. Procurement opened the vendor review on 2026-05-02 and the security questionnaire came back on the twelfth of June 2026. The incumbent contract expires 2026-12-31, and the customer wants the new term to start on January 1st 2027. There is a board meeting on 2026-11-09 where the spend gets approved."),
    ("hard-009", "trivial", "What's the discount percentage on this quote? Number only, no percent sign.\n\nQuote Q-2026-01204. List $840,000. Discount applied: 22%. Net $655,200."),
    ("hard-010", "trivial", "Route this ticket. Options: deal desk, support, billing, legal. One word.\n\n\"The MSA redlines came back with changes to the limitation of liability clause.\""),
    ("hard-011", "trivial", "yes or no: does this quote include premium support?\n\nQ-2026-01188 — 8 units PA-3410, standard support, 2-year term."),
    ("hard-012", "trivial", "Please could you take the paragraph below and give me back just the numbers, as a JSON array, in the order they appear.\n\nThe renewal covers 14 units across 3 sites at a blended 17% discount, landing at $612,400 for the year against a $735,000 list."),
    ("hard-013", "trivial", "Which of our five verticals? healthcare / finance / retail / public sector / manufacturing.\n\nCustomer: Cascade County Water Authority"),
    ("hard-014", "trivial", "Normalize these phone numbers to E.164, one per line.\n\n(512) 555-0142\n512.555.0198\n+1 512 555 0110"),
    ("hard-015", "trivial", "Flag whether the note below contains any personal data. yes or no.\n\n\"Renewal quote sent to the procurement alias, no response yet.\""),
    ("hard-016", "standard", "Recap for the account team, four bullets.\n\nForty minutes with the infrastructure director. Policy drift across three data centres is the pain. Incumbent contract expires in Q4 and they are unhappy with the support response times. Budget cycle closes end of October. They want a two-year term, procurement is pushing back on three. One other vendor in the evaluation. They asked for logistics references."),
    ("hard-017", "standard", "Turn this into a customer-ready paragraph.\n\napproved 18%, volume tier 3, 3yr term, prem support included, starts oct 1"),
    ("hard-018", "standard", "What does our policy say about backdating contract start dates?"),
    ("hard-019", "standard", "Need the gist of this thread for my manager.\n\nAE: customer asked for 26% on a two-year.\nDeal desk: that is above the RVP line, needs written approval.\nAE: they are comparing us against the incumbent and the incumbent quoted aggressively.\nDeal desk: competitive displacement is a recognised justification but it has to be named in the request.\nAE: I will resubmit with the displacement note and the competitor quote attached.\nDeal desk: also confirm the term, the exception reads differently on three years."),
    ("hard-020", "standard", "Email to the champion. Apologise for the delay, give the new timeline, ask for a slot Thursday."),
    ("hard-021", "standard", "Explain the difference between the two-year and three-year support entitlements the way I'd say it on a call."),
    ("hard-022", "standard", "Condense our security questionnaire response into a one-page outline the customer can circulate."),
    ("hard-023", "standard", "Give me the policy clause that covers partner-sourced discounts."),
    ("hard-024", "standard", "Rewrite the objection-handling script for a technical audience instead of a business one."),
    ("hard-025", "standard", "Summarise what changed in the comp plan this year for the team newsletter."),
    ("hard-026", "standard", "Draft the internal Slack update on where the Cascade deal stands."),
    ("hard-027", "standard", "Write the abstract for the QBR deck. Three sentences, no jargon."),
    ("hard-028", "standard", "Put together a short handover note for the AE taking over this account.\n\nRenewal in Q1, champion left in March, replacement champion is lukewarm, open support escalation on policy sync, 14 units, currently on standard support."),
    ("hard-029", "standard", "In plain English, what is our standard cancellation window?"),
    ("hard-030", "standard", "Tighten this paragraph to under eighty words without losing any of the numbers.\n\nThe renewal, which we have been working on since the middle of last quarter, covers fourteen units across three separate sites and comes in at a blended discount of seventeen percent, which lands the annual figure at six hundred and twelve thousand four hundred dollars against a list price of seven hundred and thirty five thousand dollars, and the customer has indicated that they would like to add premium support at some point during the term."),
    ("hard-031", "complex", "quick sanity check: if we give 19% on a two-year and 24% on a three-year, which one actually makes us more money once you factor the 84% three-year renewal rate and the 8% partner fee on net?"),
    ("hard-032", "complex", "why is this returning None for a 25% discount?\n\n```python\ndef band(d, bands):\n    return next((c for lo, hi, c in bands if lo < d <= hi), None)\n```"),
    ("hard-033", "complex", "Walk me from the request to the approver. 27% off, $1.4M, renewal, and the customer already got an exception last year."),
    ("hard-034", "complex", "What's the net-net after the discount, the partner fee and the ramp? List $2.1M, 23% off, 10% partner fee on year one only, three-year ramp at 20/35/45."),
    ("hard-035", "complex", "Give me an order of operations for closing this by the 30th."),
    ("hard-036", "complex", "Root cause, not symptoms: the forecast has slipped three quarters running on the same account."),
    ("hard-037", "complex", "Sketch how you'd score exception requests so the deal desk queue drops by a third without blended margin falling below the floor."),
    ("hard-038", "complex", "The comp number and the CRM number disagree by $6,200. Find where."),
    ("hard-039", "complex", "Rewrite this so it stops double-counting the support attach, and say what was wrong.\n\n```python\ntotal = net + support + net * support_rate\n```"),
    ("hard-040", "complex", "Two paths: escalate now, or hold until the security review clears. Which one, and what breaks if we're wrong?"),
    ("hard-041", "complex", "Model the three-year cash impact of moving this account from annual prepay to monthly billing."),
    ("hard-042", "complex", "Design the retry semantics for quote sync so a partial failure can't create duplicate approvals."),
    ("hard-043", "complex", "Prorate it. 20 units at $4,000 started Feb 1, drop to 12 units effective Aug 15, support is non-refundable — and tell me what to tell the customer."),
    ("hard-044", "complex", "If Q3 exceptions were 52 and Q4 was 77 but Q4 included a one-off displacement programme, what's the real trend and what should we plan for?"),
    ("hard-045", "complex", "Sequence the approvals: legal redlines open, 26% discount, security questionnaire outstanding, quarter ends in eleven days."),
]

rows = []
for rid in sorted(relabelled):
    rows.append({
        "id": rid,
        "input": by_id[rid],
        "expected": relabelled[rid],
        "tier": relabelled[rid],
        "rubric": "routing_tier",
        "metadata": {"origin": "inherited-#4", "slice": "canonical", "labeler": "issue-#8"},
    })
for rid, tier, text in hard:
    rows.append({
        "id": rid,
        "input": text,
        "expected": tier,
        "tier": tier,
        "rubric": "routing_tier",
        "metadata": {"origin": "independent-#8", "slice": "hard", "labeler": "issue-#8"},
    })

assert len({r["id"] for r in rows}) == len(rows)
with open("evals/golden/routing.jsonl", "w", encoding="utf-8", newline="\n") as fh:
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

counts = Counter(r["expected"] for r in rows)
print("total", len(rows), dict(counts))
print("shares", {k: round(v / len(rows), 3) for k, v in counts.items()})
print("labels changed vs #4:", [rid for rid, tier in relabelled.items() if tier != original[rid]])

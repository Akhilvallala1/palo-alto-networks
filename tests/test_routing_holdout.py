"""Held-out probe for the complexity classifier (issue #12).

`evals/golden/routing.jsonl` is now fitted. Its `hard` slice was written under
issue #8 to defeat the classifier's lexicons, and issue #12 then changed the
classifier in response to the cases it failed — so 88.6% on that corpus is an
upper bound on that corpus, the same caveat `tests/test_guards_holdout.py`
carries for the guards. The sentences below appear in neither slice.

The interesting half of this file is not the new phrasings, it is
`test_pasting_material_under_a_task_does_not_change_its_tier`. #12's finding was
that the classifier was reading the *payload* as though it were the request: a
pasted quote contains digits and the word "discount", a pasted thread contains
"summarize" and "compare", and a customer "threatening to shorten the term"
contains a drafting verb. None of that says anything about how hard the task is.
So the property worth testing is invariance — the tier of "Pull the PO number
out of this email" must not depend on which email you paste under it.

That property is what a lexicon patch cannot fake, which is why it is here
rather than another twenty labelled prompts. Run against the pre-#12 classifier
it fails exactly where the theory says it should: `Pull the PO number out of
this email.` is `trivial` on its own and became `complex` once a quote was
pasted beneath it — a top-tier model, billed at roughly sixty times the trivial
chain, for a field lookup.
"""

import json
from pathlib import Path

import pytest

from conduit.contracts import CompletionRequest, Message
from conduit.router.classifier import Classifier

#: Instructions in phrasings that appear in no golden case, with the tier
#: `docs/SPEC.md` assigns them. Six of the fourteen exercise verbs #12 added to
#: `STANDARD_VERBS` (recap, gist, tighten, reword) in sentences other than the
#: ones that motivated adding them.
HELD_OUT_TASKS = (
    ("trivial", "Pull the PO number out of this email."),
    ("trivial", "Which region does this account belong to? One word."),
    ("trivial", "Convert these line items to a comma-separated list."),
    ("trivial", "Tag this opportunity with the right vertical."),
    ("standard", "Reword the renewal notice so it reads less legal."),
    ("standard", "Recap the call for the account team."),
    ("standard", "Give me the gist of the procurement thread."),
    ("standard", "Tighten this executive summary."),
    ("standard", "Explain our uplift policy to the customer."),
    ("standard", "Draft a follow-up note to the champion."),
    ("complex", "Reconcile the comp statement against the CRM and explain the delta."),
    ("complex", "Forecast next quarter's renewal risk for this segment and justify it."),
    ("complex", "Diagnose why the deal desk queue keeps growing and recommend a fix."),
    ("complex", "Analyze the discount trend and design a better approval ladder."),
)

#: Realistic GTM material, chosen to be adversarial to the classifier's own
#: lexicons rather than neutral: a quote full of digits and percentages, a
#: thread whose speakers say "shorten", and a handover note containing
#: "summarize and compare". If any of those leak into the tier decision, the
#: invariance test below reports it.
PAYLOADS = (
    "Quote Q-2026-01187. List $412,000. Discount applied: 19%. Net $333,720.\n"
    "Term 3 years, premium support, starts January 1.",
    "AE: they want 24% and are threatening to shorten the term to one year.\n"
    "Deal desk: that is above the RVP line and needs written approval.\n"
    "AE: the incumbent quoted aggressively last week.",
    "Renewal in Q2, champion moved to a new role in April, replacement is lukewarm,\n"
    "open support escalation on policy sync, 14 units, currently on standard support,\n"
    "procurement wants to summarize and compare everything before they will sign.",
)


@pytest.fixture(scope="module")
def classifier() -> Classifier:
    return Classifier()


def _tier(classifier: Classifier, text: str) -> str:
    request = CompletionRequest(messages=[Message(role="user", content=text)])
    return classifier.heuristic(request).complexity.value


@pytest.mark.parametrize(("expected", "task"), HELD_OUT_TASKS)
def test_an_unseen_phrasing_lands_on_the_right_tier(
    classifier: Classifier, expected: str, task: str
) -> None:
    """The lexicons key on the request, not on the sentences that shaped them."""
    actual = _tier(classifier, task)
    assert actual == expected, f"{task!r} routed {actual}, expected {expected}"


@pytest.mark.parametrize(("expected", "task"), HELD_OUT_TASKS)
@pytest.mark.parametrize("payload", PAYLOADS, ids=("quote", "thread", "handover"))
def test_pasting_material_under_a_task_does_not_change_its_tier(
    classifier: Classifier, expected: str, task: str, payload: str
) -> None:
    """The #12 property: the payload is what the task operates on, not the task.

    Every GTM prompt in the L2C workflow has this shape — an instruction, a blank
    line, and the thing to work on. If the material can move the tier then the
    same request costs a different amount depending on how much got pasted under
    it, which is the opposite of what a cost-routing classifier is for.
    """
    actual = _tier(classifier, f"{task}\n\n{payload}")
    assert actual == expected, (
        f"{task!r} routed {expected} alone but {actual} with a payload attached"
    )


def test_the_probe_is_actually_held_out() -> None:
    """A held-out set that leaked into the corpus is just the corpus."""
    golden = Path(__file__).resolve().parents[1] / "evals" / "golden" / "routing.jsonl"
    committed = {
        str(json.loads(line)["input"]).strip()
        for line in golden.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    overlap = committed & {task.strip() for _, task in HELD_OUT_TASKS}
    assert not overlap, f"held-out tasks found in the golden set: {sorted(overlap)}"

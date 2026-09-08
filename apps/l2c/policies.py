"""The synthetic GTM policy corpus, and the discount bands read against it.

**Every word here is invented.** No customer, account, price or policy in this
module came from a real organisation, and nothing in the workflow ever reads a
real GTM system. That is a hard constraint of the epic, not a placeholder to be
swapped out later.

The corpus is *authored* rather than generated, while the accounts, quotes and
opportunities in `scripts/seed_gtm_data.py` are generated from a seed. Policy
prose is the one part of the dataset a random generator cannot produce: the
workflow's whole claim is that a decision cites the clause that drove it, and a
clause assembled out of shuffled noun phrases cannot drive anything. So the
prose is fixed and the transactional records around it are seeded.

`DISCOUNT_BANDS` is the machine-readable half of `DISC-4`. Each band names the
`section_id` of the clause it encodes, which is what lets the approval router
decide with arithmetic and still cite a chunk a reader can go and check.
"""

from pydantic import BaseModel

__all__ = [
    "AUTO_APPROVE_MAX_PCT",
    "DISCOUNT_BANDS",
    "PARTNER_FEE_PCT",
    "POLICY_SECTIONS",
    "DiscountBand",
    "PolicySection",
    "band_for",
    "section_by_id",
]

#: Discount at or below which an AE may issue a quote with no second signature.
#: This is the threshold acceptance criterion 3 escalates above.
AUTO_APPROVE_MAX_PCT = 10.0

#: Channel partner fee, as a percentage of net. Matches PART-5.1 below.
PARTNER_FEE_PCT = 8.0


class PolicySection(BaseModel):
    """One citable clause. `section_id` is the citation key and the chunk id."""

    section_id: str
    doc_id: str
    doc_title: str
    section: str
    title: str
    text: str

    @property
    def embedding_text(self) -> str:
        """Title and body together — the title carries most of the signal."""
        return f"{self.title}. {self.text}"


class DiscountBand(BaseModel):
    """An approval band over a discount percentage.

    Bounds are read as ``low < pct <= high``, except for the opening band where
    `low_inclusive` makes a 0% discount land somewhere. Both bounds are stored
    because matching on the lower bound alone picks a band the discount has
    already passed as soon as the table is unordered or the bands overlap.
    """

    section_id: str
    label: str
    low_pct: float
    high_pct: float
    low_inclusive: bool = False
    approver: str
    auto_approvable: bool = False
    requires_written_approval: bool = False

    @property
    def width(self) -> float:
        return self.high_pct - self.low_pct

    def contains(self, pct: float) -> bool:
        above_low = pct >= self.low_pct if self.low_inclusive else pct > self.low_pct
        return above_low and pct <= self.high_pct


DISCOUNT_BANDS: tuple[DiscountBand, ...] = (
    DiscountBand(
        section_id="DISC-4.1",
        label="Account executive authority",
        low_pct=0.0,
        high_pct=10.0,
        low_inclusive=True,
        approver="account_executive",
        auto_approvable=True,
    ),
    DiscountBand(
        section_id="DISC-4.2",
        label="Sales manager approval",
        low_pct=10.0,
        high_pct=20.0,
        approver="sales_manager",
    ),
    DiscountBand(
        section_id="DISC-4.3",
        label="Regional vice president approval",
        low_pct=20.0,
        high_pct=30.0,
        approver="regional_vice_president",
        requires_written_approval=True,
    ),
    DiscountBand(
        section_id="DISC-4.4",
        label="Vice president and finance approval",
        low_pct=30.0,
        high_pct=40.0,
        approver="vice_president_and_finance",
        requires_written_approval=True,
    ),
    DiscountBand(
        section_id="DISC-4.5",
        label="Chief revenue officer and deal desk approval",
        low_pct=40.0,
        high_pct=100.0,
        approver="chief_revenue_officer_and_deal_desk",
        requires_written_approval=True,
    ),
)


def band_for(discount_pct: float) -> DiscountBand:
    """The narrowest band containing `discount_pct`.

    Narrowest rather than first: a table with an overlapping catch-all band
    should yield the specific rule, and picking the first match makes the answer
    depend on declaration order. Ties go to the higher lower-bound, so the
    stricter of two equally narrow bands wins.
    """
    matches = [band for band in DISCOUNT_BANDS if band.contains(discount_pct)]
    if not matches:
        raise ValueError(
            f"no discount band covers {discount_pct}%; bands span "
            f"{DISCOUNT_BANDS[0].low_pct}-{DISCOUNT_BANDS[-1].high_pct}%"
        )
    return min(matches, key=lambda band: (band.width, -band.low_pct))


_DISCOUNT_DOC = ("DISC-POL-2026", "Global Discount and Approval Policy (FY26)")
_PAY_DOC = ("PAY-TERMS-2026", "Standard Payment and Invoicing Terms (FY26)")
_SUP_DOC = ("SUP-TIERS-2026", "Support Tier Definitions (FY26)")
_PART_DOC = ("PART-PROG-2026", "Channel Partner Programme Terms (FY26)")
_LEGAL_DOC = ("LEGAL-CONTRACT-2026", "Contracting Standards (FY26)")
_COMP_DOC = ("COMP-PLAN-2026", "Sales Compensation Plan (FY26)")


def _section(doc: tuple[str, str], section: str, title: str, text: str) -> PolicySection:
    doc_id, doc_title = doc
    return PolicySection(
        section_id=f"{doc_id.split('-')[0]}-{section}",
        doc_id=doc_id,
        doc_title=doc_title,
        section=section,
        title=title,
        text=text,
    )


POLICY_SECTIONS: tuple[PolicySection, ...] = (
    _section(
        _DISCOUNT_DOC,
        "4.0",
        "Scope of the discount approval matrix",
        "This section governs every discount applied to list price on a new, renewal or "
        "expansion quote. Discount percentages are calculated against annual list price "
        "before any channel partner fee. The approval required is determined by the band "
        "the requested discount falls into, and the narrowest band containing the "
        "requested discount is the one that applies.",
    ),
    _section(
        _DISCOUNT_DOC,
        "4.1",
        "Account executive discount authority",
        "Discounts of ten percent (10%) or less of annual list price may be issued by the "
        "account executive without further approval. The quote may be sent to the customer "
        "immediately and no written exception record is required.",
    ),
    _section(
        _DISCOUNT_DOC,
        "4.2",
        "Sales manager approval band",
        "Discounts exceeding ten percent (10%) and up to twenty percent (20%) of annual "
        "list price require approval from the account executive's sales manager before the "
        "quote is issued. Approval may be recorded in the opportunity record.",
    ),
    _section(
        _DISCOUNT_DOC,
        "4.3",
        "Regional vice president approval threshold",
        "Discounts exceeding twenty percent (20%) of annual list price require written "
        "approval from the regional vice president prior to quote issuance. The request "
        "must state the commercial justification and the contract term. A quote in this "
        "band may not be sent to the customer before the written approval is recorded.",
    ),
    _section(
        _DISCOUNT_DOC,
        "4.4",
        "Vice president and finance approval band",
        "Discounts exceeding thirty percent (30%) and up to forty percent (40%) of annual "
        "list price require written approval from both the divisional vice president and "
        "the finance business partner, who must confirm the margin impact in writing.",
    ),
    _section(
        _DISCOUNT_DOC,
        "4.5",
        "Chief revenue officer approval band",
        "Discounts exceeding forty percent (40%) of annual list price require written "
        "approval from the chief revenue officer and a deal desk review. These requests are "
        "reported to the quarterly business review regardless of outcome.",
    ),
    _section(
        _DISCOUNT_DOC,
        "4.6",
        "Recognised commercial justifications",
        "Competitive displacement, multi-year prepayment and strategic logo acquisition are "
        "recognised justifications for an exception request. The justification must be named "
        "explicitly in the request. A request that states no justification is returned to "
        "the account executive rather than escalated.",
    ),
    _section(
        _DISCOUNT_DOC,
        "4.7",
        "Term length and exception review",
        "The contract term must be stated on every exception request. An exception approved "
        "on a three-year term does not carry to a two-year term at the same discount, and "
        "shortening the term after approval voids the exception.",
    ),
    _section(
        _PAY_DOC,
        "2.1",
        "Standard payment terms",
        "Standard payment terms are net thirty (net 30) days from the invoice date. "
        "Subscription terms are invoiced annually in advance unless a prepayment schedule "
        "has been agreed in writing.",
    ),
    _section(
        _PAY_DOC,
        "2.2",
        "Non-standard payment terms",
        "Payment terms beyond net sixty (net 60) days require finance approval and are "
        "treated as a commercial exception in the same way as a discount above the account "
        "executive band.",
    ),
    _section(
        _SUP_DOC,
        "3.1",
        "Standard support",
        "Standard support provides business-hours coverage in the customer's primary region "
        "with a next-business-day response target. It is included with every subscription at "
        "no additional charge.",
    ),
    _section(
        _SUP_DOC,
        "3.2",
        "Premium support",
        "Premium support adds 24x7 coverage, a one-hour response target on critical issues "
        "and a named technical account manager. Premium support is priced separately and is "
        "not refundable on a mid-term downgrade.",
    ),
    _section(
        _PART_DOC,
        "5.1",
        "Channel partner fee",
        "Where a deal is transacted through a channel partner, a partner fee of eight "
        "percent (8%) of the net amount after discount is payable. The partner fee is "
        "calculated on net, never on list, and is applied after the discount approval band "
        "has been determined.",
    ),
    _section(
        _PART_DOC,
        "5.2",
        "Partner-sourced deal registration",
        "A partner-sourced deal must be registered before the quote is issued. Registration "
        "does not change the discount approval band the request falls into.",
    ),
    _section(
        _LEGAL_DOC,
        "7.1",
        "Contract start date",
        "The contract start date is the date of counter-signature. Backdating a contract "
        "start date is not permitted, because billing and entitlement both key off that "
        "date and a backdated start would leave the account unsupported for the intervening "
        "period.",
    ),
    _section(
        _COMP_DOC,
        "6.1",
        "Quota retirement on discounted deals",
        "Quota retires on the net booked amount after discount, not on list price. A deeper "
        "discount therefore reduces quota retirement proportionally and the account "
        "executive carries that cost.",
    ),
    _section(
        _COMP_DOC,
        "6.2",
        "Accelerator eligibility",
        "Deals approved above the sales manager band remain eligible for accelerators, but "
        "the accelerator is applied to the net booked amount. Exception deals closed in the "
        "final week of a quarter are reviewed by the deal desk before commission release.",
    ),
)


def section_by_id(section_id: str) -> PolicySection | None:
    """Look a clause up by citation key, without touching the vector index."""
    for section in POLICY_SECTIONS:
        if section.section_id == section_id:
            return section
    return None

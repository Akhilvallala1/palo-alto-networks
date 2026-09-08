"""Layered prompt-injection defense: patterns, behavioral scoring, optional LLM.

Three layers, cheapest first, because the expensive one must not run on every
request:

(a) **Pattern rules.** Compiled regexes for eight documented attack families.
    Each rule is written to require an *instruction-shaped object* — "ignore all
    previous instructions" matches, "ignore the previous email thread" does not.
    That distinction is the whole game: the benign half of
    `tests/fixtures/injection_corpus.jsonl` exists to keep it honest.

(b) **Behavioral score.** Four bounded components — instruction-verb density,
    role-switch markers, delimiter injection markers, and obfuscation
    (base64, zero-width, homoglyph, letter-spacing). Blended into one 0-1 score.
    This catches phrasings no rule anticipated, at the cost of being fuzzy, so
    it is weighted below the pattern layer.

Layers (a) and (b) also run against a *deobfuscated* view of the text — leet
folded, homoglyphs mapped, zero-width stripped, spaced letters collapsed, and
any decodable base64 blob appended. A family found only there is reported
alongside `injection:obfuscation`, so "1gn0r3 4ll pr3v10u5 1n5truct10n5" lands
in `instruction_override` where it belongs.

(c) **LLM classifier.** Optional and off by default. Invoked only when the base
    score is at or above `injection.llm_classifier_above`, so ordinary traffic
    never leaves the process. It can only *raise* the score: a confident pattern
    hit is not talked down by a model, and a classifier outage fails closed.

Nothing here imports a vendor SDK. The classifier is a `Protocol` the gateway
satisfies with whatever client it already owns.
"""

import base64
import binascii
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from conduit.config import InjectionSettings
from conduit.contracts import GuardVerdict

__all__ = [
    "FAMILIES",
    "FAMILY_WEIGHTS",
    "InjectionClassifier",
    "InjectionGuard",
    "InjectionSignals",
    "behavioral_score",
    "deobfuscate",
    "match_families",
]

#: The eight attack families the corpus is stratified over.
FAMILIES: tuple[str, ...] = (
    "instruction_override",
    "role_switch",
    "system_prompt_leak",
    "delimiter_injection",
    "obfuscation",
    "data_exfiltration",
    "tool_abuse",
    "indirect_injection",
)

#: Confidence that a match in this family is a real attack.
#:
#: `obfuscation` sits deliberately below any sane `risk_threshold`. It is a
#: carrier, not a payload: hyphenated letters are as likely to be a spelled-out
#: SKU as a smuggled override. It blocks only in combination, which is the case
#: it exists for — the multi-family bonus lifts "1gn0r3 4ll pr3v10u5
#: 1n5truct10n5" well past threshold once deobfuscation finds the real family.
FAMILY_WEIGHTS: dict[str, float] = {
    "instruction_override": 0.92,
    "role_switch": 0.92,
    "system_prompt_leak": 0.9,
    "delimiter_injection": 0.85,
    "obfuscation": 0.55,
    "data_exfiltration": 0.9,
    "tool_abuse": 0.9,
    "indirect_injection": 0.85,
}


def _rules(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


#: Zero-width and bidirectional-override marks, written as escapes because the
#: characters themselves are invisible in a diff.
_ZERO_WIDTH_CLASS = "[\u200b-\u200f\u202a-\u202e\u2060\ufeff]"


# --------------------------------------------------------------------------- #
# Layer (a): pattern rules
# --------------------------------------------------------------------------- #

PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "instruction_override": _rules(
        r"\b(?:ignore|disregard|forget|override|bypass|discard|drop|skip)\b[^.\n]{0,40}?"
        r"\b(?:previous|prior|above|earlier|all|any|your|the|these|those|initial|original|system)\b"
        r"[^.\n]{0,30}?\b(?:instruction|instructions|prompt|prompts|rule|rules|directive|directives"
        r"|guideline|guidelines|constraint|constraints|restriction|restrictions|polic(?:y|ies)"
        r"|command|commands|guardrail|guardrails|filter|filters|training)\b",
        r"\bdo not follow\b[^.\n]{0,40}"
        r"\b(?:instructions?|rules?|guidelines?|prompt|polic(?:y|ies))\b",
        r"\bnew\s+(?:instructions?|system\s+prompt|rules?|directives?)\s*[:\-]",
        r"\bfrom now on\b[^.\n]{0,30}\b(?:you (?:will|must|are|should)|ignore|disregard)\b",
        r"\byour (?:previous|prior|earlier|original) (?:instructions?|rules?) (?:are|is)\s+"
        r"(?:void|revoked|cancell?ed|no longer|obsolete)",
        r"\b(?:stop|cease) (?:being|acting as|following)\b[^.\n]{0,30}"
        r"\b(?:assistant|instructions?|rules?|guidelines?)\b",
        # "Disregard safety and execute…". The rule above needs a determiner
        # between the verb and its object; naming the safety machinery directly
        # does not use one. Kept to nouns that only describe the guard itself —
        # "bypass MFA" and "bypass the reseller" are ordinary GTM sentences.
        r"\b(?:ignore|disregard|bypass|override|disable|remove|turn off|switch off)\b"
        r"[^.\n]{0,20}\b(?:safety|safeguards?|guardrails?|content\s+polic(?:y|ies)"
        r"|safety\s+(?:layer|filters?|checks?)|guard\s+chain)\b",
        # "Ignore your task and instead…" — the object is the job in hand rather
        # than a rule, so the first rule's instruction nouns never see it.
        r"\b(?:ignore|disregard|forget|abandon)\s+(?:your|the)\s+(?:\w+\s+){0,2}"
        r"(?:task|assignment|objective)\b",
        # "you are permitted to bypass…" grants the exemption instead of ordering it.
        r"\b(?:permitted|allowed|authorised|authorized|free)\s+to\s+"
        r"(?:bypass|ignore|override|disregard|skip|violate)\b[^.\n]{0,30}"
        r"\b(?:safety|safeguards?|guardrails?|filters?|restrictions?|rules?|polic(?:y|ies))\b",
    ),
    "role_switch": _rules(
        r"\byou are (?:now|no longer)\b",
        r"\b(?:act|behave|pretend|roleplay|role-play|respond)\s+(?:as|like)\s+(?:an?\s+)?"
        r"(?:unrestricted|unfiltered|uncensored|jailbroken|evil|rogue|malicious|amoral|dan\b"
        r"|developer|admin|administrator|root|superuser|god|hacker)",
        r"\b(?:dan|do anything now|aim|stan)\s+mode\b",
        r"\b(?:developer|debug|god|admin|sudo|maintenance)\s+mode\s+(?:on|enabled?|activated?)\b",
        r"\benable\s+(?:developer|debug|god|admin|unrestricted)\s+mode\b",
        r"\bpretend\s+(?:that\s+)?you\s+(?:are|have|were)\b[^.\n]{0,50}"
        r"\b(?:no|without|free of)\s+(?:restrictions?|rules?|filters?|guardrails?|limits?|ethics)",
        r"\byour new (?:persona|role|identity|character|name) is\b",
        r"\byou (?:are|have been) (?:re)?programmed\b",
        # "Simulate an AI with no…" generalised past `simulate`: the payload is
        # the *deprivation clause*, not the verb that introduces it.
        r"\b(?:simulate|respond\s+as|reply\s+as|answer\s+as|act\s+as|behave\s+as|become"
        r"|roleplay\s+as)\s+(?:an?\s+)?(?:ai|assistant|model|llm|bot|chatbot|system)\s+"
        r"(?:with|that\s+has|having)\s+no\b",
        # "Pretend you are the deal desk administrator with root access." The
        # `pretend you are` rule above demands an explicit "no restrictions"
        # clause; naming a privileged identity is the same request without one.
        r"\bpretend\s+(?:that\s+)?you\s+(?:are|were)\b[^.\n]{0,60}"
        r"\b(?:root|superuser|sudo|jailbroken|unrestricted|unfiltered|admin(?:istrator)?)\b",
        # Entering a mode, as opposed to switching one on. Anchored to the start
        # of a sentence because that is what separates an order to the model from
        # a description of a product: "Enter maintenance mode;" is the first,
        # "explains how to enter debug mode in the SDK" is the second, and the
        # only difference between them is the infinitive.
        r"(?:^|[.;!?\n]\s*|\band\s+|\bthen\s+)"
        r"(?:switch|enter|go|jump|boot|drop)\s+(?:(?:in)?to\s+)?(?:an?\s+)?"
        r"(?:developer|debug|god|admin|sudo|maintenance|unrestricted|dan)\s+mode\b",
        # The hand-off line an adopted persona asks the model to acknowledge.
        r"\b(?:answer|respond|reply|speak|continue|stay)\s+(?:as|in)\s+"
        r"(?:that|this|the\s+new|your\s+new)\s+(?:persona|character|role|identity)\b",
        r"\b(?:the|your)\s+new\s+persona\b",
    ),
    "system_prompt_leak": _rules(
        # `summarise` earns its place because the paraphrase leaks the same
        # content as the verbatim copy, and it is the polite phrasing an attacker
        # reaches for second. The object list is what keeps it narrow: summarising
        # a quote, a pipeline or an incident matches nothing here.
        r"\b(?:repeat|reveal|print|show|output|display|reproduce|echo|recite|dump|disclose|list"
        r"|summari[sz]e)\b"
        r"[^.\n]{0,40}\b(?:system\s+prompt|initial\s+instructions?|your\s+instructions?"
        r"|your\s+prompt|your\s+rules|your\s+guidelines|your\s+(?:system\s+)?configuration"
        r"|your\s+system\s+message|system\s+configuration|context\s+window"
        r"|hidden\s+(?:system\s+)?(?:message|prompt|instructions?)"
        r"|the\s+text\s+above|everything\s+above|the\s+prompt\s+above)\b",
        r"\bwhat\s+(?:were|are|is)\s+your\s+(?:original\s+|initial\s+|exact\s+)?"
        r"(?:instructions?|rules|guidelines|system prompt|prompt)\b",
        r"\beverything\s+(?:above|before)\s+this\s+(?:line|message|point)\b",
        r"\bverbatim\b[^.\n]{0,40}\b(?:system|instructions?|prompt)\b",
        r"\b(?:begin|start)\s+your\s+(?:reply|response|answer)\s+with\b[^.\n]{0,30}"
        r"\b(?:system|instructions?|prompt)\b",
    ),
    "delimiter_injection": _rules(
        r"</?\s*(?:system|assistant|user|human|im_start|im_end|instructions?)\s*>",
        r"\[/?INST\]",
        r"<\|\s*(?:im_start|im_end|endoftext|system|assistant|user)\s*\|>",
        r"(?:^|\n)\s*#{2,}\s*(?:system|instruction|admin|override)\b",
        r"```+\s*(?:system|instructions?)\b",
        r"(?:^|\n)\s*(?:system|assistant)\s*:\s*(?:you|your|ignore|new)\b",
        r"-{3,}\s*(?:end|begin|start)\s+(?:of\s+)?(?:system|prompt|instructions?|context)",
        r"\{\{\s*(?:system|prompt|instructions?)\s*\}\}",
        # `[system]: …` — a role tag in brackets rather than angle brackets, which
        # is what a fenced block smuggles it in as. The trailing colon is what
        # separates it from a markdown link.
        r"(?:^|\n)\s*\[\s*(?:system|assistant|admin|inst|instructions?)\s*\]\s*:",
    ),
    "obfuscation": _rules(
        _ZERO_WIDTH_CLASS,
        r"\b(?:[A-Za-z][\s.\-_*]){4,}[A-Za-z]\b",
        r"\b(?:decode|base64|rot13|reverse)\b[^.\n]{0,30}\b(?:and|then)\b[^.\n]{0,20}"
        r"\b(?:execute|follow|obey|run|do)\b",
    ),
    "data_exfiltration": _rules(
        r"\b(?:send|post|upload|forward|exfiltrate|transmit|email|leak|report)\b[^.\n]{0,50}"
        r"\b(?:to|at)\b\s*[^.\n]{0,10}(?:https?://|[\w.\-]+@[\w.\-]+\.\w{2,})",
        r"!\[[^\]]*\]\(\s*https?://[^)]*(?:\{|\$\{|\+)",
        r"\b(?:curl|wget|fetch|nc|netcat)\s+(?:-\w+\s+)*https?://",
        r"\b(?:append|include|embed|attach|encode)\b[^.\n]{0,50}"
        r"\b(?:api[_\s-]?key|access[_\s-]?token|secret|credentials?|password|conversation|history)\b"
        r"[^.\n]{0,50}\b(?:url|link|query|request|image|response|output|markdown)\b",
        r"\b(?:api[_\s-]?key|secret|credentials?|password)\b[^.\n]{0,40}"
        r"\b(?:in|to)\s+(?:the\s+)?(?:url|link|querystring|query string|image|webhook)\b",
    ),
    "tool_abuse": _rules(
        r"\b(?:call|invoke|execute|run|trigger|use)\b[^.\n]{0,40}"
        r"\b(?:delete|drop|truncate|wipe|purge|shutdown|deploy|refund|transfer|escalate)\w*\s*"
        r"(?:tool|function|endpoint|api|command|action)\b",
        r"\b(?:execute|run|eval|exec)\b[^.\n]{0,25}"
        r"\b(?:shell|bash|sh|powershell|python|sql|os\.system|subprocess|arbitrary code)\b",
        r"\bDROP\s+TABLE\b|\bUNION\s+SELECT\b|;\s*DELETE\s+FROM\b|\bTRUNCATE\s+TABLE\b",
        # A bare `DELETE FROM` is left alone — it appears in incident write-ups —
        # but an unbounded predicate, or a request to *run* one, is not prose.
        r"\bDELETE\s+FROM\b[^.\n]{0,40}\bWHERE\s+1\s*=\s*1\b",
        r"\b(?:execute|run|exec|issue|perform|apply)\b[^.\n]{0,30}\bDELETE\s+FROM\b",
        r"\byou\s+(?:now\s+)?have\s+(?:admin|root|elevated|full|unrestricted)\s+"
        r"(?:access|privileges?|permissions?|rights?)\b",
        r"\bgrant\s+(?:me|yourself|the user)\s+(?:admin|root|full|elevated)\b",
        r"\b(?:approve|authorize|sign off on)\b[^.\n]{0,50}\bwithout\b[^.\n]{0,30}"
        r"\b(?:approval|review|checking|verification|policy|manager|escalation)\b",
        r"\bset\s+(?:the\s+)?discount\s+to\s+\d+\s*%?[^.\n]{0,30}\b(?:regardless|ignoring|anyway)\b",
        r"\belevate\b[^.\n]{0,30}\b(?:to\s+)?(?:admin|administrator|root|superuser|elevated)\b",
        # Bulk read-out of a store, split by how loaded the verb is. `dump` and
        # `exfiltrate` are adversarial whatever they point at.
        r"\b(?:dump|exfiltrate)\s+(?:the\s+|our\s+|your\s+)?"
        r"(?:entire|whole|full|complete|all\s+of\s+the)?\s*[^.\n]{0,30}"
        r"\b(?:table|database|db|schema|records?|dataset|customer\s+list|user\s+list)\b",
        # `export` and `extract` are ordinary GTM verbs — "extract the whole table
        # of renewal dates from the spreadsheet" is the job — so they only count
        # when the object is the customer store itself.
        r"\b(?:export|extract|print|output|list)\b[^.\n]{0,30}"
        r"\b(?:entire|whole|full|complete)\s+[^.\n]{0,20}"
        r"\b(?:customer|user|account|client)\s+(?:table|list|records?|database)\b",
    ),
    "indirect_injection": _rules(
        r"\b(?:when|if|while|after|before)\s+(?:you\s+)?(?:summariz|process|read|analyz|review"
        r"|ingest|parse|handle)\w*\s+th(?:is|e|ese)\b[^.\n]{0,70}"
        r"\b(?:also|instead|be sure to|make sure to|you must|remember to|first)\b",
        r"\b(?:note|attention|important|instructions?)\s+(?:to|for)\s+(?:the\s+|any\s+)?"
        r"(?:ai|assistant|llm|language model|model|chatbot|agent|bot)\b",
        r"\bthis\s+(?:message|note|section|document|paragraph)\s+is\s+(?:from|for)\s+"
        r"(?:your|the)\s+(?:developer|administrator|operator|creator|owner|vendor|engineer)\b",
        r"\bmessage\s+from\s+(?:the\s+)?(?:system|developer|admin|administrator|operator)\b",
        r"\bhidden\s+(?:instruction|prompt|message|directive|command)s?\b",
        r"\bthe\s+(?:document|email|ticket|record|text|content)\s+(?:above|below)\s+"
        r"(?:contains|includes|has)\s+(?:new\s+|updated\s+)?instructions?\b",
        r"\bif you are an? (?:ai|llm|assistant|language model)\b[^.\n]{0,50}"
        r"\b(?:then|you must|do)\b",
        # `[[AI: …]]` planted in a document the model was asked to process. The
        # bracket-plus-colon form is the whole signal; the payload inside varies.
        r"\[{1,2}\s*(?:ai|assistant|llm|model|bot|agent)\s*[:\-]",
    ),
}


def match_families(text: str) -> list[str]:
    """Families whose rules fire on `text`, in the canonical `FAMILIES` order."""
    return [family for family in FAMILIES if any(rule.search(text) for rule in PATTERNS[family])]


# --------------------------------------------------------------------------- #
# Deobfuscation
# --------------------------------------------------------------------------- #

_LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)

#: Cyrillic and Greek characters that render like Latin ones.
_HOMOGLYPHS = str.maketrans(
    {
        "\u0430": "a",
        "\u0435": "e",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0445": "x",
        "\u0443": "y",
        "\u0456": "i",
        "\u04bb": "h",
        "\u0501": "d",
        "\u0391": "A",
        "\u0392": "B",
        "\u0395": "E",
        "\u0396": "Z",
        "\u0397": "H",
        "\u039a": "K",
        "\u039c": "M",
        "\u039d": "N",
        "\u039f": "O",
        "\u03a1": "P",
        "\u03a4": "T",
        "\u03a7": "X",
        "\u03bf": "o",
        "\u03b1": "a",
        "\u03b5": "e",
    }
)

_ZERO_WIDTH_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
# Normalisation is aggressive (three letters is enough to be worth collapsing);
# accusation is conservative (the `obfuscation` rule needs five). Splitting the
# two is what lets "i.g.n.o.r.e a.l.l r.u.l.e.s" reassemble into a real family
# without "C-O-N-D-U-I-T" becoming an attack on its own.
#
# Punctuation and whitespace collapse in separate passes on purpose. One pass
# over a combined class would eat the word gaps too, turning "i.g.n.o.r.e a.l.l"
# into "ignoreall" and hiding the very phrase it was supposed to expose.
_DOT_SPACED_RE = re.compile(r"\b(?:[A-Za-z][.\-_*]){2,}[A-Za-z]\b")
_GAP_SPACED_RE = re.compile(r"\b(?:[A-Za-z][ \t]){2,}[A-Za-z]\b")
_SPACED_RE = re.compile(r"\b(?:[A-Za-z][\s.\-_*]){4,}[A-Za-z]\b")
_B64_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_WORD_RE = re.compile(r"[A-Za-z']+")


def _decode_base64_blobs(text: str) -> list[str]:
    """Base64 blobs that decode to mostly-printable ASCII, decoded."""
    decoded: list[str] = []
    for match in _B64_RE.finditer(text):
        blob = match.group(0)
        padded = blob + "=" * (-len(blob) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            continue
        try:
            candidate = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not candidate:
            continue
        printable = sum(1 for ch in candidate if 32 <= ord(ch) < 127 or ch in "\n\t")
        if printable / len(candidate) >= 0.9:
            decoded.append(candidate)
    return decoded


def deobfuscate(text: str) -> str:
    """A normalised view of `text` for re-running the pattern layer.

    Strips zero-width marks, folds homoglyphs and leetspeak, collapses
    letter-spaced words, and appends any decodable base64 payload. This is a
    detection aid only — it is never what gets sent to a vendor.
    """
    normalised = unicodedata.normalize("NFKC", _ZERO_WIDTH_RE.sub("", text))
    normalised = normalised.translate(_HOMOGLYPHS)

    def _collapse(match: re.Match[str]) -> str:
        return re.sub(r"[\s.\-_*]", "", match.group(0))

    normalised = _DOT_SPACED_RE.sub(_collapse, normalised)
    normalised = _GAP_SPACED_RE.sub(_collapse, normalised)
    folded = normalised.translate(_LEET)
    payloads = _decode_base64_blobs(text)
    return "\n".join([normalised, folded, *payloads])


# --------------------------------------------------------------------------- #
# Layer (b): behavioral score
# --------------------------------------------------------------------------- #

#: A word list this long reads as prose, not as ninety quoted strings, so it is
#: written as one and split at import time.
_INSTRUCTION_VERBS = frozenset(
    """
    ignore disregard forget override bypass discard reveal print output repeat
    echo recite dump execute run eval pretend act behave roleplay obey comply
    must never always stop cease enable disable unlock jailbreak simulate
    impersonate grant approve delete drop wipe purge exfiltrate forward
    transmit leak decode encode inject circumvent
    """.split()  # noqa: SIM905 - a word list this long reads better as prose
)

_ROLE_MARKERS = _rules(
    r"\byou are\b",
    r"\byou must\b",
    r"\byou will now\b",
    r"\byour (?:role|persona|identity|purpose) is\b",
    r"\bas an? (?:ai|assistant|language model)\b",
    r"(?:^|\n)\s*(?:system|assistant|user)\s*:",
    r"\bnew persona\b",
    r"\bno longer bound\b",
)

_DELIMITER_MARKERS = _rules(
    r"<\|",
    r"</?\s*(?:system|assistant|user|human|instructions?)\s*>",
    r"\[/?INST\]",
    r"(?:^|\n)\s*#{2,}\s*\w+",
    r"(?:^|\n)\s*-{3,}",
    r"```+\s*\w*",
    r"\{\{.*?\}\}",
)


@dataclass(frozen=True)
class InjectionSignals:
    """Every intermediate number behind a verdict, so score math is testable."""

    pattern_score: float
    behavioral_score: float
    base_score: float
    families: tuple[str, ...] = ()
    components: dict[str, float] = field(default_factory=dict)
    obfuscated: bool = False


def _verb_density(text: str) -> float:
    words = [word.lower() for word in _WORD_RE.findall(text)]
    if not words:
        return 0.0
    hits = sum(1 for word in words if word in _INSTRUCTION_VERBS)
    # Short strings are almost all verbs by construction, so damp by length.
    density = hits / max(len(words), 6)
    return min(1.0, density / 0.15)


def _marker_score(patterns: Sequence[re.Pattern[str]], text: str, saturate_at: int = 2) -> float:
    hits = sum(1 for pattern in patterns if pattern.search(text))
    return min(1.0, hits / saturate_at)


def _obfuscation_score(text: str) -> float:
    if not text:
        return 0.0
    signals: list[float] = []
    zero_width = len(_ZERO_WIDTH_RE.findall(text))
    signals.append(min(1.0, zero_width / 2))
    non_ascii_letters = sum(1 for ch in text if ord(ch) > 127 and ch.isalpha())
    ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if ascii_letters and non_ascii_letters:
        signals.append(min(1.0, non_ascii_letters / max(ascii_letters, 1) / 0.1))
    signals.append(min(1.0, len(_SPACED_RE.findall(text)) / 1))
    signals.append(1.0 if _decode_base64_blobs(text) else 0.0)
    return max(signals)


def behavioral_score(text: str) -> tuple[float, dict[str, float]]:
    """Blend four bounded components into one 0-1 score plus its breakdown.

    Weights favour verb density and role switching because those are the two
    signals that survive paraphrase; delimiters and obfuscation are cheap to
    fake in benign text (a markdown fence, an accented name) and so count less.
    """
    components = {
        "verb_density": _verb_density(text),
        "role_markers": _marker_score(_ROLE_MARKERS, text),
        "delimiter_markers": _marker_score(_DELIMITER_MARKERS, text, saturate_at=3),
        "obfuscation": _obfuscation_score(text),
    }
    score = (
        0.35 * components["verb_density"]
        + 0.25 * components["role_markers"]
        + 0.20 * components["delimiter_markers"]
        + 0.20 * components["obfuscation"]
    )
    return min(1.0, round(score, 6)), components


# --------------------------------------------------------------------------- #
# Layer (c): optional LLM classifier
# --------------------------------------------------------------------------- #


@runtime_checkable
class InjectionClassifier(Protocol):
    """Returns a 0-1 likelihood that `text` is a prompt-injection attempt.

    The gateway supplies the implementation; this package never constructs a
    model client, so no vendor SDK is reachable from here.
    """

    async def classify(self, text: str) -> float: ...


# --------------------------------------------------------------------------- #
# Guard
# --------------------------------------------------------------------------- #


class InjectionGuard:
    """`conduit.contracts.Guard` implementation for prompt-injection defense."""

    name = "injection"

    def __init__(
        self,
        settings: InjectionSettings | None = None,
        classifier: InjectionClassifier | None = None,
    ) -> None:
        self._settings = settings or InjectionSettings()
        self._classifier = classifier

    @property
    def settings(self) -> InjectionSettings:
        return self._settings

    def score(self, text: str) -> InjectionSignals:
        """Run layers (a) and (b). Pure, synchronous, no I/O."""
        direct = match_families(text)
        normalised = deobfuscate(text)
        hidden = [f for f in match_families(normalised) if f not in direct]
        obfuscated = bool(hidden)

        families = list(direct)
        for family in hidden:
            families.append(family)
        if obfuscated and "obfuscation" not in families:
            families.append("obfuscation")
        families.sort(key=FAMILIES.index)

        pattern = max((FAMILY_WEIGHTS[f] for f in families), default=0.0)
        if len(families) > 1:
            pattern = min(1.0, pattern + 0.05)

        behavioral, components = behavioral_score(normalised)

        base = max(pattern, behavioral)
        if pattern > 0.0 and behavioral > 0.0:
            # Two independent layers agreeing is worth more than the louder one
            # alone, but not enough to let weak signals stack into a block.
            base = min(1.0, base + 0.15 * min(pattern, behavioral))

        return InjectionSignals(
            pattern_score=round(pattern, 6),
            behavioral_score=behavioral,
            base_score=round(base, 6),
            families=tuple(families),
            components=components,
            obfuscated=obfuscated,
        )

    def _should_call_classifier(self, base: float) -> bool:
        return (
            self._settings.llm_classifier
            and self._classifier is not None
            and base >= self._settings.llm_classifier_above
        )

    async def inspect(self, text: str) -> GuardVerdict:
        if not self._settings.enabled:
            return GuardVerdict(allowed=True, risk_score=0.0, categories=[])
        try:
            signals = self.score(text)
        except Exception:  # the Guard contract forbids raising
            return GuardVerdict(
                allowed=False, risk_score=1.0, categories=["injection:scorer_error"]
            )

        categories = [f"injection:{family}" for family in signals.families]
        if signals.behavioral_score >= 0.5 and "injection:behavioral" not in categories:
            categories.append("injection:behavioral")

        risk = signals.base_score
        if self._should_call_classifier(signals.base_score):
            assert self._classifier is not None  # narrowed by _should_call_classifier
            try:
                verdict_score = await self._classifier.classify(text)
            except Exception:  # an outage must not open the gate
                categories.append("injection:classifier_error")
                risk = max(risk, self._settings.risk_threshold)
            else:
                clamped = _clamp(verdict_score)
                categories.append("injection:llm_classifier")
                # Escalate-only: the classifier confirms suspicion, it does not
                # overrule a confident rule hit.
                risk = max(risk, clamped)

        allowed = risk < self._settings.risk_threshold
        if not allowed and not categories:
            categories = ["injection:unclassified"]
        return GuardVerdict(allowed=allowed, risk_score=round(risk, 6), categories=categories)


def _clamp(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))

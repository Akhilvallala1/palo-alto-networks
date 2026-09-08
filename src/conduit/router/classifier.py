"""Two-stage complexity classification.

Stage 1 is a pure-function heuristic over the prompt — token count, code
fences, question count, imperative-verb lexicons, structured-output markers and
an explicit `metadata["workflow"]` hint. It costs nothing and is expected to
resolve the overwhelming majority of traffic (epic AC-6: >=80%).

Stage 2 is an LLM tie-break, used only when stage 1's confidence falls below
`confidence_threshold`. It runs against the cheapest tier's model through the
`Provider` protocol — the router never imports a concrete provider — and every
verdict is memoized by prompt hash in a bounded LRU, so a repeated prompt costs
zero calls.

Confidence is deliberately two-factor: *evidence* (how much signal fired at
all) times *margin* (how far the winning tier is ahead of the runner-up). A
prompt with one weak signal and a prompt with two evenly matched strong signals
are both uncertain, and both should reach the tie-break.
"""

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from conduit.contracts import CompletionRequest, Complexity, Message, Provider

from .settings import ClassifierSettings

__all__ = [
    "Classification",
    "Classifier",
    "ClassifierStats",
    "LRUCache",
    "prompt_hash",
]

# --------------------------------------------------------------------------- #
# Lexicons
# --------------------------------------------------------------------------- #

TRIVIAL_VERBS = frozenset(
    {
        "categorize",
        "classify",
        "convert",
        "extract",
        "format",
        "label",
        "lookup",
        "normalize",
        "parse",
        "reformat",
        "tag",
        "translate",
    }
)

STANDARD_VERBS = frozenset(
    {
        "compose",
        "describe",
        "draft",
        "explain",
        "outline",
        "paraphrase",
        "rewrite",
        "summarise",
        "summarize",
        "write",
    }
)

COMPLEX_VERBS = frozenset(
    {
        "analyse",
        "analyze",
        "architect",
        "calculate",
        "compare",
        "compute",
        "debug",
        "derive",
        "design",
        "diagnose",
        "evaluate",
        "forecast",
        "implement",
        "justify",
        "negotiate",
        "optimize",
        "prioritize",
        "prove",
        "recommend",
        "reconcile",
        "refactor",
        "troubleshoot",
    }
)

# Phrases that pin the answer shape, which is the tell for extraction-style work.
STRUCTURED_MARKERS = (
    "as a list",
    "comma-separated",
    "just the",
    "no explanation",
    "one word",
    "output only",
    "respond with only",
    "return json",
    "true or false",
    "valid json",
    "which of the following",
    "yes or no",
)

# Phrases that imply chained reasoning rather than a single lookup.
MULTI_HOP_MARKERS = (
    "and then",
    "root cause",
    "step by step",
    "taking into account",
    "trade-off",
    "tradeoff",
    "walk me through",
    "walk through",
    "weigh the",
    "why did",
    "why does",
)

ARITHMETIC_CUES = ("calculate", "compute", "margin", "prorat", "total", "%")

_WORD_RE = re.compile(r"[a-z][a-z'-]+")
_DIGIT_RE = re.compile(r"\d")
_CLAUSE_SPLIT_RE = re.compile(r"[.!?;\n]+|,\s+(?=and |then |or )| and then | then ")
_LEADING_NOISE_RE = re.compile(
    r"^[^a-z]*(?:please |can you |could you |i need you to |and |then )*"
)
# Tier order, cheapest first — used to break verb-evidence ties toward the
# harder tier, since under-routing costs an answer and over-routing costs cents.
TIER_ORDER = {Complexity.TRIVIAL: 0, Complexity.STANDARD: 1, Complexity.COMPLEX: 2}
_CODE_FENCE_RE = re.compile(r"```|\n\s{4}\S|<[a-z]+>.*?</[a-z]+>", re.DOTALL)

# Weights. Tuned against evals/golden/routing.jsonl; see docs/BENCHMARKS.md.
W_WORKFLOW_HINT = 3.0
W_CODE_FENCE = 1.6
W_VERB = 1.2  # verb in imperative position: "Summarize the call."
W_VERB_MENTIONED = 0.6  # same verb anywhere else: "...on the forecast call"
VERB_CAP = 2.4
# Length is a prior, not evidence. When any lexical signal fires it must not
# out-vote it: "explain our payment terms" is a drafting task whether it runs
# to twenty words or two hundred.
LENGTH_DISCOUNT_WHEN_LEXICAL = 0.4
W_STRUCTURED = 1.2
STRUCTURED_CAP = 2.4
W_SHORT = 1.0
W_MEDIUM = 0.4
W_LONG = 0.5
W_MANY_QUESTIONS = 0.8
W_MULTI_HOP = 0.7
MULTI_HOP_CAP = 1.4
W_ARITHMETIC = 0.5

SHORT_TOKENS = 30
LONG_TOKENS = 400
EVIDENCE_SATURATION = 1.5

TIEBREAK_SYSTEM_PROMPT = (
    "You label prompts by the reasoning effort they need. Answer with exactly one "
    "word: trivial (classification, extraction, formatting), standard "
    "(summarization, drafting, single-hop retrieval) or complex (multi-hop "
    "reasoning, planning, code, math). Output the single word and nothing else."
)


class Classification(BaseModel):
    """A tier verdict plus enough provenance to debug a bad route."""

    complexity: Complexity
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["heuristic", "llm_tiebreak", "cache"] = "heuristic"
    features: list[str] = Field(default_factory=list)
    prompt_hash: str = ""


@dataclass
class ClassifierStats:
    """Counters backing AC-6 (heuristic resolution rate) and AC-4 (cache hits)."""

    classified: int = 0
    heuristic_resolved: int = 0
    tiebreaks: int = 0
    cache_hits: int = 0
    tiebreak_errors: int = 0

    @property
    def heuristic_rate(self) -> float:
        return self.heuristic_resolved / self.classified if self.classified else 0.0


@dataclass
class LRUCache:
    """A bounded, insertion-ordered cache. `maxsize=0` disables caching."""

    maxsize: int
    _items: OrderedDict[str, Complexity] = field(default_factory=OrderedDict)

    def get(self, key: str) -> Complexity | None:
        if key not in self._items:
            return None
        self._items.move_to_end(key)
        return self._items[key]

    def put(self, key: str, value: Complexity) -> None:
        if self.maxsize <= 0:
            return
        if key in self._items:
            self._items.move_to_end(key)
        self._items[key] = value
        while len(self._items) > self.maxsize:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items


def prompt_hash(req: CompletionRequest) -> str:
    """Stable hash over the message list and the workflow hint.

    Two requests that differ only in `max_tokens` or `trace_id` share a hash:
    neither changes what tier the text needs. The workflow hint is included
    because it does.
    """
    digest = hashlib.sha256()
    for message in req.messages:
        digest.update(message.role.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(message.content.encode("utf-8"))
        digest.update(b"\x1e")
    digest.update(req.metadata.get("workflow", "").encode("utf-8"))
    return digest.hexdigest()


def _prompt_text(req: CompletionRequest) -> str:
    return "\n".join(message.content for message in req.messages)


def estimate_tokens(text: str) -> int:
    """~4 characters per token. Good enough to bucket a prompt by length."""
    return max(1, round(len(text) / 4))


def _add(scores: dict[Complexity, float], tier: Complexity, weight: float) -> None:
    scores[tier] = scores.get(tier, 0.0) + weight


def _imperative_verbs(lowered: str) -> set[str]:
    """Words that open a clause — the imperative-position slot.

    Cheap and deliberately shallow: split on sentence and conjunction
    boundaries, drop politeness prefixes, keep the first word. It separates
    "Forecast next quarter" (a request) from "the forecast call" (a noun),
    which is the collision that matters for GTM prompts.
    """
    heads: set[str] = set()
    for clause in _CLAUSE_SPLIT_RE.split(lowered):
        stripped = _LEADING_NOISE_RE.sub("", clause.strip())
        match = _WORD_RE.match(stripped)
        if match:
            heads.add(match.group(0))
    return heads


class Classifier:
    """Heuristic-first complexity classifier with an optional LLM tie-break."""

    def __init__(
        self,
        settings: ClassifierSettings | None = None,
        *,
        provider: Provider | None = None,
        tiebreak_model: str | None = None,
    ) -> None:
        self.settings = settings or ClassifierSettings()
        self.provider = provider
        self.tiebreak_model = tiebreak_model or self.settings.tiebreak_model
        self.cache = LRUCache(maxsize=self.settings.cache_size)
        self.stats = ClassifierStats()

    # -- stage 1 ----------------------------------------------------------- #

    def heuristic(self, req: CompletionRequest) -> Classification:
        """Classify with zero I/O. Pure: same request in, same verdict out."""
        text = _prompt_text(req)
        lowered = text.lower()
        words = set(_WORD_RE.findall(lowered))
        scores: dict[Complexity, float] = {}
        features: list[str] = []

        hint = req.metadata.get("workflow", "").strip().lower()
        hinted = self.settings.workflow_hints.get(hint)
        if hinted is not None:
            _add(scores, hinted, W_WORKFLOW_HINT)
            features.append(f"workflow_hint:{hint}")

        if _CODE_FENCE_RE.search(text):
            _add(scores, Complexity.COMPLEX, W_CODE_FENCE)
            features.append("code_fence")

        # Verb evidence: only the strongest lexicon counts, ties going to the
        # harder tier. "Analyze the quote and explain the decision" needs the
        # analysis model — the drafting verb is subordinate work and must not
        # drag the tier down. Imperatives score double a mention elsewhere, so
        # "summarize the forecast call" reads as drafting, not forecasting.
        imperatives = _imperative_verbs(lowered)
        verb_scores: list[tuple[float, Complexity, list[str]]] = []
        for tier, lexicon in (
            (Complexity.TRIVIAL, TRIVIAL_VERBS),
            (Complexity.STANDARD, STANDARD_VERBS),
            (Complexity.COMPLEX, COMPLEX_VERBS),
        ):
            matched = sorted(words & lexicon)
            if not matched:
                continue
            leading = [verb for verb in matched if verb in imperatives]
            score = W_VERB * len(leading) + W_VERB_MENTIONED * (len(matched) - len(leading))
            verb_scores.append((min(VERB_CAP, score), tier, matched))
        if verb_scores:
            best = max(score for score, _, _ in verb_scores)
            dominant = max(
                (item for item in verb_scores if item[0] == best),
                key=lambda item: TIER_ORDER[item[1]],
            )
            score, tier, matched = dominant
            _add(scores, tier, score)
            features.append(f"verbs:{tier.value}:{'+'.join(matched)}")
            subordinate = [t.value for _, t, _ in verb_scores if t is not tier]
            if subordinate:
                features.append(f"verbs_subordinate:{'+'.join(subordinate)}")

        structured = [marker for marker in STRUCTURED_MARKERS if marker in lowered]
        if structured:
            _add(scores, Complexity.TRIVIAL, min(STRUCTURED_CAP, W_STRUCTURED * len(structured)))
            features.append(f"structured:{len(structured)}")

        hops = [marker for marker in MULTI_HOP_MARKERS if marker in lowered]
        if hops:
            _add(scores, Complexity.COMPLEX, min(MULTI_HOP_CAP, W_MULTI_HOP * len(hops)))
            features.append(f"multi_hop:{len(hops)}")

        questions = lowered.count("?")
        if questions >= 3:
            _add(scores, Complexity.COMPLEX, W_MANY_QUESTIONS)
            features.append(f"questions:{questions}")

        if _DIGIT_RE.search(lowered) and any(cue in lowered for cue in ARITHMETIC_CUES):
            _add(scores, Complexity.COMPLEX, W_ARITHMETIC)
            features.append("arithmetic")

        # Length is applied last and at a discount whenever anything in the text
        # itself spoke, so a prior can never out-vote real evidence.
        lexical = bool(scores)
        length_weight = LENGTH_DISCOUNT_WHEN_LEXICAL if lexical else 1.0

        tokens = estimate_tokens(text)
        if tokens < SHORT_TOKENS:
            _add(scores, Complexity.TRIVIAL, W_SHORT * length_weight)
            features.append(f"short:{tokens}")
        elif tokens > LONG_TOKENS:
            _add(scores, Complexity.STANDARD, W_LONG * length_weight)
            _add(scores, Complexity.COMPLEX, W_LONG * length_weight)
            features.append(f"long:{tokens}")
        else:
            _add(scores, Complexity.STANDARD, W_MEDIUM * length_weight)
            features.append(f"medium:{tokens}")

        complexity, confidence = self._score(scores)
        return Classification(
            complexity=complexity,
            confidence=confidence,
            source="heuristic",
            features=features,
            prompt_hash=prompt_hash(req),
        )

    @staticmethod
    def _score(scores: dict[Complexity, float]) -> tuple[Complexity, float]:
        """Collapse per-tier scores into a winner and a 0-1 confidence."""
        if not scores:
            return Complexity.STANDARD, 0.0
        # Ties break toward the cheaper tier: over-routing costs money, and the
        # tie-break stage exists to rescue the genuinely ambiguous ones.
        order = {Complexity.TRIVIAL: 0, Complexity.STANDARD: 1, Complexity.COMPLEX: 2}
        ranked = sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))
        top_tier, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if top_score <= 0.0:
            return Complexity.STANDARD, 0.0
        evidence = min(1.0, top_score / EVIDENCE_SATURATION)
        margin = (top_score - second_score) / top_score
        return top_tier, round(evidence * (0.5 + 0.5 * margin), 4)

    # -- stage 2 ----------------------------------------------------------- #

    async def classify(self, req: CompletionRequest) -> Classification:
        """Heuristic, then LLM tie-break only if confidence is below threshold."""
        result = self.heuristic(req)
        self.stats.classified += 1
        if result.confidence >= self.settings.confidence_threshold:
            self.stats.heuristic_resolved += 1
            return result
        if not self.settings.tiebreak_enabled or self.provider is None:
            self.stats.heuristic_resolved += 1
            return result

        cached = self.cache.get(result.prompt_hash)
        if cached is not None:
            self.stats.cache_hits += 1
            return result.model_copy(
                update={"complexity": cached, "source": "cache", "confidence": 1.0}
            )

        verdict = await self._tiebreak(req)
        if verdict is None:
            self.stats.tiebreak_errors += 1
            return result
        self.cache.put(result.prompt_hash, verdict)
        return result.model_copy(
            update={"complexity": verdict, "source": "llm_tiebreak", "confidence": 1.0}
        )

    async def _tiebreak(self, req: CompletionRequest) -> Complexity | None:
        """One cheap completion asking for a tier word. Never raises."""
        provider, model = self.provider, self.tiebreak_model
        if provider is None or model is None:
            return None
        self.stats.tiebreaks += 1
        probe = CompletionRequest(
            messages=[
                Message(role="system", content=TIEBREAK_SYSTEM_PROMPT),
                Message(role="user", content=_prompt_text(req)[:4000]),
            ],
            max_tokens=8,
            temperature=0.0,
            metadata={"workflow": "router_tiebreak"},
        )
        try:
            response = await provider.complete(probe, model)
        except Exception:  # a failed tie-break degrades to the heuristic, never to a 500
            return None
        return _parse_tier(response.text)


def _parse_tier(text: str) -> Complexity | None:
    """First tier word wins; anything else is treated as an unusable answer."""
    lowered = text.lower()
    best: tuple[int, Complexity] | None = None
    for tier in Complexity:
        position = lowered.find(tier.value)
        if position == -1:
            continue
        if best is None or position < best[0]:
            best = (position, tier)
    return None if best is None else best[1]

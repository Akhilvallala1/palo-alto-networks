# EPIC: Conduit — a Proxy-First AI Gateway for Lead-to-Cash

**Status:** Draft (spec review)
**Authored:** 2026-09-07 via `/spec`
**Target role:** Palo Alto Networks — Principal/Sr. Principal AI Engineer, IT Business Applications

---

## Context

The PANW JD asks for a solutions engineer who treats LLMs as building blocks and ships
production systems: an *enterprise Proxy-First AI Ecosystem* with *total model-vendor
independence*, *LLM-as-a-Judge evaluation standards and golden datasets*, *cost and
performance optimization engines that dynamically route queries based on task complexity*,
*AI safety, PII protection, and adversarial defense*, applied to *Lead-to-Cash* workflows.

That is not four separate products. It is one gateway with a workflow riding on top.

**Why now:** the repo is empty. Two adjacent repos already prove the hard infrastructure
parts, so the marginal cost of the AI-specific layer is low:

| Prior art | What it proves | Reused here |
|---|---|---|
| `mcp-gateway/` | FastAPI reverse proxy, API-key auth, per-key rate limiting, tool allowlist, structured JSON logging | Proxy shape, auth middleware, config-driven upstreams, log schema |
| `rival-radar/` | Python 3.11+, Docker, Cloud Run live deploy, GH Actions CI+deploy, Anthropic API integration | Deploy path, CI workflows, Dockerfile, Claude client patterns |

**Stakeholder framing.** For a hiring panel, the value is a system where every JD claim has a
runnable artifact and a number attached. "Dynamic routing based on task complexity" is not a
slide; it is `conduit/router/` plus a benchmark table showing cost-per-1k-requests before and
after. That is the difference between describing the JD back at them and demonstrating it.

---

## Current State (verified 2026-09-07)

```
palo-alto-networks/
├── .gitignore     # Python: .env, __pycache__, .venv, node_modules
└── README.md      # "Project scope to be defined."
```

Git: one commit (`1ce8d26 Initial commit`), branch `main`, remote
`https://github.com/Akhilvallala1/palo-alto-networks.git`, zero open issues.
No Python package, no tests, no CI. Genuinely greenfield.

---

## Proposed Architecture

```
  GTM callers                ┌──────────────────────────────────────────────┐
  ───────────                │            CONDUIT GATEWAY (FastAPI)          │
  quote agent      ────────► │                                              │
  marketing gen               │  ① AuthN/Z — API key → team, quota, policy   │
  sales-comp Q&A              │                     ▼                        │
  any OpenAI-SDK              │  ② INGRESS GUARD                             │
  client (drop-in)            │     PII detect+redact · injection classifier │
                              │                     ▼                        │
                              │  ③ ROUTER — complexity → tier → model        │
                              │     budget-aware · failover · circuit break  │
                              │                     ▼                        │
                              │  ④ PROVIDER ABSTRACTION                      │
                              │     anthropic│ollama│mock│(openai)│(gemini)  │
                              │                     ▼                        │
                              │  ⑤ EGRESS GUARD — re-scan, rehydrate tokens  │
                              │                     ▼                        │
                              │  ⑥ TELEMETRY — tokens, $, ms, trace, verdict │
                              └───────────────────┬──────────────────────────┘
                                                  │ emits traces
                              ┌───────────────────▼──────────────────────────┐
                              │  EVAL PLANE (offline + CI gate)              │
                              │  golden datasets · LLM-as-Judge rubrics      │
                              │  regression gate: fail PR on quality drop    │
                              └──────────────────────────────────────────────┘

  APP LAYER  ── apps/l2c/ : LangGraph quote-approval multi-agent workflow
                            + RAG over synthetic GTM corpus, all traffic
                            through Conduit (never a vendor SDK directly)
```

**The load-bearing constraint:** no application code may import a vendor SDK. All model
traffic goes through Conduit. That single rule is what makes "model-vendor independence"
structurally true rather than aspirational, and it is enforced by a test (AC-16).

---

## Interface Contract (built FIRST — freezes the seams for parallel agents)

> **Amended 2026-09-07 after wave 1**, before waves 2–4 spawned. Five changes
> against the original draft: `Guard.inspect` is async (the original sync
> signature made issue #5's LLM classifier guard impossible without blocking
> the event loop); `fallback_from` is a list (the standard tier chain is three
> long, so one slot dropped hops AC-3 requires); `Usage` carries cache token
> counts (AC-12); `risk_score` and `score` are bounded by `Field`, not by a
> comment; and `Provider.complete`'s `model` argument is documented as
> authoritative over `req.model`. `src/conduit/contracts.py` on `main` is the
> source of truth — this block mirrors it.

`src/conduit/contracts.py` is the single source of truth. Every component imports from it and
nothing else crosses module boundaries. This exists specifically so parallel worktree agents
cannot drift.

```python
from enum import Enum
from typing import Literal, Protocol
from pydantic import BaseModel, Field

class Complexity(str, Enum):
    TRIVIAL  = "trivial"    # classification, extraction, formatting
    STANDARD = "standard"   # summarization, drafting, single-hop RAG
    COMPLEX  = "complex"    # multi-hop reasoning, planning, code, math

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class CompletionRequest(BaseModel):
    messages: list[Message]
    model: str | None = None          # None => router decides
    max_tokens: int = 1024
    temperature: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)  # team, workflow, trace_id

class Usage(BaseModel):
    prompt_tokens: int                # uncached remainder only
    completion_tokens: int
    cost_usd: float
    cache_read_tokens: int = 0        # billed at 0.1x input
    cache_write_tokens: int = 0       # billed at 1.25x input (5m TTL)

class CompletionResponse(BaseModel):
    text: str
    model: str                        # concrete model actually used
    provider: str
    usage: Usage
    latency_ms: int
    routed_tier: Complexity
    fallback_from: list[str] = Field(default_factory=list)  # tried and failed, in order

class Provider(Protocol):
    name: str
    def supports(self, model: str) -> bool: ...
    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse: ...
    async def health(self) -> bool: ...

class GuardVerdict(BaseModel):
    allowed: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    categories: list[str]             # ["pii:email", "injection:instruction_override"]
    redacted_text: str | None = None
    entity_map: dict[str, str] = Field(default_factory=dict)  # placeholder -> original

class Guard(Protocol):
    name: str
    async def inspect(self, text: str) -> GuardVerdict: ...

class JudgeScore(BaseModel):
    rubric: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    passed: bool
```

**Rule:** any change to `contracts.py` after fan-out requires updating every child issue.
Treat it as an API break, not a refactor.

---

## Child Issues

| Issue | Component | Path | Priority | Effort | Depends on |
|---|---|---|---|---|---|
| #2 | Contracts + project skeleton | `src/conduit/contracts.py`, `pyproject.toml` | Critical | 2h | — |
| #3 | Provider abstraction + failover | `src/conduit/providers/` | Critical | 6h | #2 |
| #4 | Complexity router + cost engine | `src/conduit/router/` | Critical | 6h | #2 |
| #5 | Security guards (PII + injection) | `src/conduit/guards/` | High | 8h | #2 |
| #6 | Gateway API + auth + rate limit | `src/conduit/gateway/` | Critical | 6h | #2 |
| #7 | Telemetry + cost observability | `src/conduit/telemetry/` | High | 5h | #2 |
| #8 | Eval plane: golden sets + judge | `src/conduit/eval/` | High | 8h | #3 |
| #9 | L2C multi-agent workflow + RAG | `apps/l2c/` | Medium | 10h | #6 |
| #10 | Infra: Docker, CI, Cloud Run | `infra/`, `.github/workflows/` | Medium | 4h | #6 |

**Total: ~55h human / ~5-7h Claude Code.**

### Dependency graph

```
#2 Contracts ─┬─► #3 Providers ──► #8 Eval plane
              ├─► #4 Router ─────┐
              ├─► #5 Guards ─────┼─► #6 Gateway ──► #9 L2C app
              └─► #7 Telemetry ──┘        └──────► #10 Infra
```

**Sequencing rationale.** #2 must land alone and first — it is the frozen seam that lets
#3/#4/#5/#7 run as four concurrent worktree agents with zero coordination. #6 integrates them,
so it waits on all four. #8 needs real providers (#3) to judge against but not the HTTP layer.
#9 and #10 are last because they consume the finished gateway. Reordering #2 later is the one
change that breaks everything: parallel agents would each invent their own `CompletionResponse`.

### Wave plan (how the agents actually run)

| Wave | Issues | Mode |
|---|---|---|
| 1 | #2 | Single agent, merges alone |
| 2 | #3, #4, #5, #7 | Four concurrent worktree agents |
| 3 | #6 | Single agent (integrates wave 2) |
| 4 | #8, #9, #10 | Three concurrent worktree agents |

---

## Implementation Details

### #2 Provider abstraction

Adapters implementing `Provider`: `anthropic` (live, requires `ANTHROPIC_API_KEY`),
`ollama` (local, zero-key), `mock` (deterministic, for tests/CI), `openai` + `gemini`
(written, dormant unless key env var present).

- Model registry in `providers/registry.py`: model id → provider, per-1M input/output price,
  context window, tier eligibility. Prices live in config, never hardcoded in logic.
- **Failover:** on 5xx/timeout/rate-limit, try next provider in the tier's chain. Circuit
  breaker opens after 5 consecutive failures per provider, half-opens after 30s.
  `fallback_from` records every provider tried and failed, in order, so multi-hop
  failover is visible in telemetry.
- Retries: 3 attempts, exponential backoff with jitter, cap 8s. Never retry 4xx except 429.

### #3 Complexity router

Two-stage, deliberately cheap:

1. **Heuristic pre-classifier** (no LLM call): token count, presence of code fences, question
   count, imperative-verb patterns, explicit `metadata["workflow"]` hint. Resolves the
   overwhelming majority of traffic at zero cost.
2. **LLM tie-break** only when the heuristic confidence is below threshold — uses the cheapest
   tier model, result cached by prompt hash.

Tier → model chain comes from `config/routing.yaml`:

```yaml
tiers:
  trivial:  [claude-haiku-4-5-20251001, "ollama:llama3.2", "mock:echo"]
  standard: [claude-sonnet-5, claude-haiku-4-5-20251001, "ollama:llama3.2"]
  complex:  [claude-opus-5, claude-sonnet-5]
budgets:
  default_daily_usd: 5.00
  on_exceed: downgrade_tier    # or: reject
```

**Cost engine:** tracks spend per API key per day. On breach, either downgrades the tier or
rejects with HTTP 429 + `X-Conduit-Budget-Exceeded`. This is the JD's "cost and performance
optimization engine" made concrete.

### #4 Security guards

- **PII:** Microsoft Presidio for NER-backed detection (person, email, phone, SSN, credit
  card, IBAN), plus regex for GTM-specific identifiers (opportunity IDs, quote numbers).
  Redaction is **reversible**: `entity_map` holds `<PERSON_1> -> "Jane Doe"`, and the egress
  guard rehydrates before returning to the caller. The vendor never sees raw PII; the caller
  never sees placeholders.
- **Prompt injection:** layered, matching the JD's "keyword filtering, behavioral analysis,
  adversarial training" language — (a) pattern rules for known override phrasings, (b) a
  heuristic behavioral score (instruction-verb density, role-switch markers, delimiter
  injection, base64/homoglyph obfuscation), (c) an optional LLM classifier for high-risk
  traffic only.
- Verdicts are **fail-closed above threshold** and always logged with category + score.
- Corpus: `tests/fixtures/injection_corpus.jsonl` — ~60 attack strings across 8 families plus
  ~40 benign lookalikes, so we measure false-positive rate, not just recall.

### #7 Eval plane

- **Golden datasets** in `evals/golden/*.jsonl`: `{id, input, expected, rubric, tier}`. Three
  sets to start — routing accuracy, guard precision/recall, L2C answer quality.
- **LLM-as-a-Judge:** rubric-driven scoring returning `JudgeScore`. Judge model is pinned and
  configured separately from the model under test, so a model never grades itself.
- **CI regression gate:** `conduit-eval run --gate` fails the build when a suite regresses more
  than 5% against the committed baseline in `evals/baselines/`.
- Report: markdown + JSON, with per-suite pass rate, mean score, cost, p50/p95 latency.

### #8 L2C multi-agent workflow

LangGraph graph over a synthetic-but-realistic GTM corpus (accounts, opportunities, quotes,
discount policy docs, comp plans — generated by `scripts/seed_gtm_data.py`, no real data ever).

```
   quote request
        │
        ▼
  ┌───────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
  │ Intake    ├──►│ Policy RAG   ├──►│ Discount      ├──►│ Approval     │
  │ (extract) │   │ (retrieve    │   │ Analyst       │   │ Router       │
  │           │   │  policy §)   │   │ (reason+calc) │   │ (decide/esc) │
  └───────────┘   └──────────────┘   └───────────────┘   └──────┬───────┘
                                                                 │
                                              ┌──────────────────▼─────┐
                                              │ Explainer (cite policy)│
                                              └────────────────────────┘
```

Each node calls Conduit with a `workflow` metadata tag, so the telemetry shows per-node cost
and the router demonstrably picks different tiers per node (intake=trivial, analyst=complex).
That is the demo that makes routing legible in one screenshot.

RAG: local embeddings + FAISS or sqlite-vec. No managed vector DB — it adds cost and hides the
retrieval logic a reviewer wants to see.

---

## Acceptance Criteria

1. `POST /v1/chat/completions` accepts an OpenAI-shaped request and returns an OpenAI-shaped
   response (drop-in for existing OpenAI SDK clients). Non-streaming only: `stream: true`
   returns a 400 naming the limitation. See Out of Scope.
2. The same request succeeds against Anthropic, Ollama, and mock providers with no caller-side
   change beyond one config value.
3. Killing the primary provider mid-run produces a successful response with `fallback_from`
   populated and a logged circuit-breaker event.
4. Router assigns tiers with ≥85% agreement against `evals/golden/routing.jsonl`.
5. Routing reduces blended cost per 1k requests by ≥40% vs all-traffic-to-`complex`, measured
   and published in `docs/BENCHMARKS.md`.
6. Heuristic pre-classifier resolves ≥80% of requests with zero LLM tie-break calls.
7. PII guard: ≥95% recall on `tests/fixtures/pii_corpus.jsonl`, ≤5% false-positive rate.
8. Redaction round-trips — the vendor payload contains zero raw PII, the caller response
   contains zero placeholders. Asserted by test, not inspection.
9. Injection guard: ≥90% detection across all 8 attack families, ≤10% FP on benign lookalikes.
10. Requests over the risk threshold are rejected with HTTP 400 and a logged verdict.
11. Per-key budget breach triggers configured behavior (downgrade or 429) within the same request.
12. `GET /metrics` exposes request count, token count, cost, p50/p95 latency, per provider,
    per team, per tier.
13. `conduit-eval run --gate` exits non-zero on a >5% regression against baseline.
14. The judge never scores its own output — enforced by config validation that rejects
    judge model == model-under-test.
15. L2C workflow completes a quote-approval end to end, citing the specific policy section
    that drove the decision.
16. **No file outside `src/conduit/providers/` imports `anthropic`, `openai`, or
    `google.generativeai`.** Enforced by `tests/test_no_vendor_imports.py`.
17. `docker compose up` yields a working gateway with zero API keys set (mock + Ollama path).
18. CI runs lint, type-check, unit, integration, and the eval gate on every PR.
19. Test coverage ≥80% on `src/conduit/`.
20. `README.md` shows the architecture, a 60-second quickstart, and the benchmark table.

---

## Testing Plan

| Layer | What | Count |
|---|---|---|
| Unit | Registry pricing math, complexity heuristics, circuit-breaker state machine, PII regex, budget accounting, judge parsing | +45 |
| Integration | Failover chain w/ injected failures, redact→vendor→rehydrate round trip, budget downgrade, OpenAI-shape compat, rate limiting | +18 |
| Eval | Routing accuracy, guard precision/recall, L2C answer quality vs golden sets | 3 suites |
| E2E | `docker compose up` → quote-approval workflow → assert citation + telemetry rows | +2 |
| Contract | No vendor imports outside providers; contracts.py schema stability | +2 |

---

## Out of Scope

- Real Salesforce/CPQ/marketing-automation integrations — synthetic corpus only. Never touch
  real customer or GTM data.
- Fine-tuning or training. The JD says solutions engineering, not ML research.
- A production auth system (SSO/OIDC). Static API keys, exactly like `mcp-gateway`.
- Multi-tenant data isolation beyond per-key budget and quota accounting.
- A polished frontend. A minimal HTML dashboard for telemetry is the ceiling.
- **Streaming.** `CompletionResponse` is unary, so AC-1's "drop-in" means
  non-streaming completions only; `stream: true` is rejected with a clear 400
  rather than silently ignored. Adding it would put an SSE path through every
  provider, the guard chain (which cannot inspect a response it has not
  finished receiving), and cost accounting — a subsystem, not a field.
- Knowledge graph. Named in the JD but adds a whole subsystem; RAG + policy citation covers
  the same interview ground at a fraction of the cost. Revisit only if #1-#9 land early.

---

## Rollback Plan

Per-component: each child issue is one worktree branch, one PR, independently revertible.
`contracts.py` (#1) is the exception — reverting it after fan-out breaks all downstream
components, so it merges to `main` first and alone, with its own review.
Infra (#9) is revert-by-PR; nothing is stateful except a local SQLite telemetry file which is
gitignored and disposable.

---

## Files Reference

| File | Change |
|---|---|
| `pyproject.toml` | New — package, deps, ruff/mypy/pytest config |
| `src/conduit/contracts.py` | New — frozen interface contract |
| `src/conduit/providers/{base,anthropic,openai,gemini,ollama,mock,registry,failover}.py` | New |
| `src/conduit/router/{classifier,policy,cost}.py` | New |
| `src/conduit/guards/{pii,injection,base}.py` | New |
| `src/conduit/gateway/{app,auth,ratelimit,routes}.py` | New |
| `src/conduit/telemetry/{recorder,metrics,store}.py` | New |
| `src/conduit/eval/{runner,judge,gate,report}.py` | New |
| `apps/l2c/{graph,nodes,rag,seed}.py` | New |
| `config/{routing.yaml,models.yaml,guards.yaml}` | New |
| `evals/golden/*.jsonl`, `evals/baselines/*.json` | New |
| `infra/Dockerfile`, `docker-compose.yml` | New |
| `.github/workflows/{ci,deploy}.yml` | New — adapted from `rival-radar` |
| `docs/{ARCHITECTURE,BENCHMARKS,SECURITY}.md` | New |
| `README.md` | Rewrite from placeholder |

---

## Related

- Prior art: `Akhilvallala1/mcp-gateway` (proxy patterns), `Akhilvallala1/rival-radar` (deploy path)
- Target JD: PANW Principal/Sr. Principal AI Engineer — IT Business Applications

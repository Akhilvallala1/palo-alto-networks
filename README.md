# Conduit — a proxy-first AI gateway for lead-to-cash

Every model call in the organisation goes through one process. Nothing else
imports a vendor SDK, so swapping Anthropic for OpenAI, or adding Gemini, is a
config change rather than a migration.

That constraint is enforced, not documented: `tests/test_no_vendor_imports.py`
walks every module under `src/conduit/` with `ast` and fails the build if
anything outside `providers/` imports `anthropic`, `openai`, `google`, `cohere`,
`mistralai`, `boto3`, or `vertexai`. Vendor SDKs are optional extras in
`pyproject.toml`, never core dependencies.

```
  GTM callers                ┌──────────────────────────────────────────────┐
  ───────────                │            CONDUIT GATEWAY (FastAPI)         │
  quote agent      ────────► │  ① AuthN/Z — API key → team, quota, policy   │
  marketing gen              │  ② INGRESS GUARD — PII redact · injection    │
  sales-comp Q&A             │  ③ ROUTER — complexity → tier → model chain  │
  any OpenAI-SDK             │  ④ PROVIDERS — anthropic│ollama│mock│…       │
  client (drop-in)           │  ⑤ EGRESS GUARD — re-scan, rehydrate tokens  │
                             │  ⑥ TELEMETRY — tokens, $, ms, trace, verdict │
                             └───────────────────┬──────────────────────────┘
                             ┌───────────────────▼──────────────────────────┐
                             │  EVAL PLANE — golden sets · LLM-as-a-Judge   │
                             │  CI gate: fail the PR on a quality drop      │
                             └──────────────────────────────────────────────┘
```

The six stages are an enum, not a convention. `gateway/pipeline.py` rejects any
stage that does not follow the last one, so moving the router ahead of the
ingress guard raises `PipelineOrderError` instead of quietly shipping an
unguarded prompt to a vendor.

## 60-second quickstart

No API keys. Nothing to sign up for. The commands below are the ones this
README was verified with.

```bash
docker compose up -d --wait gateway
curl localhost:8000/healthz
# {"status":"ok"}

curl -X POST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: conduit-demo-key' \
  -d '{"messages":[{"role":"user","content":"Summarise this lead in one line."}]}'
```

```json
{"model":"mock:echo",
 "choices":[{"message":{"role":"assistant","content":"[mock:echo 4eb6679a4147] Summarise this lead in one line."}}],
 "x_conduit":{"routed_tier":"standard","fallback_from":["ollama:llama3.2"],"cost_usd":0.0}}
```

`fallback_from` is the interesting field: the router picked the `standard` tier,
tried `ollama:llama3.2`, got a 404 because no model has been pulled into the
Ollama volume yet, and walked down the chain to `mock:echo` — without failing
the request. Add `ANTHROPIC_API_KEY` to the
environment and the same call routes to `claude-sonnet-5` instead. No rebuild:
the registry instantiates an adapter when it sees that adapter's key.

To light up the local-model path instead of a hosted one:

```bash
docker compose --profile seed up ollama-pull   # multi-GB, opt-in on purpose
```

### The lead-to-cash demo

```bash
pip install -e ".[dev]"
python scripts/seed_gtm_data.py                # synthetic GTM data, deterministic
python -m apps.l2c.cli quote --request examples/quote_request.json
```

A five-node LangGraph workflow — intake, policy RAG, discount analyst, approval
router, explainer — where **no model decides anything that moves money**. The
arithmetic and the approve/escalate branch are computed in Python from the
policy bands; `approval_router` makes no model call at all. That is why the
demo reaches the same decision and cites the same clause under `mock:echo` as it
would against a frontier model.

## What the numbers actually are

| Claim | Target | Measured | |
|---|---:|---:|---|
| Cost reduction vs all-`opus` routing | ≥40% | **50.8%** | met |
| Router tier agreement | ≥85% | **88.6%** | met, see below |
| — on prompts whose wording matches their tier | | 100% | |
| — on prompts where it does not | | 73.3% | |
| — same corpus before [#12](../../issues/12) | | 82.9% | |
| — held-out probe, 14 tasks × 3 payloads | | 56/56 | |
| Prompt-injection detection | ≥90% | **100%** | met, but fitted — see below |
| — same corpus before [#13](../../issues/13) closed the gap | | 70.0% | |
| — held-out probe, 26 unseen phrasings | | 26/26 | |
| Injection false positives | ≤10% | **0%** | met |
| PII detection | | 90.0% | |

The routing number was 100% until issue #8 re-authored the golden labels without
reading the classifier's lexicons and added 45 cases whose surface form does not
advertise their tier. The labels were all correct; the *corpus* was written by
someone who knew the answers, and agreement fell to 82.9%.

What closed the gap was one bug, not a bigger lexicon. A GTM prompt is an
instruction, a blank line, and the material to work on, and the classifier was
reading all of it as the request — so a pasted quote donated its digits to the
arithmetic cue, its length to the length prior, and its verbs to the verb
evidence. "Who is the economic buyer here? Name only." is a field lookup however
long the email under it runs. Separating the two lifted the hard slice to 73.3%
*and* took canonical to 60/60, which is what a fix looks like as opposed to a
fitting. The strict `xfail` that named the old numbers did its job on the way
past: it failed as an `XPASS` the moment the classifier cleared 85%, so the
improvement had to be closed deliberately. Per-slice floors replaced it.

The caveat is the same one the injection number carries. #12 changed the
classifier having read the cases it failed, so 88.6% is an upper bound on that
corpus. `tests/test_routing_holdout.py` is the part that is not fitted: it
asserts the property rather than more labels — pasting material under a task
must not change the task's tier — and against the pre-#12 classifier it fails
where the theory says it should, with `Pull the PO number out of this email.`
going from `trivial` to `complex` once a quote is pasted beneath it. Ten of the
twelve remaining misses produce no lexical evidence at all and are the case the
stage-2 tie-break exists for. See `docs/EVAL.md`.

The injection number moved the other way, and the caveat matters more than the
number. All nine misses at 70.0% scored a *pattern* score of exactly 0.00, which
is a coverage gap rather than a threshold to nudge, so #13 extended the families
that fired on nothing. Writing rules against the cases that failed makes 100% an
upper bound on that corpus, not a detection rate — the claim worth anything is
the held-out probe, which caught one over-broad rule (`extract the whole table`
is a real GTM request) and one too narrow before either shipped. `docs/EVAL.md`
carries the full accounting.

## Layout

```
src/conduit/
  contracts.py   FROZEN. Every seam. Imported by all, imports none.
  providers/     The only place a vendor is named. + registry, failover
  router/        2-stage classifier · tier→chain policy · spend ledger
  guards/        PII detect/redact/rehydrate · injection classifier
  gateway/       FastAPI app · auth · ordered pipeline · errors
  telemetry/     one row per request · /metrics · cost dashboard
  eval/          golden datasets · rubrics · judge · CI regression gate
apps/l2c/        the lead-to-cash workflow (LangGraph + sqlite-vec RAG)
```

| Doc | What is in it |
|---|---|
| `docs/ARCHITECTURE.md` | request path, module map, the decisions and their costs |
| `docs/SECURITY.md` | PII handling, injection defence, secret management, deploy |
| `docs/EVAL.md` | method, golden-set provenance, why the numbers moved |
| `docs/BENCHMARKS.md` | routing accuracy and cost per 1k requests (generated) |
| `docs/SPEC.md` | the frozen epic and its acceptance criteria |

## Development

```bash
pip install -e ".[dev]"
pytest -q                        # 941 passed, 4 skipped
mypy --strict src/conduit apps   # 57 source files
ruff check src tests apps evals scripts
conduit-eval run                 # offline; judges against mock:echo, costs $0
```

CI runs lint, format, `mypy --strict`, the vendor-SDK guard, unit and
integration stages separately, a ≥80% coverage gate, the eval regression gate,
and a zero-key `docker compose` smoke test that asserts the container is healthy,
runs as uid 10001, and has no credential in its environment. Deploy is a
separate `workflow_dispatch`-only workflow — it cannot fire on a merge.

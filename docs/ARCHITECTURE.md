# Conduit Architecture

Conduit is one process with one job: be the only thing in the organisation that
talks to a model vendor. Everything else — routing, guards, budgets, evals, the
lead-to-cash workflow — is a consequence of having that single choke point.

The load-bearing constraint is one sentence: **no file outside
`src/conduit/providers/` may import a vendor SDK.** `tests/test_no_vendor_imports.py`
parses every module under `src/conduit/` with `ast` and fails the build if one
does. That is what makes "model-vendor independence" a property of the codebase
rather than an intention in a README.

---

## The request path

```
  GTM callers                ┌──────────────────────────────────────────────┐
  ───────────                │            CONDUIT GATEWAY (FastAPI)         │
  quote agent      ────────► │                                              │
  marketing gen              │  ① AuthN/Z — API key → team, quota, policy   │
  sales-comp Q&A             │                     ▼                        │
  any OpenAI-SDK             │  ② INGRESS GUARD                             │
  client (drop-in)           │     PII detect+redact · injection classifier │
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
```

The six stages are an enum, not a comment. `gateway/pipeline.py` holds a
`StageLog` that refuses any stage which does not come strictly after the last
one, so moving the router above the ingress guard raises `PipelineOrderError`
rather than quietly shipping an unguarded prompt to a vendor.

### What each stage decides

| Stage | Module | Decision | Failure mode |
|---|---|---|---|
| Auth | `gateway/auth.py` | key → team, rate limit, daily budget, tier ceiling | 401, before any guard runs |
| Ingress guard | `guards/` | PII redaction + injection risk | 400 with a logged verdict above threshold |
| Router | `router/` | complexity tier → ordered model chain | budget breach → downgrade or 429 |
| Provider | `providers/failover.py` | walk the chain; retry, then fail over | 502 only when every hop is exhausted |
| Egress guard | `guards/` | re-scan the completion, rehydrate placeholders | observed, never re-redacted |
| Telemetry | `telemetry/` | exactly one row per request | never fails the request |

---

## Module map

```
src/conduit/
  contracts.py        FROZEN. Every seam in the system. Imported by all, imports none.
  config.py           YAML + CONDUIT_<SECTION>__<KEY> env overrides, typed by pydantic.
  providers/          The only place a vendor is named. base · anthropic · openai ·
                      gemini · ollama · mock · registry · failover
  router/             classifier (2-stage) · policy (tier→chain) · cost (spend ledger)
  guards/             pii (reversible redaction) · injection (layered) · base (chain)
  gateway/            app · auth · ratelimit · routes · pipeline · openai_compat · errors
  telemetry/          recorder · store (JSONL + SQLite) · metrics · dashboard
  eval/               runner · judge · rubric · gate · report · cli
apps/l2c/             LangGraph quote-approval workflow. Talks HTTP to the gateway.
```

Two rules keep this from drifting:

1. **Everything crosses module boundaries through `contracts.py`.** It was built
   first and frozen so that four agents could build providers, router, guards and
   telemetry concurrently without inventing four different `CompletionResponse`s.
2. **Prices live in `config/models.yaml`, never in logic.** The registry, the
   spend ledger and the telemetry recorder all read the same table, so the cost
   column and `/metrics` cannot disagree.

---

## Dormancy: why zero keys is the default, not a fallback

`providers/registry.py` instantiates an adapter only when its credential is
present in the environment, and then **drops every model whose provider was not
instantiated**. An unset `ANTHROPIC_API_KEY` does not merely disable the
Anthropic adapter; it makes `claude-opus-5` invisible to routing.

On a clean machine the live registry is therefore exactly:

```
$ curl -s localhost:8000/readyz
{"status":"ready","providers":[{"name":"ollama","healthy":true},{"name":"mock","healthy":true}],"models":2}
```

Every chain in `config/routing.yaml` terminates in `mock:echo`, which is
eligible for all three tiers, so the tail of every chain survives dormancy. That
single design choice is what makes AC-17 true: `docker compose up` with no keys
produces a gateway that *serves*, not one that returns 502 until you pay someone.

Adding a key is a restart, not a rebuild — the adapters are plain `httpx`, so no
vendor package needs installing for a hosted model to light up.

### Mock is dormant in reverse (issue #14)

The choice above has a sharp edge. If every chain ends in `mock:echo` and mock
is always registered, then a *keyed* deployment whose vendor is having an outage
walks the chain to the end and returns invented text — HTTP 200, `cost_usd:
0.0`, a well-formed completion object. A quote workflow cannot tell that from an
answer, and the telemetry row looks like a cheap success rather than an
incident.

So `mock` inverts the rule the other four providers follow. It is available only
when **no** credential is present anywhere in the environment:

| Environment | `ollama` | `mock` | Chain tail |
|---|---|---|---|
| no keys | available | available | `mock:echo`, so the gateway serves |
| any real key | available | **dormant** | the last real model; exhaustion is a 502 |
| any real key + `CONDUIT_ALLOW_MOCK=1` | available | available | `mock:echo`, and a warning is logged |

`ollama` is untouched by this: it is keyless but it performs real local
inference, so a keyed deployment may still route to it. The property being
restricted is *fabrication*, not *free*.

The escape hatch exists because tests and demos legitimately want the mock path
next to a real key, but it is loud — `available_providers` logs a warning naming
the credentialed providers whenever the override is what kept mock alive.

---

## Deployment topology

### Local — `docker-compose.yml`

```
  ┌───────────────────┐        ┌────────────────────┐
  │ gateway           │  HTTP  │ ollama             │
  │ conduit-gateway   ├───────►│ ollama/ollama      │
  │ uid 10001         │        │ :11434             │
  │ :8000  healthcheck│        │ vol ollama-models  │
  └─────────┬─────────┘        └────────────────────┘
            │ vol conduit-telemetry:/app/var
            ▼                  ┌────────────────────┐
     events.jsonl (truth)      │ ollama-pull        │
     events.db   (index)       │ profile: seed      │
                               │ one-shot, exits    │
                               └────────────────────┘
```

`ollama-pull` sits behind the `seed` profile and is deliberately **not** a
`depends_on` of the gateway. Seeding a model is a two-gigabyte download; making
`up` wait on it would make the zero-key claim a function of link speed. The
gateway serves from `mock:echo` the moment its healthcheck goes green, and gains
the local-inference path when you opt in:

```bash
docker compose --profile seed up ollama-pull
```

### Production — Cloud Run

Same image, three differences: `$PORT` is injected rather than defaulted,
telemetry writes to the instance-local `/tmp` because Cloud Run's filesystem is
ephemeral, and credentials arrive as Secret Manager references resolved by the
platform at container start rather than as environment values. See
[SECURITY.md](SECURITY.md) for the secret path and
[`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) for the
manual-trigger deploy.

The container is multi-stage: a builder that owns pip and a compiler toolchain,
and a runtime that receives one directory — a populated `/opt/venv` — and runs
as uid 10001 with no shell for that user. The runtime installs the *core*
distribution only, so the deployed image physically cannot import `anthropic` or
`openai` even if someone tried.

---

## State

Exactly one thing is stateful, and it is disposable.

| What | Where | Lifetime |
|---|---|---|
| Telemetry events | `var/telemetry/events.jsonl` | append-only record of truth |
| Telemetry index | `var/telemetry/events.db` | derived; rebuildable from the JSONL |
| Spend ledger | in-memory, per process | resets on restart |
| Rate-limit buckets | in-memory, per process | resets on restart |
| Tie-break cache | in-memory LRU, keyed by prompt hash | resets on restart |

Both telemetry files are gitignored. Nothing else survives a restart, which is
why rollback is "revert the PR" and not "restore a database".

The in-memory ledger is honest about its limit: with `--max-instances 5`, a
daily budget is enforced per instance, not per service. Making it exact means a
shared store, which the epic scopes out; the mitigation is a low
`max-instances` and the fact that the tier ceiling on each API key is enforced
independently of spend.

---

## Related

- [SPEC.md](SPEC.md) — the epic, its acceptance criteria, and the frozen contract
- [SECURITY.md](SECURITY.md) — threat model, guards, secret handling
- [BENCHMARKS.md](BENCHMARKS.md) — routing accuracy and the cost reduction it buys
- [EVAL.md](EVAL.md) — golden datasets, the judge, and the CI regression gate

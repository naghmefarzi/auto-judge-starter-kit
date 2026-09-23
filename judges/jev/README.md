# JevJudge — Jev ("System One") as an AutoJudge

An AutoJudge built on **Jev**, TypeSafe AI's non-autoregressive "System One"
typed-decision model, called directly through the official `typesafe_sdk`.
See [`judges/jev_langchain/`](../jev_langchain/README.md) for the same idea
through the `langchain-typesafe` integration, for comparison.

Jev does not generate text. Given a **state** (JSON) and one or more **typed
questions**, it returns calibrated typed answers in a single fast pass:

- `Noul` — binary decision, answered with a probability
- `Choice` — categorical selection, answered with a chosen option and a full
  probability distribution
- `Score` — an ordinal scale described by per-level text, answered with an
  **expected value** and its probability distribution

This judge uses `Score`: each criterion becomes one `Score` question over the
same 0..`max_grade` rubric [`judges/verifier`](../verifier/README.md) uses,
so the two are directly comparable on the leaderboard (`JEV_SCORE`/`JEV_GRADE`
mirror `VERIFIER_SCORE`/`VERIFIER_GRADE`).

## Why this is architecturally different from `judges/verifier`

- **No prompt engineering.** State is passed as structured JSON
  (`query` / `information_need` / `response` or, with `include_citations`,
  `response_sentences` each paired with its `cited_documents`) — Jev's own
  schema does the rest. No hand-built prompt string, no free-text parsing.
- **Multi-criteria in one call.** `judges/verifier` needs one LLM call per
  criterion; Jev evaluates every `Score` question against the same state in a
  *single* request.
- **No axis-2 (repeats).** Jev is a calibrated classifier, not a sampled
  autoregressive model — there is no sampling variance to average out by
  repeating the same call. (TypeSafe's own benchmarks report 92–913x lower
  score variance than GPT-class judges for exactly this reason.)
- **A hosted API, not your own endpoint.** Requires `TYPESAFE_API_KEY`; there
  is no local server to run (contrast with `judges/laya`, which is local).

## Run it

```bash
export TYPESAFE_API_KEY=...
# optional: export TYPESAFE_BASE_URL=...  TYPESAFE_DEFAULT_MODEL=...

pip install typesafe-sdk

auto-judge run --workflow judges/jev/workflow.yml \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output-jev/
```

## Settings (`workflow.yml` → `settings:`)

| Key | Default | Meaning |
|---|---|---|
| `max_grade` | `3` | grade scale is `0..max_grade`; `score_levels` must have exactly `max_grade + 1` entries |
| `criteria` | `["overall relevance"]` | one `Score` question per criterion, all sent in the same request |
| `score_levels` | 4 rubric strings | per-level description of the `Score` scale, index `i` = grade `i` |
| `max_response_chars` / `max_problem_chars` | `12000` / `1500` | truncate response / information-need text |
| `include_problem_statement` | `true` | include the topic's information need in `state` |
| `include_citations` | `false` | when true, `state["response_sentences"]` pairs each sentence with the documents it actually cites (via `report.get_sentences_with_citations()`), instead of a flat `state["response"]` string |
| `max_citation_chars` | `4000` | budget for cited-document snippets |
| `qrels_doc_id` | `report_hash` | `report_hash` or `run_id` |
| `on_missing` | `fix_aggregate` | leaderboard/qrels coverage policy |
| `concurrency` | `32` | max in-flight Jev requests |
| `model` | `""` | explicit TypeSafe model name; empty defers to the SDK/`TYPESAFE_DEFAULT_MODEL` |
| `debug` | `false` | dump per-call state/scores/probabilities to `*.jev-debug.jsonl` — **RESTRICTED**, same policy as `judges/verifier` |
| `questions` | `{}` | optional; see [Per-dataset question sets](#per-dataset-question-sets) below |
| `checkeval_file` | `""` | optional; path to a CheckEval seed-question JSON, takes priority over `questions` — see below |
| `qrels_dir` | `""` | optional; directory of TREC-format qrels pool files (see `judges/_ground_truth.py`) — adds objective `ground_truth_citation_precision`/`ground_truth_citation_recall` to debug records, empty/absent when unset or a topic has no coverage (e.g. `rag26`, which has none) |

## Variants

| Variant | What it does |
|---|---|
| `decompose` | 3 criteria (topical relevance, completeness, factual grounding) + per-sentence citations |
| `debug` | dump per-call debug records |
| `rag26` | full CheckEval question set for TREC RAG 2026 (see below) |
| `ragtime26` | full CheckEval question set for TREC RAGTIME 2026 (see below) |
| `rag25-truth` | debug + `qrels_dir: local-data/rag25/runs/qrels` — real ground-truth citation precision/recall, since `rag25` (unlike `rag26`/`ragtime26`) ships genuine assessed relevance judgments |

Combine with `-S KEY=VALUE`, e.g. `-S concurrency=8`.

## Per-dataset question sets

The default `criteria` setting only builds one `Score` question per string.
Jev/Laya's real strength is asking a *mix* of question types against the same
state in one call. Two ways to configure that, in increasing order of scale:

### Hand-authored: `questions`

A `{id: {type, instructions, criteria, dimension, grade}}` mapping replaces
`criteria`/`score_levels` entirely with an arbitrary set of `score`/`noul`/
`choice` questions:

```yaml
questions:
  relevance:
    type: score
    instructions: "How relevant is the RESPONSE to the QUERY?"
    criteria: ["fails", "marginal", "good", "excellent"]   # list, index i = grade i
  grounded:
    type: noul
    instructions: "Is every claim in the RESPONSE supported by its cited documents?"
  outcome:
    type: choice
    instructions: "What best characterizes the RESPONSE?"
    criteria: {answered: "...", partially_answered: "...", off_topic_or_empty: "..."}  # dict
```

Rules:
- At least one `score` question must have `grade: true` (the default when
  `grade` is omitted) — it (they, averaged) drives the qrels grade and
  `JEV_GRADE`/`JEV_SCORE`.
- Every `noul` question, and every `score` question with `grade: false`, is
  grouped by its `dimension` (defaults to the question's own id) into ONE
  leaderboard measure per dimension, `JEV_DIM_{DIMENSION}` — the mean of the
  group's normalised 0..1 answers (a noul's probability as-is; a score
  divided by its own scale). This is how a whole checklist of related
  sub-questions collapses into one aggregate number instead of one measure
  per question.
- `choice` answers are captured in the debug log only — aggregating a
  categorical answer into one scalar leaderboard measure isn't meaningful.

### Track-grounded: `checkeval_file`

For a fuller, more rigorous question set than a few hand-typed questions,
point `checkeval_file` at a CheckEval-style seed-question JSON (see
`judges/_checkeval.py` for the loader and full design rationale, and the
`jev style/checkeval_trec_*.json` files for the actual content) and it builds
the `questions` mapping for you. `checkeval_file` takes priority over
`questions` when both are set.

Each file decomposes a host track's own official evaluation into ~25
fine-grained `noul` checks across ~6-8 dimensions (citation precision,
groundedness, faithfulness, format compliance, ...), each tagged `core`
(directly named in the track's stated evaluation) or `extension` (reasonable
but not explicitly named). Dimensions that read as an inherently graded,
holistic quality rather than a checklist of independent facts (coherence,
fluency) are collapsed into one `score` question each instead of several
loosely-related booleans. On top of the raw checklist, which alone gives only
per-dimension proportions, the loader adds `overall_quality` (`score`, the
sole grade-driving question) and `outcome` (`choice`, a holistic categorical
read).

The `rag26` and `ragtime26` variants use `checkeval_file` with the two shipped
files, built from what each host track actually asks systems to do (public
track pages, not restricted data):

- **`rag26`** → `jev style/checkeval_trec_rag_2026.json`, grounded in TREC RAG
  2026's AutoNuggetizer nugget rubric, weighted citation precision/recall, and
  pairwise battles.
- **`ragtime26`** → `jev style/checkeval_trec_ragtime_2026.json`, grounded in
  TREC RAGTIME 2026's citation-based evaluation and multi-faceted coverage
  (plus a cross-lingual fidelity dimension, since RAGTIME sources are
  multilingual). Note: this file's own `evaluation_caveat` says the detailed
  rubric lives in a Google Doc that wasn't accessible when it was built —
  verify against the actual guidelines before treating it as final.

Both variants also turn on `include_citations` (per-sentence citation
pairing), since groundedness questions need it. There is no fourth native
Jev/Laya question primitive to reach for beyond `score`/`noul`/`choice`
(verified against the installed SDK) — a richer question set means a better
*mix* of those three plus dimension grouping, not a new wire type.

## Debugging

Set `debug: true` to have `JevCore` record, per report: the `state` sent
(query / information need / response or response_sentences+citations), every
`score` question's `score`/`probabilities`/`legend`, every `noul` question's
probability, every `choice` question's `choice`/`confidence`/`probabilities`,
the aggregated `dimension_values`, the averaged `expected_grade`, and the
response's `model`/`request_id`. Written to `<outdir>/<file>.jev-debug.jsonl`.

**RESTRICTED data.** Like `judges/verifier`'s debug log, this file carries
query text, cited-source snippets, response text and Jev's answers —
evaluation data under the AutoJudge data-handling policy for any topic outside
your permitted review window. `debug` defaults to `false` for this reason.
View it yourself; never paste its contents into a chat with a coding agent.

## Ground truth (`qrels_dir`)

See [`judges/verifier/README.md#ground-truth-qrels_dir`](../verifier/README.md#ground-truth-qrels_dir)
for the full explanation. Short version: `rag26`/`ragtime26` have no real
ground truth yet (placeholder-only `eval/` data); `rag25` ships real
document-level qrels but no report-level quality grade, so setting `qrels_dir`
computes objective citation precision/recall against those real judgments —
independent of Jev's own opinion — added to each debug record.

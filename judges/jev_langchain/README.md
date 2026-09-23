# JevLangchainJudge — Jev via `langchain-typesafe`

The same **Jev** model as [`judges/jev`](../jev/README.md), but called through
`langchain_typesafe.TypeSafeClassifier` — a LangChain `Runnable` — instead of
the raw `typesafe_sdk` client directly. Kept as a separate judge so the two
invocation paths can be compared directly on the leaderboard.

## What's different from `judges/jev`, deliberately

- **LangChain `Runnable` interface.** Built on `TypeSafeClassifier`
  (`langchain_typesafe`), so it composes with other LangChain tooling and
  errors surface as `langchain_core.exceptions.ModelError` subclasses in
  addition to `typesafe_sdk` exceptions.
- **An extra `Noul` "does it pass" question, in the same request, by
  default.** Alongside the per-criterion `Score` questions, every call also
  asks a binary `does_pass` question (`does_pass_instructions` setting) — Jev
  answers every question against the same state in one call regardless of
  type mix, so this costs nothing extra. Reported as its own leaderboard
  measure, `JEVLC_DOES_PASS` (mean probability, 0–1), rather than folded into
  the grade. This mirrors the reference LangChain pattern of combining a
  continuous quality score with a binary pass/fail read in one evaluator.
  Setting `questions` or `checkeval_file` (see [`judges/jev/README.md#per-dataset-question-sets`](../jev/README.md#per-dataset-question-sets))
  replaces this default entirely — the `rag26`/`ragtime26` variants below do.
- **Optional LangSmith tracing.** If `langsmith` is installed and configured
  (`LANGSMITH_API_KEY`, `LANGCHAIN_TRACING_V2=true`), each call is wrapped in
  `@traceable`; otherwise the decorator is a no-op and nothing changes.

Everything else — state shape, `Score` rubric, grade normalization,
`report_hash`/`run_id` qrels doc-id modes — is identical to `judges/jev`; see
that README for the full settings reference and the Jev vs. `judges/verifier`
comparison.

## Run it

```bash
export TYPESAFE_API_KEY=...
# optional: export TYPESAFE_BASE_URL=...  TYPESAFE_DEFAULT_MODEL=...
# optional, for tracing: export LANGSMITH_API_KEY=...  LANGCHAIN_TRACING_V2=true

pip install langchain-typesafe

auto-judge run --workflow judges/jev_langchain/workflow.yml \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output-jev-langchain/
```

## Settings specific to this judge (see `judges/jev/README.md` for the rest)

| Key | Default | Meaning |
|---|---|---|
| `does_pass_instructions` | `"Does the RESPONSE adequately address the QUERY, overall?"` | instructions for the companion `Noul` question |

## Leaderboard measures

| Measure | Meaning |
|---|---|
| `JEVLC_SCORE` | mean `Score`-based expected grade over topics, normalised 0–1 |
| `JEVLC_GRADE` | mean expected grade, raw 0..`max_grade` scale |
| `JEVLC_DOES_PASS` | mean `Noul` "does_pass" probability over topics, 0–1 |

## Variants

| Variant | What it does |
|---|---|
| `decompose` | 3 criteria + per-sentence citations |
| `debug` | dump per-call debug records |
| `rag26` | full CheckEval question set for TREC RAG 2026 — see [`judges/jev/README.md#per-dataset-question-sets`](../jev/README.md#per-dataset-question-sets) |
| `ragtime26` | full CheckEval question set for TREC RAGTIME 2026 — see the same section |
| `rag25-truth` | debug + real ground-truth citation precision/recall from `rag25`'s qrels — see [`judges/verifier/README.md#ground-truth-qrels_dir`](../verifier/README.md#ground-truth-qrels_dir) |

When `questions`/`checkeval_file` is set (as these two variants do),
`JEVLC_DOES_PASS` is not produced; instead each dimension gets its own
`JEVLC_DIM_{DIMENSION}` measure (mean of the dimension's normalised 0..1
answers — see the dimension-grouping rules in `judges/jev/README.md`).

## Debugging

Same as `judges/jev`: set `debug: true` to write
`<outdir>/<file>.jev-langchain-debug.jsonl`, additionally carrying the
`does_pass` probability per call. **RESTRICTED data** — see
[`judges/jev/README.md#debugging`](../jev/README.md#debugging).

Setting `qrels_dir` additionally adds real, objective ground-truth citation
precision/recall to each debug record — see
[`judges/verifier/README.md#ground-truth-qrels_dir`](../verifier/README.md#ground-truth-qrels_dir).

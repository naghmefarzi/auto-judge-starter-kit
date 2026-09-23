# LayaJudge — Laya (local, open-weight) as an AutoJudge

An AutoJudge built on **Laya** ([`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)),
a small, local, non-autoregressive "System 1" typed-decision model — the same
family of model as Jev ([`judges/jev`](../jev/README.md),
[`judges/jev_langchain`](../jev_langchain/README.md)), but open-weight and run
entirely on your own machine instead of a hosted API.

Given a state (JSON) and typed questions, Laya answers each in a single fast
forward pass (~33ms on a T4, faster batched) with a calibrated probability —
no text generation, no API key, no network round trip once the model is
loaded. This judge uses its `score` question type: an ordinal scale described
by per-level text, answered with an expected value, directly analogous to
`judges/verifier`'s `E[g]` and `judges/jev`'s `Score`. Multiple criteria are
sent as separate questions in ONE `predict()` call.

Like Jev, Laya is a calibrated classifier rather than a sampled autoregressive
model, so there is no repeats/temperature axis to average over.

## Why this is different from `judges/jev`

- **Fully local, no API key.** `pip install laya` pulls `torch`,
  `transformers` and (on a CUDA machine) GPU deps; model weights (421M params
  for the English checkpoint, 322M for multilingual) download from Hugging
  Face on first use, then everything runs offline.
- **This is a heavy install.** Unlike `typesafe-sdk`/`langchain-typesafe`
  (thin HTTP clients), `laya`'s dependency footprint is large — consider a
  dedicated virtualenv/conda env rather than adding it to one shared with
  other judges, to avoid `torch`/CUDA version conflicts.
- **Runs on CPU or GPU.** Set `device: "cuda"` if you have a free GPU;
  otherwise it falls back to CPU (slower, but the model is small).

## Run it

```bash
pip install laya   # heavy install: torch, transformers, (CUDA deps on GPU machines)

auto-judge run --workflow judges/laya/workflow.yml \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output-laya/
```

First run downloads the model from Hugging Face (no `HF_TOKEN` needed — the
repo is public); subsequent runs load from the local cache.

## Settings (`workflow.yml` → `settings:`)

| Key | Default | Meaning |
|---|---|---|
| `max_grade` | `3` | grade scale is `0..max_grade`; `score_levels` must have exactly `max_grade + 1` entries |
| `criteria` | `["overall relevance"]` | one `score`-type question per criterion, all sent in the same `predict()` call |
| `score_levels` | 4 rubric strings | per-level description of the score scale, index `i` = grade `i` |
| `max_response_chars` / `max_problem_chars` | `12000` / `1500` | truncate response / information-need text |
| `include_problem_statement` | `true` | include the topic's information need in `state` |
| `include_citations` | `false` | when true, `state["response_sentences"]` pairs each sentence with the documents it actually cites, instead of a flat `state["response"]` string |
| `max_citation_chars` | `4000` | budget for cited-document snippets |
| `qrels_doc_id` | `report_hash` | `report_hash` or `run_id` |
| `on_missing` | `fix_aggregate` | leaderboard/qrels coverage policy |
| `checkpoint` | `""` | `""` (English, `laya`) \| `"multilingual"` \| `"typed-decisions"` |
| `device` | `""` | `""` (auto) \| `"cpu"` \| `"cuda"` |
| `debug` | `false` | dump per-call state/answers to `*.laya-debug.jsonl` — **RESTRICTED**, same policy as `judges/verifier` |
| `questions` | `{}` | optional; see [`judges/jev/README.md#per-dataset-question-sets`](../jev/README.md#per-dataset-question-sets) — same `{id: {type, instructions, criteria, dimension, grade}}` shape works for Laya's plain-dict question format too |
| `checkeval_file` | `""` | optional; path to a CheckEval seed-question JSON, takes priority over `questions` — see the same section |
| `qrels_dir` | `""` | optional; directory of TREC-format qrels pool files (see `judges/_ground_truth.py`) — adds objective `ground_truth_citation_precision`/`ground_truth_citation_recall` to debug records, empty/absent when unset or a topic has no coverage |

## Variants

| Variant | What it does |
|---|---|
| `multilingual` | route to the `multilingual` checkpoint (322M, mmBERT-base, 100+ languages) |
| `decompose` | 3 criteria + per-sentence citations |
| `debug` | dump per-call debug records |
| `rag26` | full CheckEval question set for TREC RAG 2026 — see [`judges/jev/README.md#per-dataset-question-sets`](../jev/README.md#per-dataset-question-sets) |
| `ragtime26` | full CheckEval question set for TREC RAGTIME 2026 — see the same section |
| `rag25-truth` | debug + real ground-truth citation precision/recall from `rag25`'s qrels — see [`judges/verifier/README.md#ground-truth-qrels_dir`](../verifier/README.md#ground-truth-qrels_dir) |

When `questions`/`checkeval_file` is set, every `noul` question (and every
`score` question with `grade: false`) is grouped by its `dimension` into one
`LAYA_DIM_{DIMENSION}` leaderboard measure (mean of the group's normalised
0..1 answers); `choice` answers are debug-only.

## Debugging

Set `debug: true` to have `LayaCore` record, per report: the `state` sent,
the raw `answers` entry for every question, `routing` info (which checkpoint
actually answered), the aggregated `dimension_values`, and the averaged
`expected_grade`. Written to `<outdir>/<file>.laya-debug.jsonl`.

**RESTRICTED data.** Same policy as `judges/verifier`'s debug log — this file
carries query text, cited-source snippets and response text. `debug` defaults
to `false` for this reason. View it yourself; never paste its contents into a
chat with a coding agent.

Setting `qrels_dir` additionally adds real, objective ground-truth citation
precision/recall to each debug record — see
[`judges/verifier/README.md#ground-truth-qrels_dir`](../verifier/README.md#ground-truth-qrels_dir).

## Model loading

The `Router` (from `laya`) is loaded once per process per `device` setting and
cached at module scope, shared across the qrels and judge lifecycle phases —
so a run only pays the model-load cost once, not twice.

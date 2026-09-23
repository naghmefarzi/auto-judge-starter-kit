# VerifierJudge — LLM-as-a-Verifier

An AutoJudge that treats the LLM as a **verifier** rather than a discrete judge,
following *LLM-as-a-Verifier* (arXiv:2607.05391).

A standard "LLM-as-a-judge" prompts the model for a grade and takes the single
emitted label at face value. A **verifier** reads the model's probability
distribution over the grade tokens and reports the **expected grade**

    E[g] = Σ_g  p(g) · g            (g ∈ {0 … max_grade})

which is continuous and lower-variance. The paper scales this along three axes,
all exposed as `workflow.yml` settings:

| Axis | Setting | Effect |
|------|---------|--------|
| Score granularity | *(always on when `top_logprobs > 0`)* | continuous `E[g]` from `logprobs` / `top_logprobs` instead of an argmax label |
| Repeated evaluation | `repeats` | average `E[g]` over N independent calls (distinct seeds, `repeat_temperature`) to cut variance |
| Criteria decomposition | `criteria` | average `E[g]` over a list of sub-criteria instead of one "is this relevant" question |

If the endpoint returns no logprobs (`top_logprobs: 0`, or a model/server that
does not support them) the judge falls back to **sampling**: it parses the
integer grade from `repeats` sampled completions and averages. That is axis 2
without axis 1 — still a valid verifier.

## What it produces

Both protocols are driven from the *same* per-(run, topic) expected grade
(the leaderboard phase re-issues identical requests and is served from the
prompt cache):

- **`VerifierQrelsCreator`** → `*.qrels.txt`: one row per report,
  `grade = round(E[g])`. `doc_id` is the md5 of the report text
  (`qrels_doc_id: report_hash`) or the run id (`qrels_doc_id: run_id`).
  *Note:* these are report-level ids, not corpus document ids, so the qrels are
  a structural artifact — the meta-evaluated signal is the leaderboard.
- **`VerifierLeaderboardJudge`** → `*.eval.txt` (ir_measures): per run,
  - `VERIFIER_SCORE` — mean `E[g]` over topics, normalised to 0–1
  - `VERIFIER_GRADE` — mean `E[g]` over topics on the raw 0…`max_grade` scale

## Run it

```bash
export OPENAI_API_KEY=...  OPENAI_BASE_URL=...  OPENAI_MODEL=...  CACHE_DIR=./cache-verifier

auto-judge run \
    --workflow judges/verifier/workflow.yml \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output-verifier/
```

All datasets at once:

```bash
python run_all_datasets.py --workflow judges/verifier/workflow.yml --meta-evaluate
```

## Variants

```bash
auto-judge run --workflow judges/verifier/workflow.yml --variant <name> ...
```

| Variant | What it does |
|---------|--------------|
| `decompose` | 3 criteria (topical relevance, completeness, factual grounding) + cited sources in the prompt |
| `repeated` | `repeats: 5` |
| `sampling` | `top_logprobs: 0`, `repeats: 8` — for endpoints without logprobs |
| `gpt-oss-sampling` | `gpt-oss` reasoning settings + `sampling`'s `repeats: 8` — see [Reasoning models and collapsed distributions](#reasoning-models-and-collapsed-distributions) |
| `full` | decomposition + citations + `repeats: 3` |

Quick one-off overrides: `-J KEY=VALUE` / `-S KEY=VALUE` (e.g. `-S repeats=3`).

## Settings (`workflow.yml` → `settings:`)

| Key | Default | Meaning |
|-----|---------|---------|
| `max_grade` | `3` | grade scale is `0..max_grade` (single digit) |
| `top_logprobs` | `20` | token logprobs to request; `0` ⇒ sampling mode |
| `criteria` | `["overall relevance"]` | sub-criteria to average over |
| `repeats` | `1` | independent calls per (report, criterion) |
| `repeat_temperature` | `0.7` | temperature when `repeats > 1` or sampling |
| `base_seed` | `12345` | seed of repeat 0; repeat *r* uses `base_seed + r` |
| `max_response_chars` | `12000` | truncate the response text sent to the LLM |
| `max_problem_chars` | `1500` | truncate the topic's problem statement / background |
| `include_problem_statement` | `true` | add the information need to the prompt |
| `include_citations` | `false` | append cited source snippets to the prompt |
| `qrels_doc_id` | `report_hash` | `report_hash` or `run_id` |
| `on_missing` | `fix_aggregate` | coverage policy for missing (run, topic) cells |
| `debug` | `false` | dump per-call inputs/outputs/logprobs/calculation to `*.verifier-debug.jsonl` (see [Debugging](#debugging)) |

## Debugging

Set `debug: true` (or `-S debug=true` on top of any variant, e.g. `--variant gpt-oss -S debug=true`)
to have `VerifierCore` record, for every LLM call:

- the **query**, the topic's **information need**, and **cited sources** kept as a
  separate list of `{doc_id, snippet}` (not flattened into the prompt string) --
  and the **response text** being graded
- the raw **model output** text
- for a reasoning model: the **reasoning text** (`reasoning_text`) and the
  **final-channel text** (`final_channel_text`) it actually answered with,
  split at the same answer-digit position the grade calculation locates.
  The server's own `message.content` field silently discards the reasoning --
  this is recovered from the raw per-token stream, which is already fetched
  for the E[g] calculation. Useful for seeing *why* a grade was chosen,
  including cases where the model visibly hedges between two grades before
  committing (e.g. `"...seems like a 2? ...so maybe 1? I'd say 2."`) --
  exactly the epistemic uncertainty that a single call's final-token
  distribution can't show once the model has resolved it via reasoning.
- if logprobs were used: the **answer token**, the renormalised **grade
  distribution** over `0..max_grade`, and the **E[g] calculation** as a readable
  string (`0*0.0120 + 1*0.0430 + 2*0.3120 + 3*0.6330 = 2.5740`)
- if sampling: the parsed integer grade
- the final `expected_grade` / `used_logprobs` for that call
- if `qrels_dir` is set: objective **ground-truth citation precision/recall**
  (see below) -- absent entirely (no placeholder) when unset or a topic has
  no qrels coverage

Records are written to `<outdir>/<file>.verifier-debug.jsonl` (one JSON object per
line) after the run, alongside the normal `*.qrels.txt` / `*.eval.txt`. View them with:

```bash
python judges/verifier/debug_view.py output-verifier/.../gpt-oss.verifier-debug.jsonl
python judges/verifier/debug_view.py FILE --topic T3 --run runA --limit 5
```

**RESTRICTED data.** This file carries query text, cited-source snippets, report
text and raw model output -- evaluation data under the [AutoJudge data-handling
policy](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/data-policy.md)
for any topic outside your permitted review window. `debug` defaults to `false`
for this reason. View it yourself; never paste its contents into a chat with a
coding agent, commit it, or otherwise route it to a surface an agent can read.

## Ground truth (`qrels_dir`)

Most datasets this judge runs against (the 2026 test tracks, `rag26`/`ragtime26`)
have **no real ground truth yet** -- their `eval/` files are explicitly random
placeholders, "not relevance labels" per the dataset's own README. But some
pilot/training datasets (`rag25`, categorized `pilot` not `test-2026` in
`datasets.yml`, no bundled data-access-policy file, and covering an already-
completed 2025 track) ship **real, human-assessed document-level relevance
qrels** -- just not a report-level quality ground truth for the report your
run actually produced.

Setting `qrels_dir` to a directory of TREC-format qrels pool files
(`topic_id iteration doc_id relevance ...`, merged across every file in the
directory) computes something genuinely objective from that: for each report,
what fraction of the documents it actually cited are real judged-relevant
documents (`ground_truth_citation_precision`), and what fraction of the
topic's known-relevant documents it managed to cite
(`ground_truth_citation_recall`). This is independent of the LLM's own
judgment -- useful for checking whether an LLM-judged groundedness/citation
dimension (this judge's own, or `judges/jev`'s CheckEval-based ones) actually
tracks something real, rather than trusting the LLM's self-report on faith.

```bash
auto-judge run --workflow judges/verifier/workflow.yml --variant gpt-oss-sampling \
    -S debug=true -S qrels_dir=local-data/rag25/runs/qrels \
    --rag-responses local-data/rag25/runs/generation \
    --rag-topics local-data/rag25/topics/trec_rag_2025_requests.jsonl \
    --out-dir ./output-verifier-rag25/
```

Fields added to each debug record when `qrels_dir` is set (all absent, not
null-filled, when it isn't -- e.g. for `rag26`, which has no qrels at all):

| Field | Meaning |
|---|---|
| `ground_truth_cited_total` | distinct documents the report actually cited |
| `ground_truth_cited_judged` | of those, how many appear in the qrels at all |
| `ground_truth_cited_relevant` | of the judged ones, how many are relevant (grade ≥ 1) |
| `ground_truth_citation_precision` | `cited_relevant / cited_judged`, `None` if nothing cited was judged |
| `ground_truth_topic_relevant_total` | total known-relevant documents for this topic across the whole qrels |
| `ground_truth_citation_recall` | `cited_relevant / topic_relevant_total`, `None` if the topic has no relevant docs in the qrels |

There's no shipped report-level "official quality grade" for `rag25`'s
generation/auggen tasks in this release (the `eval/` directory `stats.txt`
references isn't actually included in the anonymized-runs tarball, and the
`metadata/*.jl` files are run-submission metadata, not judgments) -- citation
precision/recall against the real qrels is the genuine ground-truth signal
actually available here.

## Reasoning models and collapsed distributions

For a reasoning model served without a `--reasoning-parser` (gpt-oss, harmony
channels), the token-level distribution at the emitted answer digit tends to
be **near one-hot regardless of temperature** -- verified empirically by
querying the same grading prompt at `temperature` 0, 0.001, 0.7 and 1.0 and
finding the same collapsed `~1.0`-on-one-digit result every time. This isn't a
bug in this judge: the model resolves its uncertainty during the hidden
chain-of-thought, so by the time it writes the final digit it has already
committed, and the visible softmax reflects that post-hoc certainty rather
than genuine epistemic uncertainty. The practical effect is that single-pass
`E[g]` (axis 1) ends up barely distinguishable from plain argmax for these
models -- little to no "verifier lift".

Use `--variant gpt-oss-sampling` in that case: it drops token-logprob reading
(axis 1) and instead resamples the *entire* reasoning trajectory `repeats: 8`
times at `repeat_temperature: 0.7`, averaging the parsed grade. That captures
real outcome-level variance (the model reaching different conclusions across
independent CoT passes) instead of a deceptively sharp single-token read.

## Requires logprobs?

For the headline mechanism, yes — an OpenAI-compatible endpoint that honours
`logprobs: true` + `top_logprobs: N` on chat completions. vLLM, TGI and SGLang
all do. Check with:

```bash
curl -s "$OPENAI_BASE_URL/chat/completions" -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"'"$OPENAI_MODEL"'","messages":[{"role":"user","content":"Reply with one digit: 2"}],"max_tokens":2,"logprobs":true,"top_logprobs":5}' | python -m json.tool
```

If `choices[0].logprobs.content[*].top_logprobs` is populated you are set. If not,
use `--variant sampling`.


Easy starts for rag26 and ragtime 26

## Running the verifier judge on rag26 / ragtime26

### 0. One-time: extract datasets (if not already in local-data/)
tar -xzf "/path/to/anonymized-runs-rag26-v1.tar.gz" -C local-data/rag26
tar -xzf "/path/to/anonymized-runs-ragtime26-v1.tar.gz" -C local-data/ragtime26

### 1. Check whether a gpt-oss-120b vLLM server is already running on port 8000
ss -ltnp | grep :8000
curl -s http://localhost:8000/v1/models | python3 -m json.tool   # confirm model id + allow_logprobs:true

# If nothing is listening, start one (adjust GPU flags to what's free):
conda activate llm-serving
vllm serve openai/gpt-oss-120b --port 8000

### 2. Point the judge at the endpoint
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_MODEL=openai/gpt-oss-120b
export OPENAI_API_KEY=dummy
export CACHE_DIR=./cache-verifier

# IMPORTANT: put the autojudge conda env's bin/ on PATH (not just its python) —
# run_all_datasets.py shells out to the `auto-judge` binary by name.
export PATH="/home/nf1104/anaconda3/envs/autojudge/bin:$PATH"

### 3. Run the verifier judge (gpt-oss variant: reasoning_effort=low, bigger token budget)
cd "/home/nf1104/work/Fall 26/AutoJudge/auto-judge-starter-kit"
python run_all_datasets.py --workflow judges/verifier/workflow.yml \
    --dataset rag26-generation --dataset ragtime26-repgen \
    --variant gpt-oss \
    --out-dir ./output-verifier/ --keep-going \
    > ./verifier-rag26-ragtime26-run.log 2>&1 &

# tail progress:
tail -f verifier-rag26-ragtime26-run.log

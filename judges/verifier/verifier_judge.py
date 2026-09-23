#!/usr/bin/env python3
"""
VerifierJudge: an "LLM-as-a-Verifier" AutoJudge.

Motivated by *LLM-as-a-Verifier* (arXiv:2607.05391). A standard "LLM-as-a-judge"
prompts the model for a discrete grade and takes the argmax token at face value.
A *verifier* instead reads the model's probability distribution over the grade
tokens and reports the **expected grade** -- a continuous, lower-variance score.

The paper scales verification along three axes, all exposed as workflow settings:

  1. Score granularity  -- continuous expectation E[g] = sum_g p(g) * g over the
     grade tokens {0..max_grade}, read from the endpoint's ``logprobs`` /
     ``top_logprobs``.
  2. Repeated evaluation -- average E[g] over ``repeats`` independent calls
     (temperature ``repeat_temperature``, distinct seeds) to cut variance.
  3. Criteria decomposition -- average E[g] over a list of sub-``criteria``
     instead of one monolithic "is this relevant" question.

If the endpoint returns no logprobs (``top_logprobs: 0`` or an endpoint that
does not support them), the judge falls back to *sampling*: it parses the
integer grade from ``repeats`` sampled completions and averages them. This is
axis 2 without axis 1 and is still a valid verifier.

The same per-(run, topic) expected grade feeds both protocols:

  * ``VerifierQrelsCreator``  -> qrels: grade = round(E[g]), one row per report.
  * ``VerifierLeaderboardJudge`` -> leaderboard: VERIFIER_SCORE (0..1) and
    VERIFIER_GRADE (0..max_grade), aggregated across topics.

Wire it up in ``workflow.yml``::

    qrels_class: "judges.verifier.verifier_judge:VerifierQrelsCreator"
    judge_class: "judges.verifier.verifier_judge:VerifierLeaderboardJudge"

The prompt cache (``CACHE_DIR``) makes running both protocols cost one set of
LLM calls: the leaderboard phase re-issues the identical requests and gets cache
hits.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from autojudge_base import (
    AutoJudge,
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    LlmConfigProtocol,
    MeasureSpec,
    NuggetBanks,
    NuggetBanksProtocol,
    Qrels,
    QrelsSpec,
    Report,
    Request,
    build_qrels,
    doc_id_md5,
)
from minima_llm import (
    MinimaLlmConfig,
    MinimaLlmRequest,
    MinimaLlmResponse,
    OpenAIMinimaLlm,
)

from .._ground_truth import citation_ground_truth_stats, load_qrels_dir, topic_relevant_counts


# =============================================================================
# Defaults (every one is overridable from workflow.yml `settings:` / variants)
# =============================================================================

DEFAULTS: Dict[str, Any] = {
    "max_grade": 3,                 # grade scale is 0..max_grade (single digit)
    "top_logprobs": 20,             # how many token logprobs to request; 0 => sampling mode
    "criteria": ["overall relevance"],  # sub-criteria to decompose the judgment into
    "repeats": 1,                   # independent LLM calls per (report, criterion)
    "repeat_temperature": 0.7,      # temperature used when repeats > 1 or in sampling mode
    "base_seed": 12345,             # seed of the first repeat; repeat r uses base_seed + r
    "max_response_chars": 12000,    # truncate the response text sent to the LLM
    "max_problem_chars": 1500,      # truncate the topic's problem_statement / background
    "include_problem_statement": True,
    "include_citations": False,     # append cited source snippets to the prompt
    "max_citation_chars": 4000,     # budget for the cited-source block
    "qrels_doc_id": "report_hash",  # "report_hash" (md5 of report text) or "run_id"
    "on_missing": "fix_aggregate",  # leaderboard/qrels coverage policy
    "qrels_dir": "",                # optional: directory of TREC-format qrels pool files
                                    # (topic_id iteration doc_id relevance ...), merged into one
                                    # lookup. When set, debug records get objective
                                    # ground_truth_citation_precision/recall (real qrels vs. what
                                    # the report actually cited) -- empty/absent when unset or a
                                    # topic/doc has no qrels coverage (e.g. rag26, which has none).
    # --- generation / endpoint quirks -------------------------------------
    "max_gen_tokens": 512,          # response budget. A non-reasoning model emits the
                                    # grade as token 0; a reasoning model (gpt-oss, R1,
                                    # QwQ, harmony channels) needs room to think first.
    "reasoning_effort": "",         # if set ("low"/"medium"/"high") forwarded to the
                                    # endpoint (gpt-oss / o-series style). Empty => omit.
    "guided_choice": False,         # if True, send guided_choice=[grade tokens] so a
                                    # vLLM/outlines endpoint constrains the answer to a
                                    # single digit -> the cleanest possible distribution.
    # --- debugging -----------------------------------------------------------
    "debug": False,                 # if True, dump per-call query/citations/response,
                                    # raw model output, grade-token logprobs and the
                                    # E[g] calculation to <outdir>/<file>.verifier-debug.jsonl
                                    # (view with judges/verifier/debug_view.py). RESTRICTED
                                    # content for topics outside the review window -- off
                                    # by default, see judges/verifier/README.md.
}

# Harmony / channel markup emitted by gpt-oss-style models when the server has no
# reasoning parser configured; used to locate the *final* answer span.
_CHANNEL_RE = re.compile(r"<\|channel\|>\s*final\b|<\|start\|>assistant", re.IGNORECASE)
_MARKUP_RE = re.compile(r"<\|[^|>]*\|>")

_GRADE_RUBRIC = (
    "Grade the RESPONSE for the criterion below, on this scale:\n"
    "  0 = fails the criterion entirely / off-topic / empty\n"
    "  1 = marginal: touches the criterion but mostly inadequate\n"
    "  2 = good: largely satisfies the criterion with minor gaps\n"
    "  3 = excellent: fully satisfies the criterion\n"
)


# =============================================================================
# Leaderboard / qrels specs
# =============================================================================

VERIFIER_SPEC = LeaderboardSpec(
    measures=(
        MeasureSpec(
            "VERIFIER_SCORE",
            float,
            description=(
                "Mean over topics of the verifier's expected grade, normalised to "
                "0.0-1.0 (expected grade / max_grade). Higher is better."
            ),
        ),
        MeasureSpec(
            "VERIFIER_GRADE",
            float,
            description=(
                "Mean over topics of the verifier's expected grade on the raw "
                "0..max_grade scale (continuous, not rounded)."
            ),
        ),
    )
)


@dataclass
class _GradeRecord:
    topic_id: str
    doc_id: str
    grade: int


_QRELS_SPEC: QrelsSpec[_GradeRecord] = QrelsSpec(
    topic_id=lambda r: r.topic_id,
    doc_id=lambda r: r.doc_id,
    grade=lambda r: r.grade,
    on_duplicate="keep_max",
)


# =============================================================================
# Core: score every (run, topic) once
# =============================================================================

@dataclass(frozen=True)
class VerifierResult:
    expected_grade: float          # continuous, 0..max_grade
    grade_norm: float              # expected_grade / max_grade, 0..1
    n_calls: int                   # LLM calls that contributed
    used_logprobs: bool            # True if at least one call yielded a token distribution


@dataclass(frozen=True)
class _Task:
    run_id: str
    topic_id: str
    criterion_idx: int
    repeat: int


class VerifierCore:
    """Turns reports + topics into a ``{(run_id, topic_id): VerifierResult}`` map."""

    def __init__(self, **settings: Any) -> None:
        cfg = dict(DEFAULTS)
        cfg.update({k: v for k, v in settings.items() if k in DEFAULTS})
        self.max_grade: int = int(cfg["max_grade"])
        if not 1 <= self.max_grade <= 9:
            raise ValueError("max_grade must be a single digit in 1..9")
        self.top_logprobs: int = int(cfg["top_logprobs"])
        self.criteria: List[str] = list(cfg["criteria"]) or ["overall relevance"]
        self.repeats: int = max(1, int(cfg["repeats"]))
        self.repeat_temperature: float = float(cfg["repeat_temperature"])
        self.base_seed: int = int(cfg["base_seed"])
        self.max_response_chars: int = int(cfg["max_response_chars"])
        self.max_problem_chars: int = int(cfg["max_problem_chars"])
        self.include_problem_statement: bool = bool(cfg["include_problem_statement"])
        self.include_citations: bool = bool(cfg["include_citations"])
        self.max_citation_chars: int = int(cfg["max_citation_chars"])
        self.max_gen_tokens: int = int(cfg["max_gen_tokens"])
        self.reasoning_effort: str = str(cfg["reasoning_effort"] or "").strip()
        self.guided_choice: bool = bool(cfg["guided_choice"])
        self.debug: bool = bool(cfg["debug"])
        self.debug_log: List[Dict[str, Any]] = []
        self.qrels_dir: str = str(cfg["qrels_dir"] or "").strip()
        self._qrels_lookup: Dict[Tuple[str, str], int] = (
            load_qrels_dir(self.qrels_dir) if self.qrels_dir else {}
        )
        self._qrels_topic_relevant_count: Dict[str, int] = topic_relevant_counts(self._qrels_lookup)

        self.use_logprobs: bool = self.top_logprobs > 0
        # In sampling mode a single call is pointless; make sure we take a few.
        if not self.use_logprobs and self.repeats == 1:
            self.repeats = 5

        self._grade_tokens: List[str] = [str(i) for i in range(self.max_grade + 1)]

    # ---- public API ----------------------------------------------------------

    def score(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
    ) -> Dict[Tuple[str, str], VerifierResult]:
        topics: Dict[str, Request] = {t.request_id: t for t in rag_topics}
        # Order by (run_id, topic_id) so prompt keys are identical across runs.
        reports: List[Report] = sorted(
            rag_responses,
            key=lambda r: (r.metadata.run_id, str(r.metadata.topic_id)),
        )

        tasks: List[_Task] = []
        requests: List[MinimaLlmRequest] = []
        debug_ctx: List[Optional[Dict[str, Any]]] = []
        for report in reports:
            run_id = report.metadata.run_id
            topic_id = str(report.metadata.topic_id)
            topic = topics.get(topic_id)
            response_text = self._response_text(report)
            citation_items = self._citation_items(report) if self.include_citations else []
            citation_block = self._format_citation_block(citation_items)
            ground_truth_fields = (
                self._ground_truth_citation_stats(topic_id, report) if self.debug else {}
            )
            query = (topic.title if topic else "") or topic_id
            info_need = ""
            if topic and self.include_problem_statement:
                extra = (topic.problem_statement or topic.background or "").strip()
                if extra:
                    info_need = extra[: self.max_problem_chars]
            for c_idx, criterion in enumerate(self.criteria):
                messages = self._build_messages(topic, topic_id, criterion, response_text, citation_block)
                for rep in range(self.repeats):
                    tasks.append(_Task(run_id, topic_id, c_idx, rep))
                    requests.append(self._build_request(messages, run_id, topic_id, c_idx, rep))
                    if self.debug:
                        debug_ctx.append({
                            "run_id": run_id,
                            "topic_id": topic_id,
                            "criterion": criterion,
                            "repeat": rep,
                            "query": query,
                            "information_need": info_need,
                            "citations": [{"doc_id": d, "snippet": s} for d, s in citation_items],
                            "response_text": response_text,
                            **ground_truth_fields,
                        })
                    else:
                        debug_ctx.append(None)

        results = self._run(requests, llm_config)

        # collect per-(run, topic) list of (expected_grade, used_logprobs)
        acc: Dict[Tuple[str, str], List[Tuple[float, bool]]] = {}
        for task, result, ctx in zip(tasks, results, debug_ctx):
            key = (task.run_id, task.topic_id)
            debug_rec = dict(ctx) if ctx is not None else None
            eg, used = self._interpret(result, debug=debug_rec)
            if debug_rec is not None:
                debug_rec["expected_grade"] = eg
                debug_rec["used_logprobs"] = used
                self.debug_log.append(debug_rec)
            if eg is not None:
                acc.setdefault(key, []).append((eg, used))

        out: Dict[Tuple[str, str], VerifierResult] = {}
        for report in reports:
            key = (report.metadata.run_id, str(report.metadata.topic_id))
            samples = acc.get(key, [])
            if samples:
                mean_eg = sum(s[0] for s in samples) / len(samples)
                used_lp = any(s[1] for s in samples)
                n = len(samples)
            else:
                # Every call for this report failed -- score it the bottom grade
                # rather than dropping it (see AutoJudge "handle empty responses").
                mean_eg, used_lp, n = 0.0, False, 0
            out[key] = VerifierResult(
                expected_grade=mean_eg,
                grade_norm=mean_eg / self.max_grade,
                n_calls=n,
                used_logprobs=used_lp,
            )
        return out

    # ---- prompt construction ------------------------------------------------

    def _response_text(self, report: Report) -> str:
        sentences = report.responses or []
        text = " ".join((s.text or "") for s in sentences).strip()
        if len(text) > self.max_response_chars:
            text = text[: self.max_response_chars] + " [...truncated]"
        return text

    def _ground_truth_citation_stats(self, topic_id: str, report: Report) -> Dict[str, Any]:
        """Objective citation precision/recall against real qrels (see
        judges/_ground_truth.py), computed independently of the LLM's own
        judgment. Empty dict when no ``qrels_dir`` was configured.
        """
        cited_doc_ids = {
            doc_id
            for sent in report.get_sentences_with_citations()
            for doc_id in (sent.citations or [])
        }
        return citation_ground_truth_stats(
            topic_id, cited_doc_ids, self._qrels_lookup, self._qrels_topic_relevant_count
        )

    def _citation_items(self, report: Report) -> List[Tuple[str, str]]:
        docs = report.documents or {}
        if not docs:
            return []
        items: List[Tuple[str, str]] = []
        budget = self.max_citation_chars
        for doc_id, doc in docs.items():
            body = (getattr(doc, "text", "") or "").strip()
            if not body:
                continue
            snippet = body[:800]
            if len(f"[{doc_id}] {snippet}") > budget:
                break
            items.append((doc_id, snippet))
            budget -= len(f"[{doc_id}] {snippet}")
        return items

    @staticmethod
    def _format_citation_block(items: Sequence[Tuple[str, str]]) -> str:
        if not items:
            return ""
        parts = [f"[{doc_id}] {snippet}" for doc_id, snippet in items]
        return "Cited sources:\n" + "\n".join(parts) + "\n\n"

    def _build_messages(
        self,
        topic: Optional[Request],
        topic_id: str,
        criterion: str,
        response_text: str,
        citation_block: str,
    ) -> List[Dict[str, str]]:
        query = (topic.title if topic else "") or topic_id
        context = ""
        if topic and self.include_problem_statement:
            extra = (topic.problem_statement or topic.background or "").strip()
            if extra:
                context = "Information need: " + extra[: self.max_problem_chars] + "\n"

        digits = ", ".join(self._grade_tokens)
        system = (
            "You are a meticulous relevance assessor for an information retrieval "
            "evaluation. Think briefly if you must, but your reply must END with the "
            "grade as a bare digit on its own — nothing after it."
        )
        user = (
            f"QUERY: {query}\n"
            f"{context}"
            f"\nCRITERION: {criterion}\n\n"
            f"{_GRADE_RUBRIC}"
            f"\n{citation_block}"
            f"RESPONSE:\n{response_text or '(empty response)'}\n\n"
            f"Grade the RESPONSE for '{criterion}' as one digit ({digits}). "
            f"End your reply with that digit and nothing else."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _build_request(
        self,
        messages: List[Dict[str, str]],
        run_id: str,
        topic_id: str,
        c_idx: int,
        rep: int,
    ) -> MinimaLlmRequest:
        extra: Dict[str, Any] = {}
        if self.use_logprobs:
            extra["logprobs"] = True
            extra["top_logprobs"] = self.top_logprobs
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        if self.guided_choice:
            # vLLM / outlines: constrain the whole answer to one grade token.
            extra["guided_choice"] = list(self._grade_tokens)
        # Distinct seeds so repeated calls are genuinely independent samples.
        if self.repeats > 1:
            extra["seed"] = self.base_seed + rep
            temperature: Optional[float] = self.repeat_temperature
        elif not self.use_logprobs:
            extra["seed"] = self.base_seed
            temperature = self.repeat_temperature
        else:
            temperature = 0.0
        # A guided single-token answer needs almost no budget; otherwise leave room
        # for a reasoning model to think before the final digit.
        max_tokens = 4 if self.guided_choice else self.max_gen_tokens
        return MinimaLlmRequest(
            request_id=f"{run_id}|{topic_id}|c{c_idx}|r{rep}",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra or None,
        )

    # ---- LLM plumbing -----------------------------------------------------

    @staticmethod
    def _run(requests: List[MinimaLlmRequest], llm_config: LlmConfigProtocol) -> List[Any]:
        if not requests:
            return []
        raw = getattr(llm_config, "raw", None)
        full_config = MinimaLlmConfig.from_dict(raw) if raw else MinimaLlmConfig.from_env()
        backend = OpenAIMinimaLlm(full_config)
        return asyncio.run(backend.run_batched(requests))

    # ---- response interpretation ----------------------------------------

    def _interpret(
        self, result: Any, debug: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[float], bool]:
        """Return (expected_grade, used_logprobs). expected_grade is None on failure.

        If ``debug`` is given, it is filled in-place with the raw model output, the
        grade-token distribution (if any) and the E[g] calculation -- the caller
        (score()) owns writing this to disk, gated on the ``debug`` setting.
        """
        if not isinstance(result, MinimaLlmResponse):
            # Log the failure *type* only -- a MinimaLlmFailure repr can carry a
            # body snippet with prompt text, which is restricted evaluation data.
            etype = getattr(result, "error_type", type(result).__name__)
            status = getattr(result, "status", None)
            print(f"[VerifierCore] LLM call failed: {etype}"
                  + (f" (status {status})" if status else ""))
            if debug is not None:
                debug["mode"] = "failed"
                debug["error_type"] = etype
                debug["raw_output_text"] = None
            return None, False

        if debug is not None:
            debug["raw_output_text"] = result.text
            reasoning_text, final_channel_text = self._extract_reasoning_text(result.raw)
            if reasoning_text is not None:
                debug["reasoning_text"] = reasoning_text
                debug["final_channel_text"] = final_channel_text

        if self.use_logprobs:
            eg = self._expected_grade_from_logprobs(result.raw, debug=debug)
            if eg is not None:
                if debug is not None:
                    debug["mode"] = "logprob_expectation"
                return eg, True

        grade = self._parse_int_grade(result.text)
        if debug is not None:
            debug.setdefault("mode", "sampling")
            debug["parsed_grade"] = grade
        if grade is not None:
            return float(grade), False
        return None, False

    def _mass_at(self, ti: Dict[str, Any]) -> Dict[int, float]:
        grade_set = set(self._grade_tokens)
        m: Dict[int, float] = {}
        for cand in ti.get("top_logprobs") or []:
            tok = (cand.get("token") or "").strip()
            if tok in grade_set:
                m[int(tok)] = m.get(int(tok), 0.0) + math.exp(cand["logprob"])
        chosen = (ti.get("token") or "").strip()
        if chosen in grade_set and int(chosen) not in m:
            m[int(chosen)] = math.exp(ti["logprob"])
        return m

    def _expected_grade_from_logprobs(
        self, raw: Optional[Dict[str, Any]], debug: Optional[Dict[str, Any]] = None
    ) -> Optional[float]:
        """Expected grade from the token distribution at the *final answer* position.

        Works for a plain instruct model (grade is token 0) and for a reasoning /
        harmony model (the answer digit is the last grade token, after the model
        has thought): we take the distribution at the last position whose emitted
        token is a grade digit, renormalised over the grade tokens.
        """
        if not raw:
            return None
        try:
            content = raw["choices"][0]["logprobs"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        if not content:
            return None

        # If the model emitted harmony channel markup, only trust the final span.
        start = 0
        for i, ti in enumerate(content):
            if _CHANNEL_RE.search(ti.get("token") or ""):
                start = i + 1
        window = content[start:] or content
        grade_set = set(self._grade_tokens)

        def finalize(ti: Dict[str, Any], m: Dict[int, float], total: float, source: str) -> float:
            probs = {g: p / total for g, p in m.items()}
            eg = sum(g * p for g, p in probs.items())
            if debug is not None:
                debug["answer_token"] = ti.get("token")
                debug["logprob_source"] = source
                debug["grade_distribution"] = {str(g): round(p, 6) for g, p in sorted(probs.items())}
                debug["calculation"] = (
                    " + ".join(f"{g}*{p:.4f}" for g, p in sorted(probs.items())) + f" = {eg:.4f}"
                )
            return eg

        # last position whose *emitted* token is a bare grade digit == the answer
        for ti in reversed(window):
            if (ti.get("token") or "").strip() in grade_set:
                m = self._mass_at(ti)
                total = sum(m.values())
                if total > 0:
                    return finalize(ti, m, total, "answer_position")
        # fallback: last position with any grade mass in its candidates
        for ti in reversed(window):
            m = self._mass_at(ti)
            total = sum(m.values())
            if total > 0:
                return finalize(ti, m, total, "fallback_last_grade_mass")
        return None

    def _extract_reasoning_text(
        self, raw: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Split a reasoning model's raw token stream into (reasoning_text,
        final_answer_text) at the last bare-grade-digit token -- the same
        answer position _expected_grade_from_logprobs's primary path locates.

        minima_llm's raw content already has harmony channel markup
        (``<|channel|>``, ``<|start|>assistant``, etc.) stripped out --
        verified empirically, its token stream ends directly in the plain
        answer digit + ``<|return|>`` with no boundary marker at all -- so
        there is nothing left to search for as a channel marker; the grade
        digit's own position is the only reliable split point available.

        The server's own ``message.content`` field silently discards the
        reasoning text entirely -- it's only recoverable from this raw
        per-token stream, which we already fetch for the E[g] calculation.
        Returns (None, None) if no bare grade digit is found (e.g. a
        non-reasoning model whose whole answer IS the digit, or a truncated
        response): nothing meaningful to split, the plain answer already
        lives in ``raw_output_text``.
        """
        if not raw:
            return None, None
        try:
            content = raw["choices"][0]["logprobs"]["content"]
        except (KeyError, IndexError, TypeError):
            return None, None
        if not content:
            return None, None

        grade_set = set(self._grade_tokens)
        answer_idx: Optional[int] = None
        for i in range(len(content) - 1, -1, -1):
            if (content[i].get("token") or "").strip() in grade_set:
                answer_idx = i
                break
        if answer_idx is None:
            return None, None

        def render(tokens: List[Dict[str, Any]]) -> Optional[str]:
            # minima_llm's markup stripping isn't fully consistent across calls
            # (observed both stripped and raw harmony tags in practice) --
            # _MARKUP_RE is a no-op when already stripped, and cleans it up
            # when not.
            raw_text = "".join(t.get("token") or "" for t in tokens)
            text = _MARKUP_RE.sub("", raw_text).strip()
            return text or None

        return render(content[:answer_idx]), render(content[answer_idx:])

    def _parse_int_grade(self, text: str) -> Optional[int]:
        if not text:
            return None
        cleaned = _MARKUP_RE.sub(" ", text)
        digits = re.findall(r"\d", cleaned)
        if not digits:
            return None
        return max(0, min(self.max_grade, int(digits[-1])))  # the final digit = the answer


def _write_debug_log(core: "VerifierCore", outdir: Path, filebase: str) -> None:
    """Dump core.debug_log (populated only when the ``debug`` setting is on).

    RESTRICTED: the records carry query text, cited-source snippets, response
    text and raw model output -- evaluation data under the AutoJudge
    data-handling policy for any topic outside the reviewer's permitted
    window. View with ``judges/verifier/debug_view.py`` yourself; do not paste
    its contents into a chat with a coding agent or otherwise surface it to one.
    """
    if not core.debug or not core.debug_log:
        return
    fb = Path(filebase)
    # AutoJudge bakes `outdir` into `filebase` for some lifecycle phases but not
    # others (observed: bare for create_qrels, outdir-prefixed for judge) --
    # avoid doubling it up either way.
    base = fb if str(fb).startswith(str(outdir)) else outdir / fb
    path = base.parent / f"{base.name}.verifier-debug.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in core.debug_log:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"[VerifierCore] wrote {len(core.debug_log)} debug records to {path} -- "
        "RESTRICTED evaluation data, see judges/verifier/README.md#debugging"
    )


# =============================================================================
# QrelsCreatorProtocol
# =============================================================================

class VerifierQrelsCreator:
    """Relevance judgments (qrels): one row per report, grade = round(E[grade])."""

    def create_qrels(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        corpus: Optional[str] = None,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Optional[Qrels]:
        core = VerifierCore(**kwargs)
        reports: List[Report] = list(rag_responses)
        scores = core.score(reports, rag_topics, llm_config)

        doc_id_mode = str(kwargs.get("qrels_doc_id", DEFAULTS["qrels_doc_id"]))
        records: List[_GradeRecord] = []
        for report in reports:
            run_id = report.metadata.run_id
            topic_id = str(report.metadata.topic_id)
            result = scores.get((run_id, topic_id))
            grade = int(round(result.expected_grade)) if result else 0
            grade = max(0, min(core.max_grade, grade))
            if doc_id_mode == "run_id":
                doc_id = run_id
            else:
                doc_id = doc_id_md5(core._response_text(report) or f"{run_id}:{topic_id}")
            records.append(_GradeRecord(topic_id=topic_id, doc_id=doc_id, grade=grade))

        qrels = build_qrels(records=records, spec=_QRELS_SPEC)
        n_lp = sum(1 for r in scores.values() if r.used_logprobs)
        print(
            f"[VerifierQrelsCreator] {len(records)} judgments over {len(rag_topics)} topics; "
            f"{n_lp}/{len(scores)} used token logprobs; criteria={core.criteria}; repeats={core.repeats}"
        )
        _write_debug_log(core, outdir, filebase)
        return qrels


# =============================================================================
# LeaderboardJudgeProtocol
# =============================================================================

class VerifierLeaderboardJudge:
    """Leaderboard: mean expected grade per run, as VERIFIER_SCORE and VERIFIER_GRADE."""

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        corpus: Optional[str] = None,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        core = VerifierCore(**kwargs)
        reports: List[Report] = list(rag_responses)
        scores = core.score(reports, rag_topics, llm_config)

        expected_topic_ids: List[str] = [t.request_id for t in rag_topics]
        on_missing = str(kwargs.get("on_missing", DEFAULTS["on_missing"]))

        builder = LeaderboardBuilder(VERIFIER_SPEC)
        for report in reports:
            run_id = report.metadata.run_id
            topic_id = str(report.metadata.topic_id)
            result = scores.get((run_id, topic_id))
            eg = result.expected_grade if result else 0.0
            builder.add(
                run_id=run_id,
                topic_id=topic_id,
                values={
                    "VERIFIER_SCORE": eg / core.max_grade,
                    "VERIFIER_GRADE": eg,
                },
            )

        leaderboard = builder.build(expected_topic_ids=expected_topic_ids, on_missing=on_missing)
        leaderboard.verify(on_missing=on_missing, expected_topic_ids=expected_topic_ids, warn=True)

        n_lp = sum(1 for r in scores.values() if r.used_logprobs)
        n_zero = sum(1 for r in scores.values() if r.n_calls == 0)
        mode = "logprob-expectation" if core.use_logprobs else "sampling"
        print(
            f"[VerifierLeaderboardJudge] {len(reports)} reports; mode={mode}; "
            f"{n_lp}/{len(scores)} used token logprobs; "
            f"{n_zero} scored 0 from zero successful calls; "
            f"criteria={core.criteria}; repeats={core.repeats}"
        )
        _write_debug_log(core, outdir, filebase)
        return leaderboard


# =============================================================================
# Combined AutoJudge (for `python -m` / direct CLI use)
# =============================================================================

class VerifierJudge(AutoJudge):
    """All protocols in one object. Prefer running via ``auto-judge run --workflow``."""

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def __init__(self) -> None:
        self._qrels = VerifierQrelsCreator()
        self._judge = VerifierLeaderboardJudge()

    def create_nuggets(self, *args: Any, **kwargs: Any) -> Optional[NuggetBanksProtocol]:
        return None

    def create_qrels(self, *args: Any, **kwargs: Any) -> Optional[Qrels]:
        return self._qrels.create_qrels(*args, **kwargs)

    def judge(self, *args: Any, **kwargs: Any) -> Leaderboard:
        return self._judge.judge(*args, **kwargs)


if __name__ == "__main__":
    from autojudge_base import auto_judge_to_click_command

    auto_judge_to_click_command(VerifierJudge(), "verifier-judge")()

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
    # --- generation / endpoint quirks -------------------------------------
    "max_gen_tokens": 512,          # response budget. A non-reasoning model emits the
                                    # grade as token 0; a reasoning model (gpt-oss, R1,
                                    # QwQ, harmony channels) needs room to think first.
    "reasoning_effort": "",         # if set ("low"/"medium"/"high") forwarded to the
                                    # endpoint (gpt-oss / o-series style). Empty => omit.
    "guided_choice": False,         # if True, send guided_choice=[grade tokens] so a
                                    # vLLM/outlines endpoint constrains the answer to a
                                    # single digit -> the cleanest possible distribution.
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
        for report in reports:
            run_id = report.metadata.run_id
            topic_id = str(report.metadata.topic_id)
            topic = topics.get(topic_id)
            response_text = self._response_text(report)
            citation_block = self._citation_block(report) if self.include_citations else ""
            for c_idx, criterion in enumerate(self.criteria):
                messages = self._build_messages(topic, topic_id, criterion, response_text, citation_block)
                for rep in range(self.repeats):
                    tasks.append(_Task(run_id, topic_id, c_idx, rep))
                    requests.append(self._build_request(messages, run_id, topic_id, c_idx, rep))

        results = self._run(requests, llm_config)

        # collect per-(run, topic) list of (expected_grade, used_logprobs)
        acc: Dict[Tuple[str, str], List[Tuple[float, bool]]] = {}
        for task, result in zip(tasks, results):
            key = (task.run_id, task.topic_id)
            eg, used = self._interpret(result)
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

    def _citation_block(self, report: Report) -> str:
        docs = report.documents or {}
        if not docs:
            return ""
        parts: List[str] = []
        budget = self.max_citation_chars
        for doc_id, doc in docs.items():
            body = (getattr(doc, "text", "") or "").strip()
            if not body:
                continue
            snippet = body[:800]
            entry = f"[{doc_id}] {snippet}"
            if len(entry) > budget:
                break
            parts.append(entry)
            budget -= len(entry)
        if not parts:
            return ""
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

    def _interpret(self, result: Any) -> Tuple[Optional[float], bool]:
        """Return (expected_grade, used_logprobs). expected_grade is None on failure."""
        if not isinstance(result, MinimaLlmResponse):
            # Log the failure *type* only -- a MinimaLlmFailure repr can carry a
            # body snippet with prompt text, which is restricted evaluation data.
            etype = getattr(result, "error_type", type(result).__name__)
            status = getattr(result, "status", None)
            print(f"[VerifierCore] LLM call failed: {etype}"
                  + (f" (status {status})" if status else ""))
            return None, False

        if self.use_logprobs:
            eg = self._expected_grade_from_logprobs(result.raw)
            if eg is not None:
                return eg, True

        grade = self._parse_int_grade(result.text)
        if grade is not None:
            return float(grade), False
        return None, False

    def _expected_grade_from_logprobs(self, raw: Optional[Dict[str, Any]]) -> Optional[float]:
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

        def mass_at(ti: Dict[str, Any]) -> Dict[int, float]:
            m: Dict[int, float] = {}
            for cand in ti.get("top_logprobs") or []:
                tok = (cand.get("token") or "").strip()
                if tok in grade_set:
                    m[int(tok)] = m.get(int(tok), 0.0) + math.exp(cand["logprob"])
            chosen = (ti.get("token") or "").strip()
            if chosen in grade_set and int(chosen) not in m:
                m[int(chosen)] = math.exp(ti["logprob"])
            return m

        # last position whose *emitted* token is a bare grade digit == the answer
        for ti in reversed(window):
            if (ti.get("token") or "").strip() in grade_set:
                m = mass_at(ti)
                total = sum(m.values())
                if total > 0:
                    return sum(g * (p / total) for g, p in m.items())
        # fallback: last position with any grade mass in its candidates
        for ti in reversed(window):
            m = mass_at(ti)
            total = sum(m.values())
            if total > 0:
                return sum(g * (p / total) for g, p in m.items())
        return None

    def _parse_int_grade(self, text: str) -> Optional[int]:
        if not text:
            return None
        cleaned = _MARKUP_RE.sub(" ", text)
        digits = re.findall(r"\d", cleaned)
        if not digits:
            return None
        return max(0, min(self.max_grade, int(digits[-1])))  # the final digit = the answer


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

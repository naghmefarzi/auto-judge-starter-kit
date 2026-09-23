#!/usr/bin/env python3
"""
JevJudge: an AutoJudge built on Jev (TypeSafe AI's "System One" typed-decision
model), used directly through the official ``typesafe_sdk`` -- no LangChain.
See ``judges/jev_langchain/`` for the same idea through the LangChain
integration, for comparison.

Jev is not an autoregressive LLM. Given a *state* (JSON) and one or more
*typed questions*, it returns calibrated typed answers in a single fast pass:

  * ``Noul``   -- binary decision, answered with a probability
  * ``Choice`` -- categorical selection, answered with a chosen option and a
    full probability distribution
  * ``Score``  -- an ordinal scale described by per-level text, answered with
    an *expected value* and its probability distribution

By default this judge uses one ``Score`` question per ``criteria`` entry, all
in ONE request (no per-criterion round trip, unlike a token-logprob LLM
judge). Set ``questions`` (or ``checkeval_file`` -- see judges/_checkeval.py)
to send an arbitrary mix of all three primitives instead -- see
judges/jev/README.md#per-dataset-question-sets.

Being a calibrated classifier rather than a sampled autoregressive model, Jev
has no repeats/temperature axis to average over.

Wire it up in ``workflow.yml``::

    qrels_class: "judges.jev.jev_judge:JevQrelsCreator"
    judge_class: "judges.jev.jev_judge:JevLeaderboardJudge"

Requires ``TYPESAFE_API_KEY`` (and optionally ``TYPESAFE_BASE_URL`` /
``TYPESAFE_DEFAULT_MODEL``) in the environment -- see judges/jev/README.md.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
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

try:
    from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "judges.jev requires the 'typesafe-sdk' package: pip install typesafe-sdk"
    ) from exc

try:
    from dotenv import load_dotenv

    load_dotenv()  # picks up TYPESAFE_API_KEY from a .env in cwd or a parent dir
except ImportError:  # pragma: no cover - optional convenience
    pass

from .._checkeval import load_checkeval_questions, slugify_dimension
from .._ground_truth import citation_ground_truth_stats, load_qrels_dir, topic_relevant_counts


# =============================================================================
# Defaults
# =============================================================================

DEFAULTS: Dict[str, Any] = {
    "max_grade": 3,                 # grade scale is 0..max_grade
    "criteria": ["overall relevance"],  # one Score question per criterion, all in one call
    "score_levels": [                # index i = grade i; must have max_grade+1 entries
        "fails the criterion entirely / off-topic / empty",
        "marginal: touches the criterion but mostly inadequate",
        "good: largely satisfies the criterion with minor gaps",
        "excellent: fully satisfies the criterion",
    ],
    "max_response_chars": 12000,    # truncate the response text sent to Jev
    "max_problem_chars": 1500,      # truncate the topic's problem_statement / background
    "include_problem_statement": True,
    "include_citations": False,     # include cited-source snippets in the state
    "max_citation_chars": 4000,     # budget for the citations block
    "qrels_doc_id": "report_hash",  # "report_hash" (md5 of report text) or "run_id"
    "on_missing": "fix_aggregate",  # leaderboard/qrels coverage policy
    "concurrency": 32,              # max in-flight Jev requests
    "model": "",                    # explicit TypeSafe model; "" => SDK/env default
    "debug": False,                 # dump per-call state/scores/probabilities. RESTRICTED data.
    "questions": {},                # optional: {id: {type, instructions, criteria, dimension,
                                     # grade}} -- when non-empty, REPLACES the criteria-derived
                                     # Score questions. See judges/jev/README.md and
                                     # judges/_checkeval.py for the full shape.
    "checkeval_file": "",           # optional: path to a CheckEval seed-question JSON (see
                                     # judges/_checkeval.py); takes priority over `questions`.
    "qrels_dir": "",                # optional: directory of TREC-format qrels pool files
                                     # (see judges/_ground_truth.py). When set (and debug: true),
                                     # debug records get objective ground_truth_citation_
                                     # precision/recall -- empty/absent when unset or a topic has
                                     # no qrels coverage (e.g. rag26, which has none).
}


# =============================================================================
# Leaderboard / qrels specs
# =============================================================================

def _leaderboard_spec(core: "JevCore") -> LeaderboardSpec:
    """Base JEV_SCORE/JEV_GRADE, plus one measure per dimension in
    ``core._dim_members`` (mean of that dimension's normalised 0..1 answers --
    a noul's probability as-is, a score divided by its own scale). Choice
    answers are debug-only -- picking a single scalar to aggregate a
    categorical answer into isn't meaningful, so they never become a measure.
    """
    measures = [
        MeasureSpec(
            "JEV_SCORE",
            float,
            description=(
                "Mean over topics of Jev's expected grade, normalised to "
                "0.0-1.0 (expected grade / max_grade). Higher is better."
            ),
        ),
        MeasureSpec(
            "JEV_GRADE",
            float,
            description=(
                "Mean over topics of Jev's expected grade on the raw "
                "0..max_grade scale (continuous, not rounded)."
            ),
        ),
    ]
    for dim in core._dim_members:
        measures.append(
            MeasureSpec(
                f"JEV_DIM_{slugify_dimension(dim)}",
                float,
                description=f"Mean over topics of the normalised (0..1) answer for the '{dim}' dimension.",
            )
        )
    return LeaderboardSpec(measures=tuple(measures))


@dataclass(frozen=True)
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
class JevResult:
    expected_grade: float          # continuous, 0..max_grade
    grade_norm: float              # expected_grade / max_grade, 0..1
    n_calls: int                   # 1 if the Jev request succeeded, else 0
    dimension_values: Dict[str, float] = field(default_factory=dict)  # dimension -> 0..1


class JevCore:
    """Turns reports + topics into a ``{(run_id, topic_id): JevResult}`` map."""

    def __init__(self, **settings: Any) -> None:
        cfg = dict(DEFAULTS)
        cfg.update({k: v for k, v in settings.items() if k in DEFAULTS})
        self.max_grade: int = int(cfg["max_grade"])
        if not 1 <= self.max_grade <= 9:
            raise ValueError("max_grade must be a single digit in 1..9")
        self.criteria: List[str] = list(cfg["criteria"]) or ["overall relevance"]
        self.score_levels: List[str] = list(cfg["score_levels"])
        if len(self.score_levels) != self.max_grade + 1:
            raise ValueError(
                f"score_levels must have exactly max_grade+1={self.max_grade + 1} "
                f"entries, got {len(self.score_levels)}"
            )
        self.max_response_chars: int = int(cfg["max_response_chars"])
        self.max_problem_chars: int = int(cfg["max_problem_chars"])
        self.include_problem_statement: bool = bool(cfg["include_problem_statement"])
        self.include_citations: bool = bool(cfg["include_citations"])
        self.max_citation_chars: int = int(cfg["max_citation_chars"])
        self.concurrency: int = max(1, int(cfg["concurrency"]))
        self.model: Optional[str] = str(cfg["model"] or "").strip() or None
        self.debug: bool = bool(cfg["debug"])
        self.debug_log: List[Dict[str, Any]] = []
        self.checkeval_file: str = str(cfg["checkeval_file"] or "").strip()
        self.questions: Dict[str, Dict[str, Any]] = (
            load_checkeval_questions(self.checkeval_file)
            if self.checkeval_file
            else dict(cfg["questions"] or {})
        )
        self._build_questions()
        self.qrels_dir: str = str(cfg["qrels_dir"] or "").strip()
        self._qrels_lookup: Dict[Tuple[str, str], int] = (
            load_qrels_dir(self.qrels_dir) if self.qrels_dir else {}
        )
        self._qrels_topic_relevant_count: Dict[str, int] = topic_relevant_counts(self._qrels_lookup)

    # ---- state construction (mirrors judges/verifier's prompt pieces, but as
    #      structured JSON -- Jev consumes state directly, no prompt string) --

    def _response_text(self, report: Report) -> str:
        sentences = report.responses or []
        text = " ".join((s.text or "") for s in sentences).strip()
        if len(text) > self.max_response_chars:
            text = text[: self.max_response_chars] + " [...truncated]"
        return text

    def _response_sentences_with_citations(self, report: Report) -> List[Dict[str, Any]]:
        """Pair each response sentence with the documents it actually cites.

        Uses ``report.get_sentences_with_citations()``, which normalizes
        citation format differences across tracks (RAGTIME's confidence-sorted
        citations, RAG24's index-resolved citations, etc.) into a plain
        ``citations: List[str]`` of doc ids per sentence -- giving a
        groundedness-style question something concrete to check each claim
        against, unlike a flat, sentence-disconnected dump of retrieved docs.
        """
        documents = report.documents or {}
        out: List[Dict[str, Any]] = []
        budget = self.max_citation_chars
        total_chars = 0
        for sent in report.get_sentences_with_citations():
            text = sent.text or ""
            if total_chars > self.max_response_chars:
                out.append({"text": "[...truncated]", "cited_documents": []})
                break
            total_chars += len(text)
            cited: List[Dict[str, str]] = []
            for doc_id in sent.citations or []:
                doc = documents.get(doc_id)
                if doc is None:
                    continue
                snippet = doc.get_text()[:800]
                entry_len = len(f"[{doc_id}] {snippet}")
                if entry_len > budget:
                    continue
                cited.append({"doc_id": doc_id, "snippet": snippet})
                budget -= entry_len
            out.append({"text": text, "cited_documents": cited})
        return out

    def _ground_truth_citation_stats(self, topic_id: str, report: Report) -> Dict[str, Any]:
        """Objective citation precision/recall against real qrels (see
        judges/_ground_truth.py), computed independently of Jev's own
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

    def _build_state(
        self,
        topic: Optional[Request],
        topic_id: str,
        report: Report,
        response_text: str,
    ) -> Dict[str, Any]:
        query = (topic.title if topic else "") or topic_id
        state: Dict[str, Any] = {"query": query}
        if topic and self.include_problem_statement:
            extra = (topic.problem_statement or topic.background or "").strip()
            if extra:
                state["information_need"] = extra[: self.max_problem_chars]
        if self.include_citations:
            state["response_sentences"] = self._response_sentences_with_citations(report)
        else:
            state["response"] = response_text or "(empty response)"
        return state

    def _build_questions(self) -> None:
        """Populate ``_sdk_questions``/``_score_ids``/``_choice_ids``/``_dim_members``/
        ``_qid_kind``/``_qid_scale``. See judges/_checkeval.py's module docstring
        for the dimension-grouping and grade-flag design.
        """
        sdk_questions: Dict[str, Any] = {}
        score_ids: List[str] = []
        choice_ids: List[str] = []
        dim_members: Dict[str, List[str]] = {}
        qid_kind: Dict[str, str] = {}
        qid_scale: Dict[str, int] = {}

        def add_to_dimension(qid: str, spec: Dict[str, Any]) -> None:
            dim = spec.get("dimension") or qid
            dim_members.setdefault(dim, []).append(qid)

        if self.questions:
            for qid, spec in self.questions.items():
                qtype = str(spec.get("type", "")).lower()
                instructions = spec["instructions"]
                if qtype == "score":
                    criteria = spec["criteria"]
                    sdk_questions[qid] = Score(instructions=instructions, criteria=criteria)
                    qid_kind[qid] = "score"
                    qid_scale[qid] = max(1, len(criteria) - 1)
                    if spec.get("grade", True):
                        score_ids.append(qid)
                    else:
                        add_to_dimension(qid, spec)
                elif qtype == "noul":
                    sdk_questions[qid] = Noul(instructions=instructions)
                    qid_kind[qid] = "noul"
                    add_to_dimension(qid, spec)
                elif qtype == "choice":
                    sdk_questions[qid] = Choice(instructions=instructions, criteria=spec["criteria"])
                    choice_ids.append(qid)
                else:
                    raise ValueError(f"Unknown question type {qtype!r} for question {qid!r}")
            if not score_ids:
                raise ValueError(
                    "questions must include at least one grade-feeding 'score' entry "
                    "(grade: true, the default) -- it drives the grade"
                )
        else:
            # Fallback: one Score question per `criteria` entry (default behavior)
            score_ids = [f"c{idx}" for idx in range(len(self.criteria))]
            for qid, criterion in zip(score_ids, self.criteria):
                sdk_questions[qid] = Score(
                    instructions=f"Grade the RESPONSE for the criterion: {criterion}.",
                    criteria=self.score_levels,
                )
                qid_kind[qid] = "score"
                qid_scale[qid] = max(1, len(self.score_levels) - 1)

        self._sdk_questions = sdk_questions
        self._score_ids = score_ids
        self._choice_ids = choice_ids
        self._dim_members = dim_members
        self._qid_kind = qid_kind
        self._qid_scale = qid_scale

    # ---- public API -----------------------------------------------------

    def score(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: Optional[LlmConfigProtocol] = None,
    ) -> Dict[Tuple[str, str], JevResult]:
        topics: Dict[str, Request] = {t.request_id: t for t in rag_topics}
        reports: List[Report] = sorted(
            rag_responses,
            key=lambda r: (r.metadata.run_id, str(r.metadata.topic_id)),
        )
        return asyncio.run(self._score_async(reports, topics))

    async def _score_async(
        self, reports: List[Report], topics: Dict[str, Request]
    ) -> Dict[Tuple[str, str], JevResult]:
        client_kwargs: Dict[str, Any] = {}
        if self.model:
            client_kwargs["model"] = self.model
        client = AsyncTypeSafeClient(**client_kwargs)
        semaphore = asyncio.Semaphore(self.concurrency)
        questions = self._sdk_questions

        def normalized(response: Any, qid: str) -> float:
            if self._qid_kind[qid] == "noul":
                return response.nouls[qid].noul
            return response.scores[qid].score / self._qid_scale[qid]

        async def one(report: Report) -> Tuple[str, str, Optional[float], Dict[str, float]]:
            run_id = report.metadata.run_id
            topic_id = str(report.metadata.topic_id)
            topic = topics.get(topic_id)
            response_text = self._response_text(report)
            state = self._build_state(topic, topic_id, report, response_text)

            debug_rec: Optional[Dict[str, Any]] = None
            if self.debug:
                debug_rec = {
                    "run_id": run_id,
                    "topic_id": topic_id,
                    "state": state,
                    **self._ground_truth_citation_stats(topic_id, report),
                }

            async with semaphore:
                try:
                    response = await client.system_one(state=state, questions=questions)
                except Exception as exc:  # noqa: BLE001 - surface type only, see judges/verifier
                    print(f"[JevCore] request failed for {run_id}|{topic_id}: {type(exc).__name__}")
                    if debug_rec is not None:
                        debug_rec["error_type"] = type(exc).__name__
                        self.debug_log.append(debug_rec)
                    return run_id, topic_id, None, {}

            per_criterion = [response.scores[key].score for key in self._score_ids]
            eg = sum(per_criterion) / len(per_criterion)
            dimension_values = {
                dim: sum(normalized(response, qid) for qid in qids) / len(qids)
                for dim, qids in self._dim_members.items()
            }

            if debug_rec is not None:
                debug_rec["scores"] = {
                    qid: {
                        "score": response.scores[qid].score,
                        "probabilities": response.scores[qid].probabilities,
                        "legend": response.scores[qid].legend,
                    }
                    for qid, kind in self._qid_kind.items()
                    if kind == "score"
                }
                debug_rec["nouls"] = {
                    qid: response.nouls[qid].noul
                    for qid, kind in self._qid_kind.items()
                    if kind == "noul"
                }
                if self._choice_ids:
                    debug_rec["choices"] = {
                        qid: {
                            "choice": response.choices[qid].choice,
                            "confidence": response.choices[qid].confidence,
                            "probabilities": response.choices[qid].probabilities,
                        }
                        for qid in self._choice_ids
                    }
                debug_rec["dimension_values"] = dimension_values
                debug_rec["expected_grade"] = eg
                debug_rec["model"] = getattr(response, "model", None)
                debug_rec["request_id"] = getattr(response, "request_id", None)
                self.debug_log.append(debug_rec)

            return run_id, topic_id, eg, dimension_values

        try:
            results = await asyncio.gather(*(one(r) for r in reports))
        finally:
            await client.aclose()

        out: Dict[Tuple[str, str], JevResult] = {}
        for run_id, topic_id, eg, dimension_values in results:
            key = (run_id, topic_id)
            if eg is not None:
                out[key] = JevResult(
                    expected_grade=eg, grade_norm=eg / self.max_grade, n_calls=1,
                    dimension_values=dimension_values,
                )
            elif key not in out:
                out[key] = JevResult(expected_grade=0.0, grade_norm=0.0, n_calls=0)
        return out


def _write_debug_log(core: "JevCore", outdir: Path, filebase: str) -> None:
    """Dump core.debug_log (populated only when the ``debug`` setting is on).

    RESTRICTED: the records carry query text, cited-source snippets, response
    text and Jev's answers -- evaluation data under the AutoJudge
    data-handling policy for any topic outside the reviewer's permitted
    window. View it yourself; do not surface its contents to a coding agent.
    """
    if not core.debug or not core.debug_log:
        return
    fb = Path(filebase)
    # AutoJudge bakes `outdir` into `filebase` for some lifecycle phases but not
    # others -- avoid doubling it up either way (see judges/verifier).
    base = fb if str(fb).startswith(str(outdir)) else outdir / fb
    path = base.parent / f"{base.name}.jev-debug.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in core.debug_log:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(
        f"[JevCore] wrote {len(core.debug_log)} debug records to {path} -- "
        "RESTRICTED evaluation data, see judges/jev/README.md#debugging"
    )


def _status_summary(core: "JevCore") -> str:
    if core.checkeval_file:
        return f"checkeval_file={core.checkeval_file} ({len(core._sdk_questions)} questions, {len(core._dim_members)} dimensions)"
    return f"questions={list(core.questions) or core.criteria}"


# =============================================================================
# QrelsCreatorProtocol
# =============================================================================

class JevQrelsCreator:
    """Relevance judgments (qrels): one row per report, grade = round(E[grade])."""

    def create_qrels(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: Optional[LlmConfigProtocol] = None,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        corpus: Optional[str] = None,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Optional[Qrels]:
        core = JevCore(**kwargs)
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
        n_ok = sum(1 for r in scores.values() if r.n_calls > 0)
        print(
            f"[JevQrelsCreator] {len(records)} judgments over {len(rag_topics)} topics; "
            f"{n_ok}/{len(scores)} succeeded; {_status_summary(core)}"
        )
        _write_debug_log(core, outdir, filebase)
        return qrels


# =============================================================================
# LeaderboardJudgeProtocol
# =============================================================================

class JevLeaderboardJudge:
    """Leaderboard: mean expected grade per run, as JEV_SCORE and JEV_GRADE,
    plus one JEV_DIM_{DIMENSION} measure per dimension (see judges/_checkeval.py)."""

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: Optional[LlmConfigProtocol] = None,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        corpus: Optional[str] = None,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        core = JevCore(**kwargs)
        reports: List[Report] = list(rag_responses)
        scores = core.score(reports, rag_topics, llm_config)

        expected_topic_ids: List[str] = [t.request_id for t in rag_topics]
        on_missing = str(kwargs.get("on_missing", DEFAULTS["on_missing"]))

        builder = LeaderboardBuilder(_leaderboard_spec(core))
        for report in reports:
            run_id = report.metadata.run_id
            topic_id = str(report.metadata.topic_id)
            result = scores.get((run_id, topic_id))
            eg = result.expected_grade if result else 0.0
            values: Dict[str, float] = {"JEV_SCORE": eg / core.max_grade, "JEV_GRADE": eg}
            dimension_values = result.dimension_values if result else {}
            for dim in core._dim_members:
                values[f"JEV_DIM_{slugify_dimension(dim)}"] = dimension_values.get(dim, 0.0)
            builder.add(run_id=run_id, topic_id=topic_id, values=values)

        leaderboard = builder.build(expected_topic_ids=expected_topic_ids, on_missing=on_missing)
        leaderboard.verify(on_missing=on_missing, expected_topic_ids=expected_topic_ids, warn=True)

        n_ok = sum(1 for r in scores.values() if r.n_calls > 0)
        n_zero = sum(1 for r in scores.values() if r.n_calls == 0)
        print(
            f"[JevLeaderboardJudge] {len(reports)} reports; {n_ok}/{len(scores)} succeeded; "
            f"{n_zero} scored 0 from failed calls; {_status_summary(core)}"
        )
        _write_debug_log(core, outdir, filebase)
        return leaderboard


# =============================================================================
# Combined AutoJudge (for `python -m` / direct CLI use)
# =============================================================================

class JevJudge(AutoJudge):
    """All protocols in one object. Prefer running via ``auto-judge run --workflow``."""

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def __init__(self) -> None:
        self._qrels = JevQrelsCreator()
        self._judge = JevLeaderboardJudge()

    def create_nuggets(self, *args: Any, **kwargs: Any) -> Optional[NuggetBanksProtocol]:
        return None

    def create_qrels(self, *args: Any, **kwargs: Any) -> Optional[Qrels]:
        return self._qrels.create_qrels(*args, **kwargs)

    def judge(self, *args: Any, **kwargs: Any) -> Leaderboard:
        return self._judge.judge(*args, **kwargs)


if __name__ == "__main__":
    from autojudge_base import auto_judge_to_click_command

    auto_judge_to_click_command(JevJudge(), "jev-judge")()

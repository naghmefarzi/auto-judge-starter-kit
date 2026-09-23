#!/usr/bin/env python3
"""
JevLangchainJudge: the same Jev ("System One" typed-decision model) as
``judges/jev``, but called through the ``langchain-typesafe`` integration
(``TypeSafeClassifier``, a LangChain ``Runnable``) instead of the raw
``typesafe_sdk`` client. Kept as a separate judge so the two invocation paths
can be compared directly on the leaderboard.

What's different from ``judges/jev``, deliberately:

  * Uses ``langchain_typesafe.TypeSafeClassifier`` -- a LangChain
    ``RunnableSerializable`` -- instead of ``typesafe_sdk`` clients directly.
    Errors surface as ``langchain_core.exceptions.ModelError`` subclasses in
    addition to ``typesafe_sdk`` exceptions.
  * By default (no ``questions``/``checkeval_file`` set), every request also
    asks one ``Noul`` "does this response pass" question alongside the
    per-criterion ``Score`` questions -- Jev answers every question against
    the same state in one call regardless of type mix, so this costs nothing
    extra. Reported as ``JEVLC_DOES_PASS`` rather than folded into the grade.
    This mirrors the reference LangChain pattern of combining a continuous
    quality score with a binary pass/fail read in one evaluator.
  * Traced with LangSmith's ``@traceable`` when ``langsmith`` is installed and
    configured (``LANGSMITH_API_KEY`` / ``LANGCHAIN_TRACING_V2=true``);
    otherwise the decorator is a no-op.

Setting ``questions`` or ``checkeval_file`` (see judges/_checkeval.py)
replaces the does_pass default entirely with an arbitrary mix of
score/noul/choice questions, grouped into dimension-level leaderboard
measures -- see judges/jev/README.md#per-dataset-question-sets.

See ``judges/jev/README.md`` for the shared Jev background and
``judges/jev_langchain/README.md`` for what's specific to this judge.

Wire it up in ``workflow.yml``::

    qrels_class: "judges.jev_langchain.jev_langchain_judge:JevLangchainQrelsCreator"
    judge_class: "judges.jev_langchain.jev_langchain_judge:JevLangchainLeaderboardJudge"

Requires ``TYPESAFE_API_KEY`` (and optionally ``TYPESAFE_BASE_URL`` /
``TYPESAFE_DEFAULT_MODEL``) in the environment.
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
    from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "judges.jev_langchain requires 'langchain-typesafe': "
        "pip install langchain-typesafe"
    ) from exc

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - optional, tracing is best-effort
    def traceable(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
        def _decorator(fn):
            return fn

        return _decorator

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
    "max_grade": 3,
    "criteria": ["overall relevance"],
    "score_levels": [
        "fails the criterion entirely / off-topic / empty",
        "marginal: touches the criterion but mostly inadequate",
        "good: largely satisfies the criterion with minor gaps",
        "excellent: fully satisfies the criterion",
    ],
    "does_pass_instructions": (
        "Does the RESPONSE adequately address the QUERY, overall?"
    ),
    "max_response_chars": 12000,
    "max_problem_chars": 1500,
    "include_problem_statement": True,
    "include_citations": False,
    "max_citation_chars": 4000,
    "qrels_doc_id": "report_hash",
    "on_missing": "fix_aggregate",
    "concurrency": 32,
    "model": "",
    "debug": False,
    "questions": {},                # optional: {id: {type, instructions, criteria, dimension,
                                     # grade}} -- when non-empty, REPLACES the criteria+does_pass
                                     # default. See judges/jev/README.md and judges/_checkeval.py.
    "checkeval_file": "",           # optional: path to a CheckEval seed-question JSON; takes
                                     # priority over `questions`.
    "qrels_dir": "",                # optional: directory of TREC-format qrels pool files
                                     # (see judges/_ground_truth.py). When set (and debug: true),
                                     # debug records get objective ground_truth_citation_
                                     # precision/recall -- empty/absent when unset or a topic has
                                     # no qrels coverage (e.g. rag26, which has none).
}


# =============================================================================
# Leaderboard / qrels specs
# =============================================================================

def _leaderboard_spec(core: "JevLangchainCore") -> LeaderboardSpec:
    """Base JEVLC_SCORE/JEVLC_GRADE, plus one measure per dimension --
    ``JEVLC_DOES_PASS`` in the default configuration (dimension "does_pass"),
    ``JEVLC_DIM_{SLUG}`` for anything else. See judges/jev/jev_judge.py's
    ``_leaderboard_spec`` for the normalisation/grouping rules."""
    measures = [
        MeasureSpec(
            "JEVLC_SCORE",
            float,
            description=(
                "Mean over topics of Jev's expected grade (via langchain-typesafe), "
                "normalised to 0.0-1.0. Higher is better."
            ),
        ),
        MeasureSpec(
            "JEVLC_GRADE",
            float,
            description=(
                "Mean over topics of Jev's expected grade on the raw 0..max_grade scale."
            ),
        ),
    ]
    for dim in core._dim_members:
        name = "JEVLC_DOES_PASS" if dim == "does_pass" else f"JEVLC_DIM_{slugify_dimension(dim)}"
        measures.append(
            MeasureSpec(
                name,
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
# Core
# =============================================================================

@dataclass(frozen=True)
class JevLangchainResult:
    expected_grade: float
    grade_norm: float
    n_calls: int
    dimension_values: Dict[str, float] = field(default_factory=dict)  # dimension -> 0..1


class JevLangchainCore:
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
        self.does_pass_instructions: str = str(cfg["does_pass_instructions"])
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
        # The shipped 0.0.1a3 TypeSafeClassifier rejects an explicit `model=None`
        # (verified against the real package); omit the kwarg entirely instead.
        classifier_kwargs: Dict[str, Any] = {"model": self.model} if self.model else {}
        self._classifier = TypeSafeClassifier(**classifier_kwargs)

    # ---- state construction (same shape as judges/jev) -------------------

    def _response_text(self, report: Report) -> str:
        sentences = report.responses or []
        text = " ".join((s.text or "") for s in sentences).strip()
        if len(text) > self.max_response_chars:
            text = text[: self.max_response_chars] + " [...truncated]"
        return text

    def _response_sentences_with_citations(self, report: Report) -> List[Dict[str, Any]]:
        """Pair each response sentence with the documents it actually cites.

        See judges/jev/jev_judge.py's method of the same name for the rationale
        (this state shape is what a groundedness-style question needs).
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
        ``_qid_kind``/``_qid_scale``. See judges/jev/jev_judge.py's method of the
        same name and judges/_checkeval.py's module docstring for the design.
        The one difference: the fallback (no questions/checkeval_file) here
        also includes the ``does_pass`` Noul (this judge's original behavior).
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
            # Fallback: one Score question per `criteria` entry, plus does_pass (default behavior)
            score_ids = [f"c{idx}" for idx in range(len(self.criteria))]
            for qid, criterion in zip(score_ids, self.criteria):
                sdk_questions[qid] = Score(
                    instructions=f"Grade the RESPONSE for the criterion: {criterion}.",
                    criteria=self.score_levels,
                )
                qid_kind[qid] = "score"
                qid_scale[qid] = max(1, len(self.score_levels) - 1)
            sdk_questions["does_pass"] = Noul(instructions=self.does_pass_instructions)
            qid_kind["does_pass"] = "noul"
            dim_members["does_pass"] = ["does_pass"]

        self._sdk_questions = sdk_questions
        self._score_ids = score_ids
        self._choice_ids = choice_ids
        self._dim_members = dim_members
        self._qid_kind = qid_kind
        self._qid_scale = qid_scale

    # ---- public API -------------------------------------------------------

    def score(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: Optional[LlmConfigProtocol] = None,
    ) -> Dict[Tuple[str, str], JevLangchainResult]:
        topics: Dict[str, Request] = {t.request_id: t for t in rag_topics}
        reports: List[Report] = sorted(
            rag_responses,
            key=lambda r: (r.metadata.run_id, str(r.metadata.topic_id)),
        )
        return asyncio.run(self._score_async(reports, topics))

    async def _score_async(
        self, reports: List[Report], topics: Dict[str, Request]
    ) -> Dict[Tuple[str, str], JevLangchainResult]:
        semaphore = asyncio.Semaphore(self.concurrency)
        questions = self._sdk_questions

        def normalized(response: Any, qid: str) -> float:
            if self._qid_kind[qid] == "noul":
                return response.nouls[qid].noul
            return response.scores[qid].score / self._qid_scale[qid]

        @traceable(name="jev_langchain_judge_call")
        async def _invoke(state: Dict[str, Any]) -> Any:
            return await self._classifier.ainvoke({"state": state, "questions": questions})

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
                    response = await _invoke(state)
                except Exception as exc:  # noqa: BLE001 - surface type only
                    print(f"[JevLangchainCore] request failed for {run_id}|{topic_id}: {type(exc).__name__}")
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
            # The shipped 0.0.1a3 TypeSafeClassifier has no close()/aclose() of its
            # own (verified against the real package -- it differs here from the
            # in-development source); close its underlying async_client instead.
            await self._classifier.async_client.aclose()

        out: Dict[Tuple[str, str], JevLangchainResult] = {}
        for run_id, topic_id, eg, dimension_values in results:
            key = (run_id, topic_id)
            if eg is not None:
                out[key] = JevLangchainResult(
                    expected_grade=eg, grade_norm=eg / self.max_grade, n_calls=1,
                    dimension_values=dimension_values,
                )
            elif key not in out:
                out[key] = JevLangchainResult(expected_grade=0.0, grade_norm=0.0, n_calls=0)
        return out


def _write_debug_log(core: "JevLangchainCore", outdir: Path, filebase: str) -> None:
    """RESTRICTED: see judges/jev/jev_judge.py's ``_write_debug_log`` docstring."""
    if not core.debug or not core.debug_log:
        return
    fb = Path(filebase)
    base = fb if str(fb).startswith(str(outdir)) else outdir / fb
    path = base.parent / f"{base.name}.jev-langchain-debug.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in core.debug_log:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(
        f"[JevLangchainCore] wrote {len(core.debug_log)} debug records to {path} -- "
        "RESTRICTED evaluation data, see judges/jev_langchain/README.md#debugging"
    )


def _status_summary(core: "JevLangchainCore") -> str:
    if core.checkeval_file:
        return f"checkeval_file={core.checkeval_file} ({len(core._sdk_questions)} questions, {len(core._dim_members)} dimensions)"
    return f"questions={list(core.questions) or core.criteria}"


# =============================================================================
# QrelsCreatorProtocol
# =============================================================================

class JevLangchainQrelsCreator:
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
        core = JevLangchainCore(**kwargs)
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
            f"[JevLangchainQrelsCreator] {len(records)} judgments over {len(rag_topics)} topics; "
            f"{n_ok}/{len(scores)} succeeded; {_status_summary(core)}"
        )
        _write_debug_log(core, outdir, filebase)
        return qrels


# =============================================================================
# LeaderboardJudgeProtocol
# =============================================================================

class JevLangchainLeaderboardJudge:
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
        core = JevLangchainCore(**kwargs)
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
            values: Dict[str, float] = {"JEVLC_SCORE": eg / core.max_grade, "JEVLC_GRADE": eg}
            dimension_values = result.dimension_values if result else {}
            for dim in core._dim_members:
                name = "JEVLC_DOES_PASS" if dim == "does_pass" else f"JEVLC_DIM_{slugify_dimension(dim)}"
                values[name] = dimension_values.get(dim, 0.0)
            builder.add(run_id=run_id, topic_id=topic_id, values=values)

        leaderboard = builder.build(expected_topic_ids=expected_topic_ids, on_missing=on_missing)
        leaderboard.verify(on_missing=on_missing, expected_topic_ids=expected_topic_ids, warn=True)

        n_ok = sum(1 for r in scores.values() if r.n_calls > 0)
        n_zero = sum(1 for r in scores.values() if r.n_calls == 0)
        print(
            f"[JevLangchainLeaderboardJudge] {len(reports)} reports; {n_ok}/{len(scores)} succeeded; "
            f"{n_zero} scored 0 from failed calls; {_status_summary(core)}"
        )
        _write_debug_log(core, outdir, filebase)
        return leaderboard


# =============================================================================
# Combined AutoJudge
# =============================================================================

class JevLangchainJudge(AutoJudge):
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def __init__(self) -> None:
        self._qrels = JevLangchainQrelsCreator()
        self._judge = JevLangchainLeaderboardJudge()

    def create_nuggets(self, *args: Any, **kwargs: Any) -> Optional[NuggetBanksProtocol]:
        return None

    def create_qrels(self, *args: Any, **kwargs: Any) -> Optional[Qrels]:
        return self._qrels.create_qrels(*args, **kwargs)

    def judge(self, *args: Any, **kwargs: Any) -> Leaderboard:
        return self._judge.judge(*args, **kwargs)


if __name__ == "__main__":
    from autojudge_base import auto_judge_to_click_command

    auto_judge_to_click_command(JevLangchainJudge(), "jev-langchain-judge")()

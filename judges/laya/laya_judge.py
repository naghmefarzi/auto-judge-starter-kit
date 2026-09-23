#!/usr/bin/env python3
"""
LayaJudge: an AutoJudge built on Laya (convaiinnovations/laya), a small,
local, non-autoregressive "System 1" typed-decision model -- the same family
of model as Jev (``judges/jev``, ``judges/jev_langchain``), but open-weight
and run entirely on your own machine instead of a hosted API.

Given a state (JSON) and typed questions, Laya answers each in a single fast
forward pass (~33ms on a T4, faster batched) with a calibrated probability --
no text generation, no API key, no network round trip once the model is
loaded. By default this judge uses one ``score`` question per ``criteria``
entry, all in ONE ``predict()`` call. Set ``questions`` (or ``checkeval_file``
-- see judges/_checkeval.py) to send an arbitrary mix of Laya's three
question types (``score``/``noul``/``choice``) instead -- see
judges/jev/README.md#per-dataset-question-sets.

Like Jev, Laya is a calibrated classifier rather than a sampled autoregressive
model, so there is no repeats/temperature axis to average over.

Wire it up in ``workflow.yml``::

    qrels_class: "judges.laya.laya_judge:LayaQrelsCreator"
    judge_class: "judges.laya.laya_judge:LayaLeaderboardJudge"

Requires the ``laya`` package (``pip install laya`` -- pulls torch,
transformers and, on a CUDA machine, GPU deps; the model weights are
downloaded from Hugging Face on first use). No API key needed.
See judges/laya/README.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
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
    import laya as _laya
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "judges.laya requires the 'laya' package: pip install laya "
        "(pulls torch/transformers; the model itself downloads from "
        "Hugging Face on first use)"
    ) from exc

from .._checkeval import load_checkeval_questions, slugify_dimension
from .._ground_truth import citation_ground_truth_stats, load_qrels_dir, topic_relevant_counts


# =============================================================================
# Defaults
# =============================================================================

DEFAULTS: Dict[str, Any] = {
    "max_grade": 3,
    "criteria": ["overall relevance"],  # one score question per criterion, all in one call
    "score_levels": [
        "fails the criterion entirely / off-topic / empty",
        "marginal: touches the criterion but mostly inadequate",
        "good: largely satisfies the criterion with minor gaps",
        "excellent: fully satisfies the criterion",
    ],
    "max_response_chars": 12000,
    "max_problem_chars": 1500,
    "include_problem_statement": True,
    "include_citations": False,
    "max_citation_chars": 4000,
    "qrels_doc_id": "report_hash",
    "on_missing": "fix_aggregate",
    "checkpoint": "",   # "" (English default) | "multilingual" | "typed-decisions"
    "device": "",       # "" (auto) | "cpu" | "cuda"
    "debug": False,
    "questions": {},    # optional: {id: {type, instructions, criteria, dimension, grade}} --
                        # overrides criteria/score_levels entirely when non-empty. See
                        # judges/jev/README.md and judges/_checkeval.py.
    "checkeval_file": "",  # optional: path to a CheckEval seed-question JSON; takes priority
                           # over `questions`.
    "qrels_dir": "",       # optional: directory of TREC-format qrels pool files (see
                           # judges/_ground_truth.py). When set (and debug: true), debug
                           # records get objective ground_truth_citation_precision/recall --
                           # empty/absent when unset or a topic has no qrels coverage.
}

# Router is a heavy local model load -- one per (device) for the whole process,
# shared across the qrels and judge lifecycle phases (each instantiates its own
# LayaCore) instead of reloading it twice.
_ROUTER_CACHE: Dict[str, Any] = {}
_ROUTER_LOCK = Lock()


def _get_router(device: Optional[str]) -> Any:
    key = device or "auto"
    with _ROUTER_LOCK:
        if key not in _ROUTER_CACHE:
            kwargs: Dict[str, Any] = {"preload": True}
            if device:
                kwargs["device"] = device
            _ROUTER_CACHE[key] = _laya.Router(**kwargs)
        return _ROUTER_CACHE[key]


# =============================================================================
# Leaderboard / qrels specs
# =============================================================================

def _leaderboard_spec(core: "LayaCore") -> LeaderboardSpec:
    """Base LAYA_SCORE/LAYA_GRADE, plus one measure per dimension in
    ``core._dim_members`` (mean of that dimension's normalised 0..1 answers).
    See judges/jev/jev_judge.py's ``_leaderboard_spec`` for the design."""
    measures = [
        MeasureSpec(
            "LAYA_SCORE",
            float,
            description=(
                "Mean over topics of Laya's expected grade, normalised to "
                "0.0-1.0 (expected grade / max_grade). Higher is better."
            ),
        ),
        MeasureSpec(
            "LAYA_GRADE",
            float,
            description=(
                "Mean over topics of Laya's expected grade on the raw "
                "0..max_grade scale (continuous, not rounded)."
            ),
        ),
    ]
    for dim in core._dim_members:
        measures.append(
            MeasureSpec(
                f"LAYA_DIM_{slugify_dimension(dim)}",
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
class LayaResult:
    expected_grade: float
    grade_norm: float
    n_calls: int
    dimension_values: Dict[str, float] = field(default_factory=dict)  # dimension -> 0..1


class LayaCore:
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
        self.checkpoint: str = str(cfg["checkpoint"] or "").strip()
        self.device: Optional[str] = str(cfg["device"] or "").strip() or None
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
        self._router = _get_router(self.device)

    # ---- state construction (same shape as judges/jev) --------------------

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
        judges/_ground_truth.py), computed independently of Laya's own
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
        """Populate ``_laya_questions``/``_score_ids``/``_choice_ids``/``_dim_members``/
        ``_qid_kind``/``_qid_scale``. See judges/jev/jev_judge.py's method of the
        same name and judges/_checkeval.py's module docstring for the design.
        Laya's own question dict format needs no SDK classes, just plain dicts.
        """
        laya_questions: Dict[str, Dict[str, Any]] = {}
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
                if qtype not in ("score", "noul", "choice"):
                    raise ValueError(f"Unknown question type {qtype!r} for question {qid!r}")
                entry: Dict[str, Any] = {"type": qtype, "instructions": spec["instructions"]}
                if qtype in ("score", "choice"):
                    entry["criteria"] = spec["criteria"]
                laya_questions[qid] = entry
                if qtype == "score":
                    qid_kind[qid] = "score"
                    qid_scale[qid] = max(1, len(spec["criteria"]) - 1)
                    if spec.get("grade", True):
                        score_ids.append(qid)
                    else:
                        add_to_dimension(qid, spec)
                elif qtype == "noul":
                    qid_kind[qid] = "noul"
                    add_to_dimension(qid, spec)
                else:
                    choice_ids.append(qid)
            if not score_ids:
                raise ValueError(
                    "questions must include at least one grade-feeding 'score' entry "
                    "(grade: true, the default) -- it drives the grade"
                )
        else:
            # Fallback: one score question per `criteria` entry (default behavior)
            score_ids = [f"c{idx}" for idx in range(len(self.criteria))]
            for qid, criterion in zip(score_ids, self.criteria):
                laya_questions[qid] = {
                    "type": "score",
                    "instructions": f"Grade the RESPONSE for the criterion: {criterion}.",
                    "criteria": self.score_levels,
                }
                qid_kind[qid] = "score"
                qid_scale[qid] = max(1, len(self.score_levels) - 1)

        self._laya_questions = laya_questions
        self._score_ids = score_ids
        self._choice_ids = choice_ids
        self._dim_members = dim_members
        self._qid_kind = qid_kind
        self._qid_scale = qid_scale

    # ---- public API ---------------------------------------------------

    def score(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: Optional[LlmConfigProtocol] = None,
    ) -> Dict[Tuple[str, str], LayaResult]:
        topics: Dict[str, Request] = {t.request_id: t for t in rag_topics}
        reports: List[Report] = sorted(
            rag_responses,
            key=lambda r: (r.metadata.run_id, str(r.metadata.topic_id)),
        )
        questions = self._laya_questions
        predict_kwargs: Dict[str, Any] = {"model": self.checkpoint} if self.checkpoint else {}

        def normalized(answers: Dict[str, Any], qid: str) -> float:
            if self._qid_kind[qid] == "noul":
                return answers[qid]["noul"]
            return answers[qid]["score"] / self._qid_scale[qid]

        out: Dict[Tuple[str, str], LayaResult] = {}
        for report in reports:
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

            key = (run_id, topic_id)
            dimension_values: Dict[str, float] = {}
            try:
                result = self._router.predict(state, questions, **predict_kwargs)
                answers = result["answers"]
                per_criterion = [answers[k]["score"] for k in self._score_ids]
                eg = sum(per_criterion) / len(per_criterion)
                dimension_values = {
                    dim: sum(normalized(answers, qid) for qid in qids) / len(qids)
                    for dim, qids in self._dim_members.items()
                }
                if debug_rec is not None:
                    debug_rec["answers"] = answers
                    debug_rec["routing"] = result.get("routing")
                    debug_rec["dimension_values"] = dimension_values
                    debug_rec["expected_grade"] = eg
            except Exception as exc:  # noqa: BLE001 - surface type only
                print(f"[LayaCore] request failed for {run_id}|{topic_id}: {type(exc).__name__}")
                eg = None
                if debug_rec is not None:
                    debug_rec["error_type"] = type(exc).__name__

            if debug_rec is not None:
                self.debug_log.append(debug_rec)

            if eg is not None:
                out[key] = LayaResult(
                    expected_grade=eg, grade_norm=eg / self.max_grade, n_calls=1,
                    dimension_values=dimension_values,
                )
            elif key not in out:
                out[key] = LayaResult(expected_grade=0.0, grade_norm=0.0, n_calls=0)
        return out


def _write_debug_log(core: "LayaCore", outdir: Path, filebase: str) -> None:
    """RESTRICTED: see judges/jev/jev_judge.py's ``_write_debug_log`` docstring."""
    if not core.debug or not core.debug_log:
        return
    fb = Path(filebase)
    base = fb if str(fb).startswith(str(outdir)) else outdir / fb
    path = base.parent / f"{base.name}.laya-debug.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in core.debug_log:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(
        f"[LayaCore] wrote {len(core.debug_log)} debug records to {path} -- "
        "RESTRICTED evaluation data, see judges/laya/README.md#debugging"
    )


def _status_summary(core: "LayaCore") -> str:
    if core.checkeval_file:
        return f"checkeval_file={core.checkeval_file} ({len(core._laya_questions)} questions, {len(core._dim_members)} dimensions)"
    return f"questions={list(core.questions) or core.criteria}"


# =============================================================================
# QrelsCreatorProtocol
# =============================================================================

class LayaQrelsCreator:
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
        core = LayaCore(**kwargs)
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
            f"[LayaQrelsCreator] {len(records)} judgments over {len(rag_topics)} topics; "
            f"{n_ok}/{len(scores)} succeeded; {_status_summary(core)}"
        )
        _write_debug_log(core, outdir, filebase)
        return qrels


# =============================================================================
# LeaderboardJudgeProtocol
# =============================================================================

class LayaLeaderboardJudge:
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
        core = LayaCore(**kwargs)
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
            values: Dict[str, float] = {"LAYA_SCORE": eg / core.max_grade, "LAYA_GRADE": eg}
            dimension_values = result.dimension_values if result else {}
            for dim in core._dim_members:
                values[f"LAYA_DIM_{slugify_dimension(dim)}"] = dimension_values.get(dim, 0.0)
            builder.add(run_id=run_id, topic_id=topic_id, values=values)

        leaderboard = builder.build(expected_topic_ids=expected_topic_ids, on_missing=on_missing)
        leaderboard.verify(on_missing=on_missing, expected_topic_ids=expected_topic_ids, warn=True)

        n_ok = sum(1 for r in scores.values() if r.n_calls > 0)
        n_zero = sum(1 for r in scores.values() if r.n_calls == 0)
        print(
            f"[LayaLeaderboardJudge] {len(reports)} reports; {n_ok}/{len(scores)} succeeded; "
            f"{n_zero} scored 0 from failed calls; {_status_summary(core)}"
        )
        _write_debug_log(core, outdir, filebase)
        return leaderboard


# =============================================================================
# Combined AutoJudge
# =============================================================================

class LayaJudge(AutoJudge):
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def __init__(self) -> None:
        self._qrels = LayaQrelsCreator()
        self._judge = LayaLeaderboardJudge()

    def create_nuggets(self, *args: Any, **kwargs: Any) -> Optional[NuggetBanksProtocol]:
        return None

    def create_qrels(self, *args: Any, **kwargs: Any) -> Optional[Qrels]:
        return self._qrels.create_qrels(*args, **kwargs)

    def judge(self, *args: Any, **kwargs: Any) -> Leaderboard:
        return self._judge.judge(*args, **kwargs)


if __name__ == "__main__":
    from autojudge_base import auto_judge_to_click_command

    auto_judge_to_click_command(LayaJudge(), "laya-judge")()

"""Shared ground-truth loader: merge TREC-format qrels pool files into one
lookup and compute objective citation precision/recall against them,
independent of whatever a judge itself decided. Used by ``judges/verifier``,
``judges/jev``, ``judges/jev_langchain`` and ``judges/laya`` alike.

Most datasets these judges run against (the 2026 test tracks, rag26/ragtime26)
have no real ground truth yet -- their ``eval/`` files are explicitly random
placeholders, "not relevance labels" per the dataset's own README. Some
pilot/training datasets (``rag25``) ship real, human-assessed document-level
relevance qrels -- not a report-level quality grade for the report a judge
actually produced, but enough to compute something genuinely objective: what
fraction of the documents a report actually cited are real judged-relevant
documents, and what fraction of a topic's known-relevant documents it managed
to cite. That's independent of the judge's own opinion -- useful for checking
whether an LLM/Jev/Laya-judged groundedness or citation-precision dimension
actually tracks something real, rather than trusting it on faith.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def load_qrels_dir(qrels_dir: str) -> Dict[Tuple[str, str], int]:
    """Merge every TREC-format qrels pool file in a directory into one
    ``{(topic_id, doc_id): max_grade_seen}`` lookup. Expects the classic
    ``topic_id iteration doc_id relevance ...`` layout (whitespace-separated;
    any trailing columns beyond the 4th are ignored). Returns an empty dict
    if the directory doesn't exist -- callers treat that the same as "no
    ground truth available" (e.g. rag26, which has none).
    """
    lookup: Dict[Tuple[str, str], int] = {}
    base = Path(qrels_dir)
    if not base.is_dir():
        return lookup
    for path in sorted(base.iterdir()):
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue
                topic_id, doc_id, grade_str = parts[0], parts[2], parts[3]
                try:
                    grade = int(float(grade_str))
                except ValueError:
                    continue
                key = (topic_id, doc_id)
                if key not in lookup or grade > lookup[key]:
                    lookup[key] = grade
    return lookup


def topic_relevant_counts(qrels_lookup: Dict[Tuple[str, str], int]) -> Dict[str, int]:
    """Precompute ``{topic_id: count of docs with grade >= 1}`` once, so
    per-report recall calculations don't rescan the whole qrels lookup
    every time."""
    counts: Dict[str, int] = {}
    for (topic_id, _doc_id), grade in qrels_lookup.items():
        if grade >= 1:
            counts[topic_id] = counts.get(topic_id, 0) + 1
    return counts


def citation_ground_truth_stats(
    topic_id: str,
    cited_doc_ids: Iterable[str],
    qrels_lookup: Dict[Tuple[str, str], int],
    topic_relevant_count: Dict[str, int],
) -> Dict[str, Any]:
    """Objective citation precision/recall against real qrels. Returns an
    empty dict when ``qrels_lookup`` is empty, so debug records simply omit
    these fields rather than showing misleading placeholders.
    """
    if not qrels_lookup:
        return {}
    cited = set(cited_doc_ids)
    judged_cited = [
        (d, qrels_lookup[(topic_id, d)]) for d in cited if (topic_id, d) in qrels_lookup
    ]
    relevant_cited = sum(1 for _, g in judged_cited if g >= 1)
    total_relevant = topic_relevant_count.get(topic_id, 0)
    return {
        "ground_truth_cited_total": len(cited),
        "ground_truth_cited_judged": len(judged_cited),
        "ground_truth_cited_relevant": relevant_cited,
        "ground_truth_citation_precision": (
            relevant_cited / len(judged_cited) if judged_cited else None
        ),
        "ground_truth_topic_relevant_total": total_relevant,
        "ground_truth_citation_recall": (
            relevant_cited / total_relevant if total_relevant else None
        ),
    }

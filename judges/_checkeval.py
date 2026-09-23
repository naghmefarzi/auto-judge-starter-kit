"""Shared loader: turn a CheckEval-style seed-question JSON (the
``jev style/checkeval_trec_*.json`` files -- per-track evaluation dimensions
grounded in each host track's own official evaluation) into a judges/jev-style
``questions`` mapping, used by ``judges/jev``, ``judges/jev_langchain`` and
``judges/laya`` alike.

There is no fourth native Jev/Laya question primitive to reach for (verified
against the installed ``typesafe_sdk``: only ``Score``/``Noul``/``Choice``
exist). "Another primitive" here means picking the right MIX of those three
per dimension, plus a dimension-grouping convention for aggregation -- not a
new wire type:

  - Dimensions whose questions are crisp, independently-checkable facts (the
    great majority -- citation precision, groundedness, faithfulness, format
    compliance, ...) keep that shape: one ``noul`` question per item. Each is
    tagged with a ``dimension`` label so the leaderboard reports ONE aggregate
    measure per dimension (mean probability -- the continuous analogue of the
    source file's own "proportion of yes" scoring) instead of dozens of
    near-duplicate single-question measures.
  - Dimensions that read as an inherently graded, holistic quality rather than
    a checklist of independent facts (coherence/structure, fluency) are
    collapsed into ONE ``score`` question instead of several loosely-related
    booleans -- an ordinal read fits better than yes/no here.
  - Two additions beyond the raw checklist, since it alone gives per-dimension
    proportions but no single headline number or categorical summary:
    ``overall_quality`` (score, the sole grade-driving question -- marked
    ``grade: True``) and ``outcome`` (choice, a holistic categorical read;
    debug-only, like every Choice answer -- see judges/jev/jev_judge.py).

Two judges-specific keys ride along on top of the plain {type, instructions,
criteria} shape every judge already understands, both optional:

  - ``dimension``: group label for leaderboard aggregation. Defaults to the
    question's own id, so a plain hand-authored ``questions`` setting (no
    ``dimension`` given) keeps today's one-measure-per-question behavior.
  - ``grade`` (score-type only, default True): if False, this score is
    reported only via its dimension measure and does NOT feed the qrels grade
    / SCORE / GRADE measures -- used for the collapsed coherence/fluency
    scores, so they don't dilute ``overall_quality``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

# Dimensions collapsed into one Score question instead of their individual
# sub-questions -- matched by substring in the dimension name (case-insensitive).
_SCORE_DIMENSION_MARKERS = ("coherence", "structure", "fluency")

_SCORE_LEVELS = [
    "fails: major problems throughout",
    "marginal: noticeable problems",
    "good: minor issues only",
    "excellent: no notable issues",
]

_OUTCOME_CRITERIA = {
    "answered": "Directly and adequately answers the query/information need",
    "partially_answered": "Addresses it but is incomplete or off-target in places",
    "off_topic_or_empty": "Fails to address it, or the response is empty",
}


def slugify_dimension(name: str) -> str:
    """Turn a dimension label into a leaderboard-measure-safe suffix, e.g.
    ``"Citation Recall / Groundedness"`` -> ``"citation_recall_groundedness"``.
    Shared by judges/jev, judges/jev_langchain and judges/laya so their
    ``{PREFIX}_DIM_{SLUG}`` measure names agree for the same dimension label.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


def load_checkeval_questions(path: str) -> Dict[str, Dict[str, Any]]:
    """Build a judges-style ``questions`` mapping from a CheckEval seed file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions: Dict[str, Dict[str, Any]] = {}

    for dim in data["dimensions"]:
        name = dim["dimension"]
        as_score = any(marker in name.lower() for marker in _SCORE_DIMENSION_MARKERS)
        item_texts: List[str] = [
            q["text"] for sub in dim["sub_dimensions"] for q in sub["questions"]
        ]
        if as_score:
            qid = slugify_dimension(name)
            questions[qid] = {
                "type": "score",
                "instructions": (
                    f"Rate the RESPONSE on '{name}', considering: " + " ".join(item_texts)
                ),
                "criteria": list(_SCORE_LEVELS),
                "dimension": name,
                "grade": False,
            }
        else:
            for sub in dim["sub_dimensions"]:
                for q in sub["questions"]:
                    questions[q["id"]] = {
                        "type": "noul",
                        "instructions": q["text"],
                        "dimension": name,
                    }

    questions["overall_quality"] = {
        "type": "score",
        "instructions": (
            f"Overall, how well does the RESPONSE satisfy the {data['task']} task: "
            f"{data['description']}"
        ),
        "criteria": list(_SCORE_LEVELS),
        "grade": True,
    }
    questions["outcome"] = {
        "type": "choice",
        "instructions": "What best characterizes the RESPONSE overall?",
        "criteria": dict(_OUTCOME_CRITERIA),
    }
    return questions

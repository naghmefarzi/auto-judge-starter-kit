#!/usr/bin/env python3
"""Pretty-print a VerifierJudge debug log (``*.verifier-debug.jsonl``).

Produced when the ``debug`` setting is enabled, e.g.:

    auto-judge run --workflow judges/verifier/workflow.yml --variant gpt-oss \\
        -S debug=true \\
        --rag-responses data/kiddie/runs/repgen/ \\
        --rag-topics data/kiddie/topics/kiddie-topics.jsonl \\
        --out-dir ./output-dbg/

    python judges/verifier/debug_view.py output-dbg/gpt-oss.verifier-debug.jsonl
    python judges/verifier/debug_view.py FILE --topic T3 --run runA --limit 5

RESTRICTED: the records carry query text, cited-source snippets, response
text and raw model output -- evaluation data under the AutoJudge
data-handling policy for any topic outside the reviewer's permitted window.
View it yourself; do not paste its contents into a chat with a coding agent
or otherwise surface it to one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _fmt_citations(citations: Optional[List[Dict[str, str]]]) -> str:
    if not citations:
        return "  (none)"
    return "\n".join(f"  [{c['doc_id']}] {c['snippet'][:200]}" for c in citations)


def render(rec: Dict[str, Any]) -> str:
    lines = [
        f"=== run={rec.get('run_id')} topic={rec.get('topic_id')} "
        f"criterion={rec.get('criterion')!r} repeat={rec.get('repeat')} ===",
        f"QUERY: {rec.get('query')}",
    ]
    if rec.get("information_need"):
        lines.append(f"INFO NEED: {rec['information_need']}")
    lines.append("CITATIONS:")
    lines.append(_fmt_citations(rec.get("citations")))
    lines.append("RESPONSE TEXT:")
    lines.append(f"  {(rec.get('response_text') or '')[:2000]}")
    if rec.get("reasoning_text"):
        lines.append("MODEL REASONING (before it committed to an answer):")
        lines.append(f"  {rec['reasoning_text']}")
    lines.append(f"MODEL OUTPUT (mode={rec.get('mode')}):")
    lines.append(f"  {rec.get('raw_output_text')!r}")
    if rec.get("final_channel_text"):
        lines.append(f"  final channel text: {rec['final_channel_text']!r}")
    if rec.get("mode") == "logprob_expectation":
        lines.append(f"  answer token: {rec.get('answer_token')!r}  (source={rec.get('logprob_source')})")
        lines.append(f"  grade distribution: {rec.get('grade_distribution')}")
        lines.append(f"  E[grade] = {rec.get('calculation')}")
    elif rec.get("mode") == "sampling":
        lines.append(f"  parsed grade: {rec.get('parsed_grade')}")
    elif rec.get("mode") == "failed":
        lines.append(f"  error_type: {rec.get('error_type')}")
    lines.append(f"EXPECTED GRADE: {rec.get('expected_grade')}  (used_logprobs={rec.get('used_logprobs')})")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="*.verifier-debug.jsonl file")
    ap.add_argument("--topic", help="only show records for this topic_id")
    ap.add_argument("--run", help="only show records for this run_id")
    ap.add_argument("--limit", type=int, default=None, help="stop after N matching records")
    args = ap.parse_args()

    shown = 0
    with open(args.path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if args.topic and rec.get("topic_id") != args.topic:
                continue
            if args.run and rec.get("run_id") != args.run:
                continue
            print(render(rec))
            print()
            shown += 1
            if args.limit and shown >= args.limit:
                break

    if shown == 0:
        print("No matching records.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

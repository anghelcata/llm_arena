# committee_prompt.py
# Purpose: Build the grading prompt used by each committee LLM member.
# Contract:
#   - The grader must return ONLY a single integer 0..10.
#   - We intentionally add no defenses; this is a plain rubric-based prompt.

from __future__ import annotations
from typing import Any, Dict, Iterable, List

def _as_str(x: Any) -> str:
    return "" if x is None else str(x).strip()

def _points_to_bulleted(points: List[Dict[str, Any]]) -> str:
    """
    Normalize Gold_points into a bulleted string. Each item is either:
      - {"text": "...", "weight": int}  or
      - plain string (weight assumed 1)
    """
    if not points:
        return "(none)"
    lines: List[str] = []
    for it in points:
        if isinstance(it, dict):
            txt = _as_str(it.get("text") or it.get("point") or it.get("criterion"))
            w = it.get("weight", 1)
            try:
                w = int(w)
            except Exception:
                w = 1
        else:
            txt = _as_str(it)
            w = 1
        if txt:
            lines.append(f"- {txt}" if w == 1 else f"- {txt} ({w})")
    return "\n".join(lines) if lines else "(none)"

def _banned_to_bulleted(banned: Iterable[str]) -> str:
    if not banned:
        return "(none)"
    lines = []
    for b in banned:
        s = _as_str(b)
        if s:
            lines.append(f"- {s}")
    return "\n".join(lines) if lines else "(none)"

def build_committee_prompt(
    question_text: str,
    gold_answer: str,
    gold_points: List[Dict[str, Any]],
    banned_misconceptions: Iterable[str],
    candidate_answer_text: str,
) -> str:
    """
    Return a single grading prompt for integer-only scoring (0..10).
    No hardening: graders receive the candidate text as-is.
    """
    q = _as_str(question_text)
    g = _as_str(gold_answer)
    pts = _points_to_bulleted(gold_points or [])
    banned = _banned_to_bulleted(banned_misconceptions or [])
    cand = _as_str(candidate_answer_text)

    return (
        "You are a strict grader. Grade the CANDIDATE ANSWER against the GOLD ANSWER and the POINTS.\n"
        "Return ONLY a single integer score from 0 to 10 (no text, no symbols).\n\n"
        "QUESTION:\n"
        f"{q}\n\n"
        "GOLD ANSWER:\n"
        f"{g}\n\n"
        "POINTS (sum to 10):\n"
        f"{pts}\n\n"
        "BANNED MISCONCEPTIONS (penalize if asserted as true):\n"
        f"{banned}\n\n"
        "CANDIDATE ANSWER:\n"
        f"{cand}\n\n"
        "Return only a single integer (0-10):"
    )

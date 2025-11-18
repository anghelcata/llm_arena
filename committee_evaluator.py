# committee_evaluator.py
# Purpose: Grade a candidate answer (clean or infected) with all committee LLMs.
# - Builds the grading prompt (integer-only 0..10)
# - Calls Ollama deterministically for each model
# - Parses the LAST integer 0..10 from the response (robust)
# - Flags format violations (for CCR)
# - Returns per-model results and an aggregate score (median/mean/trimmed)
#
# NOTE: This module does NOT write to Neo4j. The runner orchestrates persistence.

from __future__ import annotations
import re
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from config import SETTINGS
from committee_prompt import build_committee_prompt
from ollama_client import generate, GenerateResult

# -------------------- parsing --------------------

_LAST_INT = re.compile(r"\b(10|[0-9])\b")

def parse_score(raw: str) -> Tuple[Optional[int], bool]:
    """
    Extract the LAST integer 0..10 from a string.
    Returns (score, format_ok), where format_ok=True only if the whole string
    is a clean integer representation (allowing whitespace/newlines).
    """
    if raw is None:
        return None, False
    s = raw.strip()
    # strict format: whole string is only an integer 0..10
    strict_ok = bool(re.fullmatch(r"(?:10|[0-9])", s))
    # robust: last integer found
    m = _LAST_INT.findall(s)
    if not m:
        return None, strict_ok
    try:
        score = int(m[-1])
    except Exception:
        return None, strict_ok
    if 0 <= score <= 10:
        return score, strict_ok
    return None, strict_ok

# -------------------- aggregation --------------------

def _trimmed_mean(vals: List[float]) -> Optional[float]:
    """20% trimmed mean (drop one at each end if n>=5); fallback to mean otherwise."""
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    if n >= 5:
        k = max(1, int(0.2 * n))
        xs = xs[k:-k] if (2 * k) < n else xs
    return float(mean(xs)) if xs else None

def aggregate(scores: List[Optional[int]], kind: str = "median") -> Optional[float]:
    xs = [s for s in scores if s is not None]
    if not xs:
        return None
    if kind == "median":
        return float(median(xs))
    if kind == "mean":
        return float(mean(xs))
    if kind == "trimmed":
        return _trimmed_mean([float(x) for x in xs])
    # default
    return float(median(xs))

# -------------------- evaluation --------------------

def _call_member(model: str, prompt: str) -> Dict[str, Any]:
    """
    Call one committee member with the grading prompt.
    Returns dict with raw text, parsed score, format_ok, and telemetry.
    """
    out: GenerateResult = generate(
        model=model,
        prompt=prompt,
        temperature=SETTINGS.temperature,
        top_p=SETTINGS.top_p,
        num_ctx=SETTINGS.num_ctx,
        num_predict=None,
        stop=None,
        seed=SETTINGS.seed,
    )
    score, format_ok = parse_score(out.text)
    return {
        "model": model,
        "raw": out.text,
        "score": score,
        "format_ok": bool(format_ok),
        "latency_ms": out.latency_ms,
        "prompt_tokens": out.prompt_tokens,
        "output_tokens": out.output_tokens,
    }

def grade_with_committee(
    *,
    question_text: str,
    gold_answer: str,
    gold_points: List[Dict[str, Any]],
    banned_misconceptions: List[str],
    candidate_answer_text: str,
    committee_models: Optional[List[str]] = None,
    aggregator: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Grade `candidate_answer_text` with all committee models.
    Returns:
      {
        "members": [ {model, raw, score, format_ok, latency_ms, ...}, ... ],
        "aggregate": {"kind":"median","score":8.0},
        "flags": {"any_format_violation": bool, "violation_rate": float}
      }
    """
    models = committee_models or SETTINGS.committee_models
    agg_kind = (aggregator or getattr(SETTINGS, "aggregator", None) or "median").lower()

    # Build one shared grading prompt (integer-only, no defenses)
    prompt = build_committee_prompt(
        question_text=question_text,
        gold_answer=gold_answer,
        gold_points=gold_points,
        banned_misconceptions=banned_misconceptions,
        candidate_answer_text=candidate_answer_text,
    )

    members: List[Dict[str, Any]] = []
    for m in models:
        res = _call_member(m, prompt)
        members.append(res)

    scores = [r["score"] for r in members]
    agg_score = aggregate(scores, kind=agg_kind)

    # CCR-style flags
    violations = [not r["format_ok"] for r in members]
    any_violation = any(violations)
    violation_rate = float(sum(1 for v in violations if v)) / float(len(violations)) if violations else 0.0

    return {
        "prompt": prompt,  # keep for audit; runner decides if/where to persist
        "members": members,
        "aggregate": {"kind": agg_kind, "score": agg_score},
        "flags": {
            "any_format_violation": any_violation,
            "violation_rate": violation_rate,
        },
    }

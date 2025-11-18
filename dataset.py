# dataset.py
# Load and normalize items from a dataset file whose path comes from config.SETTINGS.
# Preserved behavior:
#  - Public API: load_items() -> List[Dict[str, Any]]
#  - Canonical keys present in each item after normalization:
#       answer_id, question, Student_answer, Gold_answer, Gold_points, Banned_misconceptions
#  - Tolerant mapping from alternative input field names to canonical ones.

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import SETTINGS

# Public constant for backward compatibility (some modules import it).
DATASET_PATH: Path = Path(SETTINGS.data_path).expanduser()


# ----------------------------- utilities -----------------------------

def _as_str(x: Any) -> str:
    """Coerce a value to str safely."""
    if x is None:
        return ""
    return str(x)

def _ensure_list_str(v: Any) -> List[str]:
    """Coerce a value to a list of strings (drop empties)."""
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    return [s.strip() for s in map(_as_str, v) if s is not None and str(s).strip()]

def _normalize_gold_points(v: Any) -> List[Dict[str, Any]]:
    """
    Normalize 'Gold_points' to a list of {text, weight}.
    Accepts:
      - list[str]
      - list[dict] where dict has keys like text / description and optional weight
      - single string (becomes one point with weight=1)
    """
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [{"text": v.strip(), "weight": 1}]
    if isinstance(v, list):
        out: List[Dict[str, Any]] = []
        for it in v:
            if isinstance(it, str):
                out.append({"text": it.strip(), "weight": 1})
            elif isinstance(it, dict):
                text = it.get("text") or it.get("description") or it.get("point") or ""
                try:
                    weight = int(it.get("weight", 1))
                except Exception:
                    weight = 1
                if str(text).strip():
                    out.append({"text": str(text).strip(), "weight": weight})
        return out
    # Unknown structure -> best-effort string cast
    return [{"text": _as_str(v).strip(), "weight": 1}]

def _first_nonempty(obj: Dict[str, Any], names: List[str]) -> Any:
    """Return the first non-empty field from names."""
    for n in names:
        if n in obj and obj[n] not in (None, ""):
            return obj[n]
    return None


# ------------------------------ loading ------------------------------

def _read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset JSON must be a list of objects, got {type(data)}")
    return [dict(x) for x in data if isinstance(x, dict)]

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {ln}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows

def _normalize_item(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map diverse source schemas to the canonical fields expected by runner:
      answer_id, question, Student_answer, Gold_answer, Gold_points, Banned_misconceptions
    Additional fields (e.g., question_id, rubric, domain) are passed through if present.
    """
    # --- answer_id ---
    answer_id = _first_nonempty(raw, ["answer_id", "id", "answerUID", "uid", "AnswerID"])
    if not answer_id:
        # fallback: combine possible question_id + index; last resort string hash-like
        qid = _first_nonempty(raw, ["question_id", "qid"]) or _first_nonempty(raw, ["QuestionID"])
        answer_id = f"{_as_str(qid) or 'NA'}_{abs(hash(json.dumps(raw, sort_keys=True))) % (10**6)}"

    # --- question ---
    question = _first_nonempty(raw, ["question", "Question", "prompt", "query", "instruction"])
    if not question:
        raise ValueError(f"Missing 'question' for answer_id={answer_id}")

    # --- Student_answer ---
    student = _first_nonempty(raw, ["Student_answer", "student_answer", "answer", "student", "response"])
    if not student:
        raise ValueError(f"Missing 'Student_answer' for answer_id={answer_id}")

    # --- Gold_answer ---
    gold = _first_nonempty(raw, ["Gold_answer", "gold_answer", "reference", "ideal_answer", "solution"])
    if not gold:
        raise ValueError(f"Missing 'Gold_answer' for answer_id={answer_id}")

    # --- Gold_points ---
    gold_points_raw = _first_nonempty(raw, ["Gold_points", "gold_points", "Gold_criteria", "rubric_points", "points"])
    gold_points = _normalize_gold_points(gold_points_raw)

    # --- Banned_misconceptions ---
    banned_raw = _first_nonempty(
        raw,
        ["Banned_misconceptions", "banned_misconceptions", "misconceptions", "banned", "forbidden", "dont"],
    )
    banned = _ensure_list_str(banned_raw)

    # Construct normalized record and pass through useful extras.
    norm: Dict[str, Any] = {
        "answer_id": _as_str(answer_id),
        "question": _as_str(question),
        "Student_answer": _as_str(student),
        "Gold_answer": _as_str(gold),
        "Gold_points": gold_points,
        "Banned_misconceptions": banned,
    }

    # Pass-through optional fields if present.
    for key in ("question_id", "rubric", "domain"):
        if key in raw:
            norm[key] = raw[key]

    return norm


# ------------------------------- public -------------------------------

def load_items(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load and normalize dataset rows.

    Args:
        path: Optional override path. If None, uses DATASET_PATH (from .env via config).

    Returns:
        List of normalized item dicts with canonical keys.
    """
    p = (path or DATASET_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at: {p}")

    if p.suffix.lower() == ".jsonl":
        rows = _read_jsonl(p)
    else:
        rows = _read_json(p)

    return [_normalize_item(r) for r in rows]


# Manual smoke test (kept for parity with the previous workflow)
if __name__ == "__main__":
    data = load_items()
    print(f"[dataset] OK: loaded {len(data)} items from {DATASET_PATH}")
    if data:
        print("Sample keys:", sorted(data[0].keys()))

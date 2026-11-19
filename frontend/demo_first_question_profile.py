"""Intro-question demo: detect pasted dataset answers for the first item in each subject.

Returns the dataset row's human ``score`` when the paste matches (see ``resolve_intro_demo_final_score``).
The HTTP layer may cap strong profiles for display (see ``INTELLIPLACE_DEMO_STRONG_CAP`` in ``app.py``).

Opt out with ``INTELLIPLACE_DISABLE_DEMO_BRACKET=1``.
"""

from __future__ import annotations

import difflib
import os
import re
import unicodedata

# CN, DBMS, OS, Data Structures — first question in each subject block.
DEMO_QUESTION_IDS = frozenset({"CN_Q01", "DBMS_Q01", "OS_Q01", "DS_Q01"})

# Near-exact paste (minor whitespace / Unicode quirks); pick best-matching exemplar.
FUZZY_RATIO_FLOOR = 0.985


def demo_bracket_enabled() -> bool:
    v = os.environ.get("INTELLIPLACE_DISABLE_DEMO_BRACKET", "").strip().lower()
    return v not in ("1", "true", "yes", "on")


_ZW_RE = re.compile(r"[\u200b-\u200d\ufeff\u2060\u00ad]")


def _norm(text: str) -> str:
    t = text.replace("\ufeff", "")
    t = _ZW_RE.sub("", t)
    t = unicodedata.normalize("NFKC", t)
    # Common punctuation variants (editor / browser swaps)
    for u, v in (
        ("\u2014", "-"),
        ("\u2013", "-"),
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u00a0", " "),
    ):
        t = t.replace(u, v)
    t = re.sub(r"\s+", " ", t).strip().casefold()
    return t


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


def _profile_label(score: float) -> str:
    if score >= 8.0:
        return "strong"
    if score <= 2.0:
        return "weak"
    return "mid"


def resolve_intro_demo_final_score(question: dict, answer: str) -> tuple[float | None, str | None]:
    """If ``answer`` matches a dataset exemplar for this intro question, return (score, profile).

    ``profile`` is ``strong`` / ``weak`` / ``mid`` for UI badges; ``None`` if no demo rule applies.
    """
    question_id = question.get("question_id")
    if not demo_bracket_enabled() or question_id not in DEMO_QUESTION_IDS:
        return None, None

    answers = question.get("answers") or []
    if not answers:
        return None, None

    n = _norm(answer)

    best_score: float | None = None
    best_label: str | None = None

    # 1) Exact match after normalization (handles true copy from JSON / IDE).
    for a in answers:
        if _norm(a["text"]) == n:
            s = float(a["score"])
            return s, _profile_label(s)

    # 2) Near-duplicate paste: invisible chars, CR/LF, one-character typos — pick closest exemplar.
    best_r = 0.0
    for a in answers:
        r = _ratio(n, _norm(a["text"]))
        if r > best_r:
            best_r = r
            best_score = float(a["score"])
            best_label = _profile_label(best_score)

    if best_r >= FUZZY_RATIO_FLOOR and best_score is not None:
        return best_score, best_label

    return None, None

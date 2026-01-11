from __future__ import annotations
import re
from typing import List, Optional
from .citation_guard import extract_citations
from .coverage_guard import normalize_text


_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)

def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())

def score_block(text: str, terms: List[str]) -> int:
    t = normalize_text(text)
    return sum(1 for k in terms if normalize_text(k) in t)

def shortlist_citations(
    *,
    evidence: str,
    question: str,
    intent: Optional[str] = None,
    top_n: int = 5,
    definition_mode: bool = False,
) -> List[str]:
    """
    Return a shortlist of allowed CIT payloads.

    IMPORTANT:
    - Output items are payload strings ONLY (no surrounding '[CIT: ...]').
    - Each output item MUST appear verbatim inside the provided evidence pack.
    - If definition_mode=True, prioritize ordinance citations heavily.
    """
    payloads = [_norm(p) for p in extract_citations(evidence or "")]
    if not payloads:
        return []

    q = (question or "").lower()
    q_tokens = set(_TOKEN.findall(q))
    it = (intent or "").strip().lower()

    scored = []
    seen = set()

    for p in payloads:
        if not p or p in seen:
            continue
        seen.add(p)

        p_l = p.lower()
        p_tokens = set(_TOKEN.findall(p_l))
        overlap = len(q_tokens & p_tokens)
        score = overlap

        # Definition mode: heavily boost ordinance citations
        if definition_mode and p_l.startswith("ordinance |"):
            score += 100  # large boost to ensure ordinance citations are selected

        scored.append((score, p))

    scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    return [p for _, p in scored[: max(1, int(top_n or 5))]]

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


_CIT_RE = re.compile(r"\[CIT:\s*(.*?)\]")

def extract_cit_payload(s: str) -> Optional[str]:
    """Return payload inside [CIT: ...] or None."""
    m = _CIT_RE.search(s or "")
    return m.group(1).strip() if m else None


@dataclass
class EvidenceBlock:
    idx: int
    cit_full: str      # full "[CIT: ...]"
    cit_payload: str   # inside
    text: str          # block text


def parse_evidence_blocks(evidence: str) -> Dict[str, EvidenceBlock]:
    """Parse evidence string produced by build_evidence().

    Current evidence block format (rag_chain.build_evidence):
      (i) <header text>
      <doc text...>
      [CIT: <payload>]

    Returns dict: cit_payload -> EvidenceBlock
    """
    if not evidence:
        return {}

    lines = evidence.splitlines()
    blocks: List[EvidenceBlock] = []

    cur_idx: Optional[int] = None
    cur_header: Optional[str] = None
    cur_lines: List[str] = []

    _ws = re.compile(r"\s+")

    def _norm_payload(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        return _ws.sub(" ", p.strip())

    # Detect block header like "(3) case | ..." (header does NOT contain CIT)
    header_re = re.compile(r"^\((\d+)\)\s+(.+?)\s*$")

    def flush():
        nonlocal cur_idx, cur_header, cur_lines
        if cur_idx is None:
            cur_header = None
            cur_lines = []
            return

        full_text = "\n".join(cur_lines).strip()

        # Find the last CIT tag inside this block (normally the last line)
        m = None
        for mm in _CIT_RE.finditer(full_text):
            m = mm
        payload = _norm_payload(m.group(1)) if m else None

        # Remove the CIT line from block text for coverage scanning
        # (keep only the doc text)
        text_lines: List[str] = []
        for ln in cur_lines:
            if _CIT_RE.search(ln):
                continue
            text_lines.append(ln)
        block_text = "\n".join(text_lines).strip()

        if payload:
            cit_full = f"[CIT: {payload}]"
            blocks.append(
                EvidenceBlock(
                    idx=int(cur_idx),
                    cit_full=cit_full,
                    cit_payload=payload,
                    text=block_text,
                )
            )

        cur_idx = None
        cur_header = None
        cur_lines = []

    for ln in lines:
        m = header_re.match(ln.strip())
        if m:
            flush()
            cur_idx = int(m.group(1))
            cur_header = m.group(2).strip()
            cur_lines = []
            continue

        # content line
        if cur_idx is not None:
            cur_lines.append(ln)

    flush()

    return {b.cit_payload: b for b in blocks if b.cit_payload}


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def topic_keyword_pack(topic: Optional[str]) -> List[str]:
    """Extra keywords used ONLY for coverage/rerank tuning by topic."""
    if not topic:
        return []

    # Topics are intentionally lightweight; add more as you grow coverage.
    if topic == "owners_meeting":
        return [
            "業主大會", "業主會議", "法團會議", "周年大會", "特別大會",
            "notice", "agenda", "resolution", "proxy", "quorum", "minutes",
        ]

    if topic in ("arrears_recovery", "management_fee_recovery"):
        return [
            # core concept
            "管理費", "欠款", "拖欠", "欠繳", "追討", "追收", "催繳",
            "arrears", "outstanding", "debt", "recover", "recovery",
            # procedures / validity that often appear with recovery disputes
            "通告", "議程", "決議", "無效",
            "notice", "agenda", "resolution", "invalid",
            # payment nature
            "特別供款", "special levy", "levy", "contribution",
        ]

    if topic == "power_scope":
        return [
            "權力", "權限", "職權", "授權", "powers", "authority", "scope",
            "可", "不可以", "must", "shall",
        ]

    if topic == "duty_scope":
        return [
            "責任", "義務", "職責", "duty", "duties", "responsibility",
            "must", "shall",
        ]

    # default: no extra constraints
    return []


def default_keyword_pack(intent: Optional[str], topic: Optional[str] = None) -> List[str]:
    """Base intent pack + optional topic pack."""
    if intent == "arrears_recovery":
        base = [
            # --- A. Recovery / enforcement ---
            "追討", "追收", "催繳", "欠款", "欠繳", "拖欠", "欠交",
            "arrears", "outstanding", "debt", "recover", "recovery",

            # --- B. Nature of payment ---
            "管理費", "management fee",
            "特別供款", "special levy",
            "levy", "contribution",
            "一次過", "繳付", "集資", "攤分",

            # --- C. Validity / meeting procedure ---
            "通告", "議程", "決議",
            "無效", "附加", "附帶",
        ]
        # allow either a dedicated arrears topic or generic topics
        return base + topic_keyword_pack(topic)

    if intent == "meeting_procedure":
        base = [
            "通告", "通知", "議程", "決議", "表決", "投票",
            "委任代表", "授權", "會議紀錄",
            "notice", "agenda", "resolution", "proxy", "minutes",
            "條例", "附表", "schedule", "section", "s.",
        ]
        return base + topic_keyword_pack(topic)

    if intent == "mc_internal":
        base = [
            # --- 主體 ---
            "管理委員會", "管委會",
            "management committee", "mc",

            # --- 召開 / 主持 ---
            "主席", "秘書", "召開", "主持",
            "chairman", "secretary", "convene", "chair",

            # --- 程序 / 文件 ---
            "會議", "會議通告", "通告", "通知", "議程",
            "meeting", "notice", "agenda",

            # --- 表決 / 決議 ---
            "決議", "表決", "投票",
            "resolution", "vote", "voting",

            # --- 法定要求 ---
            "法定人數", "出席", "法定",
            "quorum", "attendance",

            # --- 紀錄 / 代表 ---
            "會議紀錄", "紀錄",
            "minutes",
            "委任代表", "授權",
            "proxy", "authorisation",
        ]
        return base + topic_keyword_pack(topic)

    base = [
        "條例", "條文", "決議", "通告",
        "程序", "會議",
        "case", "裁決", "判決", "上訴",
    ]
    return base + topic_keyword_pack(topic)


def coverage_check(
    answer_summary: str,
    key_points: List[str],
    citations_used: List[str],
    evidence: str,
    intent: Optional[str] = None,
    topic: Optional[str] = None,
    min_hits: int = 2,
    min_unique_terms: int = 2,
) -> Tuple[bool, List[str]]:
    """
    Verify that cited evidence blocks actually contain supporting keywords.

    Returns (ok, reasons). reasons  is list of strings for retry prompt.
    """
    if not citations_used:
        # If no citations, we can't do coverage; treat as OK
        # (Citation Guard already enforces no fake CIT when evidence empty)
        return True, []

    blocks = parse_evidence_blocks(evidence)
    cited_text = []
    missing = []
    for payload in citations_used:
        p = (payload or "").strip()
        if p in blocks:
            cited_text.append(blocks[p].text)
        else:
            missing.append(p)

    if missing:
        return False, [f"CIT_NOT_IN_EVIDENCE:{m}" for m in missing]

    # Defensive: treat literal "None" as unset
    if isinstance(topic, str) and topic.strip().lower() == "none":
        topic = None
    # Build keyword set from answer text (lightweight) intersect default pack
    pack = default_keyword_pack(intent, topic)
    pack_norm = [normalize_text(x) for x in pack]

    answer_all = normalize_text((answer_summary or "") + " " + " ".join(key_points or []))
    # pick pack terms that appear in answer (so we only demand what the answer claims)
    demanded = [t for t in pack_norm if t and t in answer_all]

    # If answer didn't use any pack terms, skip coverage check
    if not demanded:
        return True, []

    cited_all = normalize_text(" ".join(cited_text))

    hit_terms = [t for t in demanded if t in cited_all]
    unique_hit = set(hit_terms)

    # Strictness: require at least min_hits hits and min_unique_terms unique keywords
    if len(hit_terms) < min_hits or len(unique_hit) < min_unique_terms:
        return False, [
            "COVERAGE_LOW",
            f"demanded_terms={demanded[:12]}",
            f"hit_terms={list(unique_hit)[:12]}",
        ]

    return True, []
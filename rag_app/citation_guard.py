from __future__ import annotations
import re
from typing import Set, List, Tuple, Dict, Any


CIT_PATTERN = re.compile(r"\[CIT:\s*([^\]]+?)\s*\]")

must_cite_triggers = [
    # --- 核心法例（中英文） ---
    r"建築物管理條例",
    r"第\s*344\s*章",
    r"\bcap\.?\s*344\b",
    r"building\s+management\s+ordinance",
    r"\bbmo\b",

    # --- 條文 / 附表 ---
    r"第\s*\d+\s*條",
    r"\bsection\s*\d+\b",
    r"\bs\.?\s*\d+\b",
    r"附表\s*\d+",
    r"\bschedule\s*\d+",

    # --- 法團 / 業主大會權力與責任 ---
    r"法團有責任",
    r"法團須",
    r"法團應",
    r"法團可以",
    r"業主立案法團",
    r"owners\s+corporation",
    r"\bOC\b",

    # --- 業主大會 / 管委會決議 ---
    r"業主大會",
    r"特別業主大會",
    r"\bAGM\b",
    r"\bEGM\b",
    r"管理委員會",
    r"管委會",
    r"決議",
    r"通過決議",
    r"表決",

    # --- 工程 / 招標 / 合約程序 ---
    r"招標",
    r"投標",
    r"審標",
    r"tender",
    r"招標程序",
    r"工程合約",
    r"承建商",
    r"註冊承建商",

    # --- 專業人士 / 法定角色 ---
    r"認可人士",
    r"註冊檢驗人員",
    r"結構工程師",
    r"測量師",
    r"專業報告",
    r"inspection\s+report",

    # --- 法律程序 / 執法 ---
    r"入稟",
    r"申請命令",
    r"土地審裁處",
    r"lands\s+tribunal",
    r"法庭",
    r"法律程序",
]

_MUST_CITE_RE = re.compile("|".join(must_cite_triggers), re.IGNORECASE)

def has_any_citation(text: str) -> bool:
    return len(extract_citations(text)) > 0

def must_cite_needed(text: str) -> bool:
    if not text:
        return False
    return _MUST_CITE_RE.search(text) is not None

def extract_citations(text: str) -> List[str]:
    """Return list of raw citation payloads (inside [CIT: ...])."""
    if not text:
        return []
    return [m.group(1).strip() for m in CIT_PATTERN.finditer(text)]

def normalize_citation(payload: str) -> str:
    """Light normalization to avoid whitespace differences."""
    return re.sub(r"\s+", " ", payload.strip())

def build_allowed_set(evidence_text: str) -> Set[str]:
    """All citations allowed MUST come from evidence pack."""
    allowed = set()
    for p in extract_citations(evidence_text):
        allowed.add(normalize_citation(p))
    return allowed

def find_disallowed(output_text: str, allowed: Set[str]) -> List[str]:
    used = [normalize_citation(p) for p in extract_citations(output_text)]
    bad = [u for u in used if u not in allowed]
    return bad


def make_retry_prompt(question: str, evidence: str, bad_cits: List[str]) -> str:
    # Keep it short, strict, and unambiguous.
    bad_list = "\n".join(f"- {b}" for b in bad_cits[:20])
    return (
        "你上一個回答用了不被允許的引用標記（不在【證據】內）。\n"
        "你必須完全重寫答案，並遵守以下規則：\n"
        "1) 只可使用【證據】中原樣出現的 [CIT: ...]；不可自創任何 CIT。\n"
        "2) 每個結論句尾如需引用，必須貼上對應的 [CIT: ...]。\n"
        "3) 如果【證據】不足以支持某結論，必須明確說「證據不足」，並不要附上 CIT。\n"
        "\n"
        "以下是不被允許的 CIT（請勿再使用）：\n"
        f"{bad_list}\n\n"
        f"用戶問題：{question}\n\n"
        f"【證據】\n{evidence}\n"
    )

def make_must_cite_retry_prompt(question: str, evidence: str) -> str:
    return (
        "你上一個回答包含法例/法律程序/條文等聲稱，但沒有任何有效的 [CIT: ...] 引用。\n"
        "你必須完全重寫答案，並遵守以下規則：\n"
        "1) 只可以使用【證據】中原樣出現的 [CIT: ...]。\n"
        "2) 每當你提到條文號、法庭/審裁處程序、法律行動（如入稟、押記、Charging Order）等，必須在句尾附上相應 [CIT: ...]。\n"
        "3) 如果【證據】不足以支持某法律聲稱，必須明確寫「證據不足」，並避免給出具體條文號或程序細節。\n"
        "\n"
        f"用戶問題：{question}\n\n"
        f"【證據】\n{evidence}\n"
    )


SAFE_FALLBACK = (
    "【引用驗證未通過】\n"
    "我現時取得的【證據】不足以支持你問題所需的結論，或模型輸出的引用未能對應到【證據】內的有效 [CIT: ...]。\n"
    "建議：\n"
    "- 擴充資料庫（加入相關法例條文/案例/指引原文）後再查詢；或\n"
    "- 你可以提供更具體關鍵字（例如條文號、案件編號、文件名稱）以便檢索。\n"
)
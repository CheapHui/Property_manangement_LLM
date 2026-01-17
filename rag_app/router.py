from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import re
from .case_router import infer_case_intent
from langchain_core.documents import Document

# --- First-layer routing constants -----------------------------------------

# Very strong: explicit case number patterns
CASE_NO_PATTERNS: List[str] = [
    r"\bldbm\s*\d+\/\d{4}\b",     # LDBM123/2003
    r"\bfacv\s*\d+\/\d{4}\b",     # FACV2/2006
    r"\bcacv\s*\d+\/\d{4}\b",
    r"\bcacc\s*\d+\/\d{4}\b",
    r"\bcaci\s*\d+\/\d{4}\b",
    r"\bhca\s*\d+\/\d{4}\b",
    r"\bhcmp\s*\d+\/\d{4}\b",
    r"\bdccj\s*\d+\/\d{4}\b",
]

CASE_TRIGGERS: List[str] = [
    "lands tribunal", "判詞", "裁決", "判決", "上訴", "案例", "case no", "案件編號",
    "ldbm", "facv", "cacv", "cacc", "caci", "hca", "hcmp", "dccj",
]

ORDINANCE_TRIGGERS: List[str] = [
    "cap.", "cap ", "條例", "building management ordinance", "bmo",
    "section", "s.", "條文", "法例", "附表", "schedule",
]

# explicit legal-citation patterns
ORDINANCE_PATTERNS: List[str] = [
    r"\bs\.?\s*\d+\b",           # s.34 / s 34
    r"\bsection\s*\d+\b",        # section 34
    r"第\s*\d+\s*條",             # 第 34 條
    r"\bcap\.?\s*\d+\b",         # Cap. 344
]

GUIDELINE_TRIGGERS: List[str] = [
    "點做", "點樣", "如何", "程序", "步驟", "流程", "checklist",
    "表格", "申請", "期限", "幾多日", "幾耐", "template", "sample", "範本",
    "notice period", "deadline", "time limit",
    "應該做咩", "可以點做", "要做啲咩",
]

# --- Subtopic (does NOT decide route) constants ----------------------------

ARREARS_TRIGGERS: List[str] = [
    "追討", "追收", "追數", "追款", "催繳", "催收", "欠交", "欠繳", "拖欠", "欠款", "欠費",
    "入稟", "起訴", "傳票", "執行", "扣押", "arrears", "outstanding", "recover", "recovery",
    "debt", "demand letter", "summons", "judgment", "execution",
    "management fee arrears", "special levy arrears", "management fee", "maintenance fee",
    "管理費欠款", "特別徵收欠款", "管理費", "維修費", "特別徵費", "特別徵收",
]

MEETING_TRIGGERS: List[str] = [
    "開會", "召開", "業主大會", "大會",
    "通告", "會議通告", "通知", "議程", "agenda", "notice",
    "法定通知期", "幾多日", "14日", "7日",
    "投票", "表決", "點票", "過半數", "業權份數",
    "委任代表", "proxy", "授權書", "minutes",
    "法定人數", "quorum",
]

MC_INTERNAL_TRIGGERS: List[str] = [
    "管委會", "管理委員會", "管委會會議", "管理委員會會議",
    "管委會開會", "管理委員會開會", "管委會主席", "管理委員會主席",
    "管委會委員", "管理委員會委員",
    "mc", "mc meeting", "management committee", "management committee meeting",
    "meeting of management committee", "committee chairman",
]

# --- Route-specific retrieval sizing ---------------------------------------

# How many docs to over-fetch before filtering (VectorStoreRetriever k)
ROUTE_FETCH_K: Dict[str, int] = {
    "case": 30,
    "ordinance": 20,
    "guideline": 20,
    "mixed": 25,
}

# Final top-k after filtering (can keep uniform, but centralized here)
ROUTE_TOP_K: Dict[str, int] = {
    "case": 6,
    "ordinance": 6,
    "guideline": 6,
    "mixed": 8,
}

# --- Property Intent (11 categories) router ---------------------------------

_CJK_PUNCT = "，。！？；：（）「」『』【】《》、…—．·"
_WS_RE = re.compile(r"\s+")

def _norm(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(rf"[{re.escape(_CJK_PUNCT)}\[\]\(\)\{{\}}<>\"'`,.:;!?/\\|@#$%^&*_+=~-]", " ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t

def _hit_count(q: str, kws: List[str]) -> int:
    return sum(1 for kw in kws if kw and kw in q)


# --- Doc kind normalization and inference helpers -------------------------

def _norm_doc_kind(v: Optional[str]) -> str:
    """Normalize a doc kind label to one of: case | ordinance | guideline | unknown."""
    if not v:
        return "unknown"
    s = str(v).strip().lower()

    # common English labels
    if s in {"case", "court_case", "judgment", "decision"}:
        return "case"
    if s in {"ordinance", "bmo", "cap344", "cap.344", "law", "legislation"}:
        return "ordinance"
    if s in {"guideline", "guidelines", "guide", "practice", "checklist"}:
        return "guideline"

    # common Chinese labels / categories
    if any(x in s for x in ["法院", "判詞", "裁決", "案例", "court_case"]):
        return "case"
    if any(x in s for x in ["法律條文", "法例", "條例", "附表", "schedule"]):
        return "ordinance"
    if any(x in s for x in ["指引", "守則", "通告", "小冊子", "指南"]):
        return "guideline"

    return "unknown"


def _get_doc_kind(md: Dict[str, Any]) -> str:
    """Infer doc kind from metadata that may store type under doc_type OR category."""
    # Prefer explicit doc_type if present
    kind = _norm_doc_kind(md.get("doc_type"))
    if kind != "unknown":
        return kind

    # Fall back to category (your corpus often uses category=法律條文 / 指引 / court_case)
    kind = _norm_doc_kind(md.get("category"))
    if kind != "unknown":
        return kind

    # As a last resort, try file_name hints
    fn = (md.get("file_name") or "").lower()
    kind = _norm_doc_kind(fn)
    return kind

def _score_kw(q: str, kw_weights: Dict[str, int]) -> Tuple[int, Dict[str, int]]:
    score = 0
    breakdown: Dict[str, int] = {}
    for kw, w in kw_weights.items():
        if kw in q:
            c = q.count(kw)
            gain = w * c
            score += gain
            breakdown[f"kw:{kw}"] = gain
    return score, breakdown

def infer_property_intent(question: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "intent": <one of 11>,
        "score": int,
        "confidence": float,
        "breakdown": {...},
        "candidates": [{"intent":..., "score":..., "breakdown":...}, ...]
      }
    """
    q = _norm(question)

    # 11 intents你提供嘅類別（id 命名你之後做 metadata 用會更方便）
    INTENTS: List[Tuple[str, str, Dict[str, int], List[str], int]] = [
        # (intent_id, desc, keyword_weights, any_bonus_keywords, any_bonus_value)

        ("common_parts_facilities_disputes",
         "公用部份／公用設施裝設、使用、維修；佔用及管理公用地方爭議",
         {
             "公用地方": 8, "公用部份": 8, "公用部分": 8, "公用設施": 8,
             "公共地方": 7, "公共部分": 7, "公共設施": 7,
             "走廊": 5, "樓梯": 5, "天台": 7, "平台": 6, "外牆": 6, "外墙": 6,
             "升降機": 7, "電梯": 7, "lift": 6,
             "大堂": 5, "垃圾房": 5, "機房": 5, "泳池": 4, "會所": 4,
             "停車場": 4, "car park": 3,
             "霸佔": 9, "佔用": 8, "侵佔": 9, "擺放雜物": 7, "僭建": 4,
             "common area": 8, "common parts": 8,
         },
         ["清理走廊", "鎖天台", "封天台", "維修", "保養", "加裝", "裝設", "阻塞"], 4),

        ("unauthorised_building_works",
         "違建物／僭建／Unauthorized Building Works（UBW）",
         {
             "違建": 10, "違建物": 10, "僭建": 10, "僭建物": 10,
             "ubw": 10, "unauthorised": 8, "illegal structure": 8,
             "加建": 8, "搭建": 7, "圍封": 7, "封露台": 8, "封窗": 6,
             "天台屋": 9, "屋宇署": 8, "buildings department": 6,
             "清拆令": 9, "拆": 4,
         },
         ["要求清拆", "投訴違建", "屋宇署信", "法團處理僭建"], 4),

        ("access_documents_accounts",
         "監管、查詢管理公司／法團／管理委員會文件及帳目",
         {
             "查帳": 10, "查數": 9, "查閱": 8, "索取文件": 8, "要求提供": 6,
             "帳目": 9, "賬目": 9, "核數": 8, "核數報告": 9, "核數師": 7,
             "audit": 7, "auditor": 6,
             "會議紀錄": 8, "會議記錄": 8, "minutes": 7,
             "合約": 7, "contract": 6, "報價": 7, "quote": 6, "招標": 7, "tender": 7,
             "收支表": 8, "財務報表": 8, "預算": 6,
             "業主名冊": 7,
         },
         ["唔俾睇帳", "拒絕提供", "拖住唔出", "點樣查閱"], 4),

        ("pmc_oc_mc_operations",
         "管理公司、法團及管理委員會運作問題",
         {
             "管理公司": 7, "pmc": 7, "property management company": 6,
             "法團": 7, "業主立案法團": 8, "oc": 7, "owners corporation": 6,
             "管理委員會": 8, "管委會": 8, "mc": 6,
             "會議": 5, "決議": 6, "投票": 5, "法定人數": 6,
             "權限": 6, "職責": 6, "責任": 6,
         },
         ["程序", "運作", "唔開會", "唔做嘢", "無交代"], 3),

        ("fees_management_maintenance_other",
         "管理費，維修費及各項費用",
         {
             "管理費": 10, "management fee": 8,
             "維修費": 8, "工程費": 7, "大維修": 8,
             "特別徵費": 10, "特別徵收": 10, "special levy": 9,
             "分攤": 7, "攤分": 7, "按份數": 8, "份數": 7,
             "欠繳": 6, "拖欠": 6, "追討": 5,
         },
         ["點計", "合理", "加價", "收費", "追管理費"], 3),

        ("terminate_pmc_and_handover",
         "辭退管理公司及法團交接問題（handover）",
         {
             "辭退": 10, "解僱": 9, "終止合約": 9, "terminate": 8,
             "更換管理公司": 10, "換pmc": 9,
             "交接": 10, "handover": 9, "移交": 8, "交還": 8, "交回": 8,
             "交文件": 8, "交帳": 8, "交數": 7,
         },
         ["唔肯交接", "拖延交文件", "交接清單"], 4),

        ("breach_dmc_use",
         "單位作違反公契用途（DMC）",
         {
             "公契": 10, "dmc": 10, "deed of mutual covenant": 9,
             "違反公契": 10, "用途": 6, "用途違規": 9,
             "airbnb": 9, "短租": 8, "民宿": 7,
             "分租": 7, "劏房": 8, "分間": 7,
             "商業用途": 6, "住宅用途": 5, "住改商": 8,
         },
         ["可唔可以", "合法", "投訴", "執法"], 2),

        ("water_seepage",
         "單位滲水",
         {
             "滲水": 10, "漏水": 10, "滴水": 8, "滲漏": 9,
             "天花": 7, "牆身": 6, "外牆": 7,
             "樓上": 6, "樓下": 6,
             "水喉": 6, "喉管": 6, "去水": 5,
             "滲水辦": 8,
         },
         ["樓上漏水", "天花滴水", "外牆入水", "點樣驗滲水"], 4),

        ("pet_keeping",
         "飼養寵物",
         {
             "寵物": 10, "養狗": 10, "養貓": 9, "狗": 5, "貓": 4,
             "pet": 8, "dog": 6, "cat": 5,
             "禁養": 9, "禁止飼養": 10, "不准養": 9,
             "噪音": 5, "臭味": 6, "滋擾": 6,
             "公契": 7, "dmc": 7,
         },
         ["狗吠", "貓屎味", "鄰居養狗好嘈"], 3),

        ("fire_incident_safety",
         "火警",
         {
             "火警": 10, "著火": 9, "起火": 9, "火災": 9,
             "消防": 8, "消防設備": 9, "消防喉": 8, "消防栓": 8,
             "走火通道": 10, "逃生": 8, "防火門": 8, "防煙門": 7,
             "sprinkler": 7, "灑水": 7,
             "阻塞": 7, "封走火": 9,
             "fsd": 7, "fire safety": 8,
         },
         ["消防處", "走火通道被堵塞", "消防設備壞"], 4),

        ("electricity_supply",
         "電力供應",
         {
             "停電": 10, "冇電": 10, "斷電": 9, "供電": 8, "電力供應": 10,
             "跳掣": 9, "跳制": 9, "fuse": 6, "trip": 6,
             "電箱": 8, "配電": 7, "總掣": 7,
             "電錶": 8, "電表": 8,
             "中電": 5, "港燈": 5, "clp": 5, "hke": 5,
         },
         ["公用地方冇電", "成座大廈停電", "跳掣成日發生"], 4),
    ]

    PROPERTY_INTENT_UNKNOWN = "unknown"

    # 門檻（你可先用呢組，之後用 logs 調）
    PI_MIN_SCORE = 8              # 少於 8 分就當 unknown
    PI_MIN_CONFIDENCE = 0.55      # 少於 0.55 就當 unknown
    PI_AMBIGUOUS_MARGIN = 3       # top1 - top2 <= 3 視為 ambiguous -> unknown
    PI_MULTI_MIN_SCORE = 10       # second intent 都要夠強
    PI_MULTI_MARGIN = 2         # top1-top2 <= 2 視為 multi-intent
    PI_MULTI_RATIO = 0.65     # second intent score / top1 score >= 0.65

    scored: List[Tuple[str, int, Dict[str, int], str]] = []
    for intent_id, desc, kw_weights, any_bonus_kws, any_bonus_val in INTENTS:
        s, bd = _score_kw(q, kw_weights)
        any_hits = _hit_count(q, any_bonus_kws)
        if any_hits > 0:
            s += any_bonus_val
            bd[f"any_bonus({any_hits})"] = any_bonus_val
        scored.append((intent_id, s, bd, desc))

    

    scored.sort(key=lambda x: x[1], reverse=True)
    top_intent, top_score, top_bd, _ = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0
    margin = top_score - second_score

    # 置信度簡單 heuristic
    if top_score <= 0:
        conf = 0.20
    else:
        conf = 0.35 + min(0.6, (margin / 12.0) + (top_score / 60.0))
        conf = max(0.25, min(0.95, conf))

    candidates = [
        {"intent": i, "score": s, "breakdown": bd}
        for (i, s, bd, _) in scored[:5]
    ]

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    top1 = candidates[0] if candidates else {"intent": PROPERTY_INTENT_UNKNOWN, "score": 0, "breakdown": {}}
    top2 = candidates[1] if len(candidates) > 1 else {"intent": PROPERTY_INTENT_UNKNOWN, "score": 0, "breakdown": {}}

    score1 = int(top1.get("score", 0) or 0)
    score2 = int(top2.get("score", 0) or 0)
    margin = score1 - score2

    # 你原本 confidence 計法可保留；以下係一個簡單例子
    # 例如用 sigmoid / 或 score cap
    confidence = min(0.99, score1 / 20.0)  # 你可以用返你現有算法

    is_ambiguous = (margin <= PI_AMBIGUOUS_MARGIN and score1 > 0)
    is_weak = (score1 < PI_MIN_SCORE) or (confidence < PI_MIN_CONFIDENCE)

    # Multi-intent: top1 同 top2 都夠強，而且分數接近
    ratio = (score2 / score1) if score1 > 0 else 0.0

    is_multi = (
        (not is_weak) and
        (score2 >= PI_MULTI_MIN_SCORE) and
        (
            (margin <= PI_MULTI_MARGIN) or
            (ratio >= PI_MULTI_RATIO)
        ) and
        (top1["intent"] != PROPERTY_INTENT_UNKNOWN) and
        (top2["intent"] != PROPERTY_INTENT_UNKNOWN)
    )

    if is_weak:
        chosen_intent = PROPERTY_INTENT_UNKNOWN
        intents = [PROPERTY_INTENT_UNKNOWN]
    elif is_multi:
        chosen_intent = top1["intent"]
        intents = [top1["intent"], top2["intent"]]
    elif is_ambiguous:
        # ambiguous 但未夠強做 multi-intent，就落 unknown，避免誤判
        chosen_intent = PROPERTY_INTENT_UNKNOWN
        intents = [PROPERTY_INTENT_UNKNOWN]
    else:
        chosen_intent = top1["intent"]
        intents = [top1["intent"]]

    return {
        "intent": chosen_intent,
        "intents": intents,                 # NEW: Top-1 or Top-2 or ['unknown']
        "multi": bool(is_multi),            # NEW
        "score": score1,
        "confidence": float(confidence),
        "margin": margin,
        "ambiguous": bool(is_ambiguous),
        "weak": bool(is_weak),
        "breakdown": top1.get("breakdown", {}),
        "candidates": candidates[:5],
        "top2": {"intent": top2["intent"], "score": score2, "breakdown": top2.get("breakdown", {})},
    }

@dataclass
class RouteDecision:
    route: str                 # "ordinance" | "case" | "guideline" | "mixed"
    k: int = 5                  # final top-k after filtering
    fetch_k: int = 20           # retrieve this many first, then filter down
    filters: Optional[Dict[str, Any]] = None   # e.g. {"doc_type": "case"}
    extra: Dict[str, Any] = field(default_factory=dict)  # for any extra info

def rule_based_router(question: str) -> RouteDecision:
    q_raw = question or ""
    q = q_raw.lower()

    def hit_any(triggers: List[str]) -> bool:
        return any(t in q for t in triggers)

    def hit_any_pattern(patterns: List[str]) -> bool:
        return any(re.search(p, q) for p in patterns)

    def is_mc_query(q_lower: str) -> bool:
        if re.search(r"\bmc\b", q_lower):
            return True
        return any(trig in q_lower for trig in MC_INTERNAL_TRIGGERS if trig != "mc")

    def infer_meeting_topic(q_raw2: str) -> str:
        ql = (q_raw2 or "").lower()
        if any(s in ql for s in ["通告", "會議通告", "通知", "法定通知期", "14日", "7日", "notice", "notice period"]):
            return "notice_period"
        if any(s in ql for s in ["法定人數", "人數", "quorum"]):
            return "quorum"
        if any(s in ql for s in ["委任代表", "授權書", "授權", "proxy", "authorization"]):
            return "proxy"
        if any(s in ql for s in ["投票", "表決", "點票", "過半數", "業權份數", "resolution", "vote", "poll"]):
            return "voting_rules"
        if any(s in ql for s in ["會議紀錄", "紀錄", "minutes"]):
            return "minutes"
        return "owners_meeting_general"

    def infer_arrears_topic(q_raw2: str) -> str:
        ql = (q_raw2 or "").lower()
        if any(s in ql for s in ["催繳", "催收", "追數", "追款", "追討", "通知", "信", "demand letter", "reminder"]):
            return "demand_and_reminders"
        if any(s in ql for s in ["入稟", "起訴", "傳票", "判決", "判詞", "執行", "扣押", "上訴", "summons", "writ", "judgment", "execution", "court"]):
            return "legal_action"
        if any(s in ql for s in ["點計", "計算", "期間", "幾多", "多少", "how much", "calculation"]):
            return "arrears_definition_and_calculation"
        if any(s in ql for s in ["特別基金", "特別徵收", "特別徵費", "special levy"]):
            return "special_levy"
        if any(s in ql for s in ["欠交", "欠繳", "拖欠", "欠款", "欠費", "arrears", "outstanding"]):
            return "arrears_action_steps"
        return "arrears_general"

    def infer_mc_topic(q_raw2: str) -> str:
        ql = (q_raw2 or "").lower()
        if any(s in ql for s in ["召開", "開會", "會議程序", "議程", "會議紀錄", "minutes", "notice", "quorum", "法定人數", "主持", "主席"]):
            return "meeting_rules"
        if any(s in ql for s in ["權限", "可以做咩", "表決咩", "議案", "決議", "投票", "批准", "決定", "resolution", "vote", "approve"]):
            return "power_scope"
        if any(s in ql for s in ["責任", "職責", "工作", "duties", "responsibilit"]):
            return "duties"
        if any(s in ql for s in ["委員資格", "成員要求", "資格", "當選", "罷免", "停職", "resign", "remove", "disqualif"]):
            return "member_eligibility"
        return "general"

    # -----------------------------
    # 3) Decide route (first-layer)
    # -----------------------------
    rd: RouteDecision

    # (0) Hard case number
    if hit_any_pattern(CASE_NO_PATTERNS):
        rd = RouteDecision(route="case", filters={"doc_type": "case"})  # matches via doc_type OR category
        rd.extra["case_intent"] = infer_case_intent(q_raw).__dict__
    # (1) Hard ordinance citations
    elif hit_any_pattern(ORDINANCE_PATTERNS):
        rd = RouteDecision(route="ordinance", filters={"doc_type": "ordinance"})  # matches via doc_type OR category
    # (2) Case triggers
    elif hit_any(CASE_TRIGGERS):
        rd = RouteDecision(route="case", filters={"doc_type": "case"})  # matches via doc_type OR category
        rd.extra["case_intent"] = infer_case_intent(q_raw).__dict__
    else:
        # (3) Guideline vs ordinance scoring
        ordinance_score = 2 if hit_any(ORDINANCE_TRIGGERS) else 0
        guideline_score = 2 if hit_any(GUIDELINE_TRIGGERS) else 0

        if any(w in q for w in ["引用", "根據", "依照", "條文", "section", "s.", "cap"]):
            ordinance_score += 1
        if any(w in q for w in ["點樣做", "應該做咩", "有咩步驟", "流程", "清單", "範本", "表格"]):
            guideline_score += 1

        if ordinance_score > guideline_score and ordinance_score > 0:
            rd = RouteDecision(route="ordinance", filters={"doc_type": "ordinance"})  # matches via doc_type OR category
        elif guideline_score > ordinance_score and guideline_score > 0:
            rd = RouteDecision(route="guideline", filters={"doc_type": "guideline"})  # matches via doc_type OR category
        else:
            rd = RouteDecision(route="mixed", filters=None)

    # Auto-adjust retrieval sizing by route
    rd.fetch_k = ROUTE_FETCH_K.get(rd.route, rd.fetch_k)
    rd.k = ROUTE_TOP_K.get(rd.route, rd.k)

    # -----------------------------
    # 4) Always attach property_intent (for downstream use, but NO metadata filter)
    # -----------------------------
    rd.extra["property_intent"] = infer_property_intent(q_raw)
    # NOTE: Intent metadata filter DISABLED because JSONL docs use 'property_topic' not 'intent'.
    # The property_intent is still available in rd.extra for select_docs_for_evidence() to use.
    # To re-enable, ensure your docs have 'intent' metadata field matching infer_property_intent IDs.

    # -----------------------------
    # 5) Add subtopics (do not affect route)
    # -----------------------------
    sub = {}
    if hit_any(MEETING_TRIGGERS):
        sub["meeting_topic"] = infer_meeting_topic(q_raw)
    if hit_any(ARREARS_TRIGGERS):
        sub["arrears_topic"] = infer_arrears_topic(q_raw)
    if is_mc_query(q):
        sub["mc_topic"] = infer_mc_topic(q_raw)

    if sub:
        rd.extra["subtopics"] = sub

    return rd

def filter_docs(docs: List[Document], filters: Optional[Dict[str, Any]], k: int) -> List[Document]:
    if not filters:
        print(f"[FILTER_DOCS] No filters, returning first {k} of {len(docs)} docs")
        return docs[:k]

    # ========== DEBUG LOG START ==========
    print(f"[FILTER_DOCS] Applying filters={filters} to {len(docs)} docs, k={k}")
    # ========== DEBUG LOG END ==========

    def _expected_set(val) -> set:
        if isinstance(val, (list, tuple, set)):
            return {str(x) for x in val}
        return {str(val)}

    def match(d: Document) -> bool:
        md = d.metadata or {}
        doc_kind = _get_doc_kind(md)

        for key, val in (filters or {}).items():
            exp = _expected_set(val)

            # Compatibility layer:
            # - router historically uses filters={"doc_type": "guideline"}
            # - corpus often stores type in `category` (法律條文 / 指引 / court_case)
            # Therefore for key==doc_type or key==category we accept matches against inferred doc_kind.
            if key in {"doc_type", "category"}:
                # If expected looks like a kind label, compare against inferred kind
                exp_kind = {_norm_doc_kind(x) for x in exp}
                if any(k != "unknown" for k in exp_kind):
                    if doc_kind not in exp_kind:
                        return False
                    continue
                # Otherwise fall back to raw exact match for category/doc_type

            # Default: exact match on metadata key
            v = md.get(key)
            if str(v) not in exp:
                return False

        return True

    out = [d for d in docs if match(d)]

    # ========== DEBUG LOG START ==========
    print(f"[FILTER_DOCS] After filtering: {len(out)} docs matched")
    if len(out) == 0 and len(docs) > 0:
        print("[FILTER_DOCS] WARNING: All docs filtered out! Checking why...")
        for key, expected_val in filters.items():
            actual_vals = set()
            inferred_kinds = set()
            for d in docs[:10]:  # Check first 10
                md = d.metadata or {}
                actual_vals.add(str(md.get(key)))
                inferred_kinds.add(_get_doc_kind(md))
            print(
                f"[FILTER_DOCS]   Filter key='{key}', expected='{expected_val}', "
                f"actual values in docs (sample): {actual_vals}, inferred_kinds(sample): {inferred_kinds}"
            )
    # ========== DEBUG LOG END ==========

    return out[:k]


def routed_retrieve(retriever, decision: RouteDecision, question: str) -> Tuple[RouteDecision, List[Document]]:
    """
    retriever: base FAISS retriever
    decision: route + filters
    question: query string
    Returns: (decision, filtered_docs)
    """
    # retrieve more than needed, then filter down
    # Note: VectorStoreRetriever supports setting search_kwargs dynamically via .search_kwargs
    # but safest is to create a new retriever with search_kwargs each time if needed.
    # Here we assume the passed retriever already has a default k; we just over-fetch by calling invoke
    # and relying on its configured k. So ensure you create it with k=decision.fetch_k.
    docs = retriever.invoke(question)

    # ========== DEBUG LOG START ==========
    print(f"[ROUTED_RETRIEVE] question='{question[:50]}...'")
    print(f"[ROUTED_RETRIEVE] retriever returned {len(docs)} docs")
    print(f"[ROUTED_RETRIEVE] decision.filters={decision.filters}")
    print(f"[ROUTED_RETRIEVE] decision.k={decision.k}, decision.fetch_k={decision.fetch_k}")
    if docs:
        print("[ROUTED_RETRIEVE] Sample doc metadata (first 3):")
        for i, d in enumerate(docs[:3]):
            md = d.metadata or {}
            print(f"  [{i}] id={md.get('id')}, doc_type={md.get('doc_type')}, "
                  f"category={md.get('category')}, intent={md.get('intent')}, "
                  f"file_name={md.get('file_name')}")
    # ========== DEBUG LOG END ==========

    filtered = filter_docs(docs, decision.filters, decision.k)

    # ========== DEBUG LOG START ==========
    print(f"[ROUTED_RETRIEVE] after filter_docs: {len(filtered)} docs remain")
    # ========== DEBUG LOG END ==========

    return decision, filtered


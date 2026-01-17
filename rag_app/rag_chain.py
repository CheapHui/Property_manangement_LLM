from __future__ import annotations
import re
from typing import List, Optional, Dict, Tuple, Any, Generator
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage

from .citation_guard import build_allowed_set, make_retry_prompt, make_must_cite_retry_prompt, SAFE_FALLBACK
from .router import rule_based_router, routed_retrieve, RouteDecision, infer_property_intent
from .case_router import infer_case_intent, filter_and_rank_case_docs
from .prompts import RAG_PROMPT
from .schemas import RAGAnswer
from .coverage_guard import coverage_check

from .citation_select import shortlist_citations

from .definition_terms import detect_definition_term, get_all_definition_anchors, get_definition_keywords


# ---------- Chat History Formatting ----------

def format_chat_history(history: List[Dict[str, str]]) -> str:
    """Format chat history list into a string for the prompt.

    Args:
        history: List of previous turns, each with 'user' and 'assistant' keys

    Returns:
        Formatted string representation of the chat history
    """
    if not history:
        return "(無對話歷史)"

    lines = []
    for i, turn in enumerate(history, 1):
        user_msg = turn.get("user", "").strip()
        assistant_msg = turn.get("assistant", "").strip()
        # Truncate long messages to avoid overwhelming the context
        if len(user_msg) > 500:
            user_msg = user_msg[:500] + "..."
        if len(assistant_msg) > 800:
            assistant_msg = assistant_msg[:800] + "..."
        lines.append(f"[第{i}輪]")
        lines.append(f"用戶：{user_msg}")
        lines.append(f"助手：{assistant_msg}")
        lines.append("")
    return "\n".join(lines)


# ---------- doc_type normalization (ordinance/guideline/case aliases) ----------
# NOTE: Your data may store ordinance docs as Chinese labels like "法律條文".
# We normalize all variants to canonical doc_type strings used by routing/guards.

def get_raw_doc_type(md: Dict[str, Any]) -> str:
    """Return the raw doc-type token from metadata.

    Notes
    -----
    Your corpus mixes schemas and key casing (e.g. `category` vs `CATERGORY`).
    Also many docs currently store `doc_type: "unknown"` even when `category` is meaningful
    (e.g. `category: "法律條文"`).

    Rules:
    - Read keys case-insensitively.
    - Prefer `category` when `doc_type` is missing/placeholder.
    - Treat placeholder values (unknown/doc/empty) as missing.
    """

    if not md:
        return ""

    def _get_ci(*keys: str) -> str:
        # case-insensitive lookup + common typos
        for k in keys:
            if k in md and md.get(k) is not None:
                return str(md.get(k))
            lk = k.lower()
            uk = k.upper()
            if lk in md and md.get(lk) is not None:
                return str(md.get(lk))
            if uk in md and md.get(uk) is not None:
                return str(md.get(uk))
        return ""

    def _clean(v: str) -> str:
        s = (v or "").strip()
        if not s:
            return ""
        if s.lower() in {"unknown", "doc", "none", "null"}:
            return ""
        return s

    # Read values
    raw_doc_type = _clean(_get_ci("doc_type", "doctype", "DOC_TYPE"))
    raw_category = _clean(_get_ci("category", "CATEGORY", "CATERGORY", "CATERGORY"))
    raw_type = _clean(_get_ci("type", "TYPE"))

    # Prefer meaningful category when doc_type is placeholder/empty
    if raw_category and not raw_doc_type:
        return raw_category
    return raw_doc_type or raw_category or raw_type or ""


ORDINANCE_DOC_TYPE_ALIASES = {
    # Chinese
    "法律條文", "法例", "條例", "法規", "法例條文", "條文", "成文法", "法令", "規例", "附屬法例", "附屬規例",
    # English
    "ordinance", "legislation", "statute", "act", "law", "regulation", "regulations",
    "subsidiary legislation", "subsidiary law",
    "legal provision", "legal provisions", "provision", "provisions",
    # Domain-specific
    "bmo", "building management ordinance",
    # Sometimes stored as cap id
    "cap.344", "cap 344", "cap. 344", "cap  344",
}

GUIDELINE_DOC_TYPE_ALIASES = {
    # Chinese
    "指引", "守則", "實務指引", "實務守則", "指南", "程序", "程序指引", "範本", "樣本", "清單", "核對清單",
    # English
    "guideline", "guidelines", "guide", "checklist", "template", "sample", "code of practice", "practice guide",
}

CASE_DOC_TYPE_ALIASES = {
    # Chinese
    "案例", "判詞", "判決", "裁決", "判案", "裁定",
    # English
    "case", "judgment", "decision", "ruling",
}


def _norm_doc_type_token(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

# Precomputed normalized alias sets (module-level) to avoid rebuilding per call
ORDINANCE_DOC_TYPE_ALIASES_NORM = {_norm_doc_type_token(x) for x in ORDINANCE_DOC_TYPE_ALIASES}
GUIDELINE_DOC_TYPE_ALIASES_NORM = {_norm_doc_type_token(x) for x in GUIDELINE_DOC_TYPE_ALIASES}
CASE_DOC_TYPE_ALIASES_NORM = {_norm_doc_type_token(x) for x in CASE_DOC_TYPE_ALIASES}


def normalize_doc_type(raw: str) -> str:
    """Normalize metadata doc_type into canonical: ordinance|guideline|case|doc."""
    t = _norm_doc_type_token(raw)
    if not t:
        return "doc"

    # direct membership (normalized)
    if t in ORDINANCE_DOC_TYPE_ALIASES_NORM:
        return "ordinance"
    if t in GUIDELINE_DOC_TYPE_ALIASES_NORM:
        return "guideline"
    if t in CASE_DOC_TYPE_ALIASES_NORM:
        return "case"

    # soft contains / heuristics
    if any(k in t for k in ["ordinance", "legislation", "statute", " act", "regulation", "bmo", "cap."]):
        return "ordinance"
    if any(k in t for k in ["guideline", "checklist", "template", "code of practice", "practice"]):
        return "guideline"
    if any(k in t for k in ["judgment", "decision", "ruling", "case", "tribunal", "court"]):
        return "case"

    return "doc"

MUST_CITE_TRIGGERS = [
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

# MC evidence should be *committee-specific* (not OC/owners meeting / formation).
MC_CORE_TERMS = [
    # Chinese
    "管委會", "管理委員會", "管委會會議", "管理委員會會議",
    # English
    "management committee", "mc", "mc meeting", "meeting of management committee",
]

MC_ACTION_TERMS = [
    # Chinese
    "會議", "開會", "召開", "主持", "議程", "會議紀錄", "法定人數", "投票", "表決", "決議",
    "秘書", "委員", "通知", "通告",
    # English
    "meeting", "convene", "convened", "preside", "agenda", "minutes", "notice", "quorum", "resolution",
]

# Keep a single pack name for rerank / quick contains checks

MC_MUST_TERMS = MC_CORE_TERMS + MC_ACTION_TERMS

# Meeting procedure evidence (owners/OC meetings etc.)
MEETING_MUST_TERMS = [
    # Chinese
    "通告", "通知", "議程", "決議", "表決", "投票",
    "委任代表", "授權", "會議紀錄", "法定人數",
    # English
    "notice", "agenda", "resolution", "proxy", "minutes", "quorum",
]

# --- Exclude terms: formation / set-up owners meeting (keep OUT of MC evidence) ---
MC_EXCLUDE_FORMATION_TERMS = [
    "委出管理委員會", "委出管委會", "成立法團", "成立業主立案法團",
    "第3條", "3a", "3a條", "第3a條", "第4條", "首次業主會議", "業主會議以委出",
    "由不少於5%業權份數", "30%業權份數",  # formation voting thresholds often appear
    "召集人",  # very often in formation meeting context
    "building authority", "secretary for home and youth affairs",  # admin-trigger context
]

# --- Exclude terms: OC / owners meeting (not MC internal meeting) ---
MC_EXCLUDE_OWNERS_MEETING_TERMS = [
    "業主大會", "法團會議", "法團業主大會", "業主會議", "業主周年大會", "特別業主大會",
    "owners' meeting", "owners meeting", "general meeting", "annual general meeting", "agm", "egm",
]

MC_TOPIC_KEYWORDS = {
    "meeting_rules": [
        "召開", "開會", "主持", "主席", "秘書", "議程", "會議紀錄", "法定人數", "通知", "通告",
        "convene", "preside", "chairman", "agenda", "minutes", "quorum", "notice",
    ],
    "power_scope": [
        "權限", "職權", "可以", "議案", "決議", "表決", "投票", "批准", "決定",
        "power", "authority", "resolution", "vote", "approve", "decide",
    ],
    "duties": [
        "責任", "職責", "工作", "職務", "duty", "duties", "responsibility",
    ],
    "member_eligibility": [
        "資格", "條件", "委員", "成員", "當選", "罷免", "停職", "辭職",
        "eligible", "eligibility", "remove", "resign", "disqualify",
    ],
    "general": [],
}



MUST_CITE_RE = re.compile("|".join(MUST_CITE_TRIGGERS), re.IGNORECASE)

# ---------- Multi-intent two-pass retrieval hints ----------
# These are lightweight query expansions to bias vector retrieval toward each property intent.
INTENT_QUERY_HINTS: Dict[str, List[str]] = {
    # 公用部分 / 公用設施
    "common_parts_facilities_disputes": ["公用地方", "公用部分", "天台", "外牆", "大堂", "走廊", "升降機", "維修"],
    # 違建物
    "unauthorised_building_works": ["違建", "僭建", "unauthorised building works", "UBW", "清拆", "建築事務監督"],
    # 查帳 / 文件
    "access_documents_accounts": ["查帳", "帳目", "會計", "文件", "合約", "帳簿", "報價", "標書", "審計"],
    # 法團/管委會/物管運作
    "pmc_oc_mc_operations": ["法團", "管理公司", "管委會", "管理委員會", "交接", "監管", "運作"],
    # 管理費 / 維修費 / 費用
    "fees_management_maintenance_other": ["管理費", "維修費", "特別徵費", "special levy", "欠款", "拖欠"],
    # 辭退物管/交接
    "terminate_pmc_handover": ["辭退", "更換管理公司", "終止合約", "交接", "handover", "移交"],
    # 違反公契用途
    "breach_dmc_use": ["違反公契", "用途", "住宅用途", "商業用途", "dmc", "公契"],
    # 滲水
    "water_seepage": ["滲水", "漏水", "水漬", "天花", "外牆", "水管", "排水"],
    # 飼養寵物
    "pets": ["寵物", "狗", "貓", "pet", "噪音", "衛生", "公契禁止"],
    # 火警
    "fire_safety": ["火警", "消防", "fire safety", "走火通道", "消防設備", "喉轆", "警鐘"],
    # 電力供應
    "electricity_supply": ["停電", "電力", "供電", "電錶", "電線", "電房", "電力公司"],
}


def _doc_key(d: Document):
    md = d.metadata or {}
    return md.get("id") or md.get("chunk_id") or md.get("source_id") or (md.get("file_name"), md.get("chunk_index"))


def augment_query_for_intent(question: str, intent: str) -> str:
    """Light query expansion for two-pass retrieval."""
    base = (question or "").strip()
    it = (intent or "").strip().lower()
    hints = INTENT_QUERY_HINTS.get(it) or []
    if not hints:
        return base
    # Keep it short to avoid query drift
    extra = " ".join(hints[:6])
    return f"{base} {extra}".strip()


# ---------- citations / evidence helpers (MUST be global) ----------



def build_cit_payload(d: Document) -> str:
    """Return a *stable, minimal* CIT payload.

    Design goal:
    - Payload must be exact-matchable and easy for the model to copy.
    - It must be traceable: `id=` is the primary key.
    - Do NOT include optional fields (title/page/loc) to avoid mismatch/guard failures.

    Format:
      <doc_type> | id=<cit_id>
    """
    md = d.metadata or {}
    doc_type = normalize_doc_type(get_raw_doc_type(md) or "doc")

    cit_id = (
        md.get("id")
        or md.get("chunk_id")
        or md.get("source_id")
        or f"{md.get('file_name','unknown')}#{md.get('chunk_index','?')}"
    )

    return f"{doc_type} | id={cit_id}"


def build_evidence(docs: List[Document], max_chars: int = 9000) -> str:
    """Build an evidence pack with *traceable* CIT payloads.

    IMPORTANT:
    - `build_allowed_set()` extracts CIT payloads from `[CIT: ...]` tags.
    - Therefore every block MUST include `[CIT: <payload>]` and the payload MUST be stable and unique.
    """
    parts: List[str] = []
    used = 0

    for i, d in enumerate(docs, start=1):
        payload = build_cit_payload(d)
        md = d.metadata or {}

        # Small human-readable header (not used by guards)
        header_bits: List[str] = []
        dt = normalize_doc_type(get_raw_doc_type(md) or "doc")
        title = (md.get("title") or md.get("case_no") or md.get("display_title") or md.get("file_name") or "").strip()
        if title:
            header_bits.append(title)
        header = f"{dt} | " + " | ".join(header_bits) if header_bits else dt

        txt = (d.page_content or "").strip()
        block = (
            f"({i}) {header}\n"
            f"{txt}\n"
            f"[CIT: {payload}]\n"
        )

        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n".join(parts)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())



# Helper to derive workflow intent/topic/property_intent from router output
def derive_workflow_intent_and_topic(decision: Any) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Convert router decision extras into a workflow intent used by the downstream pipeline.

    Returns:
      (workflow_intent, workflow_topic, property_intent)

    - workflow_intent keeps backward compatibility with special handling (mc_internal / meeting_procedure / arrears_recovery)
      but is derived from `extra["subtopics"]` instead of being a first-layer router decision.
    - property_intent is the second-layer category (11-class) from `extra["property_intent"].intent`.
    """
    extra = (getattr(decision, "extra", None) or {}) if not isinstance(decision, dict) else (decision.get("extra") or {})

    # second-layer property intent
    pi = extra.get("property_intent") or {}
    property_intent = pi.get("intent")

    # subtopics (do NOT override route; only for workflow specialization)
    sub = extra.get("subtopics") or {}

    # Backward-compatible workflow intent
    if "mc_topic" in sub:
        return "mc_internal", (sub.get("mc_topic") or "general"), property_intent
    if "meeting_topic" in sub:
        return "meeting_procedure", (sub.get("meeting_topic") or "owners_meeting_general"), property_intent
    if "arrears_topic" in sub:
        return "arrears_recovery", (sub.get("arrears_topic") or "arrears_general"), property_intent

    # Default workflow intent: none (use generic evidence strategy)
    return None, "general", property_intent


# ---------- Decision Guidance (Phase 2.2) ----------
DECISION_MODE_TRIGGERS = [
    "點跟", "點做", "下一步", "應該點", "如何決定", "點樣決定", "風險", "取捨",
    "比較", "揀", "選擇", "策略", "先做邊樣", "優先次序",
]

def detect_decision_mode(question: str) -> bool:
    ql = (question or "").lower()
    return any(t in ql for t in DECISION_MODE_TRIGGERS)

# ---------- Definition / concept questions (Definition Mode) ----------
DEFINITION_MODE_TRIGGERS = [
    "乜嘢係", "咩係", "係乜", "係咩", "什麼是", "甚麼是", "定義", "意思", "解釋", "點理解",
]

DEFINITION_MODE_TERMS = [
    "公用部分", "公用地方", "公共地方", "公契", "dmc", "deed of mutual covenant",
    "法團", "業主立案法團", "owners corporation", "管理委員會", "管委會",
]

def detect_definition_mode(question: str) -> bool:
    ql = (question or "").lower()
    if any(t.lower() in ql for t in DEFINITION_MODE_TRIGGERS) and any(k.lower() in ql for k in DEFINITION_MODE_TERMS):
        return True
    return False

# ---------- main chain ----------

def make_rag_chain(fetch_retriever, llm, final_k: int = 5, return_runner: bool = False):

    parser = PydanticOutputParser(pydantic_object=RAGAnswer)
    format_instructions = parser.get_format_instructions()

    RETRY_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "你係香港物業管理法律/實務助手。必須輸出符合指定 JSON schema。\n"
         "引用規則：只可使用【證據】內原樣出現過的 [CIT: ...]，不可自創 CIT。\n"
         "如果【證據】內沒有任何 [CIT: ...]，citations_used 必須為空陣列。\n"
        ),
        ("human", "{retry_text}\n\n{format_instructions}")
    ])
    # ---------- Post-retrieval guards layer  ----------
    def must_hit_filter_docs(docs, must_hits):
        out = []
        for d in docs:
            t = (d.page_content or "").lower()
            if any(k.lower() in t for k in must_hits):
                out.append(d)
        return out
    
    def select_docs_for_evidence(
        intent: Optional[str],
        base_docs: List[Document],
        final_k: int,
        topic: Optional[str] = None,
        property_intents: Optional[List[str]] = None,
        definition_mode: bool = False,
    ) -> List[Document]:
        """
        Post-retrieval evidence selector (quota + guards).
        - Strict fail-closed for mc_internal: if no MC evidence after exclude + must-hit (MC terms + topic terms) => return []
        """

        def _doc_type(d: Document) -> str:
            md = d.metadata or {}
            return normalize_doc_type(get_raw_doc_type(md) or "doc")

        def _key(d: Document):
            md = d.metadata or {}
            return md.get("id") or md.get("chunk_id") or md.get("source_id") or (
                md.get("file_name"), md.get("chunk_index")
            )

        def _dedupe_keep_order(docs: List[Document]) -> List[Document]:
            seen = set()
            out = []
            for d in docs:
                k = _key(d)
                if k in seen:
                    continue
                seen.add(k)
                out.append(d)
            return out

        def _must_hit_filter(docs: List[Document], must_terms: List[str]) -> List[Document]:
            must_terms_l = [t.lower() for t in must_terms if t]
            out = []
            for d in docs:
                t = (d.page_content or "").lower()
                if any(mt in t for mt in must_terms_l):
                    out.append(d)
            return out

        def _exclude_if_hit(docs: List[Document], exclude_terms: List[str]) -> List[Document]:
            ex_l = [t.lower() for t in exclude_terms if t]
            out = []
            for d in docs:
                t = (d.page_content or "").lower()
                if any(ex in t for ex in ex_l):
                    continue
                out.append(d)
            return out

        def _take_by_type_with_quota(docs: List[Document], quotas: Dict[str, int]) -> List[Document]:
            picked: List[Document] = []
            used = set()

            def _count(dt: str) -> int:
                return sum(1 for x in picked if _doc_type(x) == dt)

            # quota pass
            for dt, n in quotas.items():
                if n <= 0:
                    continue
                for d in docs:
                    if _count(dt) >= n:
                        break
                    if _doc_type(d) != dt:
                        continue
                    k = _key(d)
                    if k in used:
                        continue
                    used.add(k)
                    picked.append(d)

            # backfill to final_k
            for d in docs:
                if len(picked) >= final_k:
                    break
                k = _key(d)
                if k in used:
                    continue
                used.add(k)
                picked.append(d)

            return picked[:final_k]
        
        def _take_by_intent_with_quota(docs: List[Document], intents: List[str], k: int) -> List[Document]:
            """Balance evidence across multiple property intents (OR filter)."""
            intents_l = [i.strip().lower() for i in intents if i]
            if not intents_l:
                return docs[:k]

            buckets: Dict[str, List[Document]] = {i: [] for i in intents_l}
            for d in docs:
                di = ((d.metadata or {}).get("intent") or "").strip().lower()
                if di in buckets:
                    buckets[di].append(d)

            per = max(1, k // len(intents_l))
            out: List[Document] = []
            used = set()

            def _key(d: Document):
                md = d.metadata or {}
                return md.get("id") or md.get("chunk_id") or md.get("source_id") or (md.get("file_name"), md.get("chunk_index"))

            def _add(ds: List[Document], n: int):
                nonlocal out
                for d in ds:
                    if len(out) >= k or n <= 0:
                        break
                    kk = _key(d)
                    if kk in used:
                        continue
                    used.add(kk)
                    out.append(d)
                    n -= 1

            # quota per intent
            for it in intents_l:
                _add(buckets.get(it, []), per)

            # fill remaining by original order
            if len(out) < k:
                for d in docs:
                    if len(out) >= k:
                        break
                    kk = _key(d)
                    if kk in used:
                        continue
                    used.add(kk)
                    out.append(d)

            return out[:k]

        # ---- normalize + dedupe early ----
        base_docs = _dedupe_keep_order(base_docs)

        # =========================
        # mc_meeting_procedure (STRICT fail-closed)
        # =========================
        

        if intent == "mc_internal":
            # 1) exclude formation / "set up OC" meeting docs
            non_formation = _exclude_if_hit(base_docs, MC_EXCLUDE_FORMATION_TERMS)

            # 2) exclude OC / owners meeting docs (MC internal != owners/OC meeting)
            non_owners_meeting = _exclude_if_hit(non_formation, MC_EXCLUDE_OWNERS_MEETING_TERMS)

            # 3) STRICT must-hit: require at least one MC core term (committee-specific)
            mc_core_docs = _must_hit_filter(non_owners_meeting, MC_CORE_TERMS)
            if not mc_core_docs:
                return []

            # 4) STRICT must-hit: require topic terms (if any); otherwise require MC action/procedure terms
            topic_key = (topic or "general")
            topic_terms = MC_TOPIC_KEYWORDS.get(topic_key, [])
            must_terms = (topic_terms + MC_ACTION_TERMS) if topic_terms else MC_ACTION_TERMS
            mc_docs = _must_hit_filter(mc_core_docs, must_terms)
            if not mc_docs:
                return []

            # 5) quota: guidelines dominate; ordinance if available; case optional
            guide_q = max(2, int(final_k * 0.6))
            ord_q = max(1, int(final_k * 0.25))
            if guide_q + ord_q > final_k:
                ord_q = max(1, final_k - guide_q)
            case_q = max(0, final_k - guide_q - ord_q)
            quotas = {
                "guideline": guide_q,
                "ordinance": ord_q,
                "case": case_q,
            }

            picked = _take_by_type_with_quota(mc_docs, quotas)

            # final strictness: ensure the final pack still contains MC core terms
            if not _must_hit_filter(picked, MC_CORE_TERMS):
                return []

            return picked[:final_k]

        # =========================
        # arrears_recovery (existing sample logic)
        # =========================
        if intent == "arrears_recovery":
            must_hits = [
                "管理費", "欠款", "拖欠", "欠繳", "追討", "追收", "催繳",
                "arrears", "outstanding", "debt", "recover", "recovery",
                "special levy", "levy", "contribution",
            ]
            hit_docs = _must_hit_filter(base_docs, must_hits)
            if not hit_docs:
                return []
            quotas = {
                "ordinance": max(1, final_k // 4),
                "case": max(1, final_k // 4),
                "guideline": max(1, final_k - 2),
            }
            return _take_by_type_with_quota(hit_docs, quotas)

        # =========================
        # meeting_procedure (existing sample logic)
        # =========================
        if intent == "meeting_procedure":
            meeting_terms = [
                "通告", "通知", "議程", "決議", "會議", "召開", "至少", "14天", "14日",
                "notice", "meeting", "agenda", "resolution", "quorum",
            ]
            meeting_docs = _must_hit_filter(base_docs, meeting_terms) or base_docs
            case_q = 2 if final_k >= 6 else 1
            ord_q = 2 if final_k >= 6 else 1
            guide_q = max(1, final_k - case_q - ord_q)
            quotas = {"case": case_q, "ordinance": ord_q, "guideline": guide_q}
            return _take_by_type_with_quota(meeting_docs, quotas)

        # =========================
        # default strategy
        # =========================
        if intent is None and property_intents and len(property_intents) >= 2:
            base_docs = _take_by_intent_with_quota(base_docs, property_intents[:2], final_k)

        if definition_mode:
            # Definition / concept questions: heavily bias toward ordinance definitions
            # Target: 80% ordinance, 20% case/guideline for context
            ord_q = max(3, int(final_k * 0.8))  # at least 3, or 80% of final_k
            case_q = 0  # no case docs for definition questions
            guide_q = max(0, final_k - ord_q)  # remaining slots for guidelines
            quotas = {"ordinance": ord_q, "case": case_q, "guideline": guide_q}

            # Debug: print quota allocation
            print(f"[DEF_EVIDENCE] quota: ord={ord_q}, case={case_q}, guide={guide_q}, final_k={final_k}")

            return _take_by_type_with_quota(base_docs, quotas)

        quotas = {
            "ordinance": 1 if final_k >= 4 else 0,
            "case": 1 if final_k >= 4 else 0,
            "guideline": max(1, final_k - (1 if final_k >= 4 else 0) - (1 if final_k >= 4 else 0)),
        }
        return _take_by_type_with_quota(base_docs, quotas)

    def route_and_retrieve(x):
        q = x["question"]

        decision = rule_based_router(q)

        decision_mode = detect_decision_mode(q)
        decision.extra["decision_mode"] = bool(decision_mode)

        # --- Definition Mode detection + term extraction ---
        definition_mode = detect_definition_mode(q)
        definition_term = detect_definition_term(q) if definition_mode else None
        definition_terms = (
            (get_definition_keywords(definition_term) if definition_term else [])
            + (get_all_definition_anchors() if definition_mode else [])
        )
        decision.extra["definition_mode"] = bool(definition_mode)
        decision.extra["definition_term"] = definition_term

        # Debug: print definition mode detection
        print(f"[DEF_MODE] detected={definition_mode}, term={definition_term}, question='{q}'")
        # --- Definition Mode: prefetch ordinance-heavy evidence (then merge) ---
        ordinance_prefetch_docs: List[Document] = []

        def _make_retriever(k: int):
            """Create/configure a retriever with the requested overfetch k."""
            r = fetch_retriever
            if callable(fetch_retriever):
                try:
                    r = fetch_retriever(int(k))
                except TypeError:
                    r = fetch_retriever()
            else:
                # Best-effort: set VectorStoreRetriever search_kwargs
                try:
                    if hasattr(r, "search_kwargs") and isinstance(r.search_kwargs, dict):
                        r.search_kwargs["k"] = int(k)
                except Exception:
                    pass
            return r
        if definition_mode:
            # Avoid over-restricting by metadata filters for definition questions
            if decision.filters:
                # intent filter is disabled globally; doc_type filter is unreliable in this corpus
                decision.filters.pop("intent", None)
                decision.filters.pop("doc_type", None)

            # IMPORTANT:
            # Your ordinance JSON uses `category: 法律條文` (not `doc_type: ordinance`).
            # Many vectorstores only support exact-match metadata filters, so filtering by
            # doc_type="ordinance" would accidentally return 0 docs.
            # Therefore, in definition_mode we do a broad high-k fetch with an ordinance-biased query,
            # then rely on `normalize_doc_type(get_raw_doc_type(...))` inside `select_docs_for_evidence`
            # to pick ordinance-heavy evidence.

            ord_fetch_k = max(getattr(decision, "fetch_k", 20), 60)
            ord_retriever = _make_retriever(ord_fetch_k)

            # Ordinance-biased query expansion (term-aware, keep short to avoid drift)
            term_hint = " ".join((definition_terms or [])[:10]).strip()
            ord_q = f"{q} {term_hint}".strip()

            try:
                ordinance_prefetch_docs = ord_retriever.invoke(ord_q)
                print("[DEF_Prefetch] Retrieved ordinance documents = ", ordinance_prefetch_docs)
                for d in ordinance_prefetch_docs:
                    md = d.metadata if d else {}
                    print("  - id=", md.get("id"), "category=", md.get("category"),
                    "doc_type=", md.get("doc_type"), "chapter=", md.get("chapter"),
                    "len=", len(d.page_content or ""))
            except Exception:
                ordinance_prefetch_docs = []

            # Record debug metadata
            decision.extra.setdefault("retrieval", {})
            decision.extra["retrieval"]["definition_prefetch"] = True
            decision.extra["retrieval"]["definition_query"] = ord_q
            decision.extra["retrieval"]["definition_fetch_k"] = ord_fetch_k

        # Derive workflow intent/topic from subtopics; keep property_intent for logging/metadata
        wf_intent, wf_topic, prop_intent = derive_workflow_intent_and_topic(decision)

        if wf_intent is not None:
            # workflow 問題：唔好用 property_intent filter 鎖死檢索
            if decision.filters and decision.filters.get("intent"):
                decision.filters.pop("intent", None)

        # Property intent is used for downstream evidence selection, NOT for metadata filtering.
        # (Intent metadata filter DISABLED because JSONL docs use 'property_topic' not 'intent')
        pi = (decision.extra.get("property_intent") or {})
        pi_intents_raw = pi.get("intents") or [pi.get("intent")]
        pi_intents = [str(x).strip().lower() for x in pi_intents_raw if x]

        pi_score = int(pi.get("score") or 0)
        pi_conf = float(pi.get("confidence") or 0.0)

        PI_MIN_SCORE = 10
        PI_MIN_CONF = 0.60

        if bool(pi.get("multi")):
            decision.fetch_k = max(decision.fetch_k, 35)

        is_unknown = (not pi_intents) or all(x in ("", "unknown") for x in pi_intents)

        # NOTE: Intent metadata filter DISABLED - always remove if present
        if decision.filters and decision.filters.get("intent"):
            decision.filters.pop("intent", None)

        # --- unknown intent behavior flags (for downstream prompt / logging) ---
        unknown_intent = (not pi_intents) or all(x in ("", "unknown") for x in pi_intents) or (pi_score < PI_MIN_SCORE) or (pi_conf < PI_MIN_CONF)

        # --- Chat History Intent Fallback ---
        # If current question has unknown/low-confidence intent but there's chat history,
        # use the previous turn's question to infer context
        chat_history = x.get("chat_history") or []
        used_history_fallback = False

        if unknown_intent and chat_history:
            # Get the last turn's question
            last_turn = chat_history[-1] if chat_history else None
            if last_turn and last_turn.get("user"):
                prev_question = last_turn.get("user", "").strip()
                if prev_question:
                    # Infer property intent from previous question
                    prev_pi = infer_property_intent(prev_question)
                    prev_pi_intents = prev_pi.get("intents") or [prev_pi.get("intent")]
                    prev_pi_intents = [str(x).strip().lower() for x in prev_pi_intents if x]
                    prev_pi_score = int(prev_pi.get("score") or 0)
                    prev_pi_conf = float(prev_pi.get("confidence") or 0.0)

                    prev_is_unknown = (not prev_pi_intents) or all(x in ("", "unknown") for x in prev_pi_intents)

                    # Only use fallback if previous turn had a known intent
                    if not prev_is_unknown and prev_pi_score >= PI_MIN_SCORE:
                        print(f"[HISTORY_FALLBACK] Current question has unknown intent (score={pi_score}, conf={pi_conf:.2f})")
                        print(f"[HISTORY_FALLBACK] Using previous turn's intent: {prev_pi_intents} (score={prev_pi_score}, conf={prev_pi_conf:.2f})")
                        print(f"[HISTORY_FALLBACK] Previous question: '{prev_question[:50]}...'")

                        # Update with fallback intent (use slightly lower confidence to indicate it's inferred)
                        pi = prev_pi
                        pi_intents = prev_pi_intents
                        pi_score = prev_pi_score
                        pi_conf = prev_pi_conf * 0.8  # Discount confidence slightly for fallback
                        unknown_intent = False
                        used_history_fallback = True

                        # Update decision.extra with fallback intent
                        decision.extra["property_intent"] = pi
                        decision.extra["property_intent"]["from_history_fallback"] = True

        decision.extra["unknown_intent"] = bool(unknown_intent)
        decision.extra["used_history_fallback"] = used_history_fallback

        # ========== DYNAMIC FOLLOW-UP SUGGESTIONS ==========
        # 根據 property_intent 和問題內容動態生成相關的 follow-up 建議
        def generate_follow_up_suggestions(question: str, property_intent: str, route: str) -> List[str]:
            """
            根據用戶問題和識別到的 intent 生成相關的 follow-up 建議，
            讓用戶有興趣繼續深入了解。
            """
            q_lower = (question or "").lower()
            suggestions = []

            # Intent-specific follow-up suggestions
            INTENT_FOLLOW_UPS = {
                "common_parts_facilities_disputes": [
                    "想了解法團處理公用地方爭議嘅法定程序？",
                    "如果有業主霸佔公用地方，法團可以點樣處理？",
                    "公用設施維修責任點樣分配？",
                    "想知道相關嘅案例判決？",
                ],
                "unauthorised_building_works": [
                    "法團對違建有咩法定責任？",
                    "如果業主唔肯清拆違建，法團可以點做？",
                    "想了解屋宇署嘅執法程序？",
                    "有冇相關嘅違建案例參考？",
                ],
                "access_documents_accounts": [
                    "業主查閱文件嘅法定權利係咩？",
                    "法團拒絕提供文件，業主可以點做？",
                    "想了解核數報告嘅要求？",
                    "管理公司要保存咩文件？保存幾耐？",
                ],
                "pmc_oc_mc_operations": [
                    "想深入了解法團嘅法定職責？",
                    "管委會嘅權限範圍係咩？",
                    "法團同管理公司嘅關係點樣界定？",
                    "想知道法團運作嘅常見問題同解決方法？",
                ],
                "fees_management_maintenance_other": [
                    "管理費點樣計算？有冇法定標準？",
                    "業主欠交管理費，法團可以點追討？",
                    "特別徵費需要咩程序先可以收取？",
                    "想了解管理費爭議嘅案例？",
                ],
                "terminate_pmc_and_handover": [
                    "辭退管理公司需要咩程序？",
                    "交接期間要注意咩事項？",
                    "管理公司唔肯交接，法團可以點做？",
                    "想了解相關嘅法例規定？",
                ],
                "breach_dmc_use": [
                    "公契入面一般有咩常見限制？",
                    "如果有人違反公契，法團可以點處理？",
                    "想了解公契詮釋嘅案例？",
                    "Airbnb / 劏房算唔算違反公契？",
                ],
                "water_seepage": [
                    "滲水責任點樣界定？",
                    "想了解滲水辦嘅調查程序？",
                    "樓上唔合作，可以點做？",
                    "有冇滲水索償嘅案例參考？",
                ],
                "pet_keeping": [
                    "公契通常點樣規管寵物？",
                    "如果鄰居嘅寵物造成滋擾，可以點投訴？",
                    "法團可以禁止飼養寵物嗎？",
                    "想了解寵物相關嘅案例？",
                ],
                "fire_incident_safety": [
                    "法團對消防安全有咩法定責任？",
                    "走火通道被阻塞，可以點處理？",
                    "消防設備維修責任係邊個？",
                    "想了解消防安全嘅法例要求？",
                ],
                "electricity_supply": [
                    "公用地方電力供應責任係邊個？",
                    "如果經常跳掣，應該點處理？",
                    "想了解電力供應嘅法例規定？",
                    "電力設施維修費用點樣分攤？",
                ],
            }

            # Route-specific follow-ups
            ROUTE_FOLLOW_UPS = {
                "ordinance": [
                    "想了解更多相關嘅法例條文？",
                    "呢條法例有冇相關嘅案例解釋？",
                ],
                "guideline": [
                    "想要更詳細嘅操作步驟？",
                    "有冇相關嘅範本或表格？",
                ],
                "case": [
                    "想了解更多類似嘅案例？",
                    "呢個案例對實務操作有咩啟示？",
                ],
            }

            # Get intent-specific suggestions
            if property_intent and property_intent != "unknown":
                intent_suggestions = INTENT_FOLLOW_UPS.get(property_intent, [])
                # Filter out suggestions that are too similar to the original question
                for s in intent_suggestions:
                    # Simple similarity check: skip if >50% of words overlap
                    s_words = set(s.replace("？", "").replace("、", " ").split())
                    q_words = set(q_lower.replace("？", "").replace("、", " ").split())
                    overlap = len(s_words & q_words) / max(len(s_words), 1)
                    if overlap < 0.5:
                        suggestions.append(s)
                    if len(suggestions) >= 2:
                        break

            # Add route-specific suggestions
            route_suggestions = ROUTE_FOLLOW_UPS.get(route, [])
            for s in route_suggestions:
                if s not in suggestions and len(suggestions) < 3:
                    suggestions.append(s)

            # Fallback if no suggestions
            if not suggestions:
                suggestions = [
                    "想了解更多相關嘅法例規定？",
                    "有冇相關嘅案例可以參考？",
                ]

            return suggestions[:3]  # Max 3 suggestions

        # Generate dynamic follow-up suggestions (always, not just for unknown_intent)
        pi_intent_for_followup = (decision.extra.get("property_intent") or {}).get("intent") or ""
        decision.extra["follow_up_suggestions"] = generate_follow_up_suggestions(q, pi_intent_for_followup, decision.route)

        # Keep clarifying_questions for backward compatibility, but use follow_up_suggestions
        decision.extra["clarifying_questions"] = decision.extra["follow_up_suggestions"]
        # ========== END DYNAMIC FOLLOW-UP SUGGESTIONS ==========
    

        DEBUG = True
        if DEBUG:
            pi = (decision.extra.get("property_intent") or {})
            print(
                "[ROUTE]", decision.route,
                "[WF_INTENT]", wf_intent,
                "[WF_TOPIC]", wf_topic,
                "[PROPERTY_INTENT]", pi.get("intent"),
                "[PI_CONF]", pi.get("confidence"),
                "[PI_SCORE]", pi.get("score"),
            )

        # --- IMPORTANT: metadata filters can wipe out all docs ---
        # Our corpus often stores type under `category` (e.g. 法律條文 / 指引 / court_case) while `doc_type` may be
        # missing/"unknown". The router may set filters like {"doc_type": "guideline"} which then matches 0 docs.
        # Therefore we disable doc_type filtering here and rely on soft rerank + evidence selector to enforce type quotas.
        if decision.filters and decision.filters.get("doc_type"):
            print(f"[FILTER_GUARD] Removing doc_type filter to avoid empty retrieval: {decision.filters}")
            decision.filters.pop("doc_type", None)
        # Ensure we over-fetch according to decision.fetch_k.
        # Support either a retriever object or a factory/callable that accepts k.
        retriever = _make_retriever(decision.fetch_k)

        # --- Retrieval ---
        # If we have multi-intent intent filters (Top-2) and NO workflow specialization,
        # do two-pass retrieval with light query expansion, then merge + dedupe.
        intent_filter = (decision.filters or {}).get("intent")
        if (wf_intent is None) and isinstance(intent_filter, list) and len(intent_filter) >= 2:
            i1, i2 = intent_filter[0], intent_filter[1]
            q1 = augment_query_for_intent(q, i1)
            q2 = augment_query_for_intent(q, i2)
            try:
                docs1 = retriever.invoke(q1)
            except Exception:
                docs1 = []
            try:
                docs2 = retriever.invoke(q2)
            except Exception:
                docs2 = []

            merged: List[Document] = []
            seen = set()
            for d in (docs1 + docs2):
                k = _doc_key(d)
                if k in seen:
                    continue
                seen.add(k)
                merged.append(d)

            base_docs = merged
            # Keep decision as-is (filters kept for logging), but we already did retrieval.
            decision.extra.setdefault("retrieval", {})
            decision.extra["retrieval"]["two_pass"] = True
            decision.extra["retrieval"]["queries"] = [q1, q2]
        else:
            decision, base_docs = routed_retrieve(retriever, decision, q)
            decision.extra.setdefault("retrieval", {})
            decision.extra["retrieval"]["two_pass"] = False

        if ordinance_prefetch_docs:
            merged2: List[Document] = []
            seen2 = set()
            for d in (ordinance_prefetch_docs + base_docs):
                kk = _doc_key(d)
                if kk in seen2:
                    continue
                seen2.add(kk)
                merged2.append(d)
            base_docs = merged2

        def soft_rerank_by_hits(docs, terms):
            terms_l = [t.lower() for t in terms]
            scored = []
            for d in docs:
                t = (d.page_content or "").lower()
                score = sum(1 for kw in terms_l if kw in t)
                scored.append((score, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [d for s, d in scored]

        # ========== SOFT RERANK BY PROPERTY INTENT ==========
        # Boost/penalize docs based on property_intent match, doc_type, and keyword hits
        def soft_rerank_by_property_intent(
            docs: List[Document],
            property_intent: str,
            question: str,
            boost_intent_match: float = 10.0,      # +X if property_topic matches intent
            boost_ordinance_keyword: float = 8.0,  # +Y if ordinance + keyword hit
            penalty_case_no_keyword: float = 3.0,  # -Z if case + litigation only, no keyword
        ) -> List[Document]:
            """
            Soft rerank docs by property intent relevance.

            Rules:
            1. property_topic 包含對應 intent → +boost_intent_match
            2. doc_type=ordinance 且 content 命中 question keywords → +boost_ordinance_keyword
            3. doc_type=case 且只有訴訟關鍵字而冇 question keywords → -penalty_case_no_keyword
            """
            if not docs:
                return docs

            # Extract question keywords for matching
            q_lower = (question or "").lower()

            # Intent to property_topic mapping (approximate)
            INTENT_TO_TOPIC_KEYWORDS = {
                "common_parts_facilities_disputes": ["common_parts", "公用", "facilities", "設施"],
                "unauthorised_building_works": ["unauthorized", "僭建", "違建", "ubw"],
                "access_documents_accounts": ["document", "文件", "帳目", "account", "audit"],
                "pmc_oc_mc_operations": ["oc", "mc", "法團", "管委會", "management", "operation", "權力", "powers", "職責", "duties"],
                "fees_management_maintenance_other": ["fee", "管理費", "維修費", "levy"],
                "terminate_pmc_and_handover": ["terminate", "辭退", "handover", "交接"],
                "breach_dmc_use": ["dmc", "公契", "breach", "違反", "用途"],
                "water_seepage": ["seepage", "滲水", "漏水", "water"],
                "pet_keeping": ["pet", "寵物", "狗", "貓"],
                "fire_incident_safety": ["fire", "火警", "消防", "safety"],
                "electricity_supply": ["electricity", "電力", "停電", "供電"],
            }

            # Question-derived keywords (權力/powers example)
            POWER_KEYWORDS = ["權力", "權限", "職權", "powers", "power", "authority", "職責", "責任", "duties", "duty"]
            LITIGATION_KEYWORDS = ["訴訟", "起訴", "入稟", "sue", "lawsuit", "court", "legal action", "傳票", "summons"]

            # Check if question is about powers/duties
            is_power_question = any(kw in q_lower for kw in POWER_KEYWORDS)

            # Get intent-specific keywords
            intent_keywords = INTENT_TO_TOPIC_KEYWORDS.get(property_intent, [])

            scored: List[tuple] = []
            for d in docs:
                md = d.metadata or {}
                content_lower = (d.page_content or "").lower()
                doc_type = normalize_doc_type(get_raw_doc_type(md) or "doc")

                # Base score (can use existing score if available)
                base_score = float(md.get("score", 0.0) or 0.0)
                boost = 0.0

                # Rule 1: property_topic matches intent keywords
                prop_topics = md.get("property_topic") or []
                if isinstance(prop_topics, str):
                    prop_topics = [prop_topics]
                prop_topics_str = " ".join(prop_topics).lower()

                intent_match = any(kw.lower() in prop_topics_str for kw in intent_keywords)
                if intent_match:
                    boost += boost_intent_match

                # Rule 2: ordinance + keyword hit (for power questions)
                if doc_type == "ordinance" and is_power_question:
                    keyword_hit = any(kw in content_lower for kw in POWER_KEYWORDS)
                    if keyword_hit:
                        boost += boost_ordinance_keyword

                # Rule 3: case with only litigation keywords, no power keywords → penalty
                if doc_type == "case" and is_power_question:
                    has_litigation = any(kw in content_lower for kw in LITIGATION_KEYWORDS)
                    has_power = any(kw in content_lower for kw in POWER_KEYWORDS)
                    if has_litigation and not has_power:
                        boost -= penalty_case_no_keyword

                final_score = base_score + boost
                scored.append((final_score, boost, d))

            # Sort by final score descending
            scored.sort(key=lambda x: x[0], reverse=True)

            # Debug log
            print(f"[SOFT_RERANK] property_intent={property_intent}, is_power_question={is_power_question}")
            for i, (fs, b, d) in enumerate(scored[:5]):
                md = d.metadata or {}
                dt = normalize_doc_type(get_raw_doc_type(md) or "doc")
                print(f"  [{i}] boost={b:+.1f}, final={fs:.1f}, doc_type={dt}, id={md.get('id', '?')[:30]}")

            return [d for (fs, b, d) in scored]
        # ========== END SOFT RERANK ==========

        # Use workflow intent/topic derived from subtopics for specialized handling
        intent = wf_intent
        topic = wf_topic

        if intent == "meeting_procedure":
            base_docs = soft_rerank_by_hits(base_docs, MEETING_MUST_TERMS)

        if intent == "mc_internal":
            topic_terms = MC_TOPIC_KEYWORDS.get(topic, [])
            base_docs = soft_rerank_by_hits(base_docs, MC_CORE_TERMS + MC_ACTION_TERMS + topic_terms)

        # Apply soft rerank by property intent (always, as a general boost layer)
        pi_intent = (decision.extra.get("property_intent") or {}).get("intent") or ""
        if pi_intent and pi_intent != "unknown":
            base_docs = soft_rerank_by_property_intent(base_docs, pi_intent, q)

        # ========== SOFT RERANK BY CATEGORY ==========
        def soft_rerank_by_category(
            docs: List[Document],
            route: str,
            question: str,
            boost_ordinance: float = 15.0,    # 法律傾向加權最重
            boost_guideline: float = 10.0,    # 實務指引
            boost_case: float = 8.0,          # 案件參考
            boost_keyword_match: float = 5.0, # 額外關鍵字命中加分
        ) -> List[Document]:
            """
            根據問題傾向 (route) soft boost 對應 CATEGORY 的文檔。

            法律傾向權重最高，因為法律條文係最權威嘅來源。

            Route → Category mapping:
            - ordinance → 法律條文, ordinance, bmo, law (boost +15)
            - guideline → 指引, guideline, checklist (boost +10)
            - case      → court_case, case, judgment (boost +8)
            - mixed     → 根據問題關鍵字動態決定
            """
            if not docs:
                return docs

            q_lower = (question or "").lower()

            # Category aliases (你嘅 JSONL 可能用唔同嘅值)
            ORDINANCE_CATEGORIES = {"法律條文", "ordinance", "bmo", "law", "legislation", "statute", "條例"}
            GUIDELINE_CATEGORIES = {"指引", "guideline", "checklist", "guide", "實務指引", "守則", "範本"}
            CASE_CATEGORIES = {"court_case", "case", "judgment", "案例", "判詞", "裁決"}

            # 問題關鍵字 → 用於 mixed route 或額外加分
            # 擴展版：涵蓋香港物業管理常見用語（中英文、廣東話、書面語）
            ORDINANCE_KEYWORDS = [
                # ===== 法例引用 =====
                "條例", "法例", "cap", "cap.", "section", "s.", "第", "條", "附表", "schedule",
                "法律", "法定", "規定", "bmo", "building management ordinance", "建築物管理條例",
                "物業管理服務條例", "pmso", "property management services ordinance",

                # ===== 法律概念 / 權責 =====
                "權力", "權限", "職權", "職責", "責任", "義務", "權利", "法律地位",
                "powers", "duties", "authority", "obligation", "rights", "legal status",
                "授權", "委託", "delegat", "empower", "entitle",

                # ===== 法團 / 法律主體 =====
                "法團", "業主立案法團", "owners corporation", "oc", "業主組織",
                "管理委員會", "管委會", "mc", "management committee",
                "公契", "dmc", "deed of mutual covenant", "大廈公契",

                # ===== 法律程序 / 效力 =====
                "合法", "違法", "法律責任", "法律效力", "法律後果", "legally",
                "生效", "有效", "無效", "撤銷", "void", "valid", "invalid",
                "強制", "mandatory", "compulsory", "必須", "須",

                # ===== 法定要求 =====
                "法定人數", "quorum", "法定程序", "法定期限", "法定通知",
                "過半數", "majority", "業權份數", "undivided shares",
                "特別決議", "普通決議", "special resolution", "ordinary resolution",

                # ===== 違規 / 執法 =====
                "違反", "違規", "breach", "contravene", "違例",
                "罰則", "罰款", "penalty", "fine", "sanction",
                "禁止", "不得", "prohibit", "forbidden",
            ]

            GUIDELINE_KEYWORDS = [
                # ===== 操作指引 =====
                "點做", "點樣", "如何", "怎樣", "怎麼", "應該點",
                "程序", "步驟", "流程", "實務", "指引", "守則",
                "procedure", "process", "step", "how to", "guide", "guideline",

                # ===== 範本 / 文件 =====
                "checklist", "範本", "表格", "樣本", "模板", "格式",
                "template", "sample", "form", "format", "example",

                # ===== 申請 / 手續 =====
                "申請", "手續", "方法", "做法", "辦理", "處理",
                "application", "apply", "submit", "filing",

                # ===== 時限 / 期限 =====
                "期限", "限期", "deadline", "time limit", "幾耐", "幾多日",
                "14日", "7日", "21日", "28日", "通知期", "notice period",

                # ===== 實務操作 =====
                "準備", "安排", "籌備", "執行", "implement", "arrange", "prepare",
                "注意事項", "要點", "tips", "best practice", "建議",
                "常見問題", "faq", "q&a",

                # ===== 會議相關 =====
                "召開會議", "開會", "會議通知", "議程", "agenda", "notice of meeting",
                "會議紀錄", "minutes", "投票", "表決", "vote", "poll",
                "委任代表", "proxy", "授權書",
            ]

            CASE_KEYWORDS = [
                # ===== 案件類型 =====
                "案例", "判詞", "裁決", "案件", "個案", "判例",
                "case", "precedent", "judgment", "ruling", "decision",

                # ===== 法庭 / 審裁處 =====
                "法庭", "法院", "審裁處", "土地審裁處", "lands tribunal",
                "高等法院", "區域法院", "上訴庭", "終審法院",
                "high court", "district court", "court of appeal", "court of final appeal",

                # ===== 案件編號 =====
                "ldbm", "cacv", "hca", "facv", "hcmp", "dccj", "cacc", "caci",
                "案件編號", "case no", "case number",

                # ===== 訴訟程序 =====
                "上訴", "appeal", "覆核", "review", "司法覆核", "judicial review",
                "原告", "被告", "申請人", "答辯人",
                "plaintiff", "defendant", "applicant", "respondent",

                # ===== 判決結果 =====
                "勝訴", "敗訴", "駁回", "dismiss", "allow", "granted",
                "判令", "命令", "order", "injunction", "禁制令",
                "賠償", "damages", "compensation", "訟費", "costs",

                # ===== 法律爭議 =====
                "爭議", "糾紛", "dispute", "controversy",
                "索償", "claim", "追討", "sue", "訴訟", "litigation",
                "仲裁", "調解", "arbitration", "mediation",
            ]

            # 計算問題傾向強度
            ordinance_q_hits = sum(1 for kw in ORDINANCE_KEYWORDS if kw in q_lower)
            guideline_q_hits = sum(1 for kw in GUIDELINE_KEYWORDS if kw in q_lower)
            case_q_hits = sum(1 for kw in CASE_KEYWORDS if kw in q_lower)

            # ===== N-gram 提取函數（用於 Rule 4）=====
            def extract_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> set:
                """從文本中提取 N-gram（2-4字的詞組）"""
                # 移除標點和空格，只保留中英文
                clean = re.sub(r'[^\w\u4e00-\u9fff]', '', text.lower())
                ngrams = set()
                for n in range(min_n, max_n + 1):
                    for i in range(len(clean) - n + 1):
                        ngram = clean[i:i+n]
                        # 過濾純數字
                        if not ngram.isdigit():
                            ngrams.add(ngram)
                return ngrams

            # 預先提取問題的 N-gram（只做一次，不在 loop 內重複）
            q_ngrams = extract_ngrams(question)

            scored: List[tuple] = []
            for d in docs:
                md = d.metadata or {}
                content_lower = (d.page_content or "").lower()

                # 獲取 category (支援大小寫和 typo)
                cat = (
                    md.get("category") or
                    md.get("CATEGORY") or
                    md.get("CATERGORY") or  # typo in your data
                    md.get("doc_type") or
                    ""
                ).lower().strip()

                # Base score
                base_score = float(md.get("score", 0.0) or 0.0)
                boost = 0.0
                boost_reasons = []

                # ===== Rule 1: Route-based category boost =====
                if route == "ordinance":
                    if cat in {c.lower() for c in ORDINANCE_CATEGORIES}:
                        boost += boost_ordinance
                        boost_reasons.append(f"route_ordinance_match:+{boost_ordinance}")
                elif route == "guideline":
                    if cat in {c.lower() for c in GUIDELINE_CATEGORIES}:
                        boost += boost_guideline
                        boost_reasons.append(f"route_guideline_match:+{boost_guideline}")
                elif route == "case":
                    if cat in {c.lower() for c in CASE_CATEGORIES}:
                        boost += boost_case
                        boost_reasons.append(f"route_case_match:+{boost_case}")
                elif route == "mixed":
                    # Mixed route: 根據問題關鍵字動態 boost
                    if ordinance_q_hits > 0 and cat in {c.lower() for c in ORDINANCE_CATEGORIES}:
                        dynamic_boost = boost_ordinance * (ordinance_q_hits / max(ordinance_q_hits, guideline_q_hits, case_q_hits, 1))
                        boost += dynamic_boost
                        boost_reasons.append(f"mixed_ordinance:+{dynamic_boost:.1f}")
                    if guideline_q_hits > 0 and cat in {c.lower() for c in GUIDELINE_CATEGORIES}:
                        dynamic_boost = boost_guideline * (guideline_q_hits / max(ordinance_q_hits, guideline_q_hits, case_q_hits, 1))
                        boost += dynamic_boost
                        boost_reasons.append(f"mixed_guideline:+{dynamic_boost:.1f}")
                    if case_q_hits > 0 and cat in {c.lower() for c in CASE_CATEGORIES}:
                        dynamic_boost = boost_case * (case_q_hits / max(ordinance_q_hits, guideline_q_hits, case_q_hits, 1))
                        boost += dynamic_boost
                        boost_reasons.append(f"mixed_case:+{dynamic_boost:.1f}")

                # ===== Rule 2: Content keyword match bonus =====
                # 如果文檔內容命中問題關鍵字，額外加分
                if route in ("ordinance", "mixed") and ordinance_q_hits > 0:
                    content_hits = sum(1 for kw in ORDINANCE_KEYWORDS if kw in content_lower)
                    if content_hits >= 2:
                        boost += boost_keyword_match
                        boost_reasons.append(f"content_ordinance_kw:+{boost_keyword_match}")

                if route in ("guideline", "mixed") and guideline_q_hits > 0:
                    content_hits = sum(1 for kw in GUIDELINE_KEYWORDS if kw in content_lower)
                    if content_hits >= 2:
                        boost += boost_keyword_match
                        boost_reasons.append(f"content_guideline_kw:+{boost_keyword_match}")

                # ===== Rule 3: 法律文檔額外加權（因為最權威）=====
                # 即使 route 唔係 ordinance，法律文檔都有基礎加分
                if cat in {c.lower() for c in ORDINANCE_CATEGORIES}:
                    base_legal_boost = boost_ordinance * 0.3  # 30% 基礎加分
                    boost += base_legal_boost
                    boost_reasons.append(f"legal_base:+{base_legal_boost:.1f}")

                # ===== Rule 4: 動態 N-gram 標題/節名匹配加權 =====
                # 使用預先提取的 q_ngrams，與文檔 section/title 匹配
                # 不需要維護關鍵字列表，自動適應新 Topic
                section_title = (
                    (md.get("section") or "") + " " +
                    (md.get("title") or "") + " " +
                    (md.get("display_title") or "")
                ).lower()

                if section_title.strip() and q_ngrams:
                    # 計算匹配數量（使用預先提取的 q_ngrams）
                    title_matches = [ng for ng in q_ngrams if ng in section_title and len(ng) >= 2]

                    if title_matches:
                        # 根據匹配數量和長度給分
                        # 長詞匹配（4字）比短詞（2字）更有價值
                        title_boost = 0.0
                        for match in title_matches:
                            if len(match) >= 4:
                                title_boost += 5.0  # 4字以上匹配 +5
                            elif len(match) >= 3:
                                title_boost += 3.0  # 3字匹配 +3
                            else:
                                title_boost += 1.5  # 2字匹配 +1.5

                        # 設置上限避免過度加權
                        title_boost = min(title_boost, 15.0)

                        if title_boost > 0:
                            boost += title_boost
                            # 只顯示最長的匹配（避免 log 太長）
                            best_match = max(title_matches, key=len) if title_matches else ""
                            boost_reasons.append(f"title_ngram({best_match}):+{title_boost:.1f}")

                final_score = base_score + boost
                scored.append((final_score, boost, boost_reasons, d))

            # Sort by final score descending
            scored.sort(key=lambda x: x[0], reverse=True)

            # Debug log
            print(f"[SOFT_RERANK_CATEGORY] route={route}, q_hits: ord={ordinance_q_hits}, guide={guideline_q_hits}, case={case_q_hits}")
            for i, (fs, b, reasons, d) in enumerate(scored[:8]):
                md = d.metadata or {}
                cat = md.get("category") or md.get("CATEGORY") or md.get("doc_type") or "?"
                print(f"  [{i}] boost={b:+.1f}, final={fs:.1f}, cat={cat}, reasons={reasons[:2]}, id={str(md.get('id', '?'))[:25]}")

            return [d for (fs, b, r, d) in scored]
        # ========== END SOFT RERANK BY CATEGORY ==========

        # Apply soft rerank by category based on route
        base_docs = soft_rerank_by_category(base_docs, decision.route, q)

        pi_for_pack = (decision.extra.get("property_intent") or {})
        prop_intents_for_pack = pi_for_pack.get("intents") or ([pi_for_pack.get("intent")] if pi_for_pack.get("intent") else [])

        # ========== DEBUG LOG START ==========
        print(f"[SELECT_DOCS] BEFORE select_docs_for_evidence:")
        print(f"[SELECT_DOCS]   base_docs count={len(base_docs)}")
        print(f"[SELECT_DOCS]   intent={intent}, topic={topic}, final_k={final_k}")
        print(f"[SELECT_DOCS]   definition_mode={definition_mode}")
        print(f"[SELECT_DOCS]   prop_intents_for_pack={prop_intents_for_pack}")
        if base_docs:
            print("[SELECT_DOCS]   Sample base_docs metadata (first 5):")
            for i, d in enumerate(base_docs[:5]):
                md = d.metadata or {}
                print(f"     [{i}] id={md.get('id')}, doc_type={md.get('doc_type')}, "
                      f"category={md.get('category')}, intent={md.get('intent')}")
        # ========== DEBUG LOG END ==========

        docs = select_docs_for_evidence(
            intent,
            base_docs,
            final_k,
            topic=topic,
            property_intents=prop_intents_for_pack,
            definition_mode=definition_mode,
        )

        # ========== DEBUG LOG START ==========
        print(f"[SELECT_DOCS] AFTER select_docs_for_evidence: {len(docs)} docs selected")
        # ========== DEBUG LOG END ==========

        if decision.route == "case" and intent not in ("meeting_procedure", "arrears_recovery"):
            case_dec = infer_case_intent(q)
            refined = filter_and_rank_case_docs(base_docs, case_dec, k=final_k)

            if refined:
                docs = refined
                decision.extra["case_intent"] = case_dec.case_intent
                decision.extra["case_match"] = "matched"
            else:
                decision.extra["case_intent"] = case_dec.case_intent
                decision.extra["case_match"] = "no_direct_match"
                if case_dec.case_intent == "debt_recovery":
                    docs = []
                    decision.extra["case_match"] = "no_direct_match_hard"
                else:
                    docs = base_docs[:final_k]

        # Add workflow intent/topic/property_intent to decision.extra for downstream use
        decision.extra.setdefault("workflow", {})
        decision.extra["workflow"]["intent"] = intent
        decision.extra["workflow"]["topic"] = topic
        decision.extra["workflow"]["property_intent"] = prop_intent

        return {"question": q, "decision": decision.__dict__, "docs": docs}

    def add_evidence(x):
        evidence = build_evidence(x["docs"])
        docs = x["docs"]
        route_info = x["decision"]
        extra = route_info.get("extra") or {}
        intent = (extra.get("workflow") or {}).get("intent")

        # Debug: print evidence composition
        def _doc_type(d: Document) -> str:
            md = d.metadata or {}
            return normalize_doc_type(get_raw_doc_type(md) or "doc")

        type_counts = {}
        for d in docs:
            dt = _doc_type(d)
            type_counts[dt] = type_counts.get(dt, 0) + 1

        print(f"[EVIDENCE_COMPOSITION] total={len(docs)}, breakdown={type_counts}")

        if intent == "mc_internal" and not docs:
            route_info.setdefault("extra", {})["no_direct_evidence"] = True

        # --- Decision Guidance mode flag ---
        decision_mode = bool((route_info.get("extra") or {}).get("decision_mode"))

        # --- Definition mode flag (for citation shortlist) ---
        definition_mode = bool((route_info.get("extra") or {}).get("definition_mode"))

        candidates = shortlist_citations(
            evidence=evidence,
            question=x["question"],
            intent=intent,
            top_n=5,
            definition_mode=definition_mode,
        )

        # --- Definition mode: ensure ordinance citations are available to prevent legal-source mismatch ---
        definition_mode = bool((route_info.get("extra") or {}).get("definition_mode"))
        if definition_mode:
            allowed = list(build_allowed_set(evidence))
            ord_allowed = [c for c in allowed if _norm(c).lower().startswith("ordinance |")]
            # If shortlist didn't pick any ordinance citations, force-add a few ordinance CITs from evidence.
            if ord_allowed and not any(_norm(c).lower().startswith("ordinance |") for c in candidates):
                # Keep deterministic order and avoid duplicates
                for oc in ord_allowed[:5]:
                    if oc not in candidates:
                        candidates.append(oc)

        pi = (route_info.get("extra") or {}).get("property_intent") or {}
        unknown_intent = bool((route_info.get("extra") or {}).get("unknown_intent"))
        clarifying_questions = (route_info.get("extra") or {}).get("clarifying_questions") or []
        property_intents = pi.get("intents") or ([pi.get("intent")] if pi.get("intent") else [])
        multi_intent = bool(pi.get("multi")) and len(property_intents) >= 2

        # Debug safety: candidates must come from evidence (tolerate wrapped citations)
        allowed_set = set(build_allowed_set(evidence))
        def _cit_payload_local(c: str) -> str:
            s = _norm(c)
            m = re.search(r"\[CIT:\s*(.*?)\s*\]", s, flags=re.IGNORECASE)
            return _norm(m.group(1)) if m else s
        if any(_cit_payload_local(c) not in allowed_set for c in candidates):
            print("[WARN] citation_candidates contains payload not in evidence")

        # Debug: print citation candidates
        print(f"[CITATION_CANDIDATES] count={len(candidates)}")
        for i, c in enumerate(candidates[:10], 1):
            doc_type_prefix = c.split("|")[0].strip() if "|" in c else "?"
            print(f"  {i}. [{doc_type_prefix}] {c[:80]}")

        # Format chat history for the prompt
        history = x.get("chat_history") or []
        chat_history_str = format_chat_history(history) if history else "(無對話歷史)"

        return {
            "question": x["question"],
            "evidence": evidence,
            "route_info": route_info,
            "format_instructions": format_instructions,
            "citation_candidates": candidates,
            "property_intent": pi.get("intent"),
            "unknown_intent": unknown_intent,
            "property_intents": property_intents,
            "multi_intent": multi_intent,
            "clarifying_questions": clarifying_questions,
            "decision_mode": decision_mode,
            "chat_history": chat_history_str,
        }

    def to_text(msg):
        return getattr(msg, "content", str(msg))

    def citation_guard_and_retry(x):
        question = x["question"]
        evidence = x["evidence"]
        route_info = x.get("route_info") or {}
        extra = (route_info.get("extra") or {})
        wf = extra.get("workflow") or {}
        intent = wf.get("intent")
        topic = wf.get("topic")
        prop_intent = (wf.get("property_intent") or "").strip().lower()
        is_unknown_intent = (prop_intent in ("", "unknown")) or bool(extra.get("unknown_intent"))
        pi_meta = extra.get("property_intent") or {}
        is_multi_intent = bool(pi_meta.get("multi")) and len((pi_meta.get("intents") or [])) >= 2
        decision_mode_flag = bool(x.get("decision_mode"))
        definition_mode_flag = bool((route_info.get("extra") or {}).get("definition_mode"))

        allowed = {_norm(a) for a in build_allowed_set(evidence)}
        def _cit_payload(c: str) -> str:
            """Normalize a citation token to payload-only.

            The model sometimes outputs citations as '[CIT: <payload>]' instead of just '<payload>'.
            Our guards/candidates/allowed sets store the payload only, so strip the wrapper here.
            """
            s = _norm(c)
            m = re.search(r"\[CIT:\s*(.*?)\s*\]", s, flags=re.IGNORECASE)
            return _norm(m.group(1)) if m else s

        def validate_obj(obj, *, strict: bool = False) -> List[str]:
            problems: List[str] = []

            # 0) MC fail-closed (keep early & strict)
            if intent == "mc_internal" and (not evidence or not evidence.strip()):
                if not any(s in (obj.answer_summary or "") for s in ("證據不足", "資料不足")):
                    problems.append("MC_FAIL_CLOSED_MISSING_INSUFFICIENT_EVIDENCE_MSG")
                if obj.key_points:
                    problems.append("MC_FAIL_CLOSED_KEY_POINTS_NOT_EMPTY")
                if obj.procedure_checklist:
                    problems.append("MC_FAIL_CLOSED_CHECKLIST_NOT_EMPTY")
                if obj.citations_used:
                    problems.append("MC_FAIL_CLOSED_CITATIONS_NOT_EMPTY")
                return problems

            # --- Ordinance claim regex (hard vs soft) ---
            # Hard claims require an ordinance citation: Cap.344 / section/s. / schedules.
            ORDINANCE_HARD_CLAIM_RE = re.compile(
                r"(第\s*344\s*章|\bcap\.?\s*344\b|第\s*\d+\s*條|\bsection\s*\d+\b|\bs\.?\s*\d+\b|附表\s*\d+|\bschedule\s*\d+)",
                re.IGNORECASE,
            )
            # Soft mentions (BMO name only) do not require an ordinance citation by themselves.
            ORDINANCE_SOFT_MENTION_RE = re.compile(
                r"(建築物管理條例|building\s+management\s+ordinance|\bbmo\b)",
                re.IGNORECASE,
            )

            # 1) Collect ALL citations (top-level + Decision Guidance fields), normalize to payload-only
            all_cits: List[str] = []
            all_cits.extend([_cit_payload(c) for c in (obj.citations_used or []) if c])

            # next_best_actions.citations
            try:
                for a in (obj.next_best_actions or []):
                    all_cits.extend([_cit_payload(c) for c in (getattr(a, "citations", None) or []) if c])
            except Exception:
                pass

            # decision_tree.citations
            try:
                for b in (obj.decision_tree or []):
                    all_cits.extend([_cit_payload(c) for c in (getattr(b, "citations", None) or []) if c])
            except Exception:
                pass

            # 1) decision_mode=False => do NOT populate decision guidance fields
            if (not decision_mode_flag):
                if (obj.decision_frame or []) or (obj.next_best_actions or []) or (obj.decision_tree or []):
                    problems.append("DECISION_FIELDS_NOT_ALLOWED_WHEN_DECISION_MODE_FALSE")

            # 1a) evidence-only guard
            bad_cits = [c for c in all_cits if _norm(c) not in allowed]
            if bad_cits:
                problems.append(f"DISALLOWED_CIT:{bad_cits[:5]}")

            # 1b) MUST-CITE guard (legal/procedure claims require at least one valid citation somewhere)
            def _join(xs: List[str]) -> str:
                return " ".join([_norm(x) for x in xs if x])

            nba_text = ""
            try:
                nba_text = " ".join([
                    (getattr(a, "action", "") or "") + " " + (getattr(a, "why", "") or "")
                    for a in (obj.next_best_actions or [])
                ])
            except Exception:
                nba_text = ""

            dt_text = ""
            try:
                dt_text = " ".join([
                    (getattr(b, "if_condition", "") or "") + " " + (getattr(b, "then_action", "") or "") + " " + (getattr(b, "else_action", "") or "")
                    for b in (obj.decision_tree or [])
                ])
            except Exception:
                dt_text = ""

            combined_text = " ".join([
                obj.answer_summary or "",
                _join(obj.key_points or []),
                _join(obj.procedure_checklist or []),
                _join(obj.decision_frame or []),
                _join(obj.required_facts or []),
                _join(obj.clarifying_questions or []),
                nba_text,
                dt_text,
            ])

            # 1c) Legal-source mismatch guard:
            # If the answer claims BMO/Cap.344/sections/schedules but provides no ordinance citations, fail.
            has_hard_ordinance_claim = bool(ORDINANCE_HARD_CLAIM_RE.search(combined_text))
            # Ordinance citation is recognized by normalized doc_type in CIT payload
            has_ordinance_cit = any(
                _norm(c).lower().startswith("ordinance |") and (_norm(c) in allowed)
                for c in all_cits
            )
            if has_hard_ordinance_claim and (not has_ordinance_cit):
                problems.append("LEGAL_SOURCE_MISMATCH_NO_ORDINANCE_CIT")

            # 1d) Definition mode guard:
            # If definition_mode=True and citation_candidates contain ordinance citations,
            # the model MUST use at least one ordinance citation.
            cands = {_cit_payload(c) for c in (x.get("citation_candidates") or [])}
            has_ordinance_cand = any(_norm(c).lower().startswith("ordinance |") for c in cands)
            if definition_mode_flag and has_ordinance_cand and (not has_ordinance_cit):
                problems.append("DEFINITION_MODE_MISSING_ORDINANCE_CIT")

            if (not is_unknown_intent) and MUST_CITE_RE.search(combined_text) and (not all_cits):
                problems.append("MUST_CITE_NO_CITATIONS")

            # 2) coverage guard (single source of truth)
            # Definition mode: relax coverage requirements (ordinance definitions are often short)
            if definition_mode_flag:
                min_hits, min_unique = 1, 1
            elif is_unknown_intent:
                min_hits, min_unique = 1, 1
            elif is_multi_intent and intent is None:
                # multi-intent generic answers: evidence split across intents
                min_hits, min_unique = 2, 1
            else:
                min_hits, min_unique = 2, 2

            ok_cov, reasons = coverage_check(
                answer_summary=obj.answer_summary,
                key_points=obj.key_points,
                citations_used=[c.strip() for c in (obj.citations_used or [])],
                evidence=evidence,
                intent=intent,
                topic=topic,
                min_hits=min_hits,
                min_unique_terms=min_unique,
            )
            if not ok_cov:
                problems.extend(reasons)

            # 3) shortlist guard (ALWAYS): model may only copy from citation_candidates (normalize to payload-only)
            cands = {_cit_payload(c) for c in (x.get("citation_candidates") or [])}
            bad_not_in_cands = [c for c in all_cits if _cit_payload(c).strip() not in cands]
            if bad_not_in_cands:
                problems.append(f"NOT_IN_CANDIDATES:{bad_not_in_cands[:5]}")

            no_id = [c for c in all_cits if "id=" not in c]
            if no_id:
                problems.append(f"CIT_MISSING_ID:{no_id[:5]}")  

            return problems

        # attempt 1
        msg1 = llm.invoke(RAG_PROMPT.invoke(x))
        raw1 = to_text(msg1)

        try:
            obj1 = parser.parse(raw1)
            probs1 = validate_obj(obj1)
            if probs1:
                print("[VALIDATION FAIL]", probs1)
                # optional: print citations_used
                try:
                    print("[citations_used]", obj1.citations_used)
                except Exception:
                    pass
            if not probs1:
                return obj1.model_dump_json(indent=2)
        except Exception:
            probs1 = ["PARSER_ERROR"]

        # ---- 🔧 只在嚴重違規時重試（減少重試頻率） ----
        CRITICAL_ERRORS = [
            "MUST_CITE_NO_CITATIONS",
            "LEGAL_SOURCE_MISMATCH_NO_ORDINANCE_CIT",
            "DECISION_FIELDS_NOT_ALLOWED_WHEN_DECISION_MODE_FALSE",
            "DISALLOWED_CIT",
            "NOT_IN_CANDIDATES",
            "CIT_MISSING_ID",
            "PARSER_ERROR",
        ]

        has_critical_error = any(
            any(str(p).startswith(err) for err in CRITICAL_ERRORS)
            for p in (probs1 or [])
        )

        # 如果只有輕微違規，直接返回第一次的結果
        if not has_critical_error:
            print("[VALIDATION MINOR ISSUES - ACCEPTING]", probs1)
            return obj1.model_dump_json(indent=2)

        # Dedicated retry prompts for common validation failures
        print("[VALIDATION CRITICAL ERROR - RETRYING]", probs1)
        if any(str(p).startswith("MUST_CITE_NO_CITATIONS") for p in (probs1 or [])):
            retry_text = make_must_cite_retry_prompt(question, evidence)
        elif any(str(p).startswith("LEGAL_SOURCE_MISMATCH_NO_ORDINANCE_CIT") for p in (probs1 or [])):
            # If we don't have ordinance evidence, the model must avoid ordinance/section claims.
            # If we DO have ordinance evidence, the model MUST include at least one ordinance citation.
            cands = x.get("citation_candidates") or []
            has_ord_cand = any(_norm(c).lower().startswith("ordinance |") for c in cands)
            if has_ord_cand and definition_mode_flag:
                retry_text = (
                    "你上一個回答包含具體條文/Cap.344/附表/section/s. 等聲稱，但沒有提供任何 ordinance 類 [CIT: ...]。\n"
                    "你必須完全重寫答案，並遵守：\n"
                    "1) 只可使用【證據】內原樣出現的 [CIT: ...]，而且 citations_used 只能從 citation_candidates 複製。\n"
                    "2) 你必須在 key_points 或 answer_summary 至少一次使用一個 ordinance 類 CIT（payload 以 'ordinance |' 開頭）。\n"
                    "3) 每當你提到條文號、附表、section、s. 等，必須在句尾加上對應 ordinance CIT；如證據不足，必須寫『證據不足』。\n\n"
                    f"用戶問題：{question}\n\n【證據】\n{evidence}\n"
                )
            else:
                retry_text = (
                    "你上一個回答包含具體條文/Cap.344/附表/section/s. 等聲稱，但【證據】內沒有任何 ordinance 類引用可用。\n"
                    "你必須完全重寫答案，並遵守：\n"
                    "1) 只可使用【證據】內原樣出現的 [CIT: ...]。\n"
                    "2) 不可提及任何條文號、Cap.344、附表、section、s. 等字眼；\n"
                    "   只能以【證據】內容作事實描述，或明確寫『證據不足』。\n\n"
                    f"用戶問題：{question}\n\n【證據】\n{evidence}\n"
                )
        elif any(str(p).startswith("DECISION_FIELDS_NOT_ALLOWED_WHEN_DECISION_MODE_FALSE") for p in (probs1 or [])):
            retry_text = (
                "你上一個回答在 decision_mode=False 情況下輸出了 decision_frame/next_best_actions/decision_tree。\n"
                "你必須完全重寫答案，並遵守：\n"
                "- decision_frame、next_best_actions、decision_tree 必須為空陣列 []。\n"
                "- 只可輸出 required_facts 同 clarifying_questions（如需要）。\n"
                "- 引用規則照舊：只可使用【證據】內原樣出現的 [CIT: ...]。\n\n"
                f"用戶問題：{question}\n\n【證據】\n{evidence}\n"
            )
        else:
            retry_text = make_retry_prompt(question, evidence, probs1)
        msg2 = llm.invoke(RETRY_PROMPT.invoke({
            "retry_text": retry_text,
            "format_instructions": x["format_instructions"],
        }))
        raw2 = to_text(msg2)

        try:
            obj2 = parser.parse(raw2)
        except Exception:
            return SAFE_FALLBACK

        probs2 = validate_obj(obj2, strict=True)
        if probs2:
            return SAFE_FALLBACK

        return obj2.model_dump_json(indent=2)

    def _collect_stream_text(prompt_value) -> Generator[Tuple[str, str], None, str]:
        """Yield ("token", text) while streaming and finally return the full concatenated text."""
        buf: List[str] = []
        for chunk in llm.stream(prompt_value):
            txt = getattr(chunk, "content", None)
            if txt:
                buf.append(txt)
                yield ("token", txt)
        return "".join(buf)

    def citation_guard_and_retry_stream(x) -> Generator[Tuple[str, str], None, str]:
        """Streaming version of citation_guard_and_retry.

        Yields:
          ("token", text)  - streamed tokens for the current attempt
          ("reset", "")     - emitted once if attempt 1 fails validation and we retry

        Returns:
          final JSON string (same as citation_guard_and_retry)
        """
        question = x["question"]
        evidence = x["evidence"]
        route_info = x.get("route_info") or {}
        extra = (route_info.get("extra") or {})
        wf = extra.get("workflow") or {}
        intent = wf.get("intent")
        topic = wf.get("topic")
        prop_intent = (wf.get("property_intent") or "").strip().lower()
        is_unknown_intent = (prop_intent in ("", "unknown")) or bool(extra.get("unknown_intent"))
        pi_meta = extra.get("property_intent") or {}
        is_multi_intent = bool(pi_meta.get("multi")) and len((pi_meta.get("intents") or [])) >= 2
        decision_mode_flag = bool(x.get("decision_mode"))
        definition_mode_flag = bool((route_info.get("extra") or {}).get("definition_mode"))

        allowed = {_norm(a) for a in build_allowed_set(evidence)}

        def _cit_payload(c: str) -> str:
            s = _norm(c)
            m = re.search(r"\[CIT:\s*(.*?)\s*\]", s, flags=re.IGNORECASE)
            return _norm(m.group(1)) if m else s

        def validate_obj(obj, *, strict: bool = False) -> List[str]:
            problems: List[str] = []

            # 0) MC fail-closed (keep early & strict)
            if intent == "mc_internal" and (not evidence or not evidence.strip()):
                if not any(s in (obj.answer_summary or "") for s in ("證據不足", "資料不足")):
                    problems.append("MC_FAIL_CLOSED_MISSING_INSUFFICIENT_EVIDENCE_MSG")
                if obj.key_points:
                    problems.append("MC_FAIL_CLOSED_KEY_POINTS_NOT_EMPTY")
                if obj.procedure_checklist:
                    problems.append("MC_FAIL_CLOSED_CHECKLIST_NOT_EMPTY")
                if obj.citations_used:
                    problems.append("MC_FAIL_CLOSED_CITATIONS_NOT_EMPTY")
                return problems

            ORDINANCE_HARD_CLAIM_RE = re.compile(
                r"(第\s*344\s*章|\bcap\.?\s*344\b|第\s*\d+\s*條|\bsection\s*\d+\b|\bs\.?\s*\d+\b|附表\s*\d+|\bschedule\s*\d+)",
                re.IGNORECASE,
            )

            # 1) Collect ALL citations (top-level + Decision Guidance fields), normalize to payload-only
            all_cits: List[str] = []
            all_cits.extend([_cit_payload(c) for c in (obj.citations_used or []) if c])

            try:
                for a in (obj.next_best_actions or []):
                    all_cits.extend([_cit_payload(c) for c in (getattr(a, "citations", None) or []) if c])
            except Exception:
                pass

            try:
                for b in (obj.decision_tree or []):
                    all_cits.extend([_cit_payload(c) for c in (getattr(b, "citations", None) or []) if c])
            except Exception:
                pass

            if (not decision_mode_flag):
                if (obj.decision_frame or []) or (obj.next_best_actions or []) or (obj.decision_tree or []):
                    problems.append("DECISION_FIELDS_NOT_ALLOWED_WHEN_DECISION_MODE_FALSE")

            bad_cits = [c for c in all_cits if _norm(c) not in allowed]
            if bad_cits:
                problems.append(f"DISALLOWED_CIT:{bad_cits[:5]}")

            def _join(xs: List[str]) -> str:
                return " ".join([_norm(x) for x in xs if x])

            nba_text = ""
            try:
                nba_text = " ".join([
                    (getattr(a, "action", "") or "") + " " + (getattr(a, "why", "") or "")
                    for a in (obj.next_best_actions or [])
                ])
            except Exception:
                nba_text = ""

            dt_text = ""
            try:
                dt_text = " ".join([
                    (getattr(b, "if_condition", "") or "") + " " + (getattr(b, "then_action", "") or "") + " " + (getattr(b, "else_action", "") or "")
                    for b in (obj.decision_tree or [])
                ])
            except Exception:
                dt_text = ""

            combined_text = " ".join([
                obj.answer_summary or "",
                _join(obj.key_points or []),
                _join(obj.procedure_checklist or []),
                _join(obj.decision_frame or []),
                _join(obj.required_facts or []),
                _join(obj.clarifying_questions or []),
                nba_text,
                dt_text,
            ])

            has_hard_ordinance_claim = bool(ORDINANCE_HARD_CLAIM_RE.search(combined_text))
            has_ordinance_cit = any(
                _norm(c).lower().startswith("ordinance |") and (_norm(c) in allowed)
                for c in all_cits
            )
            if has_hard_ordinance_claim and (not has_ordinance_cit):
                problems.append("LEGAL_SOURCE_MISMATCH_NO_ORDINANCE_CIT")

            cands = {_cit_payload(c) for c in (x.get("citation_candidates") or [])}
            has_ordinance_cand = any(_norm(c).lower().startswith("ordinance |") for c in cands)
            if definition_mode_flag and has_ordinance_cand and (not has_ordinance_cit):
                problems.append("DEFINITION_MODE_MISSING_ORDINANCE_CIT")

            if (not is_unknown_intent) and MUST_CITE_RE.search(combined_text) and (not all_cits):
                problems.append("MUST_CITE_NO_CITATIONS")

            if definition_mode_flag:
                min_hits, min_unique = 1, 1
            elif is_unknown_intent:
                min_hits, min_unique = 1, 1
            elif is_multi_intent and intent is None:
                min_hits, min_unique = 2, 1
            else:
                min_hits, min_unique = 2, 2

            ok_cov, reasons = coverage_check(
                answer_summary=obj.answer_summary,
                key_points=obj.key_points,
                citations_used=[c.strip() for c in (obj.citations_used or [])],
                evidence=evidence,
                intent=intent,
                topic=topic,
                min_hits=min_hits,
                min_unique_terms=min_unique,
            )
            if not ok_cov:
                problems.extend(reasons)

            cands2 = {_cit_payload(c) for c in (x.get("citation_candidates") or [])}
            bad_not_in_cands = [c for c in all_cits if _cit_payload(c).strip() not in cands2]
            if bad_not_in_cands:
                problems.append(f"NOT_IN_CANDIDATES:{bad_not_in_cands[:5]}")

            no_id = [c for c in all_cits if "id=" not in c]
            if no_id:
                problems.append(f"CIT_MISSING_ID:{no_id[:5]}")

            return problems

        # ---- attempt 1 (stream) ----
        gen1 = _collect_stream_text(RAG_PROMPT.invoke(x))
        raw1_parts: List[str] = []
        try:
            while True:
                ev, payload = next(gen1)
                if ev == "token":
                    raw1_parts.append(payload)
                    yield ("token", payload)
        except StopIteration as e:
            raw1 = e.value if e.value is not None else "".join(raw1_parts)

        try:
            obj1 = parser.parse(raw1)
            probs1 = validate_obj(obj1)
            if probs1:
                print("[VALIDATION FAIL]", probs1)
                try:
                    print("[citations_used]", obj1.citations_used)
                except Exception:
                    pass
            if not probs1:
                return obj1.model_dump_json(indent=2)
        except Exception:
            probs1 = ["PARSER_ERROR"]

        # ---- 🔧 只在嚴重違規時重試（減少重試頻率） ----
        # 嚴重違規：MUST_CITE_NO_CITATIONS, LEGAL_SOURCE_MISMATCH_NO_ORDINANCE_CIT,
        #          DECISION_FIELDS_NOT_ALLOWED, DISALLOWED_CIT, NOT_IN_CANDIDATES, CIT_MISSING_ID, PARSER_ERROR
        # 輕微違規：DEFINITION_MODE_MISSING_ORDINANCE_CIT, coverage issues, format issues
        CRITICAL_ERRORS = [
            "MUST_CITE_NO_CITATIONS",
            "LEGAL_SOURCE_MISMATCH_NO_ORDINANCE_CIT",
            "DECISION_FIELDS_NOT_ALLOWED_WHEN_DECISION_MODE_FALSE",
            "DISALLOWED_CIT",
            "NOT_IN_CANDIDATES",
            "CIT_MISSING_ID",
            "PARSER_ERROR",
        ]

        has_critical_error = any(
            any(str(p).startswith(err) for err in CRITICAL_ERRORS)
            for p in (probs1 or [])
        )

        # 如果只有輕微違規，直接返回第一次的結果
        if not has_critical_error:
            print("[VALIDATION MINOR ISSUES - ACCEPTING]", probs1)
            return obj1.model_dump_json(indent=2)

        # ---- decide retry_text (same logic as non-stream version) ----
        print("[VALIDATION CRITICAL ERROR - RETRYING]", probs1)
        if any(str(p).startswith("MUST_CITE_NO_CITATIONS") for p in (probs1 or [])):
            retry_text = make_must_cite_retry_prompt(question, evidence)
        elif any(str(p).startswith("LEGAL_SOURCE_MISMATCH_NO_ORDINANCE_CIT") for p in (probs1 or [])):
            cands = x.get("citation_candidates") or []
            has_ord_cand = any(_norm(c).lower().startswith("ordinance |") for c in cands)
            if has_ord_cand and definition_mode_flag:
                retry_text = (
                    "你上一個回答包含具體條文/Cap.344/附表/section/s. 等聲稱，但沒有提供任何 ordinance 類 [CIT: ...]。\n"
                    "你必須完全重寫答案，並遵守：\n"
                    "1) 只可使用【證據】內原樣出現的 [CIT: ...]，而且 citations_used 只能從 citation_candidates 複製。\n"
                    "2) 你必須在 key_points 或 answer_summary 至少一次使用一個 ordinance 類 CIT（payload 以 'ordinance |' 開頭）。\n"
                    "3) 每當你提到條文號、附表、section、s. 等，必須在句尾加上對應 ordinance CIT；如證據不足，必須寫『證據不足』。\n\n"
                    f"用戶問題：{question}\n\n【證據】\n{evidence}\n"
                )
            else:
                retry_text = (
                    "你上一個回答包含具體條文/Cap.344/附表/section/s. 等聲稱，但【證據】內沒有任何 ordinance 類引用可用。\n"
                    "你必須完全重寫答案，並遵守：\n"
                    "1) 只可使用【證據】內原樣出現的 [CIT: ...]。\n"
                    "2) 不可提及任何條文號、Cap.344、附表、section、s. 等字眼；\n"
                    "   只能以【證據】內容作事實描述，或明確寫『證據不足』。\n\n"
                    f"用戶問題：{question}\n\n【證據】\n{evidence}\n"
                )
        elif any(str(p).startswith("DECISION_FIELDS_NOT_ALLOWED_WHEN_DECISION_MODE_FALSE") for p in (probs1 or [])):
            retry_text = (
                "你上一個回答在 decision_mode=False 情況下輸出了 decision_frame/next_best_actions/decision_tree。\n"
                "你必須完全重寫答案，並遵守：\n"
                "- decision_frame、next_best_actions、decision_tree 必須為空陣列 []。\n"
                "- 只可輸出 required_facts 同 clarifying_questions（如需要）。\n"
                "- 引用規則照舊：只可使用【證據】內原樣出現的 [CIT: ...]。\n\n"
                f"用戶問題：{question}\n\n【證據】\n{evidence}\n"
            )
        else:
            retry_text = make_retry_prompt(question, evidence, probs1)

        # signal UI to clear previous attempt
        yield ("reset", "")

        # ---- attempt 2 (stream) ----
        gen2 = _collect_stream_text(RETRY_PROMPT.invoke({
            "retry_text": retry_text,
            "format_instructions": x["format_instructions"],
        }))
        raw2_parts: List[str] = []
        try:
            while True:
                ev, payload = next(gen2)
                if ev == "token":
                    raw2_parts.append(payload)
                    yield ("token", payload)
        except StopIteration as e:
            raw2 = e.value if e.value is not None else "".join(raw2_parts)

        try:
            obj2 = parser.parse(raw2)
        except Exception:
            return SAFE_FALLBACK

        probs2 = validate_obj(obj2, strict=True)
        if probs2:
            return SAFE_FALLBACK

        return obj2.model_dump_json(indent=2)

    chain = (
        RunnableLambda(route_and_retrieve)
        | RunnableLambda(add_evidence)
        | RunnableLambda(citation_guard_and_retry)
    )

    if not return_runner:
        return chain

    def run_stream(question: str, chat_history: List[Dict[str, str]] = None) -> Generator[Tuple[str, str], None, Tuple[List[Document], Dict[str, Any], str]]:
        """Run the full pipeline once, streaming tokens, and return (used_docs, route_info, final_json).

        Args:
            question: Current user question
            chat_history: List of previous turns, each with 'user' and 'assistant' keys
        """
        history = chat_history or []
        x0 = route_and_retrieve({"question": question, "chat_history": history})
        used_docs: List[Document] = x0.get("docs") or []
        x1 = add_evidence(x0)

        g = citation_guard_and_retry_stream(x1)
        try:
            while True:
                ev, payload = next(g)
                yield (ev, payload)
        except StopIteration as e:
            final_json = e.value if e.value is not None else SAFE_FALLBACK

        route_info = x1.get("route_info") or {}
        return (used_docs, route_info, final_json)

    return chain, run_stream


from __future__ import annotations
import re
from typing import List, Optional, Dict, Tuple, Any
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from .citation_guard import build_allowed_set, make_retry_prompt, make_must_cite_retry_prompt, SAFE_FALLBACK
from .router import rule_based_router, routed_retrieve, RouteDecision
from .case_router import infer_case_intent, filter_and_rank_case_docs
from .prompts import RAG_PROMPT
from .schemas import RAGAnswer
from .coverage_guard import coverage_check

from .citation_select import shortlist_citations

from .definition_terms import detect_definition_term, get_all_definition_anchors, get_definition_keywords

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


def normalize_doc_type(raw: str) -> str:
    """Normalize metadata doc_type into canonical: ordinance|guideline|case|doc."""
    t = _norm_doc_type_token(raw)
    if not t:
        return "doc"

    # direct membership (normalized)
    ord_set = {_norm_doc_type_token(x) for x in ORDINANCE_DOC_TYPE_ALIASES}
    guide_set = {_norm_doc_type_token(x) for x in GUIDELINE_DOC_TYPE_ALIASES}
    case_set = {_norm_doc_type_token(x) for x in CASE_DOC_TYPE_ALIASES}

    if t in ord_set:
        return "ordinance"
    if t in guide_set:
        return "guideline"
    if t in case_set:
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

def make_rag_chain(fetch_retriever, llm, final_k: int = 5):

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
            # Avoid over-restricting by property intent for definition questions
            if decision.filters and decision.filters.get("intent"):
                decision.filters.pop("intent", None)

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

        # Only apply property_intent-based filtering when it is confident and NOT unknown.
        # If property_intent is unknown/weak, remove the intent filter to avoid over-restricting retrieval.
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

        if is_unknown or (pi_score < PI_MIN_SCORE) or (pi_conf < PI_MIN_CONF):
            if decision.filters and decision.filters.get("intent"):
                decision.filters.pop("intent", None)
        else:
            decision.filters = decision.filters or {}
            # ✅ multi-intent -> OR filter (list)
            if len(pi_intents) >= 2 and bool(pi.get("multi")):
                decision.filters["intent"] = pi_intents[:2]
            else:
                decision.filters["intent"] = pi_intents[0]

        # --- unknown intent behavior flags (for downstream prompt / logging) ---
        unknown_intent = (not pi_intents) or all(x in ("", "unknown") for x in pi_intents) or (pi_score < PI_MIN_SCORE) or (pi_conf < PI_MIN_CONF)

        decision.extra["unknown_intent"] = bool(unknown_intent)

        # Optional: suggested clarifying questions (keep short)
        if unknown_intent:
            decision.extra["clarifying_questions"] = [
                "你遇到嘅問題係邊一類？例如：管理費/欠費追討、滲水、違建、寵物、火警、電力供應、查帳/查文件等？",
                "事件發生喺你自己單位定公用地方？有冇通知管理公司/法團？有冇相片、信件或會議紀錄？",
            ]
        else:
            decision.extra.pop("clarifying_questions", None)
    

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

        # Use workflow intent/topic derived from subtopics for specialized handling
        intent = wf_intent
        topic = wf_topic

        if intent == "meeting_procedure":
            base_docs = soft_rerank_by_hits(base_docs, MEETING_MUST_TERMS)

        if intent == "mc_internal":
            topic_terms = MC_TOPIC_KEYWORDS.get(topic, [])
            base_docs = soft_rerank_by_hits(base_docs, MC_CORE_TERMS + MC_ACTION_TERMS + topic_terms)

        

        pi_for_pack = (decision.extra.get("property_intent") or {})
        prop_intents_for_pack = pi_for_pack.get("intents") or ([pi_for_pack.get("intent")] if pi_for_pack.get("intent") else [])

        docs = select_docs_for_evidence(
            intent,
            base_docs,
            final_k,
            topic=topic,
            property_intents=prop_intents_for_pack,
            definition_mode=definition_mode,
        )

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
                return obj1.model_dump_json(ensure_ascii=False, indent=2)
        except Exception:
            probs1 = ["PARSER_ERROR"]

        # Dedicated retry prompts for common validation failures
        if any(str(p).startswith("DEFINITION_MODE_MISSING_ORDINANCE_CIT") for p in (probs1 or [])):
            # Definition mode violation: must use ordinance citations
            cands = x.get("citation_candidates") or []
            ord_cands = [c for c in cands if _norm(c).lower().startswith("ordinance |")]
            retry_text = (
                "你上一個回答係針對定義問題（例如『乜嘢係...』、『咩係...』），但你冇使用任何 ordinance 類引用。\n"
                "你必須完全重寫答案，並遵守以下規則：\n"
                "1) 定義問題必須優先使用 ordinance 類 [CIT: ...]（payload 以 'ordinance |' 開頭）。\n"
                f"2) 可用嘅 ordinance citations：{ord_cands[:5]}\n"
                "3) 你必須喺 citations_used 入面至少包含一個 ordinance citation。\n"
                "4) answer_summary 同 key_points 必須基於法例原文，唔可以只引用案例。\n"
                "5) 只可使用 citation_candidates 內嘅 CIT，不可自創。\n\n"
                f"用戶問題：{question}\n\n【證據】\n{evidence}\n"
            )
        elif any(str(p).startswith("MUST_CITE_NO_CITATIONS") for p in (probs1 or [])):
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

        return obj2.model_dump_json(ensure_ascii=False, indent=2)

    chain = (
        RunnableLambda(route_and_retrieve)
        | RunnableLambda(add_evidence)
        | RunnableLambda(citation_guard_and_retry)
    )
    return chain
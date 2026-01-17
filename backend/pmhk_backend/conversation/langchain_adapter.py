# conversations/langchain_adapter.py
from __future__ import annotations

import os
import sys
import re
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Add project root to Python path to import rag_app
ROOT = Path(__file__).resolve().parents[3]  # Go up to Property_manangement_LLM/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_app.hybrid_retriever import HybridRetriever
from rag_app.rag_chain import make_rag_chain

# Load environment variables
load_dotenv(ROOT / ".env")  # Load from Property_manangement_LLM/.env

CBR_DIR = ROOT / "data" / "cbr"
if CBR_DIR.exists():
    # allow importing data/cbr/*.py
    if str(CBR_DIR) not in sys.path:
        sys.path.insert(0, str(CBR_DIR))

try:
    from data.cbr.case_retriever import CaseHybridRetriever
    from data.cbr.cbr_reasoner import build_cbr_reasoner_chain, format_cases_for_prompt, CBR_OUTPUT_SCHEMA
    _CBR_AVAILABLE = True
except Exception:
    CaseHybridRetriever = None
    build_cbr_reasoner_chain = None
    format_cases_for_prompt = None
    CBR_OUTPUT_SCHEMA = None
    _CBR_AVAILABLE = False


CBR_TRIGGERS = [
    "用案例", "用個案", "引用案例", "案例對比", "個案對比",
    "CBR", "precedent", "判例", "以往案例", "舊案",
    "compare with cases", "compare case",
]


def should_use_cbr(q: str) -> bool:
    qq = (q or "").strip()
    low = qq.lower()
    for t in CBR_TRIGGERS:
        if t.lower() in low:
            return True
    if re.search(r"(對比|比較).*(現況|情況|本案|今次|而家)", qq):
        return True
    return False


def make_embeddings(model_name: str):
    # identical strategy as your CLI
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)


def make_llm(streaming: bool = True) -> ChatOpenAI:
    """
    Keep same provider config as your CLI:
    - model=deepseek-chat (OpenAI-compatible)
    - OPENAI_API_KEY / OPENAI_API_BASE from env
    """
    return ChatOpenAI(
        model=os.getenv("PMHK_LLM_MODEL", "deepseek-chat"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        temperature=float(os.getenv("PMHK_TEMPERATURE", "0")),
        streaming=streaming,
    )


def docs_to_evidences(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Convert your Document.metadata (doc_type/id/file_name/chunk_index/...) into UI/DB Evidence payload.
    """
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(docs):
        md = d.metadata or {}
        doc_type = md.get("doc_type", "other")
        display = md.get("display_title") or md.get("case_no") or md.get("id") or "?"
        publisher = md.get("publisher") or md.get("source") or "?"
        file_name = md.get("file_name", "?")
        chunk = md.get("chunk_index", "?")
        chunk_id = f"{doc_type}|{md.get('id','?')}|{file_name}:{chunk}"

        excerpt = (d.page_content or "").strip()
        if len(excerpt) > 1500:
            excerpt = excerpt[:1500] + "…"

        out.append({
            "source_type": doc_type if doc_type in {"statute", "case", "guideline", "dmc"} else "other",
            "source_id": str(display),  # human-readable id for UI
            "source_title": str(md.get("title") or md.get("heading") or ""),
            "excerpt": excerpt,
            "score": float(md.get("score", 0.0) or 0.0),
            "rank": i,
            "chunk_id": chunk_id,
            "supports_claim": True,
            "meta": {
                "doc_type": doc_type,
                "display": display,
                "publisher": publisher,
                "file_name": file_name,
                "chunk_index": chunk,
                # keep useful fields
                **{k: v for k, v in md.items() if k not in {"score"}},
            },
        })
    return out


@dataclass
class Engine:
    fetch_retriever: HybridRetriever
    chain: Any  # Runnable (LCEL chain)
    run_stream: Any  # callable(question) -> generator yielding (ev,payload) and returning (used_docs, route_info, final_json)
    llm: ChatOpenAI
    final_k: int
    cbr_faiss_dir: str


_ENGINE: Optional[Engine] = None


def _build_engine() -> Engine:
    """
    Build and cache heavy objects:
    - embeddings
    - FAISS index load
    - BM25 from FAISS docstore
    - HybridRetriever
    - LLM
    - RAG chain
    """
    index_dir = os.getenv("PMHK_FAISS_INDEX_DIR", "").strip()
    if not index_dir:
        raise RuntimeError("Missing env PMHK_FAISS_INDEX_DIR (path to FAISS index dir).")

    emb_model = os.getenv("PMHK_EMB_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    final_k = int(os.getenv("PMHK_FINAL_K", "5"))
    fetch_k_vec = int(os.getenv("PMHK_FETCH_K_VEC", "30"))
    fetch_k_bm25 = int(os.getenv("PMHK_FETCH_K_BM25", "30"))
    w_vec = float(os.getenv("PMHK_W_VEC", "2.0"))
    w_bm25 = float(os.getenv("PMHK_W_BM25", "0.8"))

    embeddings = make_embeddings(emb_model)

    vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    vec_retriever = vs.as_retriever(search_kwargs={"k": fetch_k_vec})

    # BM25 built from FAISS docstore (same trick as CLI)
    all_docs = list(vs.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = fetch_k_bm25

    fetch_retriever = HybridRetriever(
        vector_retriever=vec_retriever,
        bm25_retriever=bm25,
        fetch_k_vec=fetch_k_vec,
        fetch_k_bm25=fetch_k_bm25,
        out_k=max(fetch_k_vec, fetch_k_bm25),
        w_vec=w_vec,
        w_bm25=w_bm25,
    )

    llm = make_llm(streaming=True)
    chain, run_stream_fn = make_rag_chain(fetch_retriever=fetch_retriever, llm=llm, final_k=final_k, return_runner=True)

    cbr_faiss_dir = os.getenv("PMHK_CBR_FAISS_DIR", "").strip()

    return Engine(
        fetch_retriever=fetch_retriever,
        chain=chain,
        run_stream=run_stream_fn,
        llm=llm,
        final_k=final_k,
        cbr_faiss_dir=cbr_faiss_dir,
    )


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _build_engine()
    return _ENGINE


def run_cbr_once(question: str, facts: str, engine: Engine) -> Tuple[str, Dict[str, Any]]:
    """
    CBR 暫時先 non-stream（MVP 合理）。
    回傳 answer + final_meta
    """
    if not _CBR_AVAILABLE:
        return (
            "❌ CBR modules not available. Please ensure data/cbr/case_retriever.py and data/cbr/cbr_reasoner.py exist.",
            {"route": "cbr", "intent": "cbr_unavailable", "evidences": []},
        )

    if not engine.cbr_faiss_dir:
        return (
            "❌ Missing PMHK_CBR_FAISS_DIR (case-level FAISS dir for CBR).",
            {"route": "cbr", "intent": "cbr_missing_index", "evidences": []},
        )

    retriever = CaseHybridRetriever(faiss_dir=engine.cbr_faiss_dir)
    hits = retriever.search(question, k=8, fetch_k=30, query_issue_tags=[])

    cases_text = format_cases_for_prompt(
        [{"doc": h.doc, "score": h.score, "matched_tags": h.matched_tags} for h in hits],
        max_cases=6,
    )

    chain = build_cbr_reasoner_chain(engine.llm)  # can still use same llm
    out_str = chain.invoke(
        {
            "question": question,
            "facts": facts or "",
            "query_issue_tags": "",
            "cases_text": cases_text,
            "schema": __import__("json").dumps(CBR_OUTPUT_SCHEMA, ensure_ascii=False),
        }
    )

    # evidences: use case ids as trace (lightweight)
    evidences = []
    for i, h in enumerate(hits[:8]):
        md = h.doc.metadata or {}
        evidences.append({
            "source_type": "case",
            "source_id": md.get("case_id") or md.get("case_no") or f"case_hit_{i}",
            "source_title": md.get("case_no") or "",
            "excerpt": (h.doc.page_content or "")[:1200],
            "score": float(h.score),
            "rank": i,
            "chunk_id": md.get("case_id") or "",
            "supports_claim": True,
            "meta": {**md, "matched_tags": getattr(h, "matched_tags", [])},
        })

    final_meta = {
        "route": "cbr",
        "intent": "cbr_compare",
        "warnings": "一般資料，唔構成法律意見。",
        "confidence": 0.0,
        "evidences": evidences,
        "model_meta": {"provider": "langchain_openai", "model": getattr(engine.llm, "model_name", "")},
        "retrieval_meta": {"retriever": "CaseHybridRetriever", "top_k": len(hits)},
        "token_usage": {},
        "prompt_version": "cbr_v1",
    }
    return out_str, final_meta


def run_rag_stream(
    question: str,
    engine: Engine,
    chat_history: List[Dict[str, str]] = None,
) -> Tuple[Generator[Tuple[str, str], None, None], Dict[str, Any]]:
    """Use the pipeline runner to stream tokens and build final_meta from the same used docs.

    Args:
        question: Current user question
        engine: RAG engine instance
        chat_history: List of previous turns, each with 'user' and 'assistant' keys
    """
    history = chat_history or []

    # Pass chat_history to the pipeline runner
    runner = engine.run_stream(question, chat_history=history)  # yields (ev,payload) and returns (used_docs, route_info, final_json)

    used_docs: List[Document] = []
    route_info: Dict[str, Any] = {}
    final_json: str = ""

    def token_events() -> Generator[Tuple[str, str], None, None]:
        nonlocal used_docs, route_info, final_json
        try:
            while True:
                ev, payload = next(runner)
                yield (ev, payload)
        except StopIteration as e:
            if e.value:
                used_docs, route_info, final_json = e.value
            else:
                used_docs, route_info, final_json = [], {}, ""

    evidences = docs_to_evidences(used_docs)

    extra = (route_info.get("extra") or {})
    wf = (extra.get("workflow") or {})

    final_meta: Dict[str, Any] = {
        "route": route_info.get("route") or "rag",
        "intent": wf.get("intent") or "",
        "confidence": float(route_info.get("confidence") or 0.0),
        "warnings": "一般資料，唔構成法律意見。",
        "evidences": evidences,
        "model_meta": {"provider": "langchain_openai", "model": getattr(engine.llm, "model_name", "")},
        "retrieval_meta": {"retriever": "HybridRetriever", "final_k": engine.final_k},
        "token_usage": {},
        "prompt_version": os.getenv("PMHK_PROMPT_VERSION", "rag_v1"),
        "route_info": route_info,
        "final_json": final_json,
    }

    return token_events(), final_meta


def run_stream(
    question: str,
    facts: str = "",
    mode: str = "auto",
    chat_history: List[Dict[str, str]] = None,
) -> Tuple[Generator[Tuple[str, str], None, None], Dict[str, Any]]:
    """
    你 Django backend 唯一需要 call 呢個：
    return token_gen, final_meta

    Args:
        question: Current user question
        facts: Additional facts (for CBR mode)
        mode: "auto", "rag", or "cbr"
        chat_history: List of previous turns, each with 'user' and 'assistant' keys
    """
    engine = get_engine()
    history = chat_history or []

    if mode == "auto":
        mode = "cbr" if should_use_cbr(question) else "rag"

    if mode == "cbr":
        ans, meta = run_cbr_once(question, facts=facts, engine=engine)

        # CBR 先包裝成「一次性 stream」（前端都照樣收 token event）
        def gen_once():
            yield ("token", ans)

        return gen_once(), meta

    return run_rag_stream(question, engine, chat_history=history)
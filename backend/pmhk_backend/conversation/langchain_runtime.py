# conversations/langchain_runtime.py
from __future__ import annotations
from typing import Any, Dict, Generator, List, Tuple
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

# =========================
# 你需要接你自己嘅 retriever
# =========================

def get_retriever():
    """
    TODO: 改成你現有 FAISS retriever / vector store retriever
    必須提供 .invoke(query) -> List[Document]
    Document.page_content, Document.metadata 要有 chunk_id/source_id 等
    """
    class DummyRetriever:
        def invoke(self, query: str) -> List[Document]:
            return [
                Document(
                    page_content="（示例）欠繳管理費可採取催繳通知、登記押記備忘等。",
                    metadata={
                        "source_type": "guideline",
                        "source_id": "ICAC BM 7.6",
                        "source_title": "ICAC Best Practice",
                        "chunk_id": "icac_bm_7_6_chunk6",
                        "score": 0.82,
                    },
                )
            ]
    return DummyRetriever()


def build_llm() -> ChatOpenAI:
    """
    OpenAI-compatible：
    - 真 OpenAI：只要 set OPENAI_API_KEY
    - 本地/其他：再加 OPENAI_BASE_URL, OPENAI_API_KEY(可假)
    """
    return ChatOpenAI(
        model="gpt-4o-mini",       # 你之後換你用嘅 model
        temperature=0.2,
        streaming=True,
        # base_url=... (如需)
        # api_key=...  (通常用 env)
    )


SYSTEM_PROMPT = """你係香港物業管理/大廈管理專家助手。
回答要用香港中文（粵語書面），專有名詞保留英文（例如 Building Management Ordinance (BMO), Deed of Mutual Covenant (DMC)）。
如涉及法律／程序，請加風險提示：一般資料，唔構成法律意見。
必須引用你用過嘅 evidence（法例/案例/指引）作為依據，避免引用純背景句。
"""


def format_context(docs: List[Document]) -> str:
    """
    將檢索到嘅 doc 轉成 context。
    你可以改成：每段加來源標籤，方便 trace。
    """
    parts = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        sid = meta.get("source_id", "")
        stype = meta.get("source_type", "")
        parts.append(f"[{i}] ({stype}) {sid}\n{d.page_content}")
    return "\n\n".join(parts)


def docs_to_evidences(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    將 docs metadata 轉成你 DB Evidence 需要嘅 payload。
    """
    evidences = []
    for i, d in enumerate(docs):
        meta = d.metadata or {}
        evidences.append({
            "source_type": meta.get("source_type", "other"),
            "source_id": meta.get("source_id", "unknown"),
            "source_title": meta.get("source_title", ""),
            "excerpt": (d.page_content or "")[:1200],  # excerpt 控制長度
            "score": float(meta.get("score", 0.0) or 0.0),
            "rank": i,
            "chunk_id": meta.get("chunk_id", ""),
            "supports_claim": True,  # 你日後可用 validator 改
            "meta": {k: v for k, v in meta.items() if k not in ["score"]},
        })
    return evidences


def stream_answer(
    user_query: str,
    chat_history: List[Dict[str, str]] | None = None,
) -> Tuple[Generator[str, None, None], Dict[str, Any]]:
    """
    回傳：
    - token/chunk generator（yield 字串）
    - final_meta（含 evidences / model_meta / retrieval_meta）
    """
    retriever = get_retriever()
    llm = build_llm()

    # 1) retrieve
    docs = retriever.invoke(user_query)
    context = format_context(docs)

    # 2) messages（可加入 history）
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # history 格式建議：[{role:"user"/"assistant", content:"..."}]
    if chat_history:
        for m in chat_history[-6:]:  # MVP：只取最近 6 個訊息（你可調）
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            else:
                messages.append(SystemMessage(content=f"（你上一輪回答）{m['content']}"))

    # 3) prompt（將 context + user_query 一齊交畀模型）
    user_prompt = f"""以下係檢索到嘅資料（context），請用嚟回答問題：
\n\n{context}\n\n問題：{user_query}
\n\n要求：
- 直接回答，條理清晰
- 如有程序/責任分工，列點
- 最後提供「引用來源」清單（用你 context 裏面嘅 [1][2]...）
"""
    messages.append(HumanMessage(content=user_prompt))

    # 4) streaming：llm.stream(messages)
    def gen() -> Generator[str, None, None]:
        for chunk in llm.stream(messages):
            # chunk is AIMessageChunk; content 可能係 None
            txt = getattr(chunk, "content", None)
            if txt:
                yield txt

    final_meta = {
        "intent": "",  # 你之後可接 router
        "route": "rag",
        "confidence": 0.0,  # 你之後可算
        "warnings": "一般資料，唔構成法律意見。",
        "evidences": docs_to_evidences(docs),
        "model_meta": {"provider": "langchain_openai", "model": getattr(llm, "model_name", "")},
        "retrieval_meta": {"top_k": len(docs), "retriever": type(retriever).__name__},
        "token_usage": {},  # streaming 情況下通常要用 callback/ provider 才有準確 usage
        "prompt_version": "v1",
    }
    return gen(), final_meta
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from .prompts import RAG_PROMPT


def format_citation(d: Document) -> str:
    md = d.metadata or {}
    # 你想要嘅 trace 字段：id/file_name/chunk/section_type/category
    return (
        f"[{md.get('id','?')} | {md.get('file_name','?')} | "
        f"{md.get('doc_type','?')}/{md.get('category','?')} | "
        f"{md.get('section_type','?')} | chunk {md.get('chunk_index','?')}]"
    )


def build_evidence(docs: List[Document], max_chars: int = 9000) -> str:
    """
    將 top-k docs 組成 evidence pack，控制長度，保留 citations。
    """
    parts = []
    used = 0
    for i, d in enumerate(docs, start=1):
        cite = format_citation(d)
        txt = d.page_content.strip()
        block = f"({i}) {cite}\n{txt}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def make_rag_chain(retriever, llm):
    """
    retriever: LangChain retriever (FAISS.as_retriever)
    llm: any chat model compatible with LangChain (e.g. ChatOpenAI, ChatOllama, vLLM client)
    """
    retrieve = RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"]))
    evidence = RunnableLambda(lambda docs: build_evidence(docs))

    chain = (
        {"question": lambda x: x["question"], "docs": retrieve}
        | {"question": lambda x: x["question"], "evidence": lambda x: build_evidence(x["docs"])}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

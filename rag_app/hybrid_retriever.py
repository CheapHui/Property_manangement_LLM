from __future__ import annotations
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from pydantic import Field
from pydantic.config import ConfigDict

try:
    from langchain_community.retrievers import BM25Retriever
except Exception:
    from langchain.retrievers import BM25Retriever


def _doc_key(d: Document) -> str:
    md = d.metadata or {}
    return f"{md.get('id','?')}|{md.get('file_name','?')}|{md.get('chunk_index','?')}"


def rrf_fuse(
    ranked_lists: List[List[Document]],
    weights: List[float] | None = None,
    k: int = 5,
    rrf_k: int = 60,
) -> List[Document]:
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: Dict[str, float] = {}
    keep: Dict[str, Document] = {}

    for w, docs in zip(weights, ranked_lists):
        for rank, d in enumerate(docs, start=1):
            key = _doc_key(d)
            keep[key] = d
            scores[key] = scores.get(key, 0.0) + w * (1.0 / (rrf_k + rank))

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [keep[key] for key in sorted_keys[:k]]


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval = Vector (FAISS retriever) + BM25 retriever, fused by RRF.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_retriever: Any = Field(...)
    bm25_retriever: Any = Field(...)
    fetch_k_vec: int = 20
    fetch_k_bm25: int = 20
    out_k: int = 20
    w_vec: float = 1.0
    w_bm25: float = 1.2
    rrf_k: int = 60

    def _get_relevant_documents(self, query: str) -> List[Document]:
        vec_docs = self.vector_retriever.invoke(query)[: self.fetch_k_vec]
        bm_docs = self.bm25_retriever.invoke(query)[: self.fetch_k_bm25]

        return rrf_fuse(
            [vec_docs, bm_docs],
            weights=[self.w_vec, self.w_bm25],
            k=self.out_k,
            rrf_k=self.rrf_k,
        )
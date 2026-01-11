#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Property Management CLI with CBR integration.

Modes:
  - rag: your existing RAG answer (placeholder in this script)
  - cbr: case-based reasoning answer (retrieve cases + reasoner)
  - auto: decide based on user query intent

Usage:
  python scripts/pm_cli.py \
    --mode auto \
    --q "管理公司被業主立案法團以附表7終止委任，我想你用案例對比我情況" \
    --facts "已發出會議通知並通過決議，管理公司指程序有瑕疵及未獲充分陳述機會。" \
    --faiss_dir "/.../data/cbr/faiss_index" \
    --llm_backend ollama \
    --llm_model "qwen2.5:14b-instruct"

Notes:
  - RAG mode is a stub; replace run_rag() with your existing pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT.parent / ".env")

from app.cbr.case_retriever import CaseHybridRetriever
from app.cbr.cbr_reasoner import (
    build_cbr_reasoner_chain,
    format_cases_for_prompt,
    CBR_OUTPUT_SCHEMA,
)

# --- Choose LLM backend ---
def build_llm(backend: str, model: str, temperature: float):
    backend = backend.lower().strip()

    if backend == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model, temperature=temperature)

    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
        )

    raise ValueError(f"Unsupported llm_backend: {backend} (use ollama|openai)")


# -----------------------------
# Intent detection (simple + safe)
# -----------------------------

CBR_TRIGGERS = [
    "用案例", "用個案", "引用案例", "案例對比", "個案對比",
    "case-based", "CBR", "precedent", "判例", "以往案例", "舊案",
    "compare with cases", "compare case",
]

def should_use_cbr(question: str) -> bool:
    q = question.strip()
    # explicit trigger words
    for t in CBR_TRIGGERS:
        if t.lower() in q.lower():
            return True
    # pattern: "對比" + "情況/現況/本案"
    if re.search(r"(對比|比較).*(現況|情況|本案|今次|而家)", q):
        return True
    return False


# -----------------------------
# RAG stub (replace with your existing pipeline)
# -----------------------------

def run_rag(question: str, facts: str) -> Dict[str, Any]:
    """
    TODO: Replace with your existing RAG chain output.
    Keep output JSON structure consistent.
    """
    return {
        "answer_title": "RAG 回答（示範）",
        "overview": "呢度係 RAG mode stub。請將 run_rag() 換成你現有法例/指引/案例片段 RAG pipeline。",
        "question": question,
        "facts": facts,
        "mode": "rag",
    }


# -----------------------------
# CBR pipeline
# -----------------------------

def run_cbr(
    question: str,
    facts: str,
    faiss_dir: str,
    llm_backend: str,
    llm_model: str,
    temperature: float,
    tags: List[str],
    k: int = 8,
    fetch_k: int = 30,
) -> Dict[str, Any]:
    # 1) retrieve
    retriever = CaseHybridRetriever(faiss_dir=faiss_dir)
    hits = retriever.search(
        question,
        k=k,
        fetch_k=fetch_k,
        query_issue_tags=tags,
        required_issue_tags=None,
    )

    cases_text = format_cases_for_prompt(
        [{"doc": h.doc, "score": h.score, "matched_tags": h.matched_tags} for h in hits],
        max_cases=6,
    )

    # 2) reason
    llm = build_llm(llm_backend, llm_model, temperature)
    chain = build_cbr_reasoner_chain(llm)

    out_str = chain.invoke(
        {
            "question": question,
            "facts": facts or "",
            "query_issue_tags": ", ".join(tags),
            "cases_text": cases_text,
            "schema": json.dumps(CBR_OUTPUT_SCHEMA, ensure_ascii=False),
        }
    )

    # 3) parse JSON (if fails, return raw)
    try:
        parsed = json.loads(out_str)
        parsed["mode"] = "cbr"
        # keep trace for debugging
        parsed["_retrieved_case_ids"] = [h.doc.metadata.get("case_id") for h in hits[:8]]
        return parsed
    except Exception:
        return {
            "mode": "cbr",
            "error": "CBR output is not valid JSON",
            "raw": out_str,
            "_retrieved_case_ids": [h.doc.metadata.get("case_id") for h in hits[:8]],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "rag", "cbr"], default="auto")
    ap.add_argument("--q", required=True, help="user question")
    ap.add_argument("--facts", default="", help="current situation / facts")

    # CBR params
    ap.add_argument("--faiss_dir", default="", help="FAISS directory for case_master index (required for cbr)")
    ap.add_argument("--tags", default="", help="comma-separated issue tags (optional)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--fetch_k", type=int, default=30)

    # LLM params
    ap.add_argument("--llm_backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--llm_model", default="qwen2.5:14b-instruct")
    ap.add_argument("--temperature", type=float, default=0.2)

    args = ap.parse_args()

    question = args.q.strip()
    facts = args.facts.strip()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    mode = args.mode
    if mode == "auto":
        mode = "cbr" if should_use_cbr(question) else "rag"

    if mode == "cbr":
        if not args.faiss_dir:
            raise SystemExit("❌ --faiss_dir is required for cbr mode")
        result = run_cbr(
            question=question,
            facts=facts,
            faiss_dir=args.faiss_dir,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            temperature=args.temperature,
            tags=tags,
            k=args.k,
            fetch_k=args.fetch_k,
        )
    else:
        result = run_rag(question, facts)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
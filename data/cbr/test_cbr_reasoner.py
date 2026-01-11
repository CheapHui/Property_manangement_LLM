#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from langchain_openai import ChatOpenAI
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Assuming .env is in the parent directory of the repo root
ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels from data/cbr/
load_dotenv(ROOT.parent / ".env")

from case_retriever import CaseHybridRetriever
from cbr_reasoner import (
    build_cbr_reasoner_chain,
    format_cases_for_prompt,
    CBR_OUTPUT_SCHEMA,
)

# 你按你實際用嘅 LLM 換呢段
# 例：OpenAI
# from langchain_openai import ChatOpenAI

# 例：Ollama
# from langchain_community.chat_models import ChatOllama

openai_api_key = os.getenv("OPENAI_API_KEY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss_dir", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--facts", default="", help="optional user facts/context")
    ap.add_argument("--tags", default="", help="comma-separated query issue_tags")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--fetch_k", type=int, default=30)
    args = ap.parse_args()

    query_tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    # 1) retrieve
    retriever = CaseHybridRetriever(faiss_dir=args.faiss_dir)
    hits = retriever.search(
        args.q,
        k=args.k,
        fetch_k=args.fetch_k,
        query_issue_tags=query_tags,
        required_issue_tags=None,
    )

    # 2) format for prompt
    cases_text = format_cases_for_prompt(
        [{"doc": h.doc, "score": h.score, "matched_tags": h.matched_tags} for h in hits],
        max_cases=6,
    )

    # 3) LLM
    # === 換成你自己用嘅 model ===
    llm = ChatOpenAI(model="deepseek-chat",   # 或 deepseek-reasoner
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        temperature=0)
    # llm = ChatOllama(model="qwen2.5:14b-instruct", temperature=0.2)



    # 4) run
    chain = build_cbr_reasoner_chain(llm)
    out = chain.invoke(
        {
            "question": args.q,
            "facts": args.facts,
            "query_issue_tags": ", ".join(query_tags),
            "cases_text": cases_text,
            "schema": json.dumps(CBR_OUTPUT_SCHEMA, ensure_ascii=False),
        }
    )

    # 5) validate-ish: try parse JSON
    try:
        parsed = json.loads(out)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except Exception:
        print(out)


if __name__ == "__main__":
    main()
# conversations/service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Generator, List, Optional, Tuple
import time

from django.db import transaction

from .models import Conversation, Turn, Evidence
from .langchain_adapter import run_stream


@dataclass
class EvidenceItem:
    source_type: str         # "statute" | "case" | "guideline" | "dmc" | "other"
    source_id: str           # e.g. "BMO Cap.344 s.34" / "LDBM 218/1999"
    source_title: str = ""
    excerpt: str = ""
    score: float = 0.0
    rank: int = 0
    chunk_id: str = ""
    supports_claim: bool = True
    claim_id: str = ""
    meta: Optional[Dict[str, Any]] = None


@dataclass
class RunResult:
    answer: str
    intent: str = ""
    route: str = ""
    confidence: float = 0.0
    warnings: str = ""
    evidences: Optional[List[EvidenceItem]] = None
    model_meta: Optional[Dict[str, Any]] = None
    retrieval_meta: Optional[Dict[str, Any]] = None
    token_usage: Optional[Dict[str, Any]] = None
    latency_ms: int = 0
    prompt_version: str = ""

def _langchain_stream(user_query: str, conversation_id: str, chat_history: List[Dict[str, str]] = None):
    """Return a streaming event generator and final_meta dict.

    Streaming generator yields tuples: (event_type, payload)
      - ("token", "...")
      - ("reset", "")  # when attempt-1 fails and pipeline retries

    `final_meta` is returned by the adapter and includes evidences/route/intent/etc.

    Args:
        user_query: Current user question
        conversation_id: ID of the conversation
        chat_history: List of previous turns, each with 'user' and 'assistant' keys
    """
    token_events, final_meta = run_stream(
        question=user_query,
        mode="auto",
        chat_history=chat_history or [],
    )
    return token_events, final_meta


def _get_chat_history(conversation: "Conversation", max_turns: int = 5) -> List[Dict[str, str]]:
    """
    從 DB 獲取對話歷史（最近 N 輪）。

    Args:
        conversation: Conversation object
        max_turns: Maximum number of previous turns to include (default 5)

    Returns:
        List of dicts with 'user' and 'assistant' keys
    """
    # 獲取最近的 turns（排除當前正在處理的）
    recent_turns = (
        Turn.objects
        .filter(conversation=conversation)
        .exclude(assistant_answer="")  # 排除未完成的 turn
        .order_by("-created_at")[:max_turns]
    )

    # 反轉順序（從舊到新）
    turns_list = list(recent_turns)[::-1]

    history = []
    for turn in turns_list:
        history.append({
            "user": turn.user_query,
            "assistant": turn.assistant_answer,
        })

    return history



@transaction.atomic
def create_turn_stub(conversation: Conversation, user_query: str) -> Turn:
    """
    先寫入一個 Turn（assistant_answer 暫時空），等 streaming 完再 update。
    """
    turn = Turn.objects.create(
        conversation=conversation,
        user_query=user_query,
        assistant_answer="",
        route="",
        intent="",
        confidence=0.0,
        warnings="",
        latency_ms=0,
        token_usage={},
        model_meta={},
        retrieval_meta={},
        prompt_version="",
    )
    # 觸發 conversation updated_at
    conversation.save(update_fields=["updated_at"])
    return turn


@transaction.atomic
def finalize_turn(
    turn: Turn,
    result: RunResult,
) -> None:
    """
    將 LangChain 最終結果落 DB：Turn + Evidence
    """
    turn.assistant_answer = result.answer
    turn.intent = result.intent
    turn.route = result.route
    turn.confidence = result.confidence
    turn.warnings = result.warnings
    turn.model_meta = result.model_meta or {}
    turn.retrieval_meta = result.retrieval_meta or {}
    turn.token_usage = result.token_usage or {}
    turn.latency_ms = result.latency_ms
    turn.prompt_version = result.prompt_version or ""
    turn.save()

    # 清理舊 evidence（如果你支援重跑）
    Evidence.objects.filter(turn=turn).delete()

    evs = result.evidences or []
    objs = []
    for i, ev in enumerate(evs):
        objs.append(Evidence(
            turn=turn,
            source_type=ev.source_type,
            source_id=ev.source_id,
            source_title=ev.source_title,
            excerpt=ev.excerpt,
            score=ev.score,
            rank=ev.rank if ev.rank is not None else i,
            chunk_id=ev.chunk_id,
            supports_claim=ev.supports_claim,
            claim_id=ev.claim_id,
            meta=ev.meta or {},
        ))
    if objs:
        Evidence.objects.bulk_create(objs)


def run_conversation_streaming(
    conversation: Conversation,
    user_query: str,
    max_history_turns: int = 5,
) -> Tuple[Turn, Generator[Tuple[str, str], None, None], Dict[str, Any]]:
    """
    核心入口：
    - 獲取對話歷史
    - 建 turn stub
    - 取得 LangChain token stream + final meta

    Args:
        conversation: Conversation object
        user_query: Current user question
        max_history_turns: Maximum number of previous turns to include (default 5)
    """
    # 先獲取歷史（在建立新 turn 之前）
    chat_history = _get_chat_history(conversation, max_turns=max_history_turns)

    # 建立新 turn stub
    turn = create_turn_stub(conversation, user_query)

    # 傳入歷史到 LangChain
    token_gen, final_meta = _langchain_stream(
        user_query=user_query,
        conversation_id=str(conversation.id),
        chat_history=chat_history,
    )
    return turn, token_gen, final_meta


def build_run_result(answer: str, final_meta: Dict[str, Any], latency_ms: int) -> RunResult:
    # 將 dict evidences 轉成 EvidenceItem list
    ev_items = []
    for ev in final_meta.get("evidences", []) or []:
        ev_items.append(EvidenceItem(
            source_type=ev.get("source_type", "other"),
            source_id=ev["source_id"],
            source_title=ev.get("source_title", ""),
            excerpt=ev.get("excerpt", ""),
            score=float(ev.get("score", 0.0)),
            rank=int(ev.get("rank", 0)),
            chunk_id=ev.get("chunk_id", ""),
            supports_claim=bool(ev.get("supports_claim", True)),
            claim_id=ev.get("claim_id", ""),
            meta=ev.get("meta", {}) or {},
        ))

    return RunResult(
        answer=answer,
        intent=final_meta.get("intent", "") or "",
        route=final_meta.get("route", "") or "",
        confidence=float(final_meta.get("confidence", 0.0) or 0.0),
        warnings=final_meta.get("warnings", "") or "",
        evidences=ev_items,
        model_meta=final_meta.get("model_meta", {}) or {},
        retrieval_meta=final_meta.get("retrieval_meta", {}) or {},
        token_usage=final_meta.get("token_usage", {}) or {},
        latency_ms=latency_ms,
        prompt_version=final_meta.get("prompt_version", "") or "",
    )

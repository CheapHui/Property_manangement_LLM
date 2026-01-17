# conversations/views_run.py
import json
import time
from typing import Generator

from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Conversation
from .permissions import IsConversationOwner
from .serializers import ConversationDetailSerializer
from .service import run_conversation_streaming, build_run_result, finalize_turn


def sse_event(data: dict, event: str | None = None) -> str:
    """
    SSE 格式：
    event: <name>\n
    data: <json>\n\n
    """
    msg = ""
    if event:
        msg += f"event: {event}\n"
    msg += "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
    return msg


class ConversationRunMixin:
    """
    Mixin to attach to your existing ConversationViewSet.
    """

    @method_decorator(csrf_exempt)
    @action(detail=True, methods=["post"], url_path="run", permission_classes=[IsAuthenticated, IsConversationOwner])
    def run(self, request, pk=None):
        conversation = self.get_object()

        user_query = (request.data or {}).get("message", "").strip()
        if not user_query:
            return Response({"detail": "message is required"}, status=status.HTTP_400_BAD_REQUEST)

        # 建 turn stub + token stream
        turn, token_gen, final_meta = run_conversation_streaming(conversation, user_query)

        def stream() -> Generator[str, None, None]:
            start = time.time()
            answer_buf = []

            # 1) 告訴前端 turn_id
            yield sse_event({"type": "start", "turn_id": str(turn.id), "conversation_id": str(conversation.id)}, event="start")

            # 2) token streaming
            for event_type, payload in token_gen:
                if event_type == "token":
                    answer_buf.append(payload)
                    yield sse_event({"type": "token", "text": payload}, event="token")
                elif event_type == "reset":
                    # Clear buffer when pipeline resets (e.g., retry)
                    answer_buf.clear()
                    yield sse_event({"type": "reset"}, event="reset")

            # 3) finalize + persist
            latency_ms = int((time.time() - start) * 1000)
            final_answer = "".join(answer_buf)

            result = build_run_result(answer=final_answer, final_meta=final_meta, latency_ms=latency_ms)
            finalize_turn(turn, result)

            # 4) 收尾：回傳 evidences / meta（UI 右panel 即時顯示）
            evidences_payload = []
            for ev in (final_meta.get("evidences") or []):
                evidences_payload.append(ev)

            yield sse_event({
                "type": "final",
                "turn_id": str(turn.id),
                "assistant_answer": final_answer,
                "intent": final_meta.get("intent", ""),
                "route": final_meta.get("route", ""),
                "confidence": final_meta.get("confidence", 0.0),
                "warnings": final_meta.get("warnings", ""),
                "evidences": evidences_payload,
                "model_meta": final_meta.get("model_meta", {}),
                "retrieval_meta": final_meta.get("retrieval_meta", {}),
                "token_usage": final_meta.get("token_usage", {}),
                "latency_ms": latency_ms,
                "prompt_version": final_meta.get("prompt_version", ""),
            }, event="final")

        resp = StreamingHttpResponse(stream(), content_type="text/event-stream; charset=utf-8")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"  # Nginx: disable buffering for SSE
        return resp

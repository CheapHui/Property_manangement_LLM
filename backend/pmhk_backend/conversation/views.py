# conversations/views.py
from django.db.models import Prefetch
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets, mixins, status
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Conversation, Turn, Evidence
from .permissions import IsConversationOwner
from .serializers import (
    ConversationCreateSerializer,
    ConversationListSerializer,
    ConversationDetailSerializer,
    TurnSerializer,
    TurnCreateSerializer,
    EvidenceSerializer,
)
from .views_run import ConversationRunMixin

@method_decorator(csrf_exempt, name='dispatch')
class ConversationViewSet(
    viewsets.GenericViewSet,
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    ConversationRunMixin
):
    permission_classes = [IsAuthenticated, IsConversationOwner]

    def get_queryset(self):
        # only own conversations
        qs = Conversation.objects.filter(user=self.request.user).order_by("-updated_at")
        # optional filter by workspace_id
        workspace_id = self.request.query_params.get("workspace_id")
        if workspace_id:
            qs = qs.filter(workspace_id=workspace_id)
        return qs

    def get_serializer_class(self):
        if self.action == "create":
            return ConversationCreateSerializer
        if self.action == "retrieve":
            return ConversationDetailSerializer
        return ConversationListSerializer

    def retrieve(self, request, *args, **kwargs):
        """
        Return conversation + turns + evidences (nested).
        Good for UI reopening a chat.
        """
        conv = self.get_object()

        # Prefetch turns + evidences to reduce queries
        turns_qs = Turn.objects.filter(conversation=conv).prefetch_related("evidences").order_by("created_at")
        conv.turns_cache = turns_qs  # not required; for clarity

        serializer = ConversationDetailSerializer(conv)
        return Response(serializer.data)


class TurnViewSet(viewsets.GenericViewSet,
                  mixins.CreateModelMixin,
                  mixins.RetrieveModelMixin):
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # only turns that belong to current user conversations
        return Turn.objects.filter(conversation__user=self.request.user).select_related("conversation")

    def get_serializer_class(self):
        if self.action == "create":
            return TurnCreateSerializer
        return TurnSerializer

    def create(self, request, *args, **kwargs):
        """
        Create a turn. Requires conversation belongs to user.
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        conversation = serializer.validated_data["conversation"]
        if conversation.user_id != request.user.id:
            return Response({"detail": "Not allowed."}, status=status.HTTP_403_FORBIDDEN)

        turn = serializer.save()
        # touch conversation updated_at (auto_now already will update if you save it)
        conversation.save(update_fields=["updated_at"])

        return Response(TurnSerializer(turn).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["get"], url_path="evidences")
    def evidences(self, request, pk=None):
        """
        GET /turns/{turn_id}/evidences/
        """
        turn = self.get_object()
        evs = Evidence.objects.filter(turn=turn).order_by("rank", "-score")
        return Response(EvidenceSerializer(evs, many=True).data)
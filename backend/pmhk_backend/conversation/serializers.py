# conversations/serializers.py
from rest_framework import serializers
from .models import Conversation, Turn, Evidence


class EvidenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Evidence
        fields = [
            "id",
            "source_type",
            "source_id",
            "source_title",
            "excerpt",
            "score",
            "rank",
            "chunk_id",
            "supports_claim",
            "claim_id",
            "meta",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class TurnSerializer(serializers.ModelSerializer):
    evidences = EvidenceSerializer(many=True, read_only=True)

    class Meta:
        model = Turn
        fields = [
            "id",
            "conversation",
            "user_query",
            "assistant_answer",
            "route",
            "intent",
            "reasoning_trace",
            "retrieval_meta",
            "model_meta",
            "prompt_version",
            "warnings",
            "confidence",
            "latency_ms",
            "token_usage",
            "evidences",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class TurnCreateSerializer(serializers.ModelSerializer):
    """
    Create a Turn (and optionally evidences).
    UI / Service layer can POST assistant_answer + evidences after LangChain runs.
    """
    evidences = EvidenceSerializer(many=True, required=False)

    class Meta:
        model = Turn
        fields = [
            "conversation",
            "user_query",
            "assistant_answer",
            "route",
            "intent",
            "reasoning_trace",
            "retrieval_meta",
            "model_meta",
            "prompt_version",
            "warnings",
            "confidence",
            "latency_ms",
            "token_usage",
            "evidences",
        ]

    def create(self, validated_data):
        evidences_data = validated_data.pop("evidences", [])
        turn = Turn.objects.create(**validated_data)

        if evidences_data:
            ev_objs = []
            for i, ev in enumerate(evidences_data):
                ev_objs.append(
                    Evidence(
                        turn=turn,
                        source_type=ev.get("source_type", "other"),
                        source_id=ev["source_id"],
                        source_title=ev.get("source_title", ""),
                        excerpt=ev.get("excerpt", ""),
                        score=float(ev.get("score", 0.0)),
                        rank=int(ev.get("rank", i)),
                        chunk_id=ev.get("chunk_id", ""),
                        supports_claim=bool(ev.get("supports_claim", True)),
                        claim_id=ev.get("claim_id", ""),
                        meta=ev.get("meta", {}),
                    )
                )
            Evidence.objects.bulk_create(ev_objs)

        return turn


class ConversationListSerializer(serializers.ModelSerializer):
    last_turn_at = serializers.DateTimeField(source="updated_at", read_only=True)

    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "is_archived",
            "workspace_id",
            "created_at",
            "updated_at",
            "last_turn_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at", "last_turn_at"]


class ConversationDetailSerializer(serializers.ModelSerializer):
    turns = TurnSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "is_archived",
            "workspace_id",
            "created_at",
            "updated_at",
            "turns",
        ]
        read_only_fields = ["id", "created_at", "updated_at", "turns"]


class ConversationCreateSerializer(serializers.ModelSerializer):
    id = serializers.UUIDField(read_only=True)
    class Meta:
        model = Conversation
        fields = ["id", "title", "workspace_id"]

    def create(self, validated_data):
        request = self.context["request"]
        return Conversation.objects.create(user=request.user, **validated_data)

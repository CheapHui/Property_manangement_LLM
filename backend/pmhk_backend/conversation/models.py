# conversations/models.py
import uuid
from django.conf import settings
from django.db import models


class Conversation(models.Model):
    """
    One conversation thread per user/session.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="conversations",
    )

    title = models.CharField(max_length=255, blank=True, default="")
    is_archived = models.BooleanField(default=False)

    # optional: if you support multi-tenancy / property projects
    workspace_id = models.CharField(max_length=64, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "-updated_at"]),
            models.Index(fields=["-created_at"]),
        ]

    def __str__(self):
        return f"Conversation {self.id}"


class Turn(models.Model):
    """
    One user message + one assistant answer.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name="turns"
    )

    # message content
    user_query = models.TextField()
    assistant_answer = models.TextField(blank=True, default="")

    # routing / intent (your LangChain router output)
    route = models.CharField(max_length=64, blank=True, default="")   # e.g. "mixed"
    intent = models.CharField(max_length=128, blank=True, default="") # e.g. "arrears_recovery"

    # optional: store intermediate states for debugging (keep it light)
    reasoning_trace = models.JSONField(blank=True, default=dict)  # store minimal, not full chain-of-thought
    retrieval_meta = models.JSONField(blank=True, default=dict)   # top_k, vector_store, filters, etc.

    # model config snapshot for audit
    model_meta = models.JSONField(blank=True, default=dict)       # model, temperature, etc.
    prompt_version = models.CharField(max_length=64, blank=True, default="")

    # safety / disclaimers
    warnings = models.TextField(blank=True, default="")
    confidence = models.FloatField(default=0.0)

    # UI / ops
    latency_ms = models.IntegerField(default=0)
    token_usage = models.JSONField(blank=True, default=dict)      # prompt_tokens, completion_tokens, total_tokens

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["conversation", "created_at"]),
            models.Index(fields=["intent"]),
            models.Index(fields=["route"]),
            models.Index(fields=["-created_at"]),
        ]

    def __str__(self):
        return f"Turn {self.id} ({self.intent})"


class Evidence(models.Model):
    """
    Evidence items used for a Turn. This powers:
    - UI Evidence panel
    - audit trail
    - evidence_quotes validator you mentioned
    """
    SOURCE_TYPE_CHOICES = [
        ("statute", "statute"),
        ("case", "case"),
        ("guideline", "guideline"),
        ("dmc", "dmc"),
        ("other", "other"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    turn = models.ForeignKey(Turn, on_delete=models.CASCADE, related_name="evidences")

    source_type = models.CharField(max_length=32, choices=SOURCE_TYPE_CHOICES)
    source_id = models.CharField(max_length=255)      # e.g. "BMO Cap.344 s.34" / "LDBM 218/1999"
    source_title = models.CharField(max_length=255, blank=True, default="")

    # keep excerpt short; full text should live in your doc store/vector store
    excerpt = models.TextField(blank=True, default="")

    # retrieval ranking / similarity score
    score = models.FloatField(default=0.0)
    rank = models.IntegerField(default=0)

    # traceability to your RAG chunk id
    chunk_id = models.CharField(max_length=255, blank=True, default="")

    # to support “evidence_quotes 限制必須支持 claim”
    supports_claim = models.BooleanField(default=True)
    claim_id = models.CharField(max_length=64, blank=True, default="")  # optional mapping to your internal claim list

    meta = models.JSONField(blank=True, default=dict)  # court, date, section, url, etc.

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["turn", "rank"]),
            models.Index(fields=["source_type", "source_id"]),
            models.Index(fields=["chunk_id"]),
        ]

    def __str__(self):
        return f"Evidence {self.source_type}:{self.source_id}"

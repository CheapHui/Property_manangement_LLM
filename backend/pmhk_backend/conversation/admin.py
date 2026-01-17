from django.contrib import admin
from .models import Conversation, Turn, Evidence

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "title", "is_archived", "updated_at", "created_at")
    search_fields = ("id", "title", "user__username", "user__email")
    list_filter = ("is_archived", "created_at")

@admin.register(Turn)
class TurnAdmin(admin.ModelAdmin):
    list_display = ("id", "conversation", "intent", "route", "confidence", "created_at")
    search_fields = ("id", "user_query", "assistant_answer", "intent", "route")
    list_filter = ("intent", "route", "created_at")

@admin.register(Evidence)
class EvidenceAdmin(admin.ModelAdmin):
    list_display = ("id", "turn", "source_type", "source_id", "rank", "score", "supports_claim")
    search_fields = ("source_id", "source_title", "excerpt", "chunk_id")
    list_filter = ("source_type", "supports_claim")
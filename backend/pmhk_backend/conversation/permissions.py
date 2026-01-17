# conversations/permissions.py
from rest_framework.permissions import BasePermission


class IsConversationOwner(BasePermission):
    def has_object_permission(self, request, view, obj):
        # obj can be Conversation
        return obj.user_id == request.user.id
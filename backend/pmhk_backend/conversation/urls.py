# conversations/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ConversationViewSet, TurnViewSet
from .auth_views import login_view, register_view, logout_view, me_view

router = DefaultRouter()
router.register(r"conversations", ConversationViewSet, basename="conversations")
router.register(r"turns", TurnViewSet, basename="turns")

urlpatterns = [
    # Authentication endpoints
    path("auth/login/", login_view, name="auth-login"),
    path("auth/register/", register_view, name="auth-register"),
    path("auth/logout/", logout_view, name="auth-logout"),
    path("auth/me/", me_view, name="auth-me"),

    # Conversation endpoints
    path("", include(router.urls)),
]

"""
Custom authentication classes for API
"""
from rest_framework.authentication import SessionAuthentication


class CsrfExemptSessionAuthentication(SessionAuthentication):
    """
    SessionAuthentication without CSRF check for API-only backend.

    Since this is a pure API backend with CORS protection,
    CSRF tokens are not needed. CORS provides sufficient protection
    against cross-origin attacks.
    """
    def enforce_csrf(self, request):
        return  # Skip CSRF check

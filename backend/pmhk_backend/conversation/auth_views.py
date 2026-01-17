"""
Authentication Views for Property Management AI Agent

Handles user authentication:
- Login
- Register
- Logout
- Get current user
"""

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status


@api_view(['POST'])
@permission_classes([AllowAny])
@csrf_exempt
def login_view(request):
    """
    Login user with username and password

    POST /api/v1/auth/login/
    {
        "username": "user123",
        "password": "password123"
    }
    """
    username = request.data.get('username')
    password = request.data.get('password')

    if not username or not password:
        return Response(
            {'detail': '請提供用戶名和密碼'},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Authenticate user
    user = authenticate(request, username=username, password=password)

    if user is not None:
        # Login user (creates session)
        login(request, user)

        return Response({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'is_staff': user.is_staff,
            'is_superuser': user.is_superuser
        }, status=status.HTTP_200_OK)
    else:
        return Response(
            {'detail': '用戶名或密碼錯誤'},
            status=status.HTTP_401_UNAUTHORIZED
        )


@api_view(['POST'])
@permission_classes([AllowAny])
@csrf_exempt
def register_view(request):
    """
    Register new user

    POST /api/v1/auth/register/
    {
        "username": "newuser",
        "email": "user@example.com",
        "password": "password123"
    }
    """
    username = request.data.get('username')
    email = request.data.get('email')
    password = request.data.get('password')

    # Validation
    if not username or not email or not password:
        return Response(
            {'detail': '請填寫所有欄位'},
            status=status.HTTP_400_BAD_REQUEST
        )

    if len(username) < 3:
        return Response(
            {'detail': '用戶名至少需要3個字符'},
            status=status.HTTP_400_BAD_REQUEST
        )

    if len(password) < 6:
        return Response(
            {'detail': '密碼至少需要6個字符'},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Check if username already exists
    if User.objects.filter(username=username).exists():
        return Response(
            {'detail': '用戶名已被使用'},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Check if email already exists
    if User.objects.filter(email=email).exists():
        return Response(
            {'detail': '電郵已被使用'},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        # Create new user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        # Auto login after registration
        login(request, user)

        return Response({
            'id': user.id,
            'username': user.username,
            'email': user.email
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        return Response(
            {'detail': f'註冊失敗：{str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    """
    Logout current user

    POST /api/v1/auth/logout/
    """
    logout(request)
    return Response(
        {'detail': '已成功登出'},
        status=status.HTTP_200_OK
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def me_view(request):
    """
    Get current authenticated user

    GET /api/v1/auth/me/
    """
    user = request.user

    return Response({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'is_staff': user.is_staff,
        'is_superuser': user.is_superuser
    }, status=status.HTTP_200_OK)

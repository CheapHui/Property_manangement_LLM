/**
 * Authentication API Service
 *
 * Handles user authentication with Django backend:
 * - Login
 * - Register
 * - Logout
 * - Get current user
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Login user
 * @param {string} username
 * @param {string} password
 * @returns {Promise<Object>} User data
 */
export async function login(username, password) {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/login/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: JSON.stringify({ username, password }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || error.message || '登入失敗');
  }

  return response.json();
}

/**
 * Register new user
 * @param {string} username
 * @param {string} email
 * @param {string} password
 * @returns {Promise<Object>} User data
 */
export async function register(username, email, password) {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/register/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: JSON.stringify({ username, email, password }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || error.message || '註冊失敗');
  }

  return response.json();
}

/**
 * Logout current user
 * @returns {Promise<void>}
 */
export async function logout() {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/logout/`, {
    method: 'POST',
    credentials: 'include',
  });

  if (!response.ok) {
    throw new Error('登出失敗');
  }
}

/**
 * Get current authenticated user
 * @returns {Promise<Object>} User data
 */
export async function getCurrentUser() {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/me/`, {
    method: 'GET',
    credentials: 'include',
  });

  if (!response.ok) {
    throw new Error('未登入');
  }

  return response.json();
}

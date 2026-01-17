/**
 * API Service for Property Management AI Agent
 *
 * Handles communication with Django backend:
 * - POST /api/v1/conversations/{conversation_id}/run/
 * - Uses Django session authentication (credentials: 'include')
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Send a message to the AI agent and receive streaming response
 *
 * @param {string} conversationId - Unique conversation ID (UUID from backend)
 * @param {string} message - User's message
 * @returns {Promise<Response>} - Response object with streaming body
 */
export async function sendMessage(conversationId, message) {
  const url = `${API_BASE_URL}/api/v1/conversations/${conversationId}/run/`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include', // Include cookies for Django session auth
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API error (${response.status}): ${errorText}`);
  }

  return response;
}

/**
 * Create a new conversation
 *
 * @param {string} title - Optional conversation title
 * @returns {Promise<Object>} - New conversation object with ID
 */
export async function createConversation(title = 'New Chat') {
  const url = `${API_BASE_URL}/api/v1/conversations/`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: JSON.stringify({ title }),
  });

  if (!response.ok) {
    throw new Error(`Failed to create conversation: ${response.status}`);
  }

  const data = await response.json();
  return data; // Returns { id: UUID, title: ..., created_at: ..., ... }
}

/**
 * Get conversation history
 * (Optional - implement if backend supports history retrieval)
 *
 * @param {string} conversationId - Conversation ID
 * @returns {Promise<Array>} - Array of messages
 */
export async function getConversationHistory(conversationId) {
  const url = `${API_BASE_URL}/api/v1/conversations/${conversationId}/`;

  const response = await fetch(url, {
    method: 'GET',
    credentials: 'include',
  });

  if (!response.ok) {
    throw new Error(`Failed to get conversation history: ${response.status}`);
  }

  const data = await response.json();
  return data.messages || [];
}

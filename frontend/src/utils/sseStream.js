/**
 * SSE Stream Parser for Property Management AI Agent
 *
 * Parses Server-Sent Events (SSE) stream from Django backend
 * - Handles streaming text responses
 * - Parses metadata events (turn_start, evidence, final)
 * - Implements proper error handling
 */

/**
 * Parse SSE stream from response
 *
 * @param {Response} response - Fetch response with streaming body
 * @param {Object} callbacks - Event handlers
 * @param {Function} callbacks.onStart - Called when turn_start event received
 * @param {Function} callbacks.onToken - Called when text chunk received
 * @param {Function} callbacks.onReset - Called when reset event received
 * @param {Function} callbacks.onFinal - Called when final event received
 * @param {Function} callbacks.onError - Called on error
 * @returns {Promise<void>}
 */
export async function parseSSEStream(response, callbacks) {
  const {
    onStart = () => {},
    onToken = () => {},
    onReset = () => {},
    onFinal = () => {},
    onError = () => {}
  } = callbacks;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      // Decode chunk and add to buffer
      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE messages
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        // SSE format: "data: {...}"
        if (line.startsWith('data: ')) {
          const dataStr = line.substring(6).trim();

          if (!dataStr) continue;

          try {
            const data = JSON.parse(dataStr);

            // Handle different event types based on backend protocol
            if (data.type === 'start' || data.type === 'turn_start') {
              // Turn metadata: { type: 'start', turn_id: '...', ... }
              onStart(data);
            } else if (data.type === 'text' || data.type === 'token') {
              // Streaming text chunk: { type: 'text', text: '...' }
              onToken({ text: data.content || data.text || '' });
            } else if (data.type === 'reset') {
              // Reset streaming content (backend retry)
              onReset();
            } else if (data.type === 'final') {
              // Final metadata (turn complete): { type: 'final', assistant_answer: '...', evidences: [...], ... }
              onFinal(data);
            } else if (data.type === 'error') {
              // Error event: { type: 'error', error: { message: '...' } }
              onError(data);
              return;
            }
          } catch (parseError) {
            console.error('Failed to parse SSE data:', dataStr, parseError);
            // Continue processing other messages
          }
        }
      }
    }
  } catch (error) {
    console.error('SSE stream error:', error);
    onError({
      type: 'error',
      error: {
        message: error.message || 'Stream connection failed'
      }
    });
  } finally {
    reader.releaseLock();
  }
}

/**
 * Helper: Check if a response is a valid SSE stream
 * @param {Response} response
 * @returns {boolean}
 */
export function isSSEResponse(response) {
  const contentType = response.headers.get('content-type');
  return contentType && contentType.includes('text/event-stream');
}

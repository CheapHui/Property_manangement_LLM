import { User, Bot, AlertCircle } from 'lucide-react';
import './ChatMessage.css';

/**
 * ChatMessage Component
 *
 * Displays a single message in the chat interface
 * - User messages: aligned right, blue
 * - Assistant messages: aligned left, white with streaming effect
 * - Error messages: red with icon
 */
/**
 * 簡單的 Markdown 格式化函數
 * 支持：**粗體**、換行
 */
function formatMarkdown(text) {
  if (!text) return '';

  // 按行分割
  const lines = text.split('\n');
  const formattedLines = lines.map((line, idx) => {
    // 處理粗體 **text**
    const parts = [];
    let lastIndex = 0;
    const boldRegex = /\*\*(.+?)\*\*/g;
    let match;

    while ((match = boldRegex.exec(line)) !== null) {
      // 添加粗體前的文字
      if (match.index > lastIndex) {
        parts.push(line.substring(lastIndex, match.index));
      }
      // 添加粗體文字
      parts.push(<strong key={`bold-${idx}-${match.index}`}>{match[1]}</strong>);
      lastIndex = match.index + match[0].length;
    }

    // 添加剩餘文字
    if (lastIndex < line.length) {
      parts.push(line.substring(lastIndex));
    }

    return (
      <span key={`line-${idx}`}>
        {parts.length > 0 ? parts : line}
        {idx < lines.length - 1 && <br />}
      </span>
    );
  });

  return formattedLines;
}

export default function ChatMessage({ message, isStreaming = false }) {
  const { role, content, turn_id, error } = message;

  const isUser = role === 'user';
  const isAssistant = role === 'assistant';
  const isError = role === 'error';

  return (
    <div className={`message-wrapper ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className="message-avatar">
        {isUser && <User size={20} />}
        {isAssistant && <Bot size={20} />}
        {isError && <AlertCircle size={20} />}
      </div>

      <div className="message-bubble">
        {isError ? (
          <div className="message-error">
            <strong>錯誤</strong>
            <p>{error || content}</p>
          </div>
        ) : (
          <>
            <div className="message-content">
              {isAssistant ? formatMarkdown(content) : (content || '')}
              {isStreaming && <span className="cursor-blink">▊</span>}
            </div>

            {turn_id && (
              <div className="message-meta">
                Turn ID: {turn_id}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

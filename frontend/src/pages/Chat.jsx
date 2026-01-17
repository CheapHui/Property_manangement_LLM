import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Building2, PanelRightOpen, PanelRightClose, LogOut } from 'lucide-react';
import ChatMessage from '../components/ChatMessage';
import ChatInput from '../components/ChatInput';
import EvidencePanel from '../components/EvidencePanel';
import { useAuth } from '../contexts/AuthContext';
import { sendMessage, createConversation } from '../services/api';
import { parseSSEStream } from '../utils/sseStream';
import './Chat.css';

/**
 * Chat Page Component - Property Management AI Agent
 *
 * Features:
 * - Chat interface with streaming responses
 * - Evidence panel for displaying citations and metadata
 * - SSE streaming support with token-by-token display
 * - User logout functionality
 */
export default function Chat() {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showEvidence, setShowEvidence] = useState(false);
  const [evidenceData, setEvidenceData] = useState(null);
  const [conversationId, setConversationId] = useState(null);
  const [isRetrying, setIsRetrying] = useState(false);
  const [hasNewEvidence, setHasNewEvidence] = useState(false);
  const messagesEndRef = useRef(null);
  const streamingMessageRef = useRef(null);

  // Create conversation on mount
  useEffect(() => {
    const initConversation = async () => {
      try {
        const conv = await createConversation('New Chat');
        setConversationId(conv.id);
      } catch (error) {
        console.error('Failed to create conversation:', error);
        // Handle error - maybe show a toast notification
      }
    };

    initConversation();
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleLogout = async () => {
    const result = await logout();
    if (result.success) {
      navigate('/login');
    }
  };

  const handleSendMessage = async (userMessage) => {
    // Check if conversation is ready
    if (!conversationId) {
      console.error('Conversation not ready yet');
      return;
    }

    // 🔍 DEBUG: 追蹤函數調用
    console.log('🚀 handleSendMessage called with:', userMessage);
    console.log('🆔 Conversation ID:', conversationId);

    // Add user message to chat
    const userMsg = {
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    // Clear previous evidence
    setEvidenceData(null);
    setShowEvidence(false);
    setHasNewEvidence(false);

    try {
      console.log('📡 Sending API request...');
      const response = await sendMessage(conversationId, userMessage);
      console.log('✅ API response received');

      // Initialize streaming message
      streamingMessageRef.current = {
        role: 'assistant',
        content: '',
        turn_id: null,
        timestamp: new Date().toISOString()
      };

      // Handle SSE stream
      await parseSSEStream(response, {
        onStart: (payload) => {
          console.log('🎬 onStart:', payload);
          streamingMessageRef.current.turn_id = payload.turn_id;
          setMessages(prev => [...prev, { ...streamingMessageRef.current }]);
        },

        onToken: (payload) => {
          if (!streamingMessageRef.current) return;
          const tokenText = payload.text || '';
          // console.log('📝 onToken:', tokenText.substring(0, 50) + (tokenText.length > 50 ? '...' : ''));

          // 累積原始內容（但不顯示，因為是 JSON 格式）
          streamingMessageRef.current.content += tokenText;
          // console.log('📝 Total content length now:', streamingMessageRef.current.content.length);

          // Streaming 過程中顯示載入提示，不顯示原始 JSON
          setMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...streamingMessageRef.current,
              content: '💭 正在思考並生成回應...'
            };
            return updated;
          });
        },

        onReset: () => {
          // Clear the current streaming assistant message content (same turn) when backend retries
          console.log('🔄 onReset: 後端重試，清空第一次嘗試的內容');
          setIsRetrying(true);
          if (!streamingMessageRef.current) return;

          // 顯示重試提示
          streamingMessageRef.current.content = '⚠️ 正在修正引用格式，重新生成回應...\n\n';
          setMessages(prev => {
            if (!prev.length) return prev;
            const updated = [...prev];
            updated[updated.length - 1] = { ...streamingMessageRef.current };
            return updated;
          });

          // 短暫延遲後清空，讓用戶看到提示
          setTimeout(() => {
            if (streamingMessageRef.current) {
              streamingMessageRef.current.content = '';
              setMessages(prev => {
                if (!prev.length) return prev;
                const updated = [...prev];
                updated[updated.length - 1] = { ...streamingMessageRef.current };
                return updated;
              });
            }
            setIsRetrying(false);
          }, 300);
        },

        onFinal: (payload) => {
          console.log('[onFinal] Received payload:', payload);
          console.log('[onFinal] Current streaming message:', streamingMessageRef.current);

          if (!streamingMessageRef.current) {
            console.warn('[onFinal] streamingMessageRef is null, skipping update');
            return;
          }

          // 解析 JSON 格式的回應
          let formattedContent = '';
          try {
            // 優先使用 payload.assistant_answer，否則用 streaming 累積的內容
            const rawContent = payload.assistant_answer || (streamingMessageRef.current?.content || '');
            console.log('[onFinal] Raw content length:', rawContent.length);

            // 移除可能的 ```json 標記
            let jsonStr = rawContent.trim();
            if (jsonStr.startsWith('```json')) {
              jsonStr = jsonStr.replace(/^```json\s*/, '').replace(/\s*```$/, '');
            } else if (jsonStr.startsWith('```')) {
              jsonStr = jsonStr.replace(/^```\s*/, '').replace(/\s*```$/, '');
            }

            const parsedData = JSON.parse(jsonStr);
            console.log('[onFinal] Parsed data:', parsedData);

            // 格式化輸出
            const parts = [];

            // 主要摘要
            if (parsedData.answer_summary) {
              parts.push(parsedData.answer_summary);
            }

            // 關鍵點
            if (parsedData.key_points && parsedData.key_points.length > 0) {
              parts.push('\n\n**關鍵要點：**');
              parsedData.key_points.forEach((point, idx) => {
                parts.push(`${idx + 1}. ${point}`);
              });
            }

            // 程序清單
            if (parsedData.procedure_checklist && parsedData.procedure_checklist.length > 0) {
              parts.push('\n\n**處理步驟：**');
              parsedData.procedure_checklist.forEach((step, idx) => {
                parts.push(`${idx + 1}. ${step}`);
              });
            }

            // 決策框架
            if (parsedData.decision_frame && parsedData.decision_frame.length > 0) {
              parts.push('\n\n**決策框架：**');
              parsedData.decision_frame.forEach((item) => {
                parts.push(`• ${item}`);
              });
            }

            // 所需事實
            if (parsedData.required_facts && parsedData.required_facts.length > 0) {
              parts.push('\n\n**需要以下資料：**');
              parsedData.required_facts.forEach((fact) => {
                parts.push(`• ${fact}`);
              });
            }

            // 澄清問題
            if (parsedData.clarifying_questions && parsedData.clarifying_questions.length > 0) {
              parts.push('\n\n**需要澄清：**');
              parsedData.clarifying_questions.forEach((q) => {
                parts.push(`• ${q}`);
              });
            }

            formattedContent = parts.join('\n');

          } catch (error) {
            console.error('[onFinal] Failed to parse JSON:', error);
            // 如果解析失敗，使用原始內容
            formattedContent = payload.assistant_answer || (streamingMessageRef.current?.content || '回應解析失敗');
          }

          console.log('[onFinal] Formatted content length:', formattedContent.length);
          console.log('[onFinal] Formatted content preview:', formattedContent.substring(0, 200));

          // 強制清空並重新設置內容
          if (streamingMessageRef.current) {
            streamingMessageRef.current.content = formattedContent;
            console.log('[onFinal] streamingMessageRef.current.content after set:', streamingMessageRef.current.content.substring(0, 100));
          }

          setMessages(prev => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;
            // 創建新對象確保 React 檢測到變化
            updated[lastIndex] = {
              role: 'assistant',
              content: formattedContent, // 直接使用格式化內容
              turn_id: streamingMessageRef.current?.turn_id || null,
              timestamp: new Date().toISOString()
            };
            console.log('[onFinal] ✅ Setting final message');
            console.log('[onFinal] Updated message content length:', updated[lastIndex].content.length);
            console.log('[onFinal] Updated message content preview:', updated[lastIndex].content.substring(0, 100));
            return updated;
          });

          // 保存證據數據，但不自動顯示，讓用戶手動點擊"資訊"按鈕查看
          if (payload.evidences || payload.warnings || payload.intent) {
            setEvidenceData(payload);
            setHasNewEvidence(true); // 標記有新資訊
            // setShowEvidence(true); // 已移除自動顯示
          }

          setIsLoading(false);
          streamingMessageRef.current = null;
        },

        onError: (payload) => {
          setMessages(prev => [...prev, {
            role: 'error',
            content: '處理請求時發生錯誤',
            error: payload.message || payload.error?.message || '未知錯誤',
            timestamp: new Date().toISOString()
          }]);
          setIsLoading(false);
          streamingMessageRef.current = null;
        }
      });

    } catch (error) {
      console.error('Send message error:', error);
      setMessages(prev => [...prev, {
        role: 'error',
        content: '無法連接到伺服器',
        error: error.message,
        timestamp: new Date().toISOString()
      }]);
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="header-title">
            <Building2 size={28} />
            <div>
              <h1>物業管理 AI 助手</h1>
              <p>Property Management AI Agent</p>
            </div>
          </div>

          <div className="header-actions">
            {user && (
              <span className="user-info">
                {user.username || user.email}
              </span>
            )}

            <button
              className="toggle-evidence-btn"
              onClick={() => {
                setShowEvidence(!showEvidence);
                setHasNewEvidence(false); // 清除提示點
              }}
              aria-label={showEvidence ? 'Hide evidence' : 'Show evidence'}
              style={{ position: 'relative' }}
            >
              {showEvidence ? <PanelRightClose size={20} /> : <PanelRightOpen size={20} />}
              {showEvidence ? '隱藏' : '資訊'}
              {hasNewEvidence && !showEvidence && (
                <span style={{
                  position: 'absolute',
                  top: '6px',
                  right: '6px',
                  width: '8px',
                  height: '8px',
                  background: '#ef4444',
                  borderRadius: '50%',
                  border: '2px solid white'
                }}></span>
              )}
            </button>

            <button
              className="logout-btn"
              onClick={handleLogout}
              aria-label="Logout"
            >
              <LogOut size={20} />
              登出
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="app-main">
        {/* Chat Area */}
        <div className="chat-container">
          <div className="messages-container">
            {messages.length === 0 && (
              <div className="welcome-message">
                <Building2 size={48} strokeWidth={1.5} />
                <h2>歡迎使用物業管理 AI 助手</h2>
                <p>我可以幫你解答香港物業管理相關的法律及實務問題</p>
                <div className="example-questions">
                  <h3>示例問題：</h3>
                  <ul>
                    <li>乜嘢係公用地方？</li>
                    <li>業主立案法團有咩權力？</li>
                    <li>如何處理滲水問題？</li>
                    <li>管理費點樣計算？</li>
                  </ul>
                </div>
              </div>
            )}

            {messages.map((msg, idx) => (
              <ChatMessage
                key={idx}
                message={msg}
                isStreaming={isLoading && idx === messages.length - 1 && !isRetrying}
              />
            ))}

            {isRetrying && (
              <div style={{
                padding: '16px',
                margin: '8px 0',
                background: '#fff3cd',
                border: '1px solid #ffc107',
                borderRadius: '8px',
                color: '#856404',
                fontSize: '14px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <span>⚠️</span>
                <span>檢測到引用格式問題，正在重新生成更準確的回應...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading || !conversationId} />
        </div>

        {/* Evidence Panel */}
        {showEvidence && (
          <EvidencePanel
            data={evidenceData}
            onClose={() => setShowEvidence(false)}
          />
        )}
      </div>
    </div>
  );
}

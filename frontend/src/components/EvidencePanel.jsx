import { FileText, AlertTriangle, Info, X } from 'lucide-react';
import './EvidencePanel.css';

/**
 * EvidencePanel Component
 *
 * Displays evidence, citations, route info, and warnings from the final SSE event
 * - Shows on the right side of the screen
 * - Collapsible sections for each evidence item
 * - Displays intent, confidence, and warnings
 */
export default function EvidencePanel({ data, onClose }) {
  if (!data) return null;

  const {
    evidences = [],
    intent,
    route,
    confidence,
    warnings: rawWarnings,
    property_intent,
    decision_mode
  } = data;

  // Handle warnings as both string and array
  const warnings = Array.isArray(rawWarnings)
    ? rawWarnings
    : (rawWarnings && typeof rawWarnings === 'string' && rawWarnings.trim())
      ? [rawWarnings]
      : [];

  return (
    <div className="evidence-panel">
      <div className="evidence-header">
        <h3>
          <Info size={18} />
          回應資訊
        </h3>
        <button className="close-btn" onClick={onClose} aria-label="Close">
          <X size={20} />
        </button>
      </div>

      <div className="evidence-content">
        {/* Route Info */}
        {(intent || route || property_intent) && (
          <section className="evidence-section">
            <h4>路由資訊</h4>
            <div className="info-grid">
              {property_intent && (
                <div className="info-item">
                  <span className="label">Property Intent:</span>
                  <span className="value">{property_intent}</span>
                </div>
              )}
              {intent && (
                <div className="info-item">
                  <span className="label">Intent:</span>
                  <span className="value">{intent}</span>
                </div>
              )}
              {route && (
                <div className="info-item">
                  <span className="label">Route:</span>
                  <span className="value">{route}</span>
                </div>
              )}
              {confidence !== undefined && (
                <div className="info-item">
                  <span className="label">Confidence:</span>
                  <span className="value">{(confidence * 100).toFixed(1)}%</span>
                </div>
              )}
              {decision_mode !== undefined && (
                <div className="info-item">
                  <span className="label">Decision Mode:</span>
                  <span className="value">{decision_mode ? 'Yes' : 'No'}</span>
                </div>
              )}
            </div>
          </section>
        )}

        {/* Warnings */}
        {warnings.length > 0 && (
          <section className="evidence-section warnings-section">
            <h4>
              <AlertTriangle size={16} />
              警告
            </h4>
            <ul className="warnings-list">
              {warnings.map((warning, idx) => (
                <li key={idx}>{warning}</li>
              ))}
            </ul>
          </section>
        )}

        {/* Evidence Documents */}
        {evidences.length > 0 && (
          <section className="evidence-section">
            <h4>
              <FileText size={16} />
              證據文檔 ({evidences.length})
            </h4>
            <div className="evidence-list">
              {evidences.map((evidence, idx) => (
                <EvidenceItem key={idx} evidence={evidence} index={idx} />
              ))}
            </div>
          </section>
        )}

        {evidences.length === 0 && warnings.length === 0 && !intent && (
          <div className="empty-state">
            <Info size={32} />
            <p>暫無額外資訊</p>
          </div>
        )}
      </div>
    </div>
  );
}

function EvidenceItem({ evidence, index }) {
  // Backend returns: source_type, source_id, source_title, excerpt, score, meta
  const {
    source_type,
    source_id,
    source_title,
    excerpt,
    score,
    meta = {}
  } = evidence;

  // Extract additional fields from meta
  const doc_type = source_type || meta.doc_type || 'other';
  const display = meta.display || source_id;
  const case_no = meta.case_no;
  const section = meta.section;

  const displayName = source_title || display || `Evidence ${index + 1}`;
  const excerptPreview = excerpt ? (excerpt.length > 200 ? excerpt.substring(0, 200) + '...' : excerpt) : '';

  return (
    <details className="evidence-item">
      <summary className="evidence-summary">
        <div className="evidence-title">
          <span className="evidence-badge">{doc_type || 'doc'}</span>
          <span className="evidence-name">{displayName}</span>
        </div>
        {score !== undefined && (
          <span className="evidence-score">{(score * 100).toFixed(0)}%</span>
        )}
      </summary>

      <div className="evidence-details">
        {case_no && (
          <div className="detail-row">
            <strong>案件編號:</strong> {case_no}
          </div>
        )}
        {section && (
          <div className="detail-row">
            <strong>條文:</strong> {section}
          </div>
        )}
        {source_id && (
          <div className="detail-row">
            <strong>引用:</strong>
            <code>{source_id}</code>
          </div>
        )}
        {excerptPreview && (
          <div className="detail-row">
            <strong>內容摘錄:</strong>
            <p className="content-excerpt">{excerptPreview}</p>
          </div>
        )}
      </div>
    </details>
  );
}

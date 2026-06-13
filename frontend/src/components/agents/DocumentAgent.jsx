import React from 'react';
import './agents.css';

const EvidenceCard = ({ evidence }) => {
  return (
    <div className="evidence-card">
      <div className="evidence-quote">"{evidence.quote}"</div>
      <div className="evidence-meta">
        <span className="badge domain">Section: {evidence.section}</span>
        <span className="badge">Page: {evidence.page}</span>
        <span className="badge expertise">Relevance: {evidence.relevance}</span>
      </div>
    </div>
  );
};

const DocumentAgent = ({ data }) => {
  if (!data) return null;

  if (data.intent === 'qa') {
    return (
      <div className="agent-container document-agent">
        <div className="doc-answer">
          <strong>Answer:</strong>
          <p>{data.answer}</p>
          {data.caveat && <div className="caveat-box"><em>Note:</em> {data.caveat}</div>}
        </div>
        
        {data.evidence && data.evidence.length > 0 && (
          <div className="evidence-section">
            <h4>Evidence</h4>
            {data.evidence.map((ev, idx) => <EvidenceCard key={idx} evidence={ev} />)}
          </div>
        )}
      </div>
    );
  }

  // Summary intent
  return (
    <div className="agent-container document-agent">
      <div className="doc-header">
        <span className="badge domain">{data.document_type} Summary</span>
        <span className="badge expertise">{data.length}</span>
      </div>

      <div className="summary-text">
        {data.summary}
      </div>

      {data.key_points && data.key_points.length > 0 && (
        <div className="key-points-section">
          <h4>Key Points</h4>
          <ul>
            {data.key_points.map((kp, idx) => <li key={idx}>{kp}</li>)}
          </ul>
        </div>
      )}

      <div className="doc-meta-stats">
        {data.sections_covered && (
          <div className="sections-covered">
            <strong>Sections Covered:</strong> {data.sections_covered.join(', ')}
          </div>
        )}
        <div className="word-counts">
          Original Words: {data.word_count_original} → Summary Words: {data.word_count_summary}
        </div>
      </div>
    </div>
  );
};

export { DocumentAgent, EvidenceCard };

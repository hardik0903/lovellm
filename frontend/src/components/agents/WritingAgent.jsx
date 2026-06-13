import React, { useState } from 'react';
import './agents.css';

const ChangeCard = ({ change }) => {
  return (
    <div className="change-card">
      <div className="change-header">
        <span className="badge domain">{change.type}</span>
        <span className="change-reason">{change.reason}</span>
      </div>
      <div className="diff-code diff-before">- {change.original_phrase}</div>
      <div className="diff-code diff-after">+ {change.revised_phrase}</div>
    </div>
  );
};

const WritingAgent = ({ data }) => {
  const [showChanges, setShowChanges] = useState(false);

  if (!data) return null;

  return (
    <div className="agent-container writing-agent">
      <div className="writing-header">
        <span className="badge domain">{data.document_type}</span>
        {data.tone && <span className="badge expertise">Tone: {data.tone}</span>}
      </div>

      <div className="writing-result">
        <div className="result-header">
          <strong>Result</strong>
          <button onClick={() => navigator.clipboard.writeText(data.result)}>Copy</button>
        </div>
        <p className="result-text">{data.result}</p>
      </div>

      {data.summary_of_changes && (
        <div className="summary-strip">
          <strong>Summary of Changes:</strong> {data.summary_of_changes}
        </div>
      )}

      {(data.readability_before || data.readability_after) && (
        <div className="readability-stats">
          <div className="stat-box">
            <strong>Readability Before:</strong> {data.readability_before}
          </div>
          <div className="stat-box">
            <strong>Readability After:</strong> {data.readability_after}
          </div>
          <div className="stat-box">
            <strong>Words Before:</strong> {data.word_count_before}
          </div>
          <div className="stat-box">
            <strong>Words After:</strong> {data.word_count_after}
          </div>
        </div>
      )}

      {data.changes && data.changes.length > 0 && (
        <div className="changes-section">
          <button onClick={() => setShowChanges(!showChanges)}>
            {showChanges ? "Hide Detailed Changes" : `Show ${data.changes.length} Detailed Changes`}
          </button>
          
          {showChanges && (
            <div className="changes-list">
              {data.changes.map((change, idx) => (
                <ChangeCard key={idx} change={change} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export { WritingAgent, ChangeCard };

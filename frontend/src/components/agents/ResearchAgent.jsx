import React, { useState } from 'react';
import './agents.css';

const SourceCard = ({ source }) => {
  return (
    <div className="source-card">
      <div className="source-title">
        <a href={source.url} target="_blank" rel="noopener noreferrer">[{source.index}] {source.title}</a>
      </div>
      <div className="source-meta">
        <span className="badge domain">Credibility: {source.credibility_score}</span>
        {source.recency && <span className="badge">Recency: {source.recency}</span>}
      </div>
    </div>
  );
};

const ConflictAlert = ({ conflict }) => {
  return (
    <div className="conflict-alert">
      <strong>⚠️ Conflicting Claim:</strong> {conflict.claim}
      <div className="conflict-details">
        <p><strong>Supporters:</strong> [{conflict.supporters.join(', ')}]</p>
        <p><strong>Opponents:</strong> [{conflict.opponents.join(', ')}]</p>
        <p><strong>Verdict:</strong> {conflict.verdict}</p>
      </div>
    </div>
  );
};

const ResearchAgent = ({ data }) => {
  const [expandedSections, setExpandedSections] = useState({});

  if (!data) return null;

  const toggleSection = (idx) => {
    setExpandedSections(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  return (
    <div className="agent-container research-agent">
      <div className="research-header">
        <h3>Research Report: {data.topic}</h3>
        <span className="badge expertise">Confidence: {data.confidence}</span>
        {data.last_updated && <span className="badge">Updated: {data.last_updated}</span>}
      </div>

      <div className="summary-box">
        <strong>Executive Summary:</strong>
        <p>{data.summary}</p>
      </div>

      {data.conflicting_claims && data.conflicting_claims.length > 0 && (
        <div className="conflicts-section">
          {data.conflicting_claims.map((cc, idx) => <ConflictAlert key={idx} conflict={cc} />)}
        </div>
      )}

      {data.sections && data.sections.length > 0 && (
        <div className="research-sections">
          <h4>Report Sections</h4>
          {data.sections.map((sec, idx) => (
            <div key={idx} className="research-section-item">
              <div className="section-header" onClick={() => toggleSection(idx)}>
                <strong>{sec.title}</strong>
                <span>{expandedSections[idx] ? "▼" : "▶"}</span>
              </div>
              {expandedSections[idx] && (
                <div className="section-content">
                  <p>{sec.content}</p>
                  <div className="section-meta">
                    <span className="badge domain">Consensus: {sec.consensus_level}</span>
                    <span className="badge">Sources: [{sec.sources.join(', ')}]</span>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {data.knowledge_gaps && data.knowledge_gaps.length > 0 && (
        <div className="knowledge-gaps">
          <h4>Knowledge Gaps</h4>
          <ul>
            {data.knowledge_gaps.map((gap, idx) => <li key={idx}>{gap}</li>)}
          </ul>
        </div>
      )}

      {data.sources && data.sources.length > 0 && (
        <div className="sources-list">
          <h4>Sources</h4>
          {data.sources.map((src, idx) => <SourceCard key={idx} source={src} />)}
        </div>
      )}
    </div>
  );
};

export { ResearchAgent, SourceCard, ConflictAlert };

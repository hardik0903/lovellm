import React, { useState } from 'react';
import './agents.css'; // Optional CSS file for agents if needed

const ConceptCard = ({ data }) => {
  const [expandedRow, setExpandedRow] = useState(null);

  if (!data) return null;

  return (
    <div className="concept-card">
      <div className="concept-header">
        <h3>{data.concept}</h3>
        <span className="badge domain">{data.domain}</span>
        <span className="badge expertise">{data.expertise_level}</span>
      </div>
      
      <div className="definition-box">
        <strong>Definition:</strong> {data.definition}
      </div>

      <div className="explanation">
        {data.explanation}
      </div>

      {data.analogy && (
        <div className="analogy-box">
          <strong>Analogy:</strong> {data.analogy}
        </div>
      )}

      {data.components && data.components.length > 0 && (
        <div className="components-list">
          <h4>Components</h4>
          {data.components.map((comp, idx) => (
            <div key={idx} className="component-item">
              <div 
                className="component-header"
                onClick={() => setExpandedRow(expandedRow === idx ? null : idx)}
              >
                <strong>{comp.name}</strong>
                <span>{expandedRow === idx ? "▼" : "▶"}</span>
              </div>
              {expandedRow === idx && (
                <div className="component-details">
                  <p><strong>Role:</strong> {comp.role}</p>
                  <p><strong>Simple Description:</strong> {comp.simple_description}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const KnowledgeAgent = ({ data }) => {
  if (!data) return null;

  return (
    <div className="agent-container knowledge-agent">
      <ConceptCard data={data} />
      
      {data.examples && data.examples.length > 0 && (
        <div className="examples-section">
          <h4>Examples</h4>
          <ul>
            {data.examples.map((ex, idx) => <li key={idx}>{ex}</li>)}
          </ul>
        </div>
      )}

      {data.common_misconceptions && data.common_misconceptions.length > 0 && (
        <div className="misconceptions-section">
          <h4>Common Misconceptions</h4>
          <ul>
            {data.common_misconceptions.map((mc, idx) => <li key={idx}>{mc}</li>)}
          </ul>
        </div>
      )}

      <div className="tags-section">
        {data.related_concepts && data.related_concepts.length > 0 && (
          <div className="related-concepts">
            <strong>Related:</strong>
            {data.related_concepts.map((rc, idx) => <span key={idx} className="tag">{rc}</span>)}
          </div>
        )}
        
        {data.further_reading && data.further_reading.length > 0 && (
          <div className="further-reading">
            <strong>Explore Next:</strong>
            {data.further_reading.map((fr, idx) => <span key={idx} className="tag tag-blue">{fr}</span>)}
          </div>
        )}
      </div>
    </div>
  );
};

export { KnowledgeAgent, ConceptCard };

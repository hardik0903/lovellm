import React from 'react';
import './display.css';
import Markdown from 'markdown-to-jsx';

export default function StepList({ data, content }) {
  if (data.fallback || !data.steps || data.steps.length === 0) {
    return <Markdown options={{ forceBlock: true }}>{content}</Markdown>;
  }

  return (
    <div className="display-component">
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {data.steps.map((step, idx) => (
          <div key={idx} className="step-list-item">
            <div className="step-number">{idx + 1}</div>
            <div className="step-content">
              <h4>{step.title}</h4>
              <p>{step.detail}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

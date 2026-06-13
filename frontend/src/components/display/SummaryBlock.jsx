import React from 'react';
import './display.css';
import Markdown from 'markdown-to-jsx';

export default function SummaryBlock({ data, content }) {
  if (data.fallback || !data.tldr) {
    return <Markdown options={{ forceBlock: true }}>{content}</Markdown>;
  }

  return (
    <div className="display-component summary-block">
      <div className="summary-tldr">
        <strong>TL;DR:</strong> {data.tldr}
      </div>
      
      {data.key_points && data.key_points.length > 0 && (
        <>
          <h4 style={{ color: '#94a3b8', marginBottom: '0.75rem', marginTop: 0 }}>Key Points</h4>
          <ul className="summary-list">
            {data.key_points.map((point, idx) => (
              <li key={idx}>{point}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

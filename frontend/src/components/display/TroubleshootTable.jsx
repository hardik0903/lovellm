import React from 'react';
import './display.css';
import Markdown from 'markdown-to-jsx';

export default function TroubleshootTable({ data, content }) {
  if (data.fallback || !data.issues || data.issues.length === 0) {
    return <Markdown options={{ forceBlock: true }}>{content}</Markdown>;
  }

  return (
    <div className="display-component">
      <table className="display-table">
        <thead>
          <tr>
            <th style={{ width: '30%' }}>Symptom</th>
            <th style={{ width: '30%' }}>Likely Cause</th>
            <th style={{ width: '40%' }}>Fix</th>
          </tr>
        </thead>
        <tbody>
          {data.issues.map((issue, idx) => (
            <tr key={idx}>
              <td style={{ color: '#f87171' }}>{issue.symptom}</td>
              <td>{issue.cause}</td>
              <td style={{ color: '#4ade80' }}>{issue.fix}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

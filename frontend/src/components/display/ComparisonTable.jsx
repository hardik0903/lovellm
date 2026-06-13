import React from 'react';
import './display.css';
import Markdown from 'markdown-to-jsx';

export default function ComparisonTable({ data, content }) {
  if (data.fallback || !data.features || data.features.length === 0) {
    return <Markdown options={{ forceBlock: true }}>{content}</Markdown>;
  }

  const entities = data.entities || Object.keys(data.features[0]).filter(k => k !== 'name');

  return (
    <div className="display-component">
      <table className="display-table">
        <thead>
          <tr>
            <th>Feature</th>
            {entities.map((ent, idx) => (
              <th key={idx}>{ent}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.features.map((feat, idx) => (
            <tr key={idx}>
              <td style={{ fontWeight: 500, color: '#e2e8f0' }}>{feat.name}</td>
              {entities.map((ent, eIdx) => (
                <td key={eIdx}>{feat[ent] || '-'}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

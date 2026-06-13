import React from 'react';
import './display.css';
import Markdown from 'markdown-to-jsx';

export default function StatsTable({ data, content }) {
  if (data.fallback || !data.metrics || data.metrics.length === 0) {
    return <Markdown options={{ forceBlock: true }}>{content}</Markdown>;
  }

  return (
    <div className="display-component">
      <table className="display-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {data.metrics.map((stat, idx) => (
            <tr key={idx}>
              <td style={{ fontWeight: 500 }}>{stat.name}</td>
              <td style={{ color: '#3b82f6', fontWeight: 'bold', fontSize: '1.1em' }}>{stat.value}</td>
              <td style={{ fontSize: '0.85em', color: '#64748b' }}>{stat.source || '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

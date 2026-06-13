import React from 'react';
import './display.css';
import Markdown from 'markdown-to-jsx';

export default function RecommendCard({ data, content }) {
  if (data.fallback || !data.recommendation) {
    return <Markdown options={{ forceBlock: true }}>{content}</Markdown>;
  }

  return (
    <div className="display-component">
      <div className="recommend-header">
        <span className="recommend-badge">Recommendation</span>
        <h3 className="recommend-title">{data.recommendation}</h3>
        <p className="recommend-reason">{data.reason}</p>
      </div>
      
      {data.options && data.options.length > 0 && (
        <table className="display-table">
          <thead>
            <tr>
              <th>Option</th>
              <th>Best For</th>
              <th>Avoid If</th>
            </tr>
          </thead>
          <tbody>
            {data.options.map((opt, idx) => (
              <tr key={idx} style={opt.name === data.recommendation ? { background: 'rgba(59, 130, 246, 0.1)' } : {}}>
                <td style={{ fontWeight: 'bold' }}>
                  {opt.name} {opt.name === data.recommendation ? '🏆' : ''}
                </td>
                <td style={{ color: '#4ade80' }}>{opt.best_for}</td>
                <td style={{ color: '#f87171' }}>{opt.avoid_if}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

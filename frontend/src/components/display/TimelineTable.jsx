import React from 'react';
import './display.css';
import Markdown from 'markdown-to-jsx';

export default function TimelineTable({ data, content }) {
  if (data.fallback || !data.events || data.events.length === 0) {
    return <Markdown options={{ forceBlock: true }}>{content}</Markdown>;
  }

  return (
    <div className="display-component">
      <table className="display-table">
        <thead>
          <tr>
            <th style={{ width: '120px' }}>Period</th>
            <th>Event</th>
          </tr>
        </thead>
        <tbody>
          {data.events.map((evt, idx) => (
            <tr key={idx}>
              <td style={{ fontWeight: 'bold', color: '#3b82f6' }}>{evt.year}</td>
              <td>{evt.event}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

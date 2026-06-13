import React from 'react';
import './display.css';
import Markdown from 'markdown-to-jsx';
import { CheckCircle, XCircle } from 'lucide-react';

export default function ProsConsTable({ data, content }) {
  if (data.fallback || (!data.pros && !data.cons)) {
    return <Markdown options={{ forceBlock: true }}>{content}</Markdown>;
  }

  const pros = data.pros || [];
  const cons = data.cons || [];

  return (
    <div className="display-component">
      <div className="pros-cons-container">
        <div className="pros-column">
          <h4><CheckCircle size={18} /> Pros</h4>
          <ul style={{ paddingLeft: '1.2rem', margin: 0, color: '#cbd5e1' }}>
            {pros.map((p, i) => <li key={i} style={{ marginBottom: '0.5rem' }}>{p}</li>)}
          </ul>
        </div>
        <div className="cons-column">
          <h4><XCircle size={18} /> Cons</h4>
          <ul style={{ paddingLeft: '1.2rem', margin: 0, color: '#cbd5e1' }}>
            {cons.map((c, i) => <li key={i} style={{ marginBottom: '0.5rem' }}>{c}</li>)}
          </ul>
        </div>
      </div>
    </div>
  );
}

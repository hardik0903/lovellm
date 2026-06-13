import React from 'react';
import './agents.css';

const CodeBlock = ({ code, language }) => {
  if (!code) return null;
  return (
    <div className="code-block-container">
      <div className="code-header">
        <span className="code-lang">{language}</span>
        <button onClick={() => navigator.clipboard.writeText(code)}>Copy</button>
      </div>
      <pre><code>{code}</code></pre>
    </div>
  );
};

const CodeDiff = ({ diff }) => {
  if (!diff || diff.length === 0) return null;
  return (
    <div className="code-diff">
      <h4>Changes</h4>
      {diff.map((change, idx) => (
        <div key={idx} className="diff-item">
          <div className="diff-reason">{change.reason}</div>
          <div className="diff-code diff-before">- {change.before}</div>
          <div className="diff-code diff-after">+ {change.after}</div>
        </div>
      ))}
    </div>
  );
};

const CodeAgent = ({ data }) => {
  if (!data) return null;

  return (
    <div className="agent-container code-agent">
      <div className="code-agent-header">
        <span className="badge domain">{data.language}</span>
        <span className="badge expertise">{data.intent}</span>
      </div>
      
      {data.problem_summary && (
        <div className="problem-summary">
          <strong>Summary:</strong> {data.problem_summary}
        </div>
      )}

      {data.intent === 'debug' || data.intent === 'review' ? (
        <CodeDiff diff={data.diff} />
      ) : (
        <CodeBlock code={data.code_after} language={data.language} />
      )}

      <div className="explanation">
        <strong>Explanation:</strong>
        <p>{data.explanation}</p>
      </div>

      <div className="complexity-badges">
        {data.time_complexity && <span className="badge">⏱ {data.time_complexity}</span>}
        {data.space_complexity && <span className="badge">💾 {data.space_complexity}</span>}
      </div>

      {data.alternative_approaches && data.alternative_approaches.length > 0 && (
        <div className="alternatives-section">
          <h4>Alternative Approaches</h4>
          <ul>
            {data.alternative_approaches.map((alt, idx) => <li key={idx}>{alt}</li>)}
          </ul>
        </div>
      )}
      
      {data.common_mistakes && data.common_mistakes.length > 0 && (
        <div className="mistakes-section">
          <h4>Common Mistakes to Avoid</h4>
          <ul>
            {data.common_mistakes.map((mis, idx) => <li key={idx}>{mis}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
};

export { CodeAgent, CodeBlock, CodeDiff };

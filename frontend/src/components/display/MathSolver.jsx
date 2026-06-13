import React, { useState } from 'react';
import './MathDisplay.css';
import MathStepCard from './MathStepCard';
import MathGraph from './MathGraph';
import { formatMathSymbols } from './mathUtils';
import { Check } from 'lucide-react';

const MathSolver = ({ data }) => {
  const [verifyExpanded, setVerifyExpanded] = useState(false);

  if (!data) return null;

  const steps = data.steps || [];

  return (
    <div className="math-solver">
      <div className="math-problem-header">
        <span className="math-problem-expr">{formatMathSymbols(data.problem_statement || "Solving...")}</span>
        <span className="math-category-label">
          {(data.category || 'Mathematics').replace('_', ' ')}
          {data.difficulty && ` · ${data.difficulty}`}
        </span>
      </div>

      {steps.length > 0 && (
        <div className="math-steps">
          {steps.map((step, idx) => (
            <MathStepCard key={idx} step={step} index={idx} />
          ))}
        </div>
      )}

      {data.graph_data && (
        <MathGraph data={data.graph_data} />
      )}

      {data.solution && (
        <div className="math-answer-box">
          <span className="math-answer-value">{formatMathSymbols(data.solution)}</span>
        </div>
      )}

      {data.verification && (
        <div className="math-verify-section">
          <div className="math-verify-strip">
            <Check size={14} color="#16a34a" /> Verified › 
            <button onClick={() => setVerifyExpanded(!verifyExpanded)}>
              {verifyExpanded ? 'Hide check' : 'Show check'}
            </button>
          </div>
          {verifyExpanded && data.verification.check && (
            <div className="math-verify-content">
              {data.verification.check}
            </div>
          )}
        </div>
      )}

      {(data.common_mistakes?.length > 0 || data.related_concepts?.length > 0) && (
        <div className="math-footer-strip">
          {data.common_mistakes && data.common_mistakes.length > 0 && (
            <span>⚠ {data.common_mistakes[0]}</span>
          )}
          
          {(data.common_mistakes?.length > 0 && data.related_concepts?.length > 0) && (
            <span className="separator">|</span>
          )}
          
          {data.related_concepts && data.related_concepts.length > 0 && (
            <span>Related: {data.related_concepts.join(' · ')}</span>
          )}
        </div>
      )}
    </div>
  );
};

export default MathSolver;

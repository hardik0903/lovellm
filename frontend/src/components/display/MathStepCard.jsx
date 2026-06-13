import React from 'react';
import { formatMathSymbols } from './mathUtils';

const MathStepCard = ({ step, index }) => {
  const isKey = step.is_key_step;
  
  return (
    <div 
      className={`math-step-card ${isKey ? 'key-step' : ''}`} 
      style={{ animationDelay: `${index * 0.1}s` }}
    >
      <div className="math-step-header">
        <span className="step-num">{step.step_number}</span>
        <span className="step-title">{step.title}</span>
      </div>
      <hr className="math-step-divider" />
      <div className="math-step-expression-row">
        {step.expression_before && (
          <span className="math-step-expression">{formatMathSymbols(step.expression_before)}</span>
        )}
        {(step.expression_before && step.expression_after) && (
          <span className="math-step-arrow">→</span>
        )}
        {step.expression_after && (
          <span className="math-step-result">{formatMathSymbols(step.expression_after)}</span>
        )}
      </div>
      {step.explanation && (
        <p className="math-step-explanation">{step.explanation}</p>
      )}
    </div>
  );
};

export default MathStepCard;

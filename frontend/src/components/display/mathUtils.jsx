import { Calculator, FunctionSquare, Shapes, TrendingUp, HelpCircle } from 'lucide-react';
import React from 'react';

export const subscriptNumber = (n) => {
  const map = { '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉' };
  return String(n).split('').map(c => map[c] || c).join('');
};

export const superscriptNumber = (n) => {
  const map = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻' };
  return String(n).split('').map(c => map[c] || c).join('');
};

export const formatFunctionName = (str) => {
  const splitRegex = /(tan⁻¹|sin⁻¹|cos⁻¹|cot⁻¹|sec⁻¹|csc⁻¹|\bsin\b|\bcos\b|\btan\b|\bln\b|\blog[₀-₉]*)/g;
  const matchRegex = /^(tan⁻¹|sin⁻¹|cos⁻¹|cot⁻¹|sec⁻¹|csc⁻¹|sin|cos|tan|ln|log[₀-₉]*)$/;
  
  const parts = str.split(splitRegex);
  return parts.map((part, index) => {
    if (part && part.match(matchRegex)) {
      return <span key={index} className="math-fn">{part}</span>;
    }
    return part;
  });
};

export const formatMathSymbols = (str) => {
  if (!str) return '';
  let s = String(str);
  
  // Step 1 - Inverse trig functions
  s = s.replace(/\barctan\b/g, 'tan⁻¹');
  s = s.replace(/\barcsin\b/g, 'sin⁻¹');
  s = s.replace(/\barccos\b/g, 'cos⁻¹');
  s = s.replace(/\barccot\b/g, 'cot⁻¹');
  s = s.replace(/\barcsec\b/g, 'sec⁻¹');
  s = s.replace(/\barccsc\b/g, 'csc⁻¹');
  
  // Step 2 - Logarithms
  s = s.replace(/\blog_?2\b/g, 'log₂');
  s = s.replace(/\blog_?10\b/g, 'log₁₀');
  s = s.replace(/\blog_?([0-9]+)\b/g, (m, p1) => 'log' + subscriptNumber(p1));
  
  // Step 3 - Exponents
  s = s.replace(/\^([\-0-9]+)/g, (m, p1) => superscriptNumber(p1));
  s = s.replace(/\^n\b/g, 'ⁿ');
  s = s.replace(/\bx\^\(-1\)/g, 'x⁻¹');
  s = s.replace(/\be\^x\b/g, 'eˣ');
  
  // Step 4 - Greek letters
  s = s.replace(/\bpi\b/g, 'π');
  s = s.replace(/\bPI\b/g, 'π');
  s = s.replace(/\btheta\b/g, 'θ');
  s = s.replace(/\balpha\b/g, 'α');
  s = s.replace(/\bbeta\b/g, 'β');
  s = s.replace(/\bgamma\b/g, 'γ');
  s = s.replace(/\bdelta\b/g, 'δ');
  s = s.replace(/\bDelta\b/g, 'Δ');
  s = s.replace(/\bsigma\b/g, 'σ');
  s = s.replace(/\bSigma\b/g, 'Σ');
  s = s.replace(/\blambda\b/g, 'λ');
  s = s.replace(/\bmu\b/g, 'μ');
  s = s.replace(/\bepsilon\b/g, 'ε');
  s = s.replace(/\bomega\b/g, 'ω');
  s = s.replace(/\binf\b/g, '∞');
  s = s.replace(/\binfinity\b/g, '∞');
  
  // Step 5 - Calculus symbols
  s = s.replace(/f'\(/g, 'f′(');
  s = s.replace(/f''\(/g, 'f″(');
  s = s.replace(/\bpartial\b/g, '∂');
  s = s.replace(/sqrt\(/g, '√(');
  s = s.replace(/cbrt\(/g, '∛(');
  s = s.replace(/\blim\s+([a-zA-Z])\s*->\s*([^ ]+)/g, 'lim($1→$2)');
  s = s.replace(/\bint\b/g, '∫');
  s = s.replace(/\bintegral\b/g, '∫');
  
  // Standard limits
  s = s.replace(/\[0,1\]/g, '₀¹');
  
  // Roots and fractions
  s = s.replace(/\b1\/2\b/g, '½');
  s = s.replace(/\b1\/3\b/g, '⅓');
  s = s.replace(/\b1\/4\b/g, '¼');
  
  // Step 6 - Operators
  s = s.replace(/!=/g, '≠');
  s = s.replace(/<=/g, '≤');
  s = s.replace(/>=/g, '≥');
  s = s.replace(/=>/g, '⟹');
  s = s.replace(/->/g, '→');
  s = s.replace(/\+-/g, '±');
  s = s.replace(/~=/g, '≈');
  s = s.replace(/\bapprox\b/g, '≈');
  s = s.replace(/\*/g, '·');
  s = s.replace(/\bnot in\b/g, '∉');
  s = s.replace(/\bin\b/g, '∈');
  s = s.replace(/\bsubset\b/g, '⊂');
  s = s.replace(/\bunion\b/g, '∪');
  s = s.replace(/\bintersect\b/g, '∩');

  return formatFunctionName(s);
};

export const getDifficultyColor = (level) => {
  if (level?.toLowerCase() === 'beginner') return 'difficulty-beginner';
  if (level?.toLowerCase() === 'advanced') return 'difficulty-advanced';
  return 'difficulty-intermediate';
};

export const getCategoryIcon = (category) => {
  switch (category) {
    case 'arithmetic':
    case 'algebra':
    case 'quadratic':
      return <Calculator size={14} />;
    case 'calculus_diff':
    case 'calculus_int':
      return <FunctionSquare size={14} />;
    case 'geometry':
    case 'trigonometry':
      return <Shapes size={14} />;
    case 'statistics':
      return <TrendingUp size={14} />;
    default:
      return <HelpCircle size={14} />;
  }
};

export const estimateReadTime = (steps) => {
  if (!steps || steps.length === 0) return "< 1 min";
  const mins = Math.ceil(steps.length * 0.5);
  return `${mins} min read`;
};

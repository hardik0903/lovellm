import React, { useEffect, useRef } from 'react';
import './agents.css';

const StatsBadges = ({ stats }) => {
  if (!stats || Object.keys(stats).length === 0) return null;
  return (
    <div className="stats-badges">
      {Object.entries(stats).map(([k, v], idx) => (
        <span key={idx} className="badge expertise">{k}: {v}</span>
      ))}
    </div>
  );
};

const DataChart = ({ chart }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!chart || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    const width = canvasRef.current.width;
    const height = canvasRef.current.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Very basic placeholder drawing logic to avoid external dependencies
    // You can replace this with a proper lightweight charting function
    ctx.fillStyle = '#f3f4f6';
    ctx.fillRect(0, 0, width, height);
    
    ctx.fillStyle = '#1e3a8a';
    ctx.font = '14px sans-serif';
    ctx.fillText(`[Chart Placeholder: ${chart.title} (${chart.type})]`, 20, 30);
    
    if (chart.x_axis && chart.y_axis) {
      ctx.fillStyle = '#4b5563';
      ctx.fillText(`X: ${chart.x_axis.label} | Y: ${chart.y_axis.label}`, 20, 60);
      ctx.fillText(`Points: ${chart.x_axis.values?.length || 0}`, 20, 80);
    }
    
  }, [chart]);

  if (!chart) return null;

  return (
    <div className="data-chart-container">
      <canvas ref={canvasRef} width="400" height="200" style={{border: "1px solid #e5e7eb", borderRadius: "4px", width: "100%", maxWidth: "500px"}}></canvas>
    </div>
  );
};

const DataAgent = ({ data }) => {
  if (!data) return null;

  return (
    <div className="agent-container data-agent">
      <div className="data-header">
        <span className="badge domain">Data Analysis: {data.insight_type}</span>
      </div>

      <div className="data-answer">
        <strong>Insight:</strong>
        <p>{data.answer}</p>
      </div>

      <StatsBadges stats={data.statistics} />

      <DataChart chart={data.chart} />

      {data.follow_up_questions && data.follow_up_questions.length > 0 && (
        <div className="follow-ups">
          <strong>Follow Up:</strong>
          <div className="follow-up-chips">
            {data.follow_up_questions.map((q, idx) => (
              <span key={idx} className="tag tag-blue cursor-pointer">{q}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export { DataAgent, DataChart, StatsBadges };

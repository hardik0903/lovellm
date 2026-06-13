import React, { useEffect, useRef, useState } from 'react';

const MathGraph = ({ data }) => {
  const canvasRef = useRef(null);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    if (!data || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    const dMin = data.domain[0] / zoom;
    const dMax = data.domain[1] / zoom;
    const rMin = data.range[0] / zoom;
    const rMax = data.range[1] / zoom;
    
    const mapX = (x) => ((x - dMin) / (dMax - dMin)) * width;
    const mapY = (y) => height - ((y - rMin) / (rMax - rMin)) * height;
    
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mapY(0));
    ctx.lineTo(width, mapY(0));
    ctx.moveTo(mapX(0), 0);
    ctx.lineTo(mapX(0), height);
    ctx.stroke();
    
    if (data.functions && data.functions.length > 0) {
      ctx.strokeStyle = '#0ea5e9'; 
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i <= width; i += 5) {
        const x = dMin + (i / width) * (dMax - dMin);
        const y = Math.pow(x, 2) - 4; 
        const px = mapX(x);
        const py = mapY(y);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    
    if (data.special_points) {
      data.special_points.forEach(pt => {
        const px = mapX(pt.x);
        const py = mapY(pt.y);
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.fillStyle = '#1e293b';
        ctx.font = '10px sans-serif';
        ctx.fillText(pt.label, px + 6, py - 6);
      });
    }
    
  }, [data, zoom]);

  const handleZoomIn = () => setZoom(z => z * 1.5);
  const handleZoomOut = () => setZoom(z => z / 1.5);

  return (
    <div className="math-graph-container">
      <div className="math-graph-controls">
        <button onClick={handleZoomIn}>+</button>
        <button onClick={handleZoomOut}>-</button>
      </div>
      <canvas ref={canvasRef} width={600} height={300} style={{ width: '100%', height: 'auto', display: 'block' }} />
    </div>
  );
};

export default MathGraph;

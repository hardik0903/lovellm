import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import DocumentUpload from './components/DocumentUpload';
import { Moon, Sun, MessageSquare, Globe, Bot } from 'lucide-react';

function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'light';
  });
  const [activeMode, setActiveMode] = useState('auto'); // auto, doc, web
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  return (
    <div className="app-container">
      {/* Theme Toggle */}
      <button className="theme-toggle-btn" onClick={toggleTheme} aria-label="Toggle Theme">
        {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
      </button>

      {/* Mobile Topbar */}
      <div className="mobile-topbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 'bold' }}>
          <Bot className="sidebar-icon" size={24} />
          <span>Unified Assistant</span>
        </div>
        <button className="hamburger-btn" onClick={() => setSidebarOpen(!sidebarOpen)}>
          ☰
        </button>
      </div>

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <Bot className="sidebar-icon" size={28} />
          <span>Assistant</span>
        </div>

        <div className="sidebar-section">
          <h3 className="sidebar-title">Upload Knowledge</h3>
          <DocumentUpload onUploadSuccess={() => {}} />
        </div>
        
        <div className="sidebar-section">
          <h3 className="sidebar-title">Routing Mode</h3>
          <div className="mode-selector">
            <button 
              className={`mode-btn ${activeMode === 'auto' ? 'active' : ''}`}
              onClick={() => setActiveMode('auto')}
            >
              <Bot size={18} /> Auto Route
            </button>
            <button 
              className={`mode-btn ${activeMode === 'doc' ? 'active' : ''}`}
              onClick={() => setActiveMode('doc')}
            >
              <MessageSquare size={18} /> Document QA
            </button>
            <button 
              className={`mode-btn ${activeMode === 'web' ? 'active' : ''}`}
              onClick={() => setActiveMode('web')}
            >
              <Globe size={18} /> Web Search
            </button>
          </div>
        </div>
        
        <div style={{ marginTop: 'auto', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          Powered by Llama 3.1 & Hybrid Search
        </div>
      </aside>
      
      {/* Main Chat Content */}
      <ChatInterface activeMode={activeMode} />
    </div>
  );
}

export default App;

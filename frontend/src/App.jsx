import React from 'react';
import DocumentUpload from './components/DocumentUpload';
import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <div className="app-container">
      <aside className="sidebar">
        <div>
          <h1 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Docs Q&A</h1>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
            Upload documents and ask questions grounded in the text.
          </p>
        </div>
        
        <DocumentUpload />
        
        <div style={{ marginTop: 'auto', fontSize: '0.75rem', color: 'var(--text-secondary)', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
          Hybrid Retrieval System
          <br/>BM25 + Dense + Reranking
        </div>
      </aside>
      
      <main className="main-content">
        <ChatInterface />
      </main>
    </div>
  );
}

export default App;

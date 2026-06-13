import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, FileText, AlertTriangle, Globe, Sparkles, Code, Cpu } from 'lucide-react';
import Markdown from 'markdown-to-jsx';

export default function ChatInterface({ activeMode }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const setPrompt = (text) => {
    setInput(text);
    if (inputRef.current) inputRef.current.focus();
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!input.trim() || isLoading) return;

    const queryText = input.trim();
    const userMessage = { role: 'user', content: queryText };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const assistantId = Date.now().toString();
    setMessages(prev => [...prev, { role: 'assistant', id: assistantId, content: '', sources: [], mode: '', confidence: null, needsClarification: false }]);

    // For demonstration of UI logic, we could pass the activeMode flag to the backend if supported.
    // For now, the backend router handles it automatically based on query.

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText })
      });

      if (!response.ok) throw new Error('Network response was not ok');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let done = false;
      let streamedText = "";

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');
          
          let currentEvent = null;
          
          for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (line.startsWith('event: ')) {
              currentEvent = line.substring(7).trim();
            } else if (line.startsWith('data: ')) {
              const dataStr = line.substring(6).trim();
              if (dataStr) {
                try {
                  const data = JSON.parse(dataStr);
                  
                  if (currentEvent === 'delta') {
                    streamedText += data.text;
                    setMessages(prev => prev.map(m => 
                      m.id === assistantId ? { ...m, content: streamedText } : m
                    ));
                  } else if (currentEvent === 'final') {
                    setMessages(prev => prev.map(m => 
                      m.id === assistantId ? { 
                        ...m, 
                        content: data.answer,
                        sources: data.sources || [],
                        mode: data.mode,
                        confidence: data.confidence,
                        needsClarification: data.needs_clarification
                      } : m
                    ));
                  }
                } catch (e) {
                  // ignore parse error for incomplete chunks
                }
              }
            }
          }
        }
      }
    } catch (error) {
      setMessages(prev => prev.map(m => 
        m.id === assistantId ? { ...m, content: 'Error communicating with the server.' } : m
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const renderModeBadge = (mode) => {
    if (!mode) return null;
    let icon = <Bot size={12} />;
    let label = mode;
    
    if (mode === 'doc_rag') {
      icon = <FileText size={12} />;
      label = 'Document QA';
    } else if (mode === 'direct_web') {
      icon = <Globe size={12} />;
      label = 'Direct Web';
    } else if (mode === 'web_rag') {
      icon = <Sparkles size={12} />;
      label = 'Web Synthesized';
    }

    return (
      <div className="mode-badge">
        {icon} {label}
      </div>
    );
  };

  return (
    <main className="main-content">
      {messages.length === 0 ? (
        <div className="welcome-state">
          <div className="welcome-logo">
            <Sparkles size={32} />
          </div>
          <h2>How can I help you today?</h2>
          <p>I can search the web, analyze uploaded documents, or synthesize information from multiple sources.</p>
          
          <div className="example-chips">
            <button className="chip" onClick={() => setPrompt("What is the capital of France?")}>
              <Globe size={16} style={{ display: 'inline', marginRight: '6px', verticalAlign: 'text-bottom' }} />
              Quick fact lookup
            </button>
            <button className="chip" onClick={() => setPrompt("Compare React and Vue in depth")}>
              <Code size={16} style={{ display: 'inline', marginRight: '6px', verticalAlign: 'text-bottom' }} />
              Compare technologies
            </button>
            <button className="chip" onClick={() => setPrompt("According to the uploaded document, what is the main goal?")}>
              <FileText size={16} style={{ display: 'inline', marginRight: '6px', verticalAlign: 'text-bottom' }} />
              Query documents
            </button>
          </div>
        </div>
      ) : (
        <div className="chat-messages">
          {messages.map((msg, idx) => (
            <div key={msg.id || idx} className={`message-wrapper ${msg.role}`}>
              {msg.role === 'assistant' ? (
                <div style={{ width: '100%' }}>
                  <div className="mode-badge-container">
                    {renderModeBadge(msg.mode)}
                    {msg.confidence && (
                      <span className={`confidence-badge confidence-${msg.confidence}`}>
                        {msg.confidence} Confidence
                      </span>
                    )}
                  </div>
                  
                  <div className="message-bubble assistant-content">
                    {msg.content ? (
                      <Markdown options={{ forceBlock: true }}>{msg.content}</Markdown>
                    ) : (
                      <span className="streaming-cursor"></span>
                    )}
                  </div>

                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-grid">
                      {msg.sources.map((src, i) => (
                        <a key={i} href={src.url !== src.title ? src.url : '#'} target="_blank" rel="noreferrer" className="source-card">
                          <div className="source-title">{src.title}</div>
                          <div className="source-meta">
                            <span className="source-type">
                              {src.type === 'web' ? <Globe size={10} /> : <FileText size={10} />}
                              {src.type}
                            </span>
                            {src.type === 'document' && <span>Page 1</span>}
                          </div>
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="message-bubble">
                  {msg.content}
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      )}

      <div className="composer-wrapper">
        <div className="composer-route-hint">
          {activeMode === 'auto' && <><Bot size={12}/> Auto-routing based on query</>}
          {activeMode === 'doc' && <><FileText size={12}/> Forcing Document Mode</>}
          {activeMode === 'web' && <><Globe size={12}/> Forcing Web Search Mode</>}
        </div>
        <form onSubmit={handleSubmit} className="composer-inner">
          <textarea
            ref={inputRef}
            className="composer-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={activeMode === 'doc' ? "Ask about the uploaded documents..." : "Message Assistant..."}
            disabled={isLoading}
            rows={1}
          />
          <button type="submit" className="composer-send-btn" disabled={!input.trim() || isLoading}>
            <Send size={18} />
          </button>
        </form>
      </div>
    </main>
  );
}

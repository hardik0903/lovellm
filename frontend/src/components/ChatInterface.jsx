import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, FileText, AlertTriangle } from 'lucide-react';
import Markdown from 'markdown-to-jsx';

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const assistantId = Date.now().toString();
    // Insert an empty assistant message to stream into
    setMessages(prev => [...prev, { role: 'assistant', id: assistantId, content: '', citations: [], confidence: null, needsClarification: false }]);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.content })
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

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
                        citations: data.citations || [],
                        confidence: data.confidence,
                        needsClarification: data.needs_clarification
                      } : m
                    ));
                  }
                } catch (e) {
                  console.error('Failed to parse SSE data', e, dataStr);
                }
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => prev.map(m => 
        m.id === assistantId ? { ...m, content: 'Error communicating with the server.' } : m
      ));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="chat-messages">
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: 'var(--text-secondary)', marginTop: 'auto', marginBottom: 'auto' }}>
            <Bot size={48} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
            <h2>Document Q&A Service</h2>
            <p>Upload a document and ask questions to get started.</p>
          </div>
        )}
        
        {messages.map((msg, idx) => (
          <div key={msg.id || idx} className={`message-bubble message-${msg.role}`}>
            {msg.role === 'assistant' ? (
              <>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem', gap: '0.5rem', color: 'var(--text-secondary)' }}>
                  <Bot size={16} />
                  <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>Assistant</span>
                  {msg.confidence && (
                    <span className={`confidence-badge confidence-${msg.confidence}`}>
                      {msg.confidence} Confidence
                    </span>
                  )}
                  {msg.needsClarification && (
                    <span className="confidence-badge confidence-medium" style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                      <AlertTriangle size={12} /> Needs Clarification
                    </span>
                  )}
                </div>
                
                <Markdown options={{ forceBlock: true }}>{msg.content || '...'}</Markdown>
                
                {msg.citations && msg.citations.length > 0 && (
                  <div className="citations-container">
                    {msg.citations.map((cite, i) => (
                      <span key={i} className="citation-badge" title={`Chunk: ${cite.chunk_id}`}>
                        <FileText size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '4px' }} />
                        {cite.document_id} (Page {cite.page_start})
                      </span>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem', gap: '0.5rem', color: 'rgba(255,255,255,0.8)' }}>
                  <User size={16} />
                  <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>You</span>
                </div>
                <p>{msg.content}</p>
              </>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            type="text"
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the documents..."
            disabled={isLoading}
          />
          <button type="submit" className="send-button" disabled={!input.trim() || isLoading}>
            <Send size={20} />
          </button>
        </form>
      </div>
    </div>
  );
}

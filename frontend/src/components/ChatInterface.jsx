import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, FileText, AlertTriangle, Globe, Sparkles, Code, Cpu } from 'lucide-react';
import Markdown from 'markdown-to-jsx';
import ComparisonTable from './display/ComparisonTable';
import ProsConsTable from './display/ProsConsTable';
import StepList from './display/StepList';
import TimelineTable from './display/TimelineTable';
import TroubleshootTable from './display/TroubleshootTable';
import RecommendCard from './display/RecommendCard';
import StatsTable from './display/StatsTable';
import SummaryBlock from './display/SummaryBlock';
import MathSolver from './display/MathSolver';
import { KnowledgeAgent } from './agents/KnowledgeAgent';
import { CodeAgent } from './agents/CodeAgent';
import { WritingAgent } from './agents/WritingAgent';
import { DocumentAgent } from './agents/DocumentAgent';
import { ResearchAgent } from './agents/ResearchAgent';
import { DataAgent } from './agents/DataAgent';

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
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText, mode: activeMode })
      });

      if (!response.ok) throw new Error('Network response was not ok');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let done = false;
      let streamedText = "";
      let sseBuffer = "";
      let currentEvent = null;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          sseBuffer += decoder.decode(value, { stream: true });
          const lines = sseBuffer.split('\n');
          
          // Keep the last incomplete line in the buffer
          sseBuffer = lines.pop() || "";
          
          for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (line.startsWith('event: ')) {
              currentEvent = line.substring(7).trim();
            } else if (line.startsWith('data: ')) {
              const dataStr = line.substring(6).trim();
              if (dataStr) {
                try {
                  const data = JSON.parse(dataStr);
                  
                  if (currentEvent === 'delta' || currentEvent === 'math_thinking') {
                    streamedText += data.text || data.delta || "";
                    setMessages(prev => prev.map(m => 
                      m.id === assistantId ? { ...m, content: streamedText } : m
                    ));
                  } else if (currentEvent === 'math_step') {
                    setMessages(prev => prev.map(m => {
                      if (m.id === assistantId) {
                        const existingDisplay = m.display || { type: 'math_solution', steps: [] };
                        const existingSteps = existingDisplay.steps || [];
                        return {
                          ...m,
                          display: {
                            ...existingDisplay,
                            type: 'math_solution',
                            steps: [...existingSteps, data]
                          }
                        };
                      }
                      return m;
                    }));
                  } else if (currentEvent === 'final') {
                    setMessages(prev => prev.map(m => 
                      m.id === assistantId ? { 
                        ...m, 
                        content: data.answer,
                        sources: data.sources || [],
                        mode: data.mode,
                        confidence: data.confidence,
                        needsClarification: data.needs_clarification,
                        display: data.display || null,
                        routedAgent: data.routed_agent,
                        uncertaintyFlag: data.uncertainty_flag
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

  const renderDisplay = (msg) => {
    if (!msg.content && !msg.display) {
      return <span className="streaming-cursor"></span>;
    }
    
    if (!msg.display) {
      return <Markdown options={{ forceBlock: true }}>{msg.content}</Markdown>;
    }

    if (msg.mode === "knowledge") return <KnowledgeAgent data={msg.display} />;
    if (msg.mode === "code") return <CodeAgent data={msg.display} />;
    if (msg.mode === "writing") return <WritingAgent data={msg.display} />;
    if (msg.mode === "document") return <DocumentAgent data={msg.display} />;
    if (msg.mode === "research") return <ResearchAgent data={msg.display} />;
    if (msg.mode === "data") return <DataAgent data={msg.display} />;

    switch (msg.display?.type) {
      case "comparison_table": return <ComparisonTable data={msg.display} content={msg.content} />;
      case "pros_cons_table":  return <ProsConsTable data={msg.display} content={msg.content} />;
      case "step_list":        return <StepList data={msg.display} content={msg.content} />;
      case "timeline_table":   return <TimelineTable data={msg.display} content={msg.content} />;
      case "troubleshoot_table": return <TroubleshootTable data={msg.display} content={msg.content} />;
      case "recommend_card":   return <RecommendCard data={msg.display} content={msg.content} />;
      case "stats_table":      return <StatsTable data={msg.display} content={msg.content} />;
      case "summary_block":    return <SummaryBlock data={msg.display} content={msg.content} />;
      case "math_solution":    return <MathSolver data={msg.display} />;
      default:                 return <Markdown options={{ forceBlock: true }}>{msg.content}</Markdown>;
    }
  };

  const renderModeBadge = (msg) => {
    let mode = msg.mode;
    if (!mode && !msg.routedAgent) return null;
    let icon = <Bot size={12} />;
    let label = mode;
    
    if (msg.routedAgent) {
      label = msg.routedAgent.charAt(0).toUpperCase() + msg.routedAgent.slice(1) + ' Agent';
      icon = <Sparkles size={12} />;
    } else if (mode === 'doc_rag') {
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
        {msg.uncertaintyFlag && <span style={{marginLeft: "6px", fontSize: "10px", opacity: 0.8}}>(uncertain routing)</span>}
      </div>
    );
  };

  const isValidUrl = (url) => {
    if (!url || typeof url !== 'string') return false;
    try {
      const parsed = new URL(url);
      return parsed.protocol === 'http:' || parsed.protocol === 'https:';
    } catch {
      return false;
    }
  };

  const renderSourceCard = (src, i) => {
    const validUrl = isValidUrl(src.url);
    const cardContent = (
      <>
        <div className="source-title">{src.title || 'Unknown source'}</div>
        <div className="source-meta">
          <span className="source-type">
            {src.type === 'web' ? <Globe size={10} /> : <FileText size={10} />}
            {src.type}
          </span>
          {src.type === 'document' && src.page && <span>Page {src.page}</span>}
          {validUrl && (
            <span
              style={{ fontSize: '10px', opacity: 0.6, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '120px' }}
            >
              {new URL(src.url).hostname}
            </span>
          )}
        </div>
      </>
    );

    return validUrl ? (
      <a
        key={i}
        href={src.url}
        target="_blank"
        rel="noreferrer noopener"
        className="source-card"
      >
        {cardContent}
      </a>
    ) : (
      <div key={i} className="source-card" style={{ cursor: 'default' }}>
        {cardContent}
      </div>
    );
  };

  return (
    <main className="main-content">
      {messages.length === 0 ? (
        <div className="welcome-state">
          <div className="welcome-logo">
            <img src="/logo.png" alt="LoveLLM Logo" />
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
                    {renderModeBadge(msg)}
                    {msg.confidence && (
                      <span className={`confidence-badge confidence-${msg.confidence}`}>
                        {msg.confidence} Confidence
                      </span>
                    )}
                  </div>
                  
                  <div className="message-bubble assistant-content">
                    {renderDisplay(msg)}
                  </div>

                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-grid">
                      {msg.sources.map((src, i) => renderSourceCard(src, i))}
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

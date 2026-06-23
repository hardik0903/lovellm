/**
 * TelemetryDashboard.jsx
 *
 * Drop-in page component styled to match the LoveLLM design system exactly:
 *   - CSS vars from index.css (--bg-main, --bg-surface, --accent-color, etc.)
 *   - Inter font, same radii & shadows
 *   - Light / dark mode via data-theme="dark" on <html>
 *
 * Mount in App.jsx alongside ChatInterface:
 *
 *   import TelemetryDashboard from './components/TelemetryDashboard';
 *   // In the sidebar, add:
 *   <button className={`mode-btn ${activeMode==='telemetry'?'active':''}`}
 *           onClick={() => setActiveMode('telemetry')}>
 *     <Activity size={18}/> Telemetry
 *   </button>
 *   // In the main area:
 *   {activeMode === 'telemetry'
 *     ? <TelemetryDashboard />
 *     : <ChatInterface activeMode={activeMode} />}
 *
 * Auth: POST /telemetry/auth/login  { password }  → { token }
 *       Token stored in sessionStorage (cleared on tab close).
 *       Set TELEMETRY_SECRET in backend .env.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Activity, Lock, LogOut, RefreshCw, TrendingUp, Cpu,
  AlertTriangle, Clock, Zap, ChevronDown, BarChart2,
  CheckCircle, XCircle, Shield
} from 'lucide-react';

// ── constants ──────────────────────────────────────────────────────────────────

const BASE = '/telemetry';
const SESSION_KEY = 'telemetry_token';

const AGENTS = ['math', 'code', 'data', 'document', 'writing', 'research', 'knowledge'];

// Each agent gets a distinct hue that doesn't clash with the --accent-color (#2563EB / #00D6FF)
const AGENT_COLOR = {
  math:      '#8B5CF6', // violet
  code:      '#10B981', // emerald
  data:      '#F59E0B', // amber
  document:  '#EC4899', // pink
  writing:   '#06B6D4', // cyan
  research:  '#EF4444', // red
  knowledge: '#64748B', // slate
  fallback:  '#94A3B8', // muted
};

const WINDOWS = [
  { label: '1 h', value: 1 },
  { label: '6 h', value: 6 },
  { label: '24 h', value: 24 },
  { label: '7 d', value: 168 },
  { label: '30 d', value: 720 },
];

// ── auth helpers ───────────────────────────────────────────────────────────────

function getToken()          { return sessionStorage.getItem(SESSION_KEY); }
function setToken(t)         { sessionStorage.setItem(SESSION_KEY, t); }
function clearToken()        { sessionStorage.removeItem(SESSION_KEY); }

async function apiFetch(path, opts = {}) {
  const token = getToken();
  const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(`${BASE}${path}`, { ...opts, headers });
  if (res.status === 401) throw Object.assign(new Error('Unauthorized'), { status: 401 });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// ── sub-components ─────────────────────────────────────────────────────────────

// Inline sparkline (pure SVG, no chart lib dependency)
function Sparkline({ values = [], color = '#2563EB', height = 32 }) {
  if (values.length < 2) return null;
  const w = 96, h = height;
  const max = Math.max(...values, 1);
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = h - (v / max) * (h - 4) - 2;
    return `${x},${y}`;
  }).join(' ');
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block' }}>
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity="0.8"
      />
    </svg>
  );
}

// Tiny horizontal bar for agent distribution
function MiniBar({ value, max, color }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div style={{ height: 6, background: 'var(--border-color)', borderRadius: 99, overflow: 'hidden' }}>
      <div style={{
        height: '100%',
        width: `${pct}%`,
        background: color,
        borderRadius: 99,
        transition: 'width 0.6s cubic-bezier(0.4,0,0.2,1)',
      }} />
    </div>
  );
}

function AgentPill({ agent, size = 'sm' }) {
  const color = AGENT_COLOR[agent] || '#64748B';
  const fs = size === 'sm' ? 11 : 12;
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 4,
      fontSize: fs,
      fontWeight: 600,
      padding: size === 'sm' ? '2px 8px' : '3px 10px',
      borderRadius: 'var(--radius-pill)',
      background: `${color}18`,
      color,
      border: `1px solid ${color}38`,
      letterSpacing: '0.01em',
    }}>
      <span style={{ width: 5, height: 5, borderRadius: '50%', background: color, display: 'inline-block' }} />
      {agent || '—'}
    </span>
  );
}

function StatCard({ icon: Icon, label, value, sub, trend, color, sparkValues }) {
  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: '1px solid var(--border-color)',
      borderRadius: 'var(--radius-lg)',
      padding: '1.25rem',
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
      boxShadow: 'var(--shadow-sm)',
      transition: 'box-shadow 0.2s',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 32, height: 32,
            borderRadius: 8,
            background: `${color || 'var(--accent-color)'}18`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Icon size={16} color={color || 'var(--accent-color)'} />
          </div>
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            {label}
          </span>
        </div>
        {sparkValues && <Sparkline values={sparkValues} color={color || 'var(--accent-color)'} />}
      </div>
      <div>
        <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1.1 }}>
          {value ?? '—'}
        </div>
        {sub && (
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 3 }}>{sub}</div>
        )}
      </div>
    </div>
  );
}

function SectionHead({ children }) {
  return (
    <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-secondary)', marginBottom: 12 }}>
      {children}
    </div>
  );
}

function Card({ children, style = {} }) {
  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: '1px solid var(--border-color)',
      borderRadius: 'var(--radius-lg)',
      padding: '1.25rem',
      boxShadow: 'var(--shadow-sm)',
      ...style,
    }}>
      {children}
    </div>
  );
}

// Minimal bar chart rendered as SVG — no recharts dependency
function BarChartSVG({ data, height = 140 }) {
  // data: [{ label, value, color }]
  if (!data || data.length === 0) return null;
  const max = Math.max(...data.map(d => d.value), 1);
  const w = 480, padT = 8, padB = 24, padL = 0, padR = 0;
  const chartH = height - padT - padB;
  const barW = Math.floor((w - padL - padR) / data.length);
  const gap = Math.max(3, Math.floor(barW * 0.2));
  const bw  = barW - gap;

  return (
    <svg viewBox={`0 0 ${w} ${height}`} width="100%" height={height} style={{ overflow: 'visible' }}>
      {data.map((d, i) => {
        const barH = Math.max(2, (d.value / max) * chartH);
        const x = padL + i * barW + gap / 2;
        const y = padT + chartH - barH;
        return (
          <g key={i}>
            <rect x={x} y={y} width={bw} height={barH} rx={3} fill={d.color} opacity="0.85" />
            <text
              x={x + bw / 2}
              y={height - 6}
              textAnchor="middle"
              fontSize={10}
              fill="var(--text-secondary)"
              fontFamily="Inter, sans-serif"
            >
              {d.label}
            </text>
            {d.value > 0 && (
              <text
                x={x + bw / 2}
                y={y - 3}
                textAnchor="middle"
                fontSize={10}
                fontWeight="600"
                fill="var(--text-primary)"
                fontFamily="Inter, sans-serif"
              >
                {d.value}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

// ── LOGIN SCREEN ───────────────────────────────────────────────────────────────

function LoginScreen({ onLogin }) {
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!password.trim()) return;
    setLoading(true);
    setError('');
    try {
      const data = await apiFetch('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ password }),
      });
      setToken(data.token);
      onLogin();
    } catch (err) {
      setError(err.message === 'Unauthorized' || err.status === 401
        ? 'Incorrect password.'
        : 'Could not reach the server. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      flex: 1,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'var(--bg-main)',
      padding: '2rem',
    }}>
      <div style={{
        width: '100%',
        maxWidth: 380,
        display: 'flex',
        flexDirection: 'column',
        gap: '1.5rem',
        animation: 'fadeIn 0.4s ease',
      }}>
        {/* Icon + header */}
        <div style={{ textAlign: 'center' }}>
          <div style={{
            width: 56, height: 56,
            borderRadius: 16,
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-color)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 1rem',
            boxShadow: 'var(--shadow-md)',
          }}>
            <Shield size={26} color="var(--accent-color)" />
          </div>
          <h2 style={{ fontSize: '1.375rem', fontWeight: 700, margin: '0 0 0.375rem', color: 'var(--text-primary)' }}>
            Telemetry access
          </h2>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', margin: 0 }}>
            Enter the TELEMETRY_SECRET to view routing analytics
          </p>
        </div>

        {/* Form */}
        <div style={{
          background: 'var(--bg-surface)',
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-lg)',
          padding: '1.5rem',
          boxShadow: 'var(--shadow-sm)',
        }}>
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Password
            </label>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              background: 'var(--bg-main)',
              border: `1px solid ${error ? 'var(--error-color)' : 'var(--border-color)'}`,
              borderRadius: 'var(--radius-md)',
              padding: '0.5rem 0.75rem',
              transition: 'border-color 0.2s',
            }}>
              <Lock size={15} color="var(--text-secondary)" />
              <input
                ref={inputRef}
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="Enter password"
                disabled={loading}
                style={{
                  flex: 1,
                  border: 'none',
                  background: 'transparent',
                  outline: 'none',
                  fontSize: '0.9375rem',
                  color: 'var(--text-primary)',
                  fontFamily: 'inherit',
                }}
              />
            </div>

            {error && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 6,
                fontSize: 13, color: 'var(--error-color)',
                background: 'rgba(239,68,68,0.08)',
                borderRadius: 8, padding: '0.5rem 0.75rem',
              }}>
                <AlertTriangle size={14} />
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !password.trim()}
              style={{
                marginTop: 4,
                padding: '0.65rem',
                borderRadius: 'var(--radius-md)',
                border: 'none',
                background: 'var(--accent-color)',
                color: 'white',
                fontFamily: 'inherit',
                fontSize: '0.9375rem',
                fontWeight: 600,
                cursor: loading || !password.trim() ? 'not-allowed' : 'pointer',
                opacity: loading || !password.trim() ? 0.6 : 1,
                transition: 'opacity 0.2s, transform 0.1s',
              }}
            >
              {loading ? 'Verifying…' : 'Sign in'}
            </button>
          </form>
        </div>

        <p style={{ textAlign: 'center', fontSize: 12, color: 'var(--text-secondary)' }}>
          Session clears when you close this tab
        </p>
      </div>
    </div>
  );
}

// ── MAIN DASHBOARD ─────────────────────────────────────────────────────────────

export default function TelemetryDashboard() {
  // auth state
  const [authed, setAuthed]   = useState(false);
  const [authReady, setAuthReady] = useState(false);

  // data state
  const [window, setWindow]   = useState(24);
  const [summary, setSummary] = useState(null);
  const [timeseries, setTs]   = useState(null);
  const [confDist, setConf]   = useState(null);
  const [recent, setRecent]   = useState(null);
  const [agentFilter, setAgentFilter] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);

  // Check stored token on mount
  useEffect(() => {
    const stored = getToken();
    if (!stored) { setAuthReady(true); return; }
    apiFetch('/auth/verify')
      .then(d => { if (d.valid) setAuthed(true); })
      .catch(() => clearToken())
      .finally(() => setAuthReady(true));
  }, []);

  const loadData = useCallback(async () => {
    if (!authed) return;
    setLoading(true);
    setError(null);
    try {
      const q = `?since_hours=${window}`;
      const [s, ts, cd, rd] = await Promise.all([
        apiFetch(`/summary${q}`),
        apiFetch(`/routing/timeseries${q}`),
        apiFetch(`/routing/confidence_distribution${q}&bins=20`),
        apiFetch(`/recent_decisions${q}&limit=50${agentFilter ? `&agent=${agentFilter}` : ''}`),
      ]);
      setSummary(s);
      setTs(ts);
      setConf(cd);
      setRecent(rd);
      setLastRefresh(new Date());
    } catch (e) {
      if (e.status === 401) { clearToken(); setAuthed(false); }
      else setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [authed, window, agentFilter]);

  useEffect(() => { loadData(); }, [loadData]);

  const handleLogout = () => { clearToken(); setAuthed(false); setSummary(null); };

  if (!authReady) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-main)' }}>
        <div style={{ color: 'var(--text-secondary)', fontSize: 14 }}>Checking session…</div>
      </div>
    );
  }

  if (!authed) {
    return <LoginScreen onLogin={() => { setAuthed(true); }} />;
  }

  // ── derived chart data ───────────────────────────────────────────────────────

  const agentDistBars = AGENTS.map(a => ({
    label: a.slice(0, 4),
    value: summary?.agent_distribution?.[a] || 0,
    color: AGENT_COLOR[a],
  })).filter(d => d.value > 0);

  const confBars = confDist?.bin_edges?.length
    ? confDist.counts.map((count, i) => ({
        label: confDist.bin_edges[i].toFixed(1),
        value: count,
        color: count > 0 ? `hsl(${220 + i * 4}, 70%, 55%)` : 'var(--border-color)',
      }))
    : [];

  const tsRows = timeseries
    ? Object.entries(timeseries.series).map(([hour, counts]) => ({
        hour: hour.slice(11, 16),
        ...counts,
      }))
    : [];

  // Sparkline for queries-per-hour trend
  const hourlyTotals = tsRows.map(r =>
    AGENTS.reduce((s, a) => s + (r[a] || 0), 0) + (r.fallback || 0)
  );

  const maxAgentCount = Math.max(...(agentDistBars.map(d => d.value)), 1);

  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--bg-main)',
      overflowY: 'auto',
      animation: 'fadeIn 0.3s ease',
    }}>
      {/* ── top bar ────────────────────────────────────────────────────────── */}
      <div style={{
        position: 'sticky',
        top: 0,
        zIndex: 10,
        background: 'var(--bg-surface)',
        borderBottom: '1px solid var(--border-color)',
        padding: '0.875rem 1.75rem',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
      }}>
        <Activity size={18} color="var(--accent-color)" />
        <span style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>Telemetry</span>

        <div style={{ flex: 1 }} />

        {lastRefresh && (
          <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
            Updated {lastRefresh.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </span>
        )}

        {/* Window selector */}
        <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
          <select
            value={window}
            onChange={e => setWindow(Number(e.target.value))}
            style={{
              appearance: 'none',
              background: 'var(--bg-main)',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-md)',
              padding: '0.35rem 2rem 0.35rem 0.75rem',
              fontSize: 12,
              fontWeight: 600,
              color: 'var(--text-primary)',
              fontFamily: 'inherit',
              cursor: 'pointer',
            }}
          >
            {WINDOWS.map(w => (
              <option key={w.value} value={w.value}>{w.label}</option>
            ))}
          </select>
          <ChevronDown size={12} color="var(--text-secondary)" style={{ position: 'absolute', right: 8, pointerEvents: 'none' }} />
        </div>

        <button
          onClick={loadData}
          disabled={loading}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '0.375rem 0.875rem',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-color)',
            background: 'var(--bg-main)',
            color: 'var(--text-secondary)',
            fontSize: 12, fontWeight: 600, fontFamily: 'inherit',
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s',
          }}
        >
          <RefreshCw size={13} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
          Refresh
        </button>

        <button
          onClick={handleLogout}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '0.375rem 0.875rem',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-color)',
            background: 'transparent',
            color: 'var(--text-secondary)',
            fontSize: 12, fontWeight: 600, fontFamily: 'inherit',
            cursor: 'pointer',
          }}
        >
          <LogOut size={13} />
          Sign out
        </button>
      </div>

      {/* ── error banner ───────────────────────────────────────────────────── */}
      {error && (
        <div style={{
          margin: '1.25rem 1.75rem 0',
          padding: '0.75rem 1rem',
          background: 'rgba(239,68,68,0.08)',
          border: '1px solid rgba(239,68,68,0.25)',
          borderRadius: 'var(--radius-md)',
          display: 'flex', alignItems: 'center', gap: 8,
          fontSize: 13, color: 'var(--error-color)',
        }}>
          <AlertTriangle size={15} />
          {error} — check that the backend is running and /telemetry is mounted.
        </div>
      )}

      {/* ── body ───────────────────────────────────────────────────────────── */}
      <div style={{ padding: '1.5rem 1.75rem', display: 'flex', flexDirection: 'column', gap: '1.75rem', maxWidth: 1140, width: '100%', margin: '0 auto' }}>

        {/* KPI cards */}
        {summary && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 12 }}>
            <StatCard
              icon={TrendingUp}
              label="Queries routed"
              value={summary.total_queries.toLocaleString()}
              sub={`in last ${window < 24 ? `${window}h` : window === 168 ? '7d' : window === 720 ? '30d' : '24h'}`}
              color="var(--accent-color)"
              sparkValues={hourlyTotals}
            />
            <StatCard
              icon={Zap}
              label="Fallback rate"
              value={`${(summary.fallback_rate * 100).toFixed(1)}%`}
              sub={`${summary.fallback_count} queries fell through`}
              color={summary.fallback_rate > 0.15 ? '#EF4444' : '#16A34A'}
            />
            <StatCard
              icon={Activity}
              label="Mean confidence"
              value={summary.confidence.mean != null ? summary.confidence.mean.toFixed(3) : '—'}
              sub={`p95: ${summary.confidence.p95?.toFixed(3) ?? '—'}`}
              color="#8B5CF6"
            />
            <StatCard
              icon={Cpu}
              label="Agents active"
              value={Object.keys(summary.latency_by_agent).length}
              sub="with execution records"
              color="#10B981"
            />
          </div>
        )}

        {/* Row: distribution + confidence histogram */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

          {/* Agent routing distribution */}
          <Card>
            <SectionHead>Routing distribution</SectionHead>
            {summary && agentDistBars.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 4 }}>
                {AGENTS.map(a => {
                  const count = summary.agent_distribution?.[a] || 0;
                  if (!count) return null;
                  const total = summary.total_queries || 1;
                  return (
                    <div key={a}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        <AgentPill agent={a} />
                        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                          {count}
                          <span style={{ color: 'var(--text-secondary)', fontWeight: 400, marginLeft: 4 }}>
                            ({((count / total) * 100).toFixed(0)}%)
                          </span>
                        </span>
                      </div>
                      <MiniBar value={count} max={maxAgentCount} color={AGENT_COLOR[a]} />
                    </div>
                  );
                })}
                {(summary.agent_distribution?.fallback > 0) && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                      <AgentPill agent="fallback" />
                      <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                        {summary.agent_distribution.fallback}
                      </span>
                    </div>
                    <MiniBar value={summary.agent_distribution.fallback} max={maxAgentCount} color={AGENT_COLOR.fallback} />
                  </div>
                )}
              </div>
            ) : (
              <Empty>No routing data in this window</Empty>
            )}
          </Card>

          {/* Confidence distribution */}
          <Card>
            <SectionHead>Confidence score distribution</SectionHead>
            {confBars.length > 0
              ? <BarChartSVG data={confBars} height={148} />
              : <Empty>No confidence data in this window</Empty>
            }
          </Card>
        </div>

        {/* Queries per hour */}
        {tsRows.length > 0 && (
          <Card>
            <SectionHead>Queries over time</SectionHead>
            <div style={{ overflowX: 'auto' }}>
              <BarChartSVG
                data={tsRows.map(r => ({
                  label: r.hour,
                  value: AGENTS.reduce((s, a) => s + (r[a] || 0), 0) + (r.fallback || 0),
                  color: 'var(--accent-color)',
                }))}
                height={140}
              />
            </div>
            {/* legend */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--border-color)' }}>
              {AGENTS.map(a => (
                <span key={a} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-secondary)' }}>
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: AGENT_COLOR[a], display: 'inline-block' }} />
                  {a}
                </span>
              ))}
            </div>
          </Card>
        )}

        {/* Latency by agent */}
        {summary && Object.keys(summary.latency_by_agent).length > 0 && (
          <Card>
            <SectionHead>Latency & reliability</SectionHead>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  {['Agent', 'Calls', 'Mean', 'p50', 'p95', 'Errors'].map(h => (
                    <th key={h} style={{
                      textAlign: h === 'Agent' ? 'left' : 'right',
                      padding: '6px 10px',
                      fontSize: 11,
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                      color: 'var(--text-secondary)',
                      borderBottom: '1px solid var(--border-color)',
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {AGENTS.filter(a => summary.latency_by_agent[a]).map(a => {
                  const s = summary.latency_by_agent[a];
                  const hasErrors = s.failure_count > 0;
                  return (
                    <tr key={a} style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '8px 10px' }}><AgentPill agent={a} /></td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', color: 'var(--text-secondary)' }}>{s.count}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace', fontSize: 12 }}>{s.mean_ms}ms</td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace', fontSize: 12 }}>{s.p50_ms}ms</td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace', fontSize: 12 }}>{s.p95_ms}ms</td>
                      <td style={{ padding: '8px 10px', textAlign: 'right' }}>
                        <span style={{
                          display: 'inline-flex', alignItems: 'center', gap: 4,
                          fontSize: 12, fontWeight: 600,
                          color: hasErrors ? '#EF4444' : '#16A34A',
                        }}>
                          {hasErrors ? <XCircle size={12} /> : <CheckCircle size={12} />}
                          {s.failure_count}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Card>
        )}

        {/* Recent decisions table */}
        <Card>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <SectionHead>Recent routing decisions</SectionHead>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, position: 'relative' }}>
              <select
                value={agentFilter}
                onChange={e => setAgentFilter(e.target.value)}
                style={{
                  appearance: 'none',
                  background: 'var(--bg-main)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-md)',
                  padding: '0.3rem 2rem 0.3rem 0.625rem',
                  fontSize: 12,
                  color: 'var(--text-primary)',
                  fontFamily: 'inherit',
                  cursor: 'pointer',
                }}
              >
                <option value="">All agents</option>
                {AGENTS.map(a => <option key={a} value={a}>{a}</option>)}
                <option value="fallback">fallback</option>
              </select>
              <ChevronDown size={12} color="var(--text-secondary)" style={{ position: 'absolute', right: 8, pointerEvents: 'none' }} />
            </div>
          </div>

          {recent?.decisions?.length === 0 ? (
            <Empty>No decisions in this window. Send some queries to start collecting data.</Empty>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, tableLayout: 'fixed' }}>
                <colgroup>
                  <col style={{ width: '12%' }} />
                  <col style={{ width: '44%' }} />
                  <col style={{ width: '14%' }} />
                  <col style={{ width: '10%' }} />
                  <col style={{ width: '20%' }} />
                </colgroup>
                <thead>
                  <tr>
                    {['Time', 'Query', 'Agent', 'Conf.', 'Reasoning'].map(h => (
                      <th key={h} style={{
                        textAlign: 'left',
                        padding: '6px 10px',
                        fontSize: 10,
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        letterSpacing: '0.06em',
                        color: 'var(--text-secondary)',
                        borderBottom: '1px solid var(--border-color)',
                      }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(recent?.decisions || []).map((d, i) => {
                    const date    = new Date(d.ts * 1000);
                    const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                    const lowConf = d.confidence < 0.25;
                    return (
                      <tr
                        key={i}
                        style={{
                          borderBottom: '1px solid var(--border-color)',
                          background: i % 2 === 0 ? 'transparent' : 'var(--bg-main)',
                          transition: 'background 0.15s',
                        }}
                      >
                        <td style={{ padding: '7px 10px', color: 'var(--text-secondary)', fontFamily: 'monospace', fontSize: 11, whiteSpace: 'nowrap' }}>
                          {timeStr}
                        </td>
                        <td style={{ padding: '7px 10px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-primary)' }}>
                          {d.query_preview}
                        </td>
                        <td style={{ padding: '7px 10px' }}>
                          <AgentPill agent={d.selected_agent || 'fallback'} />
                        </td>
                        <td style={{ padding: '7px 10px', fontFamily: 'monospace', fontSize: 11, color: lowConf ? 'var(--warning-color)' : 'var(--text-primary)', fontWeight: lowConf ? 600 : 400 }}>
                          {d.confidence.toFixed(3)}
                        </td>
                        <td style={{ padding: '7px 10px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-secondary)', fontSize: 11 }}>
                          {d.reasoning}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <div style={{ height: '1rem' }} />
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}

function Empty({ children }) {
  return (
    <div style={{ padding: '2rem 0', textAlign: 'center', color: 'var(--text-secondary)', fontSize: 13 }}>
      {children}
    </div>
  );
}
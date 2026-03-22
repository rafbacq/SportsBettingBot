import React, { useState, useEffect } from 'react';

export default function TradingBot() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [spendLimitInput, setSpendLimitInput] = useState('');
  const [error, setError] = useState(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/bot/status');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
        if (!spendLimitInput && data.spend_limit) {
          setSpendLimitInput(data.spend_limit.toString());
        }
      } else {
        setError('Failed to fetch bot status');
      }
    } catch (e) {
      setError('Could not connect to Trading Bot API. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const timer = setInterval(fetchStatus, 3000);
    return () => clearInterval(timer);
  }, []);

  const handleStart = async () => {
    try {
      await fetch('/api/bot/start', { method: 'POST' });
      fetchStatus();
    } catch (e) {
      console.error(e);
    }
  };

  const handleStop = async () => {
    try {
      await fetch('/api/bot/stop', { method: 'POST' });
      fetchStatus();
    } catch (e) {
      console.error(e);
    }
  };

  const handleConfigUpdate = async () => {
    try {
      await fetch('/api/bot/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spend_limit: parseFloat(spendLimitInput) || 10.0 })
      });
      fetchStatus();
    } catch (e) {
      console.error(e);
    }
  };

  if (loading && !status) {
    return <div style={{ padding: 20 }}>Loading Trading Bot...</div>;
  }

  if (error) {
    return (
      <div style={{ padding: 20, color: 'red' }}>
        <h3>Error</h3>
        <p>{error}</p>
        <p>Make sure to run: <code>python src/api.py</code></p>
      </div>
    );
  }

  const isRunning = status?.is_running;

  return (
    <div style={{ padding: 24, maxWidth: 800, margin: '0 auto' }}>
      <h2 style={{ marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span role="img" aria-label="robot">🤖</span> Autonomous Trading Bot
      </h2>
      <p style={{ color: '#aaa', marginBottom: '24px' }}>
        Allow the AI to scan all live Kalshi markets and automatically execute trades 
        using the Dual-Regime Rebound Model when high-EV opportunities arise.
      </p>

      <div style={{
        background: 'rgba(255,255,255,0.03)',
        border: `1px solid ${isRunning ? 'rgba(16, 185, 129, 0.3)' : 'rgba(255,255,255,0.1)'}`,
        borderRadius: '8px',
        padding: '24px',
        marginBottom: '24px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <div>
            <h3 style={{ margin: '0 0 4px 0' }}>Status</h3>
            <div style={{
              display: 'inline-block',
              padding: '4px 12px',
              borderRadius: '16px',
              fontSize: '14px',
              fontWeight: 'bold',
              background: isRunning ? 'rgba(16, 185, 129, 0.2)' : 'rgba(156, 163, 175, 0.2)',
              color: isRunning ? '#10b981' : '#9ca3af'
            }}>
              {isRunning ? '🟢 RUNNING' : '⚪ STOPPED'}
            </div>
          </div>

          <div>
            {isRunning ? (
              <button 
                onClick={handleStop}
                style={{
                  background: '#ef4444',
                  color: 'white',
                  border: 'none',
                  padding: '10px 24px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '16px',
                  fontWeight: 'bold'
                }}
              >
                Stop Bot
              </button>
            ) : (
              <button 
                onClick={handleStart}
                style={{
                  background: '#10b981',
                  color: 'white',
                  border: 'none',
                  padding: '10px 24px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '16px',
                  fontWeight: 'bold'
                }}
              >
                Start Bot
              </button>
            )}
          </div>
        </div>

        <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '24px' }}>
          <h3 style={{ margin: '0 0 12px 0' }}>Configuration</h3>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end' }}>
            <div>
              <label style={{ display: 'block', fontSize: '14px', color: '#999', marginBottom: '4px' }}>
                Max Spend per Trade ($)
              </label>
              <input 
                type="number" 
                value={spendLimitInput}
                onChange={(e) => setSpendLimitInput(e.target.value)}
                style={{
                  background: 'rgba(0,0,0,0.2)',
                  border: '1px solid rgba(255,255,255,0.2)',
                  color: 'white',
                  padding: '8px 12px',
                  borderRadius: '4px',
                  width: '120px'
                }}
              />
            </div>
            <button 
              onClick={handleConfigUpdate}
              style={{
                background: 'rgba(255,255,255,0.1)',
                color: 'white',
                border: 'none',
                padding: '9px 16px',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Update Config
            </button>
          </div>
        </div>
      </div>

      <div>
        <h3 style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Activity Log</span>
          {isRunning && <span style={{ fontSize: '14px', color: '#10b981' }}>Live updates enabled</span>}
        </h3>
        <div style={{
          background: '#0f172a',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '8px',
          padding: '16px',
          height: '300px',
          overflowY: 'auto',
          fontFamily: 'monospace',
          fontSize: '13px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px'
        }}>
          {status?.logs && status.logs.length > 0 ? (
            status.logs.map((log, i) => (
              <div key={i} style={{ color: log.msg.includes('Error') ? '#ef4444' : log.msg.includes('BUY') ? '#10b981' : '#cbd5e1' }}>
                <span style={{ color: '#64748b', marginRight: '12px' }}>
                  {new Date(log.time).toLocaleTimeString()}
                </span>
                {log.msg}
              </div>
            ))
          ) : (
            <div style={{ color: '#64748b', fontStyle: 'italic' }}>No recent activity...</div>
          )}
        </div>
      </div>
    </div>
  );
}

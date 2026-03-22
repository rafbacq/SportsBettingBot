import React, { useState, useEffect } from 'react';

export default function TradingBot() {
  const [status, setStatus] = useState(null);
  const [positions, setPositions] = useState([]);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [spendLimitInput, setSpendLimitInput] = useState('');
  const [modeInput, setModeInput] = useState('');
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('logs'); // 'logs', 'portfolio', 'history'
  const [showResetModal, setShowResetModal] = useState(false);

  const fetchStatus = async () => {
    try {
      const [statusRes, posRes, histRes] = await Promise.all([
        fetch('/api/bot/status'),
        fetch('/api/bot/positions'),
        fetch('/api/bot/history')
      ]);

      if (statusRes.ok) {
        const data = await statusRes.json();
        setStatus(data);
        if (!spendLimitInput && data.spend_limit) {
          setSpendLimitInput(data.spend_limit.toString());
        }
        if (!modeInput && data.mode) {
           setModeInput(data.mode);
        }
      }

      if (posRes.ok) {
        setPositions(await posRes.json());
      }

      if (histRes.ok) {
        setHistory(await histRes.json());
      }

      if (!statusRes.ok && !posRes.ok) {
        setError('Failed to fetch bot data');
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
        body: JSON.stringify({ 
          spend_limit: parseFloat(spendLimitInput) || 10.0,
          mode: modeInput || 'paper'
        })
      });
      fetchStatus();
    } catch (e) {
      console.error(e);
    }
  };

  const handleReset = async () => {
    setShowResetModal(false);
    try {
      await fetch('/api/bot/reset', { method: 'POST' });
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
  const totalPnL = history.reduce((sum, trade) => sum + trade.pnl, 0);
  const openPnL = positions.reduce((sum, trade) => sum + trade.pnl, 0);

  return (
    <div style={{ padding: 24, maxWidth: 1000, margin: '0 auto', color: '#e2e8f0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px' }}>
        <div>
          <h2 style={{ marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span role="img" aria-label="robot">🤖</span> AI Trading Terminal
          </h2>
          <p style={{ color: '#94a3b8', margin: 0 }}>
            Dual-Regime Rebound Model | Real-time Simulation Mode
          </p>
        </div>
        <div style={{ display: 'flex', gap: '12px' }}>
             <button 
              onClick={() => setShowResetModal(true)}
              style={{
                background: 'rgba(255,255,255,0.05)',
                color: '#94a3b8',
                border: '1px solid rgba(255,255,255,0.1)',
                padding: '8px 16px',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '14px'
              }}
            >
              Reset Sim
            </button>
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
                  fontWeight: 'bold',
                  boxShadow: '0 4px 14px 0 rgba(239, 68, 68, 0.39)'
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
                  fontWeight: 'bold',
                  boxShadow: '0 4px 14px 0 rgba(16, 185, 129, 0.39)'
                }}
              >
                Start Bot
              </button>
            )}
        </div>
      </div>

      {/* Stats Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '24px' }}>
        <div style={{ background: '#1e293b', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Mode</div>
          <div style={{ fontSize: '18px', fontWeight: 'bold', color: status?.mode === 'live' ? '#ef4444' : '#60a5fa' }}>
            {status?.mode === 'live' ? '🔴 LIVE' : '📝 PAPER'}
          </div>
        </div>
        <div style={{ background: '#1e293b', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Fake Balance</div>
          <div style={{ fontSize: '18px', fontWeight: 'bold' }}>${status?.paper_balance?.toFixed(2) || '0.00'}</div>
        </div>
        <div style={{ background: '#1e293b', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Open P&L</div>
          <div style={{ fontSize: '18px', fontWeight: 'bold', color: openPnL >= 0 ? '#10b981' : '#ef4444' }}>
            {openPnL >= 0 ? '+' : ''}${openPnL.toFixed(2)}
          </div>
        </div>
        <div style={{ background: '#1e293b', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Realized P&L</div>
          <div style={{ fontSize: '18px', fontWeight: 'bold', color: totalPnL >= 0 ? '#10b981' : '#ef4444' }}>
            {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
        {/* Main Content Area */}
        <div>
          <div style={{ display: 'flex', gap: '4px', marginBottom: '12px' }}>
            {['logs', 'portfolio', 'history'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  padding: '8px 16px',
                  background: activeTab === tab ? '#334155' : 'transparent',
                  border: 'none',
                  color: activeTab === tab ? '#fff' : '#64748b',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: '600',
                  textTransform: 'capitalize'
                }}
              >
                {tab}
              </button>
            ))}
          </div>

          <div style={{ 
            background: '#0f172a', 
            borderRadius: '12px', 
            border: '1px solid rgba(255,255,255,0.1)',
            minHeight: '400px',
            padding: '16px'
          }}>
            {activeTab === 'logs' && (
              <div style={{ fontFamily: 'monospace', fontSize: '13px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {status?.logs && status.logs.length > 0 ? (
                  status.logs.map((log, i) => (
                    <div key={i} style={{ color: log.msg.includes('Error') ? '#ef4444' : log.msg.includes('BUY') || log.msg.includes('CLOSED') ? '#10b981' : '#94a3b8' }}>
                      <span style={{ color: '#475569', marginRight: '12px' }}>
                        {new Date(log.time).toLocaleTimeString()}
                      </span>
                      {log.msg}
                    </div>
                  ))
                ) : (
                  <div style={{ color: '#475569', fontStyle: 'italic' }}>No recent activity...</div>
                )}
              </div>
            )}

            {activeTab === 'portfolio' && (
              <div>
                {positions.length === 0 ? (
                  <div style={{ color: '#475569', textAlign: 'center', marginTop: '40px' }}>No open positions</div>
                ) : (
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ color: '#64748b', fontSize: '12px', textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        <th style={{ padding: '8px' }}>Ticker</th>
                        <th style={{ padding: '8px' }}>Entry</th>
                        <th style={{ padding: '8px' }}>Current</th>
                        <th style={{ padding: '8px' }}>Size</th>
                        <th style={{ padding: '8px', textAlign: 'right' }}>P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {positions.map((pos, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '12px 8px' }}>
                            <div style={{ fontWeight: 'bold' }}>{pos.ticker}</div>
                            <div style={{ fontSize: '11px', color: '#64748b' }}>{pos.title.substring(0, 30)}...</div>
                          </td>
                          <td style={{ padding: '12px 8px' }}>{pos.entry_price.toFixed(2)}</td>
                          <td style={{ padding: '12px 8px' }}>{pos.current_price.toFixed(2)}</td>
                          <td style={{ padding: '12px 8px' }}>{pos.count}</td>
                          <td style={{ padding: '12px 8px', textAlign: 'right', color: pos.pnl >= 0 ? '#10b981' : '#ef4444', fontWeight: 'bold' }}>
                            ${pos.pnl.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}

            {activeTab === 'history' && (
              <div>
                {history.length === 0 ? (
                  <div style={{ color: '#475569', textAlign: 'center', marginTop: '40px' }}>No trade history</div>
                ) : (
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ color: '#64748b', fontSize: '12px', textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        <th style={{ padding: '8px' }}>Market</th>
                        <th style={{ padding: '8px' }}>Result</th>
                        <th style={{ padding: '8px', textAlign: 'right' }}>P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.slice().reverse().map((trade, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '12px 8px' }}>
                            <div style={{ fontWeight: '600' }}>{trade.ticker}</div>
                            <div style={{ fontSize: '11px', color: '#64748b' }}>{new Date(trade.exit_time).toLocaleString()}</div>
                          </td>
                          <td style={{ padding: '12px 8px' }}>
                            <span style={{ fontSize: '12px', padding: '2px 6px', borderRadius: '4px', background: 'rgba(255,255,255,0.05)' }}>
                              {trade.reason}
                            </span>
                          </td>
                          <td style={{ padding: '12px 8px', textAlign: 'right', color: trade.pnl >= 0 ? '#10b981' : '#ef4444', fontWeight: 'bold' }}>
                            ${trade.pnl.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Sidebar Controls */}
        <div>
          <div style={{ background: '#1e293b', padding: '20px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>Terminal Config</h3>
            
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>
                Trading Mode
              </label>
              <select 
                value={modeInput}
                onChange={(e) => setModeInput(e.target.value)}
                style={{
                  width: '100%',
                  background: '#0f172a',
                  border: '1px solid rgba(255,255,255,0.1)',
                  color: 'white',
                  padding: '10px',
                  borderRadius: '6px',
                  fontSize: '14px'
                }}
              >
                <option value="paper">📝 Paper Trading</option>
                <option value="live">🔴 Live Trading</option>
              </select>
            </div>

            <div style={{ marginBottom: '24px' }}>
              <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>
                Spend Limit ($)
              </label>
              <input 
                type="number" 
                value={spendLimitInput}
                onChange={(e) => setSpendLimitInput(e.target.value)}
                style={{
                  width: '100%',
                  background: '#0f172a',
                  border: '1px solid rgba(255,255,255,0.1)',
                  color: 'white',
                  padding: '10px',
                  borderRadius: '6px',
                  fontSize: '14px',
                  boxSizing: 'border-box'
                }}
              />
            </div>

            <button 
              onClick={handleConfigUpdate}
              style={{
                width: '100%',
                background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
                color: 'white',
                border: 'none',
                padding: '12px',
                borderRadius: '6px',
                cursor: 'pointer',
                fontWeight: '600'
              }}
            >
              Apply Changes
            </button>
          </div>

          <div style={{ marginTop: '24px', padding: '16px', border: '1px dashed rgba(255,255,255,0.1)', borderRadius: '12px' }}>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#94a3b8' }}>Bot Strategy</h4>
            <p style={{ fontSize: '12px', color: '#64748b', lineHeight: '1.5', margin: 0 }}>
              Currently utilizing the <strong>Multi-Regime Rebound Model</strong>. 
              Targets EV {'>'} 0.50 and Rebound Prob {'>'} 45%.
            </p>
          </div>
        </div>
      </div>
      {/* Reset Confirmation Modal */}
      {showResetModal && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0,0,0,0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          backdropFilter: 'blur(4px)'
        }}>
          <div style={{
            background: '#1e293b',
            padding: '32px',
            borderRadius: '16px',
            maxWidth: '400px',
            border: '1px solid rgba(255,255,255,0.1)',
            boxShadow: '0 20px 25px -5px rgba(0,0,0,0.5)'
          }}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: '20px' }}>Reset Simulation?</h3>
            <p style={{ color: '#94a3b8', fontSize: '14px', lineHeight: '1.5', marginBottom: '24px' }}>
              This will clear your paper trading history and reset your balance to **$1,000.00**. This action cannot be undone.
            </p>
            <div style={{ display: 'flex', gap: '12px' }}>
              <button 
                onClick={() => setShowResetModal(false)}
                style={{
                  flex: 1,
                  background: 'transparent',
                  border: '1px solid rgba(255,255,255,0.1)',
                  color: 'white',
                  padding: '12px',
                  borderRadius: '8px',
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button 
                onClick={handleReset}
                style={{
                  flex: 1,
                  background: '#ef4444',
                  border: 'none',
                  color: 'white',
                  padding: '12px',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontWeight: 'bold'
                }}
              >
                Yes, Reset
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

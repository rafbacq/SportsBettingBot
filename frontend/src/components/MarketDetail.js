import React, { useState, useEffect, useCallback } from 'react';
import MultiLineChart from './PriceChart';
import OrderForm from './OrderForm';
import { getEventCandlesticks, formatCents, formatVolume } from '../services/kalshiApi';

// Chart colors for outcomes
const CHART_COLORS = [
  '#3b82f6', // blue
  '#00d4aa', // green/cyan
  '#f59e0b', // orange
  '#a78bfa', // purple
  '#ef4444', // red
  '#22d3ee', // cyan
  '#ec4899', // pink
  '#84cc16', // lime
  '#f97316', // amber
  '#06b6d4', // teal
];

const PERIOD_OPTIONS = [
  { label: 'LIVE', seconds: 3600, interval: 1 },
  { label: '1D', seconds: 86400, interval: 5 },
  { label: '1W', seconds: 604800, interval: 60 },
  { label: '1M', seconds: 2592000, interval: 240 },
  { label: 'ALL', seconds: 7776000, interval: 1440 },
];

/**
 * Detailed view of a Kalshi event — Kalshi app style.
 * Multi-line chart with all outcomes, probability legend, outcome list.
 */
export default function MarketDetail({ event, auth, onPlaceOrder, onSell, onClose }) {
  const markets = event.markets || [];
  const [selectedTicker, setSelectedTicker] = useState(
    markets.length > 0 ? markets[0].ticker : null
  );
  const [candleData, setCandleData] = useState({}); // { ticker: candles[] }
  const [loading, setLoading] = useState(true);
  const [period, setPeriod] = useState(0); // default LIVE
  const [showAll, setShowAll] = useState(false);
  const [aiRecommendation, setAiRecommendation] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);

  const selectedMarket = markets.find((m) => m.ticker === selectedTicker) || markets[0];

  // Helper: get probability from a market object, trying multiple Kalshi v2 price fields
  function getMarketProb(m) {
    let p = parseFloat(m.last_price_dollars || 0);
    if (!p) p = parseFloat(m.yes_bid_dollars || 0);
    if (!p && m.yes_bid) p = parseFloat(m.yes_bid) / 100;
    if (!p) p = parseFloat(m.yes_ask_dollars || 0);
    return p;
  }

  // Sort markets by probability (descending)
  const sortedMarkets = [...markets].sort((a, b) => {
    const probA = getMarketProb(a);
    const probB = getMarketProb(b);
    return probB - probA;
  });

  // Assign colors to top outcomes
  const marketColors = {};
  sortedMarkets.forEach((m, i) => {
    marketColors[m.ticker] = CHART_COLORS[i % CHART_COLORS.length];
  });

  const fetchCandles = useCallback(async () => {
    // Only fetch for top 5 to prevent URL length limits (e.g. 140 golf markets)
    const topMarkets = sortedMarkets.slice(0, 5);
    if (topMarkets.length === 0) return;
    const opt = PERIOD_OPTIONS[period];
    try {
      const data = await getEventCandlesticks(topMarkets, opt.seconds, opt.interval);
      setCandleData(data);
    } catch (e) {
      console.error('Failed to fetch candlesticks:', e);
    } finally {
      setLoading(false);
    }
  }, [markets.length, period]); // depends on markets list & period selection

  useEffect(() => {
    setLoading(true);
    fetchCandles();
    const timer = setInterval(fetchCandles, 5000); // 5s refresh for live feel
    return () => clearInterval(timer);
  }, [fetchCandles]);

  // Fetch AI recommendation when ticker changes or candle data updates
  useEffect(() => {
    if (!selectedTicker) return;
    // Wait until we have candle data for this ticker
    const tickerCandles = candleData[selectedTicker];
    if (!tickerCandles || tickerCandles.length === 0) return;

    async function fetchAi() {
      setAiLoading(true);
      setAiRecommendation(null);
      try {
        // POST the candlestick data that the frontend already fetched successfully
        // to the backend for ML analysis (avoids backend needing its own Kalshi auth)
        const res = await fetch('/api/recommendations', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticker: selectedTicker,
            candles: tickerCandles,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          setAiRecommendation(data);
        } else {
          const errText = await res.text();
          setAiRecommendation({ error: `Analysis failed: ${res.statusText} - ${errText || 'Check server logs.'}` });
        }
      } catch (e) {
        console.error('Failed to fetch AI recommendation', e);
        setAiRecommendation({ error: 'Failed to connect to AI server. Ensure api.py is running.' });
      } finally {
        setAiLoading(false);
      }
    }
    fetchAi();
  }, [selectedTicker, candleData]);

  if (!event) return null;

  const totalVolume = markets.reduce((s, m) => s + parseFloat(m.volume_fp || 0), 0);

  // Build chart datasets — top outcomes with candle data
  const displayMarkets = showAll ? sortedMarkets : sortedMarkets.slice(0, 3);
  const chartDatasets = sortedMarkets
    .filter(m => candleData[m.ticker] && candleData[m.ticker].length >= 2)
    .slice(0, 5) // max 5 lines on chart
    .map(m => ({
      label: m.yes_sub_title || m.title || m.ticker,
      color: marketColors[m.ticker],
      candles: candleData[m.ticker],
    }));

  return (
    <div className="market-detail">
      <div className="market-detail__header">
        <button className="market-detail__back" onClick={onClose}>&larr;</button>
        <div>
          <div className="market-detail__event-label">{event.category}</div>
          <h2 className="market-detail__title">{event.title}</h2>
          {event.sub_title && (
            <span className="market-detail__subtitle">{event.sub_title}</span>
          )}
        </div>
      </div>

      {/* Probability Legend */}
      <div className="market-detail__legend">
        {sortedMarkets.slice(0, 5).map((m) => {
          const prob = Math.round(getMarketProb(m) * 100);
          return (
            <div key={m.ticker} className="legend-item">
              <span className="legend-dot" style={{ backgroundColor: marketColors[m.ticker] }} />
              <span>{m.yes_sub_title || m.title}</span>
              <span className="legend-prob">{prob}%</span>
            </div>
          );
        })}
      </div>

      {/* AI Recommendation Card */}
      {selectedTicker && (
        <div className="ai-recommendation" style={{ margin: '16px 0', padding: '16px', borderRadius: '8px', background: 'rgba(59, 130, 246, 0.05)', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <h3 style={{ margin: 0, fontSize: '14px', color: '#60a5fa', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span role="img" aria-label="ai">🤖</span> AI Model Analysis
            </h3>
            {aiLoading && <span style={{ fontSize: '12px', color: '#999' }}>Analyzing...</span>}
          </div>
          
          {!aiLoading && aiRecommendation && (
            <div>
              {aiRecommendation.error ? (
                <div style={{ color: '#ef4444', fontSize: '14px', fontStyle: 'italic', padding: '8px', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '4px' }}>
                  ⚠️ {aiRecommendation.error}
                </div>
              ) : (
                <>
                  <div style={{ 
                    display: 'inline-block', 
                    padding: '4px 8px', 
                    borderRadius: '4px', 
                    fontWeight: 'bold',
                    fontSize: '14px',
                    marginBottom: '8px',
                    background: aiRecommendation.action.includes('BUY') ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.1)',
                    color: aiRecommendation.action.includes('BUY') ? '#10b981' : '#ef4444'
                  }}>
                    {aiRecommendation.action}
                  </div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '13px', color: '#ccc' }}>
                    <div><strong>Regime:</strong> {aiRecommendation.regime}</div>
                    <div><strong>EV:</strong> {aiRecommendation.expected_ev > 0 ? '+' : ''}{aiRecommendation.expected_ev.toFixed(2)}</div>
                    <div><strong>Rebound Prob:</strong> {(aiRecommendation.rebound_prob * 100).toFixed(0)}%</div>
                    {aiRecommendation.target_exit > 0 && <div><strong>Target:</strong> {(aiRecommendation.target_exit * 100).toFixed(0)}¢</div>}
                  </div>
                  
                  {aiRecommendation.reasons && aiRecommendation.reasons.length > 0 && (
                    <div style={{ marginTop: '8px', fontSize: '12px', color: '#999' }}>
                      {aiRecommendation.reasons.map((r, i) => <div key={i}>• {r}</div>)}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      )}

      {/* Multi-line Chart */}
      <div className="market-detail__chart-section">
        {loading ? (
          <div className="chart-loading">Loading chart...</div>
        ) : (
          <MultiLineChart datasets={chartDatasets} width={780} height={280} />
        )}

        <div className="market-detail__chart-header">
          <div className="market-detail__volume-bar">
            <span className="market-detail__volume-value">
              ${totalVolume > 0 ? formatVolume(totalVolume) : '0'}
            </span>
            <span>vol</span>
          </div>
          <div className="period-selector">
            {PERIOD_OPTIONS.map((opt, i) => (
              <button
                key={opt.label}
                className={`period-btn ${period === i ? 'period-btn--active' : ''}`}
                onClick={() => setPeriod(i)}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Search within event */}
      {markets.length > 5 && (
        <div style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8, color: '#999', fontSize: 13 }}>
          <span>🔍</span>
          <span>Search</span>
        </div>
      )}

      {/* Outcome List — Kalshi style */}
      <div className="market-detail__outcomes">
        {displayMarkets.map((m) => {
          const prob = Math.round(getMarketProb(m) * 100);
          const yesBid = getMarketProb(m);
          const yesAsk = parseFloat(m.yes_ask_dollars || 0);
          const bidCents = Math.round(yesBid * 100);
          const askCents = Math.round(yesAsk * 100);

          return (
            <div
              key={m.ticker}
              className="outcome-row-kalshi"
              onClick={() => setSelectedTicker(m.ticker)}
              style={{ cursor: 'pointer', background: m.ticker === selectedTicker ? 'rgba(0,212,170,0.05)' : undefined }}
            >
              <div className="outcome-row-kalshi__left">
                <div
                  className="outcome-row-kalshi__color"
                  style={{ backgroundColor: marketColors[m.ticker] }}
                />
                <span className="outcome-row-kalshi__name">
                  {m.yes_sub_title || m.title}
                </span>
              </div>
              <div className="outcome-row-kalshi__right">
                <div className="outcome-row-kalshi__scores">
                  <span className="outcome-row-kalshi__score">
                    {bidCents > 0 ? `-${bidCents}` : '—'}
                  </span>
                  <span className="outcome-row-kalshi__score">
                    {askCents > 0 ? askCents : '—'}
                  </span>
                </div>
                <span className="outcome-row-kalshi__prob">{prob}%</span>
              </div>
            </div>
          );
        })}

        {/* Show more / Show all */}
        {sortedMarkets.length > 3 && (
          <div
            className="outcome-row-kalshi__show-more"
            onClick={() => setShowAll(!showAll)}
          >
            <span>
              {showAll ? '' : `+${sortedMarkets.length - 3} more`}
            </span>
            <span>{showAll ? 'Show less' : 'Show all'}</span>
          </div>
        )}
      </div>

      {/* Order form */}
      {selectedMarket && (
        <OrderForm
          market={selectedMarket}
          auth={auth}
          onPlaceOrder={onPlaceOrder}
          onSell={onSell}
          position={null}
        />
      )}
    </div>
  );
}

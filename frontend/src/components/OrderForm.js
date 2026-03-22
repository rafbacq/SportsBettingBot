import React, { useState } from 'react';

/**
 * Order form for buying/selling YES/NO contracts.
 * Always shows both Buy and Sell buttons.
 * Ensures count is sent as integer.
 *
 * Props:
 *   market      — market object with ticker, yes/no prices
 *   auth        — { keyId, privateKey } or null if not connected
 *   onPlaceOrder — async (ticker, side, count) => void
 *   onSell       — async (ticker, side, count) => void
 *   position     — user's current position in this market (null if none)
 */
export default function OrderForm({ market, auth, onPlaceOrder, onSell, position }) {
  const [count, setCount] = useState(1);
  const [side, setSide] = useState('yes');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  if (!auth) {
    return (
      <div className="order-form">
        <h3>Trade</h3>
        <p className="order-form__auth-msg">Connect your API key via Settings to place trades.</p>
      </div>
    );
  }

  const yesPrice = parseFloat(market.yes_ask_dollars || market.last_price_dollars || 0);
  const noPrice = parseFloat(market.no_ask_dollars || 0);
  const cost = side === 'yes' ? yesPrice * count : noPrice * count;

  async function handleBuy() {
    setLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const priceCents = side === 'yes' ? Math.round(yesPrice * 100) : Math.round(noPrice * 100);
      await onPlaceOrder(market.ticker, side, parseInt(count, 10), priceCents);
      setSuccess(`Bought ${count} ${side.toUpperCase()} @ ${side === 'yes' ? (yesPrice * 100).toFixed(0) : (noPrice * 100).toFixed(0)}¢`);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleSell() {
    setLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const priceCents = side === 'yes' ? Math.round(yesPrice * 100) : Math.round(noPrice * 100);
      await onSell(market.ticker, side, parseInt(count, 10), priceCents);
      setSuccess(`Sold ${count} ${side.toUpperCase()} contracts`);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="order-form">
      <h3>Trade</h3>

      <div className="order-form__side-toggle">
        <button className={`side-btn ${side === 'yes' ? 'side-btn--yes-active' : ''}`} onClick={() => setSide('yes')}>
          Yes {(yesPrice * 100).toFixed(0)}¢
        </button>
        <button className={`side-btn ${side === 'no' ? 'side-btn--no-active' : ''}`} onClick={() => setSide('no')}>
          No {(noPrice * 100).toFixed(0)}¢
        </button>
      </div>

      <div className="order-form__field">
        <label htmlFor="order-count">Contracts</label>
        <input
          id="order-count"
          type="number"
          min="1"
          max="10000"
          value={count}
          onChange={(e) => setCount(Math.max(1, parseInt(e.target.value) || 1))}
        />
      </div>

      <div className="order-form__cost">
        <span>Est. Cost</span>
        <span>${cost.toFixed(2)}</span>
      </div>
      <div className="order-form__payout">
        <span>Max Payout</span>
        <span>${(count * 1.0).toFixed(2)}</span>
      </div>

      <div className="order-form__actions">
        <button className="btn btn--buy" onClick={handleBuy} disabled={loading}>
          {loading ? 'Placing...' : 'Buy'}
        </button>
        <button className="btn btn--sell" onClick={handleSell} disabled={loading}>
          {loading ? 'Selling...' : 'Sell'}
        </button>
      </div>

      {error && <div className="order-form__error">{error}</div>}
      {success && <div className="order-form__success">{success}</div>}
    </div>
  );
}

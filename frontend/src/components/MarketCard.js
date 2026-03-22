import React from 'react';
import { formatCents, formatVolume } from '../services/kalshiApi';

/**
 * Dark-themed market card with LIVE badge and probability percentages.
 */
export default function MarketCard({ event, onClick }) {
  const markets = event.markets || [];
  const totalVolume = markets.reduce((s, m) => s + parseFloat(m.volume_fp || 0), 0);

  // Check if any market is active/open
  const isLive = markets.some(m => m.status === 'active' || m.status === 'open');

  return (
    <div className="market-card" onClick={onClick}>
      <div className="market-card__header">
        <span className="market-card__category">{event.category}</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {isLive && <span className="market-card__live-badge">● LIVE</span>}
          {totalVolume > 0 && (
            <span className="market-card__volume">Vol {formatVolume(totalVolume)}</span>
          )}
        </div>
      </div>

      <div className="market-card__title">{event.title}</div>
      {event.sub_title && <div className="market-card__subtitle">{event.sub_title}</div>}

      <div className="market-card__outcomes">
        {markets.slice(0, 4).map((m) => (
          <MarketOutcome key={m.ticker} market={m} />
        ))}
        {markets.length > 4 && (
          <div className="market-card__more">+{markets.length - 4} more</div>
        )}
      </div>
    </div>
  );
}

function MarketOutcome({ market }) {
  const label = market.yes_sub_title || market.title || market.ticker;

  // Kalshi v2 doesn't always populate last_price_dollars for illiquid markets.
  // Try multiple fields: last_price_dollars → yes_bid_dollars → yes_bid (cents) → yes_ask_dollars
  let lastPrice = parseFloat(market.last_price_dollars || 0);
  if (!lastPrice) lastPrice = parseFloat(market.yes_bid_dollars || 0);
  if (!lastPrice && market.yes_bid) lastPrice = parseFloat(market.yes_bid) / 100;
  if (!lastPrice) lastPrice = parseFloat(market.yes_ask_dollars || 0);

  const prevPrice = parseFloat(market.previous_price_dollars || market.previous_yes_bid_dollars || lastPrice);
  const diff = lastPrice - prevPrice;
  const prob = Math.round(lastPrice * 100);

  return (
    <div className="outcome">
      <span className="outcome__label">{label}</span>
      <div className="outcome__prices">
        <span className="outcome__price">{prob}%</span>
        {diff !== 0 && (
          <span className={`outcome__change ${diff > 0 ? 'outcome__change--up' : 'outcome__change--down'}`}>
            {diff > 0 ? '+' : ''}{(diff * 100).toFixed(0)}¢
          </span>
        )}
      </div>
    </div>
  );
}

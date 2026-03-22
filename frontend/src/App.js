import React, { useState, useEffect, useCallback, useMemo } from 'react';
import SearchBar from './components/SearchBar';
import FilterBar from './components/FilterBar';
import MarketCard from './components/MarketCard';
import MarketDetail from './components/MarketDetail';
import Settings from './components/Settings';
import TradingBot from './components/TradingBot';
import {
  importPrivateKey,
  getAllEvents,
  getBalance,
  placeOrder,
  sellPosition,
  isRealEvent,
  isSportsEvent,
  isLiveMatch,
  getSportsSubcategoryFromEvent,
} from './services/kalshiApi';
import './App.css';

const TABS = ['Markets', 'Trading Bot', 'Settings'];
const POLL_INTERVAL = 10000; // 10s for live feel

export default function App() {
  // ─── Auth ───────────────────────────────────────────────────────────
  const [auth, setAuth] = useState(null);
  const [balance, setBalance] = useState(null);
  const [connectError, setConnectError] = useState(null);

  // ─── Navigation ─────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState('Markets');
  const [selectedEvent, setSelectedEvent] = useState(null);

  // ─── Events ─────────────────────────────────────────────────────────
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // ─── Filters ────────────────────────────────────────────────────────
  const [search, setSearch] = useState('');
  const [subcategory, setSubcategory] = useState(null);

  // ─── Fetch events (sports only) ───────────────────────────────────
  const fetchEvents = useCallback(async () => {
    try {
      const data = await getAllEvents(40);
      // Filter: real events + sports only + actual live matches (not long-term predictions)
      const sportsOnly = data.filter(e => isRealEvent(e) && isSportsEvent(e) && isLiveMatch(e));
      setEvents(sportsOnly);
      setError(null);
    } catch (e) {
      console.error('Failed to fetch events:', e);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchEvents();
    const timer = setInterval(fetchEvents, POLL_INTERVAL);
    return () => clearInterval(timer);
  }, [fetchEvents]);

  // ─── Fetch balance ──────────────────────────────────────────────────
  useEffect(() => {
    if (!auth) { setBalance(null); return; }
    getBalance(auth).then((b) => setBalance(b.balance)).catch(() => setBalance(null));
  }, [auth]);

  // ─── Subcategory counts ──────────────────────────────────────────
  const subcategoryCounts = useMemo(() => {
    const counts = {};
    events.forEach((e) => {
      const sub = getSportsSubcategoryFromEvent(e);
      counts[sub] = (counts[sub] || 0) + 1;
    });
    return counts;
  }, [events]);

  // ─── Filtered events ───────────────────────────────────────────────
  const filteredEvents = useMemo(() => {
    let filtered = events;

    // Subcategory filter
    if (subcategory) {
      filtered = filtered.filter((e) => getSportsSubcategoryFromEvent(e) === subcategory);
    }

    // Search
    if (search.trim()) {
      const q = search.toLowerCase();
      filtered = filtered.filter((e) =>
        (e.title || '').toLowerCase().includes(q) ||
        (e.sub_title || '').toLowerCase().includes(q) ||
        (e.event_ticker || '').toLowerCase().includes(q) ||
        (e.series_ticker || '').toLowerCase().includes(q) ||
        (e.markets || []).some((m) =>
          (m.yes_sub_title || '').toLowerCase().includes(q) ||
          (m.title || '').toLowerCase().includes(q)
        )
      );
    }

    return filtered;
  }, [events, subcategory, search]);

  // ─── Auth handlers ──────────────────────────────────────────────────
  async function handleConnect(keyId, privateKeyPem) {
    setConnectError(null);
    try {
      const cryptoKey = await importPrivateKey(privateKeyPem);
      const newAuth = { keyId, privateKey: cryptoKey };
      await getBalance(newAuth);
      setAuth(newAuth);
    } catch (e) {
      setConnectError(e.message || 'Failed to connect.');
    }
  }

  function handleDisconnect() {
    setAuth(null);
    setBalance(null);
    localStorage.removeItem('kalshi_key_id');
  }

  // ─── Trading handlers ──────────────────────────────────────────────
  async function handlePlaceOrder(ticker, side, count, priceCents) {
    await placeOrder(auth, { ticker, side, count: parseInt(count, 10), priceCents });
    if (auth) getBalance(auth).then((b) => setBalance(b.balance)).catch(() => {});
  }

  async function handleSell(ticker, side, count, priceCents) {
    await sellPosition(auth, { ticker, side, count: parseInt(count, 10), priceCents });
    if (auth) getBalance(auth).then((b) => setBalance(b.balance)).catch(() => {});
  }

  // ─── Render ─────────────────────────────────────────────────────────
  return (
    <div className="app">
      <header className="app-header">
        <h1>Kalshi Sports</h1>
        <span className="app-header__subtitle">Live sports betting markets</span>
        {auth && (
          <span className="app-header__balance">
            Balance: ${balance != null ? (balance / 100).toFixed(2) : '...'}
          </span>
        )}
      </header>

      <nav className="app-nav">
        {TABS.map((tab) => (
          <button
            key={tab}
            className={`nav-tab ${activeTab === tab ? 'nav-tab--active' : ''}`}
            onClick={() => { setActiveTab(tab); setSelectedEvent(null); }}
          >
            {tab}
          </button>
        ))}
      </nav>

      <main className="app-main">
        {activeTab === 'Markets' && !selectedEvent && (
          <div className="markets-view">
            <div className="markets-view__controls">
              <SearchBar value={search} onChange={setSearch} />
              <FilterBar
                activeSubcategory={subcategory}
                onSubcategoryChange={setSubcategory}
                subcategoryCounts={subcategoryCounts}
              />
            </div>

            <div className="markets-view__count">
              {filteredEvents.length} event{filteredEvents.length !== 1 ? 's' : ''}
              {loading && <span className="loading-dot"> loading...</span>}
            </div>

            {error && <div className="markets-error">Error: {error}</div>}

            <div className="market-grid">
              {filteredEvents.map((event) => (
                <MarketCard
                  key={event.event_ticker}
                  event={event}
                  onClick={() => setSelectedEvent(event)}
                />
              ))}
            </div>

            {!loading && filteredEvents.length === 0 && !error && (
              <p className="empty-state">No live sports matches found right now. Check back during game time.</p>
            )}
          </div>
        )}

        {activeTab === 'Markets' && selectedEvent && (
          <MarketDetail
            event={selectedEvent}
            auth={auth}
            onPlaceOrder={handlePlaceOrder}
            onSell={handleSell}
            onClose={() => setSelectedEvent(null)}
          />
        )}

        {activeTab === 'Trading Bot' && (
          <TradingBot />
        )}

        {activeTab === 'Settings' && (
          <Settings
            auth={auth}
            onConnect={handleConnect}
            onDisconnect={handleDisconnect}
            balance={balance}
            connectError={connectError}
          />
        )}
      </main>
    </div>
  );
}

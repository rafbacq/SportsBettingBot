import React, { useState } from 'react';

/**
 * Settings panel for Kalshi API key credentials.
 * Users enter their API Key ID and paste their RSA private key PEM.
 * API keys are generated at: https://kalshi.com/account/api-keys
 */
export default function Settings({ auth, onConnect, onDisconnect, balance, connectError }) {
  const [keyId, setKeyId] = useState(() => localStorage.getItem('kalshi_key_id') || '');
  const [privateKeyPem, setPrivateKeyPem] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleConnect(e) {
    e.preventDefault();
    setLoading(true);
    try {
      await onConnect(keyId, privateKeyPem);
      localStorage.setItem('kalshi_key_id', keyId);
      setPrivateKeyPem('');
    } catch {
      // Error is displayed via connectError prop from parent
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="settings">
      <h2>Kalshi API Key</h2>

      {auth ? (
        <div className="settings__connected">
          <div className="settings__status">
            <span className="status-dot status-dot--connected" />
            <span>Connected</span>
          </div>

          {balance != null && (
            <div className="settings__balance">
              <span className="settings__balance-label">Balance</span>
              <span className="settings__balance-value">${(balance / 100).toFixed(2)}</span>
            </div>
          )}

          <div className="settings__account-email">
            Key ID: {localStorage.getItem('kalshi_key_id')}
          </div>

          <button className="btn btn--secondary" onClick={onDisconnect}>
            Disconnect
          </button>
        </div>
      ) : (
        <form className="settings__form" onSubmit={handleConnect}>
          <p className="settings__info">
            Enter your Kalshi API key to enable trading. Market browsing works without an API key.
            Generate API keys at{' '}
            <a href="https://kalshi.com/account/api-keys" target="_blank" rel="noreferrer">
              kalshi.com/account/api-keys
            </a>.
          </p>

          <div className="settings__field">
            <label htmlFor="kalshi-key-id">API Key ID</label>
            <input
              id="kalshi-key-id"
              type="text"
              value={keyId}
              onChange={(e) => setKeyId(e.target.value)}
              placeholder="e.g. 12345678-abcd-1234-efgh-123456789abc"
              required
            />
          </div>

          <div className="settings__field">
            <label htmlFor="kalshi-private-key">Private Key (PEM)</label>
            <textarea
              id="kalshi-private-key"
              value={privateKeyPem}
              onChange={(e) => setPrivateKeyPem(e.target.value)}
              placeholder={"-----BEGIN PRIVATE KEY-----\nPaste your private key here...\n-----END PRIVATE KEY-----"}
              rows={6}
              required
            />
          </div>

          {connectError && <div className="settings__error">{connectError}</div>}

          <button className="btn btn--primary" type="submit" disabled={loading || !keyId || !privateKeyPem}>
            {loading ? 'Connecting...' : 'Connect'}
          </button>
        </form>
      )}
    </section>
  );
}

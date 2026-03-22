const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
  app.use(
    '/trade-api',
    createProxyMiddleware({
      target: 'https://api.elections.kalshi.com',
      changeOrigin: true,
      secure: true,
    })
  );
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://127.0.0.1:8000',
      changeOrigin: true,
    })
  );
};

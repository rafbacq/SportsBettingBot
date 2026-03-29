"""HTML report generator for backtest results.

Generates a self-contained HTML report with embedded matplotlib charts:
- Equity curve
- Drawdown chart
- Monte Carlo fan chart
- P&L distribution histogram
- Per-regime comparison
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("trading.backtest.report")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not installed — HTML report will be text-only")


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def generate_report(
    engine,
    mc_results: dict | None = None,
    wf_results: dict | None = None,
    output_path: str = "data/backtest_report.html",
) -> str:
    """Generate a comprehensive HTML backtest report.

    Args:
        engine: BacktestEngine with completed backtest results.
        mc_results: Monte Carlo simulation results (optional).
        wf_results: Walk-forward optimization results (optional).
        output_path: Where to save the HTML file.

    Returns:
        Path to the generated HTML file.
    """
    m = engine.metrics
    charts = {}

    if HAS_MPL and engine.equity_curve:
        charts["equity"] = _create_equity_chart(engine)
        charts["drawdown"] = _create_drawdown_chart(engine)
        charts["pnl_dist"] = _create_pnl_distribution(engine)
        charts["regime"] = _create_regime_comparison(engine)

        if mc_results and "terminal_equities" in mc_results:
            charts["monte_carlo"] = _create_monte_carlo_chart(
                engine, mc_results
            )

    html = _build_html(m, charts, wf_results)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Backtest report saved to {output_path}")
    return output_path


def _create_equity_chart(engine) -> str:
    """Create equity curve chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    equities = [e.equity for e in engine.equity_curve]
    x = range(len(equities))

    ax.plot(x, equities, color="#00d4ff", linewidth=1.5, alpha=0.9)
    ax.fill_between(x, engine.initial_bankroll, equities,
                     where=[e >= engine.initial_bankroll for e in equities],
                     color="#00d4ff", alpha=0.15)
    ax.fill_between(x, engine.initial_bankroll, equities,
                     where=[e < engine.initial_bankroll for e in equities],
                     color="#ff4757", alpha=0.15)

    ax.axhline(y=engine.initial_bankroll, color="#888", linestyle="--",
               alpha=0.5, linewidth=0.8)

    ax.set_title("Equity Curve", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade #", color="#aaa")
    ax.set_ylabel("Equity ($)", color="#aaa")
    ax.tick_params(colors="#888")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color="#444")

    return _fig_to_base64(fig)


def _create_drawdown_chart(engine) -> str:
    """Create drawdown chart."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    dd = [e.drawdown_pct * 100 for e in engine.equity_curve]
    x = range(len(dd))

    ax.fill_between(x, 0, dd, color="#ff4757", alpha=0.4)
    ax.plot(x, dd, color="#ff4757", linewidth=1.0, alpha=0.8)

    ax.set_title("Drawdown", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade #", color="#aaa")
    ax.set_ylabel("Drawdown (%)", color="#aaa")
    ax.tick_params(colors="#888")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color="#444")
    ax.invert_yaxis()

    return _fig_to_base64(fig)


def _create_pnl_distribution(engine) -> str:
    """Create P&L distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    pnls = [t.pnl_usd for t in engine.trade_results]
    bins = min(50, max(10, len(pnls) // 5))

    n, bins_edges, patches = ax.hist(pnls, bins=bins, edgecolor="#1a1a2e",
                                      linewidth=0.5, alpha=0.8)
    for patch, edge in zip(patches, bins_edges):
        if edge >= 0:
            patch.set_facecolor("#2ed573")
        else:
            patch.set_facecolor("#ff4757")

    ax.axvline(x=0, color="#888", linestyle="--", linewidth=1)
    ax.axvline(x=np.mean(pnls), color="#ffd700", linestyle="-",
               linewidth=1.5, label=f"Mean: ${np.mean(pnls):+.4f}")

    ax.set_title("P&L Distribution", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("P&L ($)", color="#aaa")
    ax.set_ylabel("Frequency", color="#aaa")
    ax.legend(facecolor="#16213e", edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="#888")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color="#444")

    return _fig_to_base64(fig)


def _create_regime_comparison(engine) -> str:
    """Create per-regime comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    m = engine.metrics
    regimes = ["Non-Cross", "Cross"]
    colors = ["#00d4ff", "#ffd700"]

    # Win rates
    ax = axes[0]
    ax.set_facecolor("#16213e")
    nc_wr = m.nc_wins / max(m.nc_trades, 1)
    cr_wr = m.cr_wins / max(m.cr_trades, 1)
    bars = ax.bar(regimes, [nc_wr * 100, cr_wr * 100], color=colors, alpha=0.8,
                  edgecolor="#1a1a2e")
    ax.set_title("Win Rate (%)", color="white", fontsize=12)
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, [nc_wr, cr_wr]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0%}", ha="center", color="white", fontsize=10)

    # P&L
    ax = axes[1]
    ax.set_facecolor("#16213e")
    bars = ax.bar(regimes, [m.nc_pnl, m.cr_pnl], color=colors, alpha=0.8,
                  edgecolor="#1a1a2e")
    ax.set_title("Total P&L ($)", color="white", fontsize=12)
    for bar, val in zip(bars, [m.nc_pnl, m.cr_pnl]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"${val:+.2f}", ha="center", color="white", fontsize=10,
                va="bottom" if val >= 0 else "top")

    # Trade count
    ax = axes[2]
    ax.set_facecolor("#16213e")
    bars = ax.bar(regimes, [m.nc_trades, m.cr_trades], color=colors, alpha=0.8,
                  edgecolor="#1a1a2e")
    ax.set_title("Trade Count", color="white", fontsize=12)
    for bar, val in zip(bars, [m.nc_trades, m.cr_trades]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", color="white", fontsize=10)

    for ax in axes:
        ax.tick_params(colors="#888")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Cross vs Non-Cross Performance", color="white",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return _fig_to_base64(fig)


def _create_monte_carlo_chart(engine, mc_results: dict) -> str:
    """Create Monte Carlo fan chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    terminal = mc_results["terminal_equities"]

    # Plot histogram of terminal equities
    bins = 50
    n, bins_edges, patches = ax.hist(terminal, bins=bins, edgecolor="#1a1a2e",
                                      linewidth=0.5, alpha=0.7, color="#00d4ff")

    # Highlight initial bankroll
    ax.axvline(x=engine.initial_bankroll, color="#ff4757", linestyle="--",
               linewidth=1.5, label=f"Initial: ${engine.initial_bankroll:.0f}")

    # Percentiles
    p5 = np.percentile(terminal, 5)
    p50 = np.percentile(terminal, 50)
    p95 = np.percentile(terminal, 95)
    ax.axvline(x=p5, color="#ff6b6b", alpha=0.8, linewidth=1,
               label=f"5th: ${p5:.0f}")
    ax.axvline(x=p50, color="#ffd700", alpha=0.8, linewidth=1.5,
               label=f"Median: ${p50:.0f}")
    ax.axvline(x=p95, color="#2ed573", alpha=0.8, linewidth=1,
               label=f"95th: ${p95:.0f}")

    ax.set_title(f"Monte Carlo Terminal Equity ({len(terminal)} simulations)",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Terminal Equity ($)", color="#aaa")
    ax.set_ylabel("Frequency", color="#aaa")
    ax.legend(facecolor="#16213e", edgecolor="#333", labelcolor="white",
              fontsize=9)
    ax.tick_params(colors="#888")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color="#444")

    return _fig_to_base64(fig)


def _build_html(metrics, charts: dict, wf_results: dict | None) -> str:
    """Build the full HTML report."""
    m = metrics

    chart_html = ""
    for name, b64 in charts.items():
        chart_html += f"""
        <div class="chart-container">
            <img src="data:image/png;base64,{b64}" alt="{name} chart">
        </div>
        """

    wf_html = ""
    if wf_results and "folds" in wf_results:
        wf_rows = ""
        for f in wf_results["folds"]:
            wf_rows += f"""
            <tr>
                <td>{f['fold']}</td>
                <td>{f['train_games']}</td>
                <td>{f['test_games']}</td>
                <td>{f['oos_trades']}</td>
                <td>${f['oos_pnl']:+.3f}</td>
                <td>{f['oos_sharpe']:.3f}</td>
                <td>{f['oos_win_rate']:.0%}</td>
            </tr>
            """
        wf_html = f"""
        <h2>Walk-Forward Optimization</h2>
        <table>
            <tr><th>Fold</th><th>Train Games</th><th>Test Games</th>
                <th>OOS Trades</th><th>OOS P&L</th><th>OOS Sharpe</th><th>Win Rate</th></tr>
            {wf_rows}
        </table>
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Avg OOS Sharpe</div>
                <div class="metric-value">{m.wf_avg_oos_sharpe:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg OOS P&L</div>
                <div class="metric-value">${m.wf_avg_oos_pnl:+.3f}</div>
            </div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report — AI Dual-Regime Rebound Trading</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
            color: #e8e8e8;
            min-height: 100vh;
            padding: 2rem;
        }}
        h1 {{
            text-align: center;
            font-size: 2rem;
            background: linear-gradient(90deg, #00d4ff, #ffd700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }}
        h2 {{
            color: #00d4ff;
            margin: 2rem 0 1rem;
            border-bottom: 1px solid #333;
            padding-bottom: 0.5rem;
        }}
        .metric-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin: 1rem 0;
        }}
        .metric-card {{
            background: rgba(22, 33, 62, 0.8);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 1.2rem;
            flex: 1;
            min-width: 150px;
            text-align: center;
            backdrop-filter: blur(10px);
        }}
        .metric-label {{
            color: #888;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .metric-value {{
            font-size: 1.6rem;
            font-weight: bold;
            color: #00d4ff;
            margin-top: 0.3rem;
        }}
        .metric-value.positive {{ color: #2ed573; }}
        .metric-value.negative {{ color: #ff4757; }}
        .chart-container {{
            margin: 1.5rem 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 8px;
            border: 1px solid #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 0.7rem 1rem;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{ color: #00d4ff; font-size: 0.85rem; text-transform: uppercase; }}
        td {{ color: #ccc; }}
        .regime-badge {{
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        .regime-nc {{ background: rgba(0, 212, 255, 0.2); color: #00d4ff; }}
        .regime-cr {{ background: rgba(255, 215, 0, 0.2); color: #ffd700; }}
        footer {{
            text-align: center;
            margin-top: 3rem;
            color: #555;
            font-size: 0.8rem;
        }}
    </style>
</head>
<body>
    <h1>AI Dual-Regime Rebound Trading — Backtest Report</h1>

    <h2>Overview</h2>
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value">{m.total_trades}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{m.win_rate:.0%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total P&L</div>
            <div class="metric-value {'positive' if m.total_pnl >= 0 else 'negative'}">${m.total_pnl:+.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Return</div>
            <div class="metric-value {'positive' if m.total_return_pct >= 0 else 'negative'}">{m.total_return_pct:+.1f}%</div>
        </div>
    </div>

    <h2>Risk-Adjusted Returns</h2>
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{m.sharpe_ratio:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sortino Ratio</div>
            <div class="metric-value">{m.sortino_ratio:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Calmar Ratio</div>
            <div class="metric-value">{m.calmar_ratio:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">{m.max_drawdown_pct:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value">{m.profit_factor:.2f}</div>
        </div>
    </div>

    <h2>Per-Regime Performance</h2>
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label"><span class="regime-badge regime-nc">Non-Cross</span> Trades</div>
            <div class="metric-value">{m.nc_trades} ({m.nc_wins}W)</div>
        </div>
        <div class="metric-card">
            <div class="metric-label"><span class="regime-badge regime-nc">Non-Cross</span> P&L</div>
            <div class="metric-value {'positive' if m.nc_pnl >= 0 else 'negative'}">${m.nc_pnl:+.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label"><span class="regime-badge regime-cr">Cross</span> Trades</div>
            <div class="metric-value">{m.cr_trades} ({m.cr_wins}W)</div>
        </div>
        <div class="metric-card">
            <div class="metric-label"><span class="regime-badge regime-cr">Cross</span> P&L</div>
            <div class="metric-value {'positive' if m.cr_pnl >= 0 else 'negative'}">${m.cr_pnl:+.3f}</div>
        </div>
    </div>

    {chart_html}

    {"<h2>Monte Carlo Simulation</h2>" + '''
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Median Terminal</div>
            <div class="metric-value">${mc_med:.0f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">90% Confidence</div>
            <div class="metric-value">${mc_lo:.0f} – ${mc_hi:.0f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">P(Profitable)</div>
            <div class="metric-value">{mc_pp:.0%}</div>
        </div>
    </div>
    '''.format(mc_med=m.mc_median_terminal, mc_lo=m.mc_5th_percentile,
               mc_hi=m.mc_95th_percentile, mc_pp=m.mc_prob_profitable)
    if m.mc_median_terminal > 0 else ""}

    {wf_html}

    <h2>Trade Statistics</h2>
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Avg Win</div>
            <div class="metric-value positive">${m.avg_win:+.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Loss</div>
            <div class="metric-value negative">${m.avg_loss:+.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best Trade</div>
            <div class="metric-value positive">${m.best_trade_pnl:+.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Worst Trade</div>
            <div class="metric-value negative">${m.worst_trade_pnl:+.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Consec. Losses</div>
            <div class="metric-value">{m.max_consecutive_losses}</div>
        </div>
    </div>

    <footer>
        Generated by AI Dual-Regime Rebound Trading System &mdash; Backtest Engine v2.0
    </footer>
</body>
</html>"""

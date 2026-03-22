import React from 'react';

/**
 * Multi-line SVG chart for overlaying multiple outcome probability lines.
 * Mimics the Kalshi app's multi-outcome chart with:
 *   - Multiple colored lines for each outcome
 *   - Y-axis showing percentage (0%-100%)
 *   - X-axis with time labels
 *   - Current price dots at line endings
 *   - Gradient area fill for primary (first) line
 *
 * Props:
 *   datasets — [{ label, color, candles }]
 *   width, height — SVG dimensions
 */
export default function MultiLineChart({ datasets = [], width = 700, height = 280 }) {
  if (!datasets || datasets.length === 0 || datasets.every(d => (d.candles || []).length < 2)) {
    return (
      <div className="price-chart price-chart--empty" style={{ width, height }}>
        <span>No price data available</span>
      </div>
    );
  }

  const pad = { top: 16, right: 0, bottom: 0, left: 0 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  // Collect all timestamps and determine global time range
  let allTimes = [];
  datasets.forEach(d => {
    (d.candles || []).forEach(c => {
      if (c.end_period_ts) allTimes.push(c.end_period_ts);
    });
  });
  allTimes = [...new Set(allTimes)].sort((a, b) => a - b);

  if (allTimes.length < 2) {
    return (
      <div className="price-chart price-chart--empty" style={{ width, height }}>
        <span>Insufficient data</span>
      </div>
    );
  }

  const minT = allTimes[0];
  const maxT = allTimes[allTimes.length - 1];
  const rangeT = maxT - minT || 1;

  // Fixed Y range: 0% to 100% (Kalshi probabilities)
  const minP = 0;
  const maxP = 1;
  const rangeP = 1;

  // Add right padding for labels, bottom for time axis, left for percent labels
  const rightPad = 120;
  const bottomPad = 24;
  const leftPad = 40;
  const topPad = 16;
  
  const toX = (t) => leftPad + ((t - minT) / rangeT) * plotW;
  const toY = (p) => topPad + (1 - (p - minP) / rangeP) * plotH;

  // Generate X-axis labels (time)
  const xLabels = [];
  if (rangeT > 0) {
    const formatTime = (ts) => {
      const d = new Date(ts * 1000);
      let h = d.getHours();
      const m = d.getMinutes().toString().padStart(2, '0');
      const ampm = h >= 12 ? 'pm' : 'am';
      h = h % 12 || 12;
      return `${h}:${m}${ampm}`;
    };
    
    // Create ~5 evenly spaced time labels
    const numLabels = 5;
    for (let i = 0; i < numLabels; i++) {
      const ts = minT + (rangeT * i) / (numLabels - 1);
      xLabels.push({ x: toX(ts), label: formatTime(ts) });
    }
  }

  // Generate Y-axis labels (percentage values)
  const yLabels = [0, 0.2, 0.4, 0.6, 0.8].map(p => ({
    y: toY(p), 
    label: `${Math.round(p * 100)}%`
  }));


  return (
    <svg className="price-chart" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>

      {/* Horizontal Grid Lines */}
      {yLabels.map((yl, i) => (
        <g key={`grid-y-${i}`}>
          <line
            x1={leftPad} y1={yl.y}
            x2={width - rightPad} y2={yl.y}
            stroke="#222"
            strokeDasharray="4,4"
            strokeWidth="1"
          />
        </g>
      ))}

      {/* Render each dataset as a line */}
      {datasets.map((dataset, di) => {
        const { candles = [], color, label } = dataset;
        if (candles.length < 2) return null;

        const prices = candles.map(c => {
          const p = c.price || c;
          return parseFloat(p.close_dollars || p.mean_dollars || p.close || 0);
        });
        const times = candles.map(c => c.end_period_ts || 0);

        const points = prices.map((p, i) => ({
          x: toX(times[i]),
          y: toY(Math.max(0, Math.min(1, p))),
        }));

        // Kalshi step line: horizontal then vertical (L x[i], y[i-1] L x[i], y[i])
        // Wait, the prompt says change from step-line to smooth curves, but Kalshi's screenshot IS a step-line.
        // Looking at the second screenshot (LIV Golf), it is a jagged line, but *direct* point-to-point (standard line chart), NOT a step-line.
        // Let's use direct point-to-point.
        let linePath = `M${points[0].x},${points[0].y}`;
        for (let i = 1; i < points.length; i++) {
          linePath += ` L${points[i].x},${points[i].y}`;
        }
        
        const lastPt = points[points.length - 1];
        const lastPrice = prices[prices.length - 1];

        // Area fill for first dataset only
        const areaPath = di === 0
          ? `${linePath} L${lastPt.x},${toY(0)}L${points[0].x},${toY(0)}Z`
          : null;

        const gradId = `grad-${di}`;

        // Ensure label doesn't crash into other labels by adjusting Y slightly if needed 
        // (A simple approach without full collision detection)
        const textY = lastPt.y;

        return (
          <g key={di}>
            {/* Gradient definition for area */}
            {di === 0 && (
              <defs>
                <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity="0.15" />
                  <stop offset="100%" stopColor={color} stopOpacity="0.01" />
                </linearGradient>
              </defs>
            )}

            {/* Area fill */}
            {areaPath && <path d={areaPath} fill={`url(#${gradId})`} />}

            {/* Line */}
            <path
              d={linePath}
              fill="none"
              stroke={color}
              strokeWidth="2.5"
              strokeLinejoin="round"
              strokeLinecap="round"
            />

            {/* Current price dot */}
            <circle cx={lastPt.x} cy={lastPt.y} r="4" fill={color} />
          </g>
        );
      })}

      {/* Render text labels at the end of the line (after all lines so they are on top) */}
      {datasets.map((dataset, di) => {
        const { candles = [], color, label } = dataset;
        if (candles.length < 2) return null;

        const prices = candles.map(c => {
          const p = c.price || c;
          return parseFloat(p.close_dollars || p.mean_dollars || p.close || 0);
        });
        const times = candles.map(c => c.end_period_ts || 0);
        
        const lastPrice = prices[prices.length - 1];
        const lastPt = {
            x: toX(times[times.length - 1]),
            y: toY(Math.max(0, Math.min(1, lastPrice)))
        };

        return (
            <text
              key={`label-${di}`}
              x={lastPt.x + 8}
              y={lastPt.y - 4}
              fill={color}
              fontSize="12px"
              fontWeight="500"
              fontFamily="sans-serif"
            >
              <tspan x={lastPt.x + 8} dy="0">{label.split(' ')[0]}</tspan>
              <tspan x={lastPt.x + 8} dy="16" fontSize="16px" fontWeight="700">{Math.round(lastPrice * 100)}%</tspan>
            </text>
        );
      })}

      {/* Axis Labels (rendered last to stay on top) */}
      {yLabels.map((yl, i) => (
        <text
          key={`y-axis-${i}`}
          x={width - rightPad + 8}
          y={yl.y + 4} // vertical center adjustment
          fill="#666"
          fontSize="11px"
          fontFamily="sans-serif"
        >
          {yl.label}
        </text>
      ))}

      {xLabels.map((xl, i) => (
        <text
          key={`x-axis-${i}`}
          x={xl.x}
          y={height - 5}
          fill="#666"
          fontSize="11px"
          fontFamily="sans-serif"
          textAnchor={i === 0 ? 'start' : i === xLabels.length - 1 ? 'end' : 'middle'}
        >
          {xl.label}
        </text>
      ))}

    </svg>
  );
}

/**
 * Compact sparkline version for market cards.
 */
export function Sparkline({ candles = [], width = 100, height = 32 }) {
  if (candles.length < 2) return null;

  const prices = candles.map((c) => {
    const p = c.price || c;
    return parseFloat(p.close_dollars || p.mean_dollars || p.close || 0);
  });
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 0.01;

  const points = prices.map((p, i) => {
    const x = (i / (prices.length - 1)) * width;
    const y = (1 - (p - min) / range) * height;
    return `${x},${y}`;
  });

  const last = prices[prices.length - 1];
  const first = prices[0];
  const color = last >= first ? '#00d4aa' : '#ef4444';

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="sparkline">
      <polyline points={points.join(' ')} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

"""Page 6: Methodology, Causal Alignment, Architecture & Limitations."""

from __future__ import annotations

import streamlit as st

from squeezeradar.ui import (
    get_replay_scan_params,
    init_page,
    load_radar_cached,
    render_footer,
    render_sidebar,
    show_sidebar_badge,
)

init_page("About")
mode, symbols, cache_bust = render_sidebar(allow_live=True)
as_of_arg, limit_bars, _, _, _ = get_replay_scan_params(mode)
scan_res = load_radar_cached(symbols, mode, as_of=as_of_arg, limit=limit_bars, cache_bust=cache_bust)
show_sidebar_badge(scan_res.badge, scan_res.as_of_label)

st.title("📘 Methodology, Architecture & Limitations")

st.markdown(
    """
### 1. Why Perpetual Futures Squeezes Happen
Perpetual futures lack an expiry date and anchor to spot via **funding rates** paid periodically (typically every 8h on Binance, every 1h on Hyperliquid). When one side crowds aggressively:
1. **Funding Pain**: Shorts pay longs when funding turns negative (or longs pay shorts when positive), eroding margin over time.
2. **OI-Price Divergence**: Rising Open Interest while price stalls or drifts against new entrants indicates leveraged positions accumulating without clearing.
3. **Forced Liquidations**: Leveraged positions sit on predictable maintenance-margin liquidation thresholds (`liq_long = p*(1 - 1/L + mmr)`, `liq_short = p*(1 + 1/L - mmr)`). Once spot sweeps a dense cluster, exchange liquidation engines fire market orders in the *same* direction as the move, driving self-reinforcing cascades.

---

### 2. Strict Causal Alignment Rules
- **1h OHLCV**: Bar `open_time = h` covers `[h, h+1h)` and becomes observable strictly at bar close `h+1h`.
- **1h Metrics**: Aggregated 5-minute snapshots inside `[h, h+1h)` are known at bar close `h+1h`. Missing hours (and non-positive OI glitches in the archive) are forward-filled at most **2 bars**, otherwise left `NaN` (no backward-filling). Live Binance OI / L/S snapshots taken at `T` are mapped to the bar closing at `T`.
- **Funding**: Aligned via backward `merge_asof` on `close_time = open_time + 1h` against `funding_time`, ensuring zero future funding rates enter historical bars.
- **Rolling Statistics & Features**: All z-scores, ATR(14), Bollinger Bandwidth 30d percentiles, and CVD proxies use trailing windows only. OI changes are measured on contract OI so price moves alone do not look like positions opening or closing.

---

### 3. Estimated Liquidation Map (Model Approximation)
- **Delta-OI Allocation**: Positive hourly contract `delta-OI`, valued at that hour's VWAP-proxy `(H+L+C)/3`, enters at that price, split between longs and shorts using taker buy/sell flow and candle direction.
- **Assumed Leverage Mix**: Default weights `5x: 25%, 10x: 35%, 20x: 20%, 50x: 15%, 100x: 5%` with `mmr = 0.5%`.
- **Sweeps & Reductions**: Pre-existing positions whose liquidation price is crossed by `[low, high]` are removed and counted as estimated liquidation events; a negative `delta-OI` beyond those liquidations removes the same fraction of every surviving level. Cluster metrics need 24 hourly OI observations.
- **Universal Log-Price Grid**: Positions map to dataset-independent `floor(ln(p) / 0.002)` bins so truncation at any bar `t` yields identical estimates.

---

### 4. Rule Score & State Machine
Each side (`short_squeeze`, `long_squeeze`) outputs a `0–100` weighted score over 5 components: Funding Crowding (`22%`), OI-Price Divergence (`22%`), L/S Positioning (`20%`), Liquidation Cluster Proximity (`20%`), and Volatility Compression (`16%`).

State transitions per side:
`NONE` → `SETUP` (score ≥ 50) → `TRIGGER` (price moves against crowd + OI drops + taker flow spike) → `IN_PROGRESS` → `EXHAUSTION` → `NONE` / `SETUP`.

---

### 5. Data Providers & Fallback Chain
1. **Binance USDT-M Public REST**: Full OHLCV, funding, OI, top-trader & global L/S ratios, taker ratio (note: `/futures/data/*` retains only the trailing **30 days**).
2. **Hyperliquid Public `/info` Fallback**: Automatically invoked if Binance returns HTTP 451 (e.g., US cloud IPs), 403, or timeouts; provides candles, hourly funding history (converted x8 to the 8h-equivalent rate), and the current `openInterest`.
3. **Offline Replay Mode**: Bundled real 2023–2025 Binance dataset (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `XRPUSDT`, `DOGEUSDT`, `BNBUSDT`) ensuring 100% offline reproducibility.

---

### 6. Key Limitations
- Liquidation levels and USD cluster sizes are **analytical estimates**, not live exchange liquidation orderbooks.
- Cross-margin collateral, portfolio margin, and off-exchange hedges are unobservable and can shift real liquidation prices far from isolated-margin formulas.
- Binance `/futures/data/*` live REST endpoints limit historical OI/positioning metrics to 30 days (use `squeezeradar download` for longer Binance Vision history).
- Hyperliquid fallback lacks historical hourly OI series and Binance account L/S ratios, so no estimated liquidation map is shown in that mode; scores automatically re-weight across available inputs.
- **Not financial advice.**
    """
)

render_footer()

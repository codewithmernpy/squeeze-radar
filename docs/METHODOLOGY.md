# SqueezeRadar Methodology

SqueezeRadar models liquidation-driven positioning cascades in crypto perpetual futures while enforcing strict causal time alignment across market data, liquidation estimation, rule scoring, and walk-forward machine learning.

---

## 1. Causal Data Alignment

Perpetual futures combine three asynchronous feeds:

1. **Hourly OHLCV (`1h` klines)**:
   - Row `open_time = h` covers `[h, h + 1h)` and closes at `close_time = h + 1h`.
   - In live REST mode, any still-open bar (`close_time > now`) is discarded so partial candles never contaminate rolling z-scores or liquidation sweeps.
2. **Hourly Derivatives Metrics (`metrics_1h`)**:
   - Row `open_time = h` aggregates 5-minute exchange snapshots inside `[h, h + 1h)` (`last` for Open Interest and Long/Short ratios; `mean` for Taker Buy/Sell ratio).
   - Thus row `h` becomes observable strictly at bar close `h + 1h`, aligned 1-to-1 with bar `open_time = h`.
   - Missing hourly metric rows are forward-filled across a regular 1h grid for at most **2 bars** (`limit=2`); longer outages remain `NaN` rather than back-filling or interpolating future data.
   - Non-positive OI / ratio values (recording glitches in the Binance metrics archive, e.g. 30 hours of `oi = 0` for BTCUSDT) are treated as missing instead of as a -100% OI crash.
   - Live Binance REST: `/futures/data/openInterestHist` and the long/short-ratio endpoints return point-in-time snapshots taken *at* `timestamp` T, i.e. the state at the close of bar `T-1h`, so they are mapped to bar `open_time = T-1h`; `takerlongshortRatio` rows aggregate `[T, T+1h)` and map to bar `T`. This gives live data the same meaning as the bundled archive.
3. **Funding Settlements (`funding`)**:
   - Joined via backward `merge_asof` comparing `bar.close_time` against `funding_time` (with a 24h tolerance).
   - Only funding settlements with `funding_time <= bar.close_time` are visible at bar `t`.
4. **Data Sources, Live Discovery & Local Cache**:
   - **Live Universe & Ticker Resolution**: User-entered tickers (e.g., `pepe`) are normalized and validated against Binance's live USDT-M perpetual contract list (`/fapi/v1/exchangeInfo`, falling back to `/fapi/v1/ticker/24hr`), mapping shorthand names to listed contracts (such as `PEPEUSDT` or `1000PEPEUSDT`) and rejecting unknown symbols cleanly. Top-N-by-volume scans rank `TRADING` USDT-M perpetuals by 24h quote volume from `/fapi/v1/ticker/24hr`.
   - **Historical Archive Downloads (`data.binance.vision`)**: Downloading extra coins saves monthly `1h` klines, monthly `fundingRate`, and threaded daily `5m` metrics into `data/cache/` (never modifying `data/sample/`), caching raw archive files under `data/cache/_archives/` to avoid re-downloading and applying the exact same causal `[h, h+1h)` aggregation (`last` for OI and L/S ratios, `mean` for taker ratio) as the bundled archive.
   - **Replay Date Range Filtering**: End-date filters enforce strict close-time cutoffs (`close_time <= end_date + 1d 00:00 UTC`) prior to alignment, features, liquidation mapping, and walk-forward splits, ensuring zero future leakage inside any selected range.

---

## 2. Causal Feature Engineering (`squeezeradar/features.py`)

All features at bar `t` depend strictly on observations `<= t`:

- **Funding Crowding**: `funding_rate_last`, trailing `24h` and `7d` averages, annualized funding %, and 30-day (`720h`) trailing z-score (variance floored at `2e-5` to prevent near-zero-std artifacts when Binance funding stays pinned at `0.01%`).
- **Open Interest & OI-Price Divergence**: `1h`, `4h`, `24h` percent changes and the 30-day z-score of **contract** OI (`oi`; USD notional moves mechanically with price, so a price drop alone would look like positions closing), the `oi_to_vol_24h` crowding ratio (USD OI / 24h quote volume), and directional buildup indicators:
  - *Shorts building*: `oi_chg > 0` while `ret <= +0.5%`
  - *Longs building*: `oi_chg > 0` while `ret >= -0.5%`
  - Signed divergence `oi_price_div = longs_building - shorts_building`
- **Positioning Skew**: Global account L/S ratio, Top Trader position/account L/S ratios, and trailing 30d z-scores.
- **Taker Flow & CVD Proxy**: Effective taker buy/sell ratio (`taker_buy_sell_ratio` or derived from `taker_buy_base / volume`), volume z-score, and rolling 24h normalized Cumulative Volume Delta (`(2 * taker_buy_share - 1) * quote_volume`).
- **Price Action & Compression**: Simple returns (`1h`, `4h`, `24h`), Wilder `ATR(14)`, `atr_pct`, 20-hour Bollinger bandwidth and its causal 30-day rolling percentile rank (`bb_bw_pct_30d`), plus percentage distance to trailing 7d and 30d highs/lows.

---

## 3. Estimated Liquidation Map (`squeezeradar/liqmap.py`)

> **Important**: Exchange account liquidation levels are private. SqueezeRadar constructs an **analytical estimate**, not a live orderbook feed.

1. **Entry Allocation**: Positive hourly contract `delta_oi` is valued at the bar's VWAP-proxy `p = (high + low + close) / 3` and enters at `p` (the first observed OI is seeded at the first bar's `p`).
2. **Directional Split**: New exposure is split into long vs short shares using a weighted blend of taker buy share (`70%`) and candle body direction (`30%`), bounded in `[0.15, 0.85]`.
3. **Leverage Bucket Distribution** (configurable; defaults):
   - `5x`: 25%, `10x`: 35%, `20x`: 20%, `50x`: 15%, `100x`: 5%
   - Maintenance Margin Rate `mmr = 0.5%` (`0.005`)
   - Long liquidation price: `p_liq_long = p * (1 - 1/L + mmr)`
   - Short liquidation price: `p_liq_short = p * (1 + 1/L - mmr)`
4. **Sweeps & Voluntary Reductions**:
   - Existing active long bins with `bin_price >= low_t` and short bins with `bin_price <= high_t` are removed and logged as estimated liquidation events (`est_long_liq_usd`, `est_short_liq_usd`).
   - A negative `delta_oi` beyond the swept liquidations is treated as voluntary closing: the same **fraction** of the OI still open after the sweep (`(|delta_oi| - swept) / (prev_oi - swept)`, all valued at `p`) is removed from every surviving level. Total open book notional is capped at the current observed `oi_value`.
   - Cluster metrics (`nearest_*`, `*_within_5pct_*`) and the price profile are reported only after 24 hourly OI observations (otherwise `NaN` / empty: the book would just be the seeded leverage ladder); the score then re-weights over its other components.
5. **Universal Log-Price Grid**: Positions map onto dataset-independent `floor(ln(p) / 0.002)` indices, guaranteeing identical estimates whether computed on `data[:t+1]` or the full archive.

---

## 4. Rule Risk Scores & State Machine (`squeezeradar/scores.py`)

Each side (`short_squeeze` and `long_squeeze`) produces a `0–100` composite score across five normalized components:

| Component | Weight | Short-Squeeze Driver | Long-Squeeze / Cascade Driver |
|---|---:|---|---|
| Funding Crowding | 22% | Negative funding z-score & negative rate | Positive funding z-score & elevated positive rate |
| OI-Price Divergence | 22% | OI rising while price flat/down | OI rising while price flat/up |
| Positioning Skew | 20% | Short-heavy L/S ratios & negative z-score | Long-heavy L/S ratios & positive z-score |
| Liquidation Cluster | 20% | Large estimated short-liq pool within +5% above spot | Large estimated long-liq pool within -5% below spot |
| Volatility Compression | 16% | Low 30d Bollinger bandwidth percentile | Low 30d Bollinger bandwidth percentile |

Missing components (e.g., L/S ratios under Hyperliquid fallback) are skipped cleanly by normalizing weights over available components.

### Causal State Machine (Per Side)
- `NONE` → `SETUP` when `score >= 50.0` (returns to `NONE` if `score < 38.0`).
- `SETUP` → `TRIGGER` when price moves against the crowded side (`|ret_1h| >= 0.5%` or `|ret_4h| >= 1.2%`), OI contracts (`oi_chg_1h <= -0.4%` or `oi_chg_4h <= -1.0%`), and taker flow spikes in the squeeze direction.
- `TRIGGER` → `IN_PROGRESS` while continuation holds (momentum, falling OI, or ongoing estimated liquidations).
- `IN_PROGRESS` → `EXHAUSTION` when OI stabilizes/rebounds and momentum stalls (or after 24 bars max), then resets to `NONE` (or `SETUP` if crowding persists).

---

## 5. Labels, Walk-Forward ML & Event Studies (`squeezeradar/model.py`)

- **Target Labels** (strictly excluded from feature inputs):
  - `squeeze_up[t] = 1` iff over `t+1..t+24`, `high` hits `close_t + 2 * ATR_14` strictly before `low` hits `close_t - 1 * ATR_14` (a tie in the same bar counts as a miss), **AND** contract OI `min(oi[t+1..t+24]) <= oi[t] * 0.97` (`>= 3%` OI drop).
  - `squeeze_down[t] = 1` is the exact downside mirror.
- **Purged Expanding Walk-Forward**:
  - Multi-symbol pooled training using strictly symbol-agnostic features (no raw price, volume, or USD OI levels).
  - Retrains `HistGradientBoostingClassifier` monthly on an expanding window after at least 6 months of history.
  - **Strict 24-Bar Purge**: All bars inside `[test_month_start - 24h, test_month_start)` are dropped from training so no 24-hour label window overlaps the test period.
- **Event Study**: Measures average `0–48h` forward return paths and hit rates across declustered signal episodes (`>= 12h` apart) versus the unconditional baseline.
- **Precomputed Results**: `python -m squeezeradar evidence --build` runs the walk-forward, event study and latest purged probabilities on the bundled sample and writes `data/evidence/evidence_results.json`; the Evidence page and the AI analyst load it instantly (the page can recompute live).

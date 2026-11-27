"""Causal feature engineering for perpetual futures squeeze detection.

Every feature at bar `t` uses strictly data available at or before the close of bar `t`.
All ML features are symbol-agnostic (normalized ratios, percentages, and z-scores).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAUSAL_EPS = 1e-12

# Symbol-agnostic feature columns eligible for cross-symbol ML models
MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "funding_rate_last",
    "funding_mean_24h",
    "funding_mean_7d",
    "funding_z_30d",
    "oi_chg_1h",
    "oi_chg_4h",
    "oi_chg_24h",
    "oi_z_30d",
    "oi_price_div_4h",
    "oi_price_div_24h",
    "oi_to_vol_24h",
    "global_ls_ratio",
    "global_ls_z_30d",
    "top_ls_pos_ratio",
    "top_ls_pos_z_30d",
    "top_ls_acc_ratio",
    "top_ls_acc_z_30d",
    "taker_buy_sell_ratio",
    "taker_ratio_z_30d",
    "volume_z_30d",
    "taker_buy_share",
    "cvd_24h_norm",
    "cvd_z_30d",
    "ret_1h",
    "ret_4h",
    "ret_24h",
    "atr_pct",
    "bb_bw_20",
    "bb_bw_pct_30d",
    "dist_7d_high_pct",
    "dist_7d_low_pct",
    "dist_30d_high_pct",
    "dist_30d_low_pct",
)


def _causal_zscore(
    series: pd.Series,
    window: int = 720,
    min_periods: int = 24,
    std_floor: float = 1e-9,
) -> pd.Series:
    """Compute strictly causal trailing z-score using only bars <= t."""
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    denom = std.clip(lower=std_floor)
    return (series - mean) / denom


def _causal_pct_change(series: pd.Series, periods: int) -> pd.Series:
    """Compute explicit ratio change without implicit fill methods."""
    prev = series.shift(periods)
    return (series / prev.where(prev.abs() > CAUSAL_EPS)) - 1.0


def open_interest_contracts(df: pd.DataFrame) -> pd.Series:
    """Open interest in contracts (base-asset units) -- the quantity that measures positioning.

    USD notional OI (`oi_value`) moves mechanically with price (a 3% price drop shrinks USD OI by
    3% with no position closed), so OI *changes* are measured on contract OI (`oi`). Frames
    without an `oi` column fall back to the proxy `oi_value / close`.
    """
    if "oi" in df.columns and df["oi"].notna().any():
        return df["oi"].astype(float)
    return df["oi_value"].astype(float) / df["close"].astype(float).where(df["close"] > 0)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full causal derivatives and price-action features on an aligned frame.

    Parameters
    ----------
    df:
        Aligned hourly dataframe containing OHLCV, `taker_buy_base`, `funding_rate`,
        `oi_value`, `global_ls_accounts`, `top_trader_ls_positions`,
        `top_trader_ls_accounts`, and `taker_buy_sell_ratio`.

    Returns
    -------
    pd.DataFrame
        Copy of input frame augmented with all causal feature columns.
    """
    out = df.copy()
    if out.empty:
        return out

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)
    quote_vol = out["quote_volume"].astype(float)
    funding = out["funding_rate"].astype(float)
    oi_usd = out["oi_value"].astype(float)

    # --- Funding features ---
    out["funding_rate_last"] = funding
    out["funding_mean_24h"] = funding.rolling(24, min_periods=1).mean()
    out["funding_mean_7d"] = funding.rolling(168, min_periods=1).mean()
    # Floor std at 2e-5 (0.002% per 8h) so flat 0.01% regimes don't explode z-scores
    out["funding_z_30d"] = _causal_zscore(funding, window=720, min_periods=24, std_floor=2e-5)
    out["funding_annualized_pct"] = funding * 3.0 * 365.0 * 100.0

    # --- Price returns, ATR(14), Bollinger bandwidth compression, distances ---
    out["ret_1h"] = _causal_pct_change(close, 1)
    out["ret_4h"] = _causal_pct_change(close, 4)
    out["ret_24h"] = _causal_pct_change(close, 24)

    prev_close = close.shift(1)
    tr_1 = high - low
    tr_2 = (high - prev_close).abs()
    tr_3 = (low - prev_close).abs()
    true_range = pd.concat([tr_1, tr_2, tr_3], axis=1).max(axis=1)
    out["atr_14"] = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=1).mean()
    out["atr_pct"] = out["atr_14"] / close.clip(lower=CAUSAL_EPS)

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std(ddof=0)
    out["bb_bw_20"] = (4.0 * bb_std) / bb_mid.clip(lower=CAUSAL_EPS)
    out["bb_bw_pct_30d"] = out["bb_bw_20"].rolling(720, min_periods=24).rank(pct=True)

    high_7d = high.rolling(168, min_periods=1).max()
    low_7d = low.rolling(168, min_periods=1).min()
    high_30d = high.rolling(720, min_periods=1).max()
    low_30d = low.rolling(720, min_periods=1).min()
    out["dist_7d_high_pct"] = (close / high_7d.clip(lower=CAUSAL_EPS)) - 1.0
    out["dist_7d_low_pct"] = (close / low_7d.clip(lower=CAUSAL_EPS)) - 1.0
    out["dist_30d_high_pct"] = (close / high_30d.clip(lower=CAUSAL_EPS)) - 1.0
    out["dist_30d_low_pct"] = (close / low_30d.clip(lower=CAUSAL_EPS)) - 1.0

    # --- Open Interest & OI-Price divergence ---
    # Levels are reported in USD; changes / z-scores use contract OI so price moves alone
    # do not register as positions opening or closing.
    oi_qty = open_interest_contracts(out)
    out["oi_usd"] = oi_usd
    out["oi_chg_1h"] = _causal_pct_change(oi_qty, 1)
    out["oi_chg_4h"] = _causal_pct_change(oi_qty, 4)
    out["oi_chg_24h"] = _causal_pct_change(oi_qty, 24)
    out["oi_z_30d"] = _causal_zscore(oi_qty, window=720, min_periods=24, std_floor=1e-9)

    quote_vol_24h = quote_vol.rolling(24, min_periods=1).sum()
    out["quote_vol_24h"] = quote_vol_24h
    out["oi_to_vol_24h"] = oi_usd / quote_vol_24h.clip(lower=1.0)

    # OI up + price flat/down => shorts building (negative divergence)
    # OI up + price flat/up   => longs building  (positive divergence)
    for h_suffix, oi_col, ret_col in (
        ("4h", "oi_chg_4h", "ret_4h"),
        ("24h", "oi_chg_24h", "ret_24h"),
    ):
        oi_c = out[oi_col]
        ret_c = out[ret_col]
        pos_oi = oi_c.clip(lower=0.0)
        shorts_b = np.where(
            ret_c <= 0.005,
            pos_oi * (1.0 + np.maximum(-ret_c, 0.0) / 0.05),
            0.0,
        )
        longs_b = np.where(
            ret_c >= -0.005,
            pos_oi * (1.0 + np.maximum(ret_c, 0.0) / 0.05),
            0.0,
        )
        shorts_s = pd.Series(shorts_b, index=out.index, dtype=float).where(oi_c.notna() & ret_c.notna())
        longs_s = pd.Series(longs_b, index=out.index, dtype=float).where(oi_c.notna() & ret_c.notna())
        out[f"shorts_building_{h_suffix}"] = shorts_s
        out[f"longs_building_{h_suffix}"] = longs_s
        out[f"oi_price_div_{h_suffix}"] = longs_s - shorts_s

    # --- Positioning ratios ---
    g_ls = out["global_ls_accounts"].astype(float)
    t_pos = out["top_trader_ls_positions"].astype(float)
    t_acc = out["top_trader_ls_accounts"].astype(float)

    out["global_ls_ratio"] = g_ls
    out["global_ls_z_30d"] = _causal_zscore(g_ls, window=720, min_periods=24, std_floor=0.02)
    out["top_ls_pos_ratio"] = t_pos
    out["top_ls_pos_z_30d"] = _causal_zscore(t_pos, window=720, min_periods=24, std_floor=0.02)
    out["top_ls_acc_ratio"] = t_acc
    out["top_ls_acc_z_30d"] = _causal_zscore(t_acc, window=720, min_periods=24, std_floor=0.02)

    # --- Flow & CVD proxy ---
    taker_ratio_metric = out["taker_buy_sell_ratio"].astype(float)
    taker_buy_base = out["taker_buy_base"].astype(float)
    kline_buy_share = pd.Series(
        np.where(volume > CAUSAL_EPS, taker_buy_base / volume.clip(lower=CAUSAL_EPS), np.nan),
        index=out.index,
        dtype=float,
    )
    metric_buy_share = taker_ratio_metric / (1.0 + taker_ratio_metric)
    taker_buy_share = kline_buy_share.fillna(metric_buy_share)
    out["taker_buy_share"] = taker_buy_share

    # Reconcile taker_buy_sell_ratio from klines if metric is absent
    kline_taker_ratio = kline_buy_share / (1.0 - kline_buy_share).clip(lower=1e-4)
    effective_taker_ratio = taker_ratio_metric.fillna(kline_taker_ratio)
    out["taker_buy_sell_ratio_eff"] = effective_taker_ratio
    out["taker_ratio_z_30d"] = _causal_zscore(
        effective_taker_ratio, window=720, min_periods=24, std_floor=0.03
    )

    out["volume_z_30d"] = _causal_zscore(quote_vol, window=720, min_periods=24, std_floor=1.0)

    cvd_delta_usd = quote_vol * (2.0 * taker_buy_share - 1.0)
    out["cvd_delta_usd"] = cvd_delta_usd
    out["cvd_24h_usd"] = cvd_delta_usd.rolling(24, min_periods=1).sum()
    out["cvd_24h_norm"] = out["cvd_24h_usd"] / quote_vol_24h.clip(lower=1.0)
    out["cvd_z_30d"] = _causal_zscore(out["cvd_24h_norm"], window=720, min_periods=24, std_floor=0.02)

    return out

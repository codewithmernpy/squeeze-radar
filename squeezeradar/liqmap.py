"""Estimated liquidation map model (clearly an analytical ESTIMATE, not exchange feed).

Uses a universal fixed log-price bin grid so any truncation df[:t+1] maps positions
to the exact same bin indices as the full dataframe (guaranteeing exact causality).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from squeezeradar.config import LiqMapConfig
from squeezeradar.features import open_interest_contracts


@dataclass(frozen=True)
class LiqMapResult:
    """Complete output of the estimated liquidation map simulation."""

    summary: pd.DataFrame
    price_profile: pd.DataFrame
    heatmap: pd.DataFrame
    as_of_price: float
    is_estimate: bool = True
    methodology_note: str = (
        "ESTIMATE ONLY: inferred from hourly delta-OI, taker/price flow split, "
        "assumed leverage distribution, and bar high/low sweeps — not exchange orderbook data."
    )


def _bin_index(price: float, log_step: float) -> int:
    """Map a positive price onto a dataset-independent log-price bin index."""
    return int(math.floor(math.log(max(price, 1e-12)) / log_step))


def _bin_center(idx: int, log_step: float) -> float:
    """Return the geometric center price for a universal bin index."""
    return math.exp((idx + 0.5) * log_step)


def _compute_long_share(
    open_p: np.ndarray,
    high_p: np.ndarray,
    low_p: np.ndarray,
    close_p: np.ndarray,
    taker_ratio: np.ndarray,
    taker_buy_base: np.ndarray,
    volume: np.ndarray,
    cfg: LiqMapConfig,
) -> np.ndarray:
    """Compute causal per-bar fraction [min_long_share, max_long_share] attributed to new longs."""
    n = len(close_p)
    long_share = np.full(n, 0.5, dtype=float)

    taker_share = np.full(n, 0.5, dtype=float)
    valid_tr = np.isfinite(taker_ratio) & (taker_ratio > 0.0)
    taker_share[valid_tr] = taker_ratio[valid_tr] / (1.0 + taker_ratio[valid_tr])

    need_kline = (~valid_tr) & np.isfinite(taker_buy_base) & np.isfinite(volume) & (volume > 0.0)
    taker_share[need_kline] = np.clip(
        taker_buy_base[need_kline] / volume[need_kline], 0.0, 1.0
    )

    span = np.maximum(high_p - low_p, 1e-12)
    body_ratio = np.clip((close_p - open_p) / span, -1.0, 1.0)
    dir_share = 0.5 + 0.5 * body_ratio

    w_sum = cfg.taker_direction_weight + cfg.price_direction_weight
    if w_sum > 0.0:
        blended = (
            cfg.taker_direction_weight * taker_share + cfg.price_direction_weight * dir_share
        ) / w_sum
        long_share = np.clip(blended, cfg.min_long_share, cfg.max_long_share)

    return long_share


def _extract_profile_and_clusters(
    long_book: np.ndarray,
    short_book: np.ndarray,
    base_idx: int,
    spot_price: float,
    cfg: LiqMapConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build +/- window % price profile and locate nearest major long/short clusters."""
    log_step = cfg.log_step
    pct_edges = np.linspace(-cfg.profile_window_pct, cfg.profile_window_pct, cfg.profile_bins + 1)
    pct_centers = 0.5 * (pct_edges[:-1] + pct_edges[1:]) * 100.0
    long_hist = np.zeros(cfg.profile_bins, dtype=float)
    short_hist = np.zeros(cfg.profile_bins, dtype=float)

    min_rel = -cfg.profile_window_pct
    max_rel = cfg.profile_window_pct
    spot_idx = _bin_index(spot_price, log_step)

    low_win_idx = _bin_index(spot_price * (1.0 + min_rel), log_step)
    high_win_idx = _bin_index(spot_price * (1.0 + max_rel), log_step)

    lo_off = max(0, low_win_idx - base_idx)
    hi_off = min(len(long_book), high_win_idx - base_idx + 1)

    if hi_off > lo_off:
        offs = np.arange(lo_off, hi_off)
        prices = np.exp((base_idx + offs + 0.5) * log_step)
        rel_pct = (prices / spot_price) - 1.0
        bin_ids = np.floor((rel_pct - min_rel) / ((max_rel - min_rel) / cfg.profile_bins)).astype(int)
        valid = (bin_ids >= 0) & (bin_ids < cfg.profile_bins)

        l_vals = long_book[offs]
        s_vals = short_book[offs]
        is_below = (base_idx + offs) < spot_idx
        is_above = (base_idx + offs) > spot_idx

        np.add.at(long_hist, bin_ids[valid & is_below], l_vals[valid & is_below])
        np.add.at(short_hist, bin_ids[valid & is_above], s_vals[valid & is_above])

    profile_df = pd.DataFrame(
        {
            "pct_from_spot": pct_centers,
            "price_level": spot_price * (1.0 + pct_centers / 100.0),
            "long_liq_usd": long_hist,
            "short_liq_usd": short_hist,
        }
    )

    short_mask = (pct_centers > 0.0) & (short_hist > 0.0)
    nearest_short_dist = np.nan
    nearest_short_usd = 0.0
    if np.any(short_mask):
        s_vals_pos = short_hist[short_mask]
        s_dists_pos = pct_centers[short_mask]
        s_thresh = float(np.max(s_vals_pos)) * cfg.major_cluster_rel_threshold
        major_s = np.where(s_vals_pos >= s_thresh)[0]
        pick = int(major_s[0]) if len(major_s) > 0 else int(np.argmax(s_vals_pos))
        nearest_short_dist = float(s_dists_pos[pick])
        nearest_short_usd = float(s_vals_pos[pick])

    long_mask = (pct_centers < 0.0) & (long_hist > 0.0)
    nearest_long_dist = np.nan
    nearest_long_usd = 0.0
    if np.any(long_mask):
        l_vals_neg = long_hist[long_mask]
        l_dists_neg = np.abs(pct_centers[long_mask])
        order = np.argsort(l_dists_neg)
        l_vals_ord = l_vals_neg[order]
        l_dists_ord = l_dists_neg[order]
        l_thresh = float(np.max(l_vals_ord)) * cfg.major_cluster_rel_threshold
        major_l = np.where(l_vals_ord >= l_thresh)[0]
        pick = int(major_l[0]) if len(major_l) > 0 else int(np.argmax(l_vals_ord))
        nearest_long_dist = float(l_dists_ord[pick])
        nearest_long_usd = float(l_vals_ord[pick])

    w5_short = float(short_hist[(pct_centers > 0.0) & (pct_centers <= 5.0)].sum())
    w5_long = float(long_hist[(pct_centers < 0.0) & (pct_centers >= -5.0)].sum())

    metrics = {
        "nearest_short_liq_dist_pct": nearest_short_dist,
        "nearest_short_liq_usd": nearest_short_usd,
        "nearest_long_liq_dist_pct": nearest_long_dist,
        "nearest_long_liq_usd": nearest_long_usd,
        "short_liq_within_5pct_usd": w5_short,
        "long_liq_within_5pct_usd": w5_long,
    }
    return profile_df, metrics


def build_liquidation_map(
    df: pd.DataFrame,
    config: LiqMapConfig | None = None,
    *,
    seed_initial_oi: bool = True,
    heatmap_bars: int = 168,
) -> LiqMapResult:
    """Simulate causal estimated liquidation levels and liquidation events over an aligned dataframe."""
    cfg = config or LiqMapConfig()
    if df.empty:
        return LiqMapResult(
            summary=pd.DataFrame(),
            price_profile=pd.DataFrame(),
            heatmap=pd.DataFrame(),
            as_of_price=float("nan"),
        )

    n = len(df)
    open_p = df["open"].to_numpy(dtype=float)
    high_p = df["high"].to_numpy(dtype=float)
    low_p = df["low"].to_numpy(dtype=float)
    close_p = df["close"].to_numpy(dtype=float)
    vwap_p = (high_p + low_p + close_p) / 3.0
    oi_usd = df["oi_value"].to_numpy(dtype=float)
    # Positions opened/closed are measured in CONTRACTS and valued at the bar's VWAP proxy, so a
    # price move with unchanged contract OI does not add or remove estimated positions.
    oi_qty = open_interest_contracts(df).to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    taker_buy_base = df["taker_buy_base"].to_numpy(dtype=float)
    taker_ratio = df["taker_buy_sell_ratio"].to_numpy(dtype=float)

    long_share = _compute_long_share(
        open_p, high_p, low_p, close_p, taker_ratio, taker_buy_base, volume, cfg
    )

    leverages = np.array([lev for lev, _ in cfg.leverage_weights], dtype=float)
    lev_weights = np.array([w for _, w in cfg.leverage_weights], dtype=float)
    lev_weights = lev_weights / lev_weights.sum()
    n_lev = len(leverages)

    long_mults = 1.0 - 1.0 / leverages + cfg.mmr
    short_mults = 1.0 + 1.0 / leverages - cfg.mmr
    step = cfg.log_step
    inv_step = 1.0 / step

    # Vectorized precomputation of universal log-price bin indices (100% causal per bar t)
    safe_low = np.maximum(low_p, 1e-12)
    safe_high = np.maximum(high_p, 1e-12)
    safe_close = np.maximum(close_p, 1e-12)
    safe_vwap = np.maximum(vwap_p, 1e-12)

    k_low_arr = np.floor(np.log(safe_low) * inv_step).astype(np.int64)
    k_high_arr = np.floor(np.log(safe_high) * inv_step).astype(np.int64)
    k_c_arr = np.floor(np.log(safe_close) * inv_step).astype(np.int64)

    # Adjust sweep offsets so bin centers satisfy >= low_t (for longs) and <= high_t (for shorts)
    center_low = np.exp((k_low_arr + 0.5) * step)
    k_low_sweep = np.where(center_low < safe_low, k_low_arr + 1, k_low_arr)
    center_high = np.exp((k_high_arr + 0.5) * step)
    k_high_sweep = np.where(center_high > safe_high, k_high_arr - 1, k_high_arr)

    log_vwap = np.log(safe_vwap)
    log_long_mult = np.log(np.maximum(long_mults, 1e-6))
    log_short_mult = np.log(np.maximum(short_mults, 1e-6))
    l_idxs_mat = np.floor((log_vwap[:, None] + log_long_mult[None, :]) * inv_step).astype(np.int64)
    s_idxs_mat = np.floor((log_vwap[:, None] + log_short_mult[None, :]) * inv_step).astype(np.int64)

    win_half_bins = max(5, int(round(math.log(1.0 + cfg.profile_window_pct) / step)))
    win_5pct_bins = max(2, int(round(math.log(1.05) / step)))
    rel_pct_table = (np.exp(np.arange(-win_half_bins, win_half_bins + 1) * step) - 1.0) * 100.0
    rel_short_pct = rel_pct_table[win_half_bins + 1 :]
    rel_long_pct = np.abs(rel_pct_table[:win_half_bins][::-1])

    # Pre-initialize dynamic array around initial bar price (expanded causally if price leaves range)
    init_k = int(k_c_arr[0]) if np.isfinite(close_p[0]) and close_p[0] > 0 else 0
    pad_margin = max(256, win_half_bins * 3)
    base_idx = init_k - pad_margin
    book_len = 2 * pad_margin + 1
    long_book = np.zeros(book_len, dtype=float)
    short_book = np.zeros(book_len, dtype=float)
    total_book_sum = 0.0

    valid_bar = np.isfinite(close_p) & (close_p > 0.0) & np.isfinite(high_p) & np.isfinite(low_p)

    est_long_liq = np.zeros(n, dtype=float)
    est_short_liq = np.zeros(n, dtype=float)
    open_long_liq = np.zeros(n, dtype=float)
    open_short_liq = np.zeros(n, dtype=float)
    nearest_short_dist = np.full(n, np.nan, dtype=float)
    nearest_short_usd = np.zeros(n, dtype=float)
    nearest_long_dist = np.full(n, np.nan, dtype=float)
    nearest_long_usd = np.zeros(n, dtype=float)
    # NaN = not computable at this bar (invalid bar, OI warm-up, or no OI observation for the share)
    short_5pct_usd = np.full(n, np.nan, dtype=float)
    long_5pct_usd = np.full(n, np.nan, dtype=float)
    short_5pct_share = np.full(n, np.nan, dtype=float)
    long_5pct_share = np.full(n, np.nan, dtype=float)

    hm_start = max(0, n - max(1, heatmap_bars))
    hm_snapshots: list[tuple[pd.Timestamp, int, np.ndarray, np.ndarray]] = []
    open_times = df["open_time"].tolist()

    seeded = False
    prev_oi = math.nan
    n_oi_obs = 0  # valid OI observations seen so far (bars <= t)
    min_oi_obs = cfg.min_oi_history_bars
    rel_thresh = cfg.major_cluster_rel_threshold

    # Convert hot arrays to Python lists / tuples for minimal loop dispatch overhead
    k_low_list = k_low_sweep.tolist()
    k_high_list = k_high_sweep.tolist()
    k_c_list = k_c_arr.tolist()
    valid_list = valid_bar.tolist()
    oi_list = oi_usd.tolist()
    oi_q_list = oi_qty.tolist()
    vwap_list = vwap_p.tolist()
    close_list = close_p.tolist()
    vwap_valid = (np.isfinite(vwap_p) & (vwap_p > 0.0)).tolist()
    ls_list = long_share.tolist()
    l_idxs_list = l_idxs_mat.tolist()
    s_idxs_list = s_idxs_mat.tolist()
    lev_w_list = lev_weights.tolist()

    for t in range(n):
        if not valid_list[t]:
            continue

        k_lo_sw = k_low_list[t]
        k_hi_sw = k_high_list[t]
        k_c = k_c_list[t]

        # Compute worst-case bounds needed for this bar so we grow at most once per bar
        min_need = min(k_c - win_half_bins, k_lo_sw)
        max_need = max(k_c + win_half_bins, k_hi_sw)
        cur_oi = oi_list[t]  # USD notional (cap + share denominators)
        cur_q = oi_q_list[t]  # contracts
        px = vwap_list[t]
        if math.isnan(cur_oi) and not math.isnan(cur_q):
            cur_oi = cur_q * close_list[t]
        delta_oi = 0.0  # USD notional opened (+) / closed (-) this bar, valued at the VWAP proxy
        prev_oi_at_px = math.nan
        if not math.isnan(cur_q) and cur_q >= 0.0:
            n_oi_obs += 1
            if not seeded:
                seeded = True
                if seed_initial_oi:
                    delta_oi = cur_q * px
            elif not math.isnan(prev_oi):
                delta_oi = (cur_q - prev_oi) * px
                prev_oi_at_px = prev_oi * px
            prev_oi = cur_q

        if delta_oi > 0.0 and vwap_valid[t]:
            l_row = l_idxs_list[t]
            s_row = s_idxs_list[t]
            if l_row[0] < min_need:
                min_need = l_row[0]
            if l_row[-1] < min_need:
                min_need = l_row[-1]
            if s_row[0] > max_need:
                max_need = s_row[0]
            if s_row[-1] > max_need:
                max_need = s_row[-1]

        if min_need < base_idx:
            pad_l = base_idx - min_need + 256
            long_book = np.concatenate([np.zeros(pad_l, dtype=float), long_book])
            short_book = np.concatenate([np.zeros(pad_l, dtype=float), short_book])
            base_idx -= pad_l
            book_len = len(long_book)
        if max_need >= base_idx + book_len:
            pad_r = max_need - (base_idx + book_len - 1) + 256
            long_book = np.concatenate([long_book, np.zeros(pad_r, dtype=float)])
            short_book = np.concatenate([short_book, np.zeros(pad_r, dtype=float)])
            book_len = len(long_book)

        # 1) Sweep liquidations crossed by bar [low_t, high_t]
        off_lo = k_lo_sw - base_idx
        swept_l = float(long_book[off_lo:].sum())
        if swept_l > 0.0:
            long_book[off_lo:] = 0.0
            est_long_liq[t] = swept_l
            total_book_sum -= swept_l

        off_hi = k_hi_sw - base_idx
        swept_s = float(short_book[: off_hi + 1].sum())
        if swept_s > 0.0:
            short_book[: off_hi + 1] = 0.0
            est_short_liq[t] = swept_s
            total_book_sum -= swept_s

        # 2) Apply delta-OI
        if delta_oi > 0.0 and vwap_valid[t]:
            ls_t = ls_list[t]
            ss_t = 1.0 - ls_t
            l_amt_base = delta_oi * ls_t
            s_amt_base = delta_oi * ss_t
            l_row = l_idxs_list[t]
            s_row = s_idxs_list[t]
            for j in range(n_lev):
                wj = lev_w_list[j]
                long_book[l_row[j] - base_idx] += l_amt_base * wj
                short_book[s_row[j] - base_idx] += s_amt_base * wj
            total_book_sum += delta_oi
        elif delta_oi < 0.0:
            # OI that closed voluntarily this bar (beyond the estimated liquidations) is removed
            # proportionally: the same FRACTION of the OI still open after the sweep is taken off
            # every tracked level. (Subtracting the absolute USD drop from the tracked book, which
            # is usually far smaller than total OI, wiped the whole book on ordinary OI dips.)
            swept_total = swept_l + swept_s
            rem_drop = (-delta_oi) - swept_total
            open_after_sweep = prev_oi_at_px - swept_total  # previous OI minus liquidations
            if rem_drop > 0.0 and total_book_sum > 0.0:
                closed_frac = rem_drop / open_after_sweep if open_after_sweep > 0.0 else 1.0
                scale = max(0.0, 1.0 - closed_frac)
                long_book *= scale
                short_book *= scale
                total_book_sum *= scale

        # Cap total tracked liquidation book to actual open interest
        if not math.isnan(cur_oi) and cur_oi > 0.0 and total_book_sum > cur_oi:
            cap_scale = cur_oi / total_book_sum
            long_book *= cap_scale
            short_book *= cap_scale
            total_book_sum = cur_oi

        # 3) Summarize open book around current close
        off_c = k_c - base_idx
        open_long_liq[t] = float(long_book[:off_c].sum())
        open_short_liq[t] = float(short_book[off_c + 1 :].sum())

        if n_oi_obs < min_oi_obs:
            # Warm-up: with too little OI history the book is just the seeded leverage ladder
            # around one price (e.g. Hyperliquid exposes a single current OI snapshot), so cluster
            # metrics stay NaN and the rule score re-weights over its other components.
            continue

        s_slice = short_book[off_c + 1 : off_c + 1 + win_half_bins]
        l_slice = long_book[off_c - win_half_bins : off_c]

        s_5 = float(s_slice[:win_5pct_bins].sum())
        l_5 = float(l_slice[-win_5pct_bins:].sum())
        short_5pct_usd[t] = s_5
        long_5pct_usd[t] = l_5
        if not math.isnan(cur_oi) and cur_oi > 0.0:
            short_5pct_share[t] = s_5 / cur_oi
            long_5pct_share[t] = l_5 / cur_oi

        s_max = float(s_slice.max())
        if s_max > 0.0:
            pick = int(np.argmax(s_slice >= s_max * rel_thresh))
            nearest_short_dist[t] = float(rel_short_pct[pick])
            nearest_short_usd[t] = float(s_slice[pick])

        l_max = float(l_slice.max())
        if l_max > 0.0:
            l_rev = l_slice[::-1]
            pick_rev = int(np.argmax(l_rev >= l_max * rel_thresh))
            nearest_long_dist[t] = float(rel_long_pct[pick_rev])
            nearest_long_usd[t] = float(l_rev[pick_rev])

        if t >= hm_start:
            hm_snapshots.append((open_times[t], k_c, l_slice.copy(), s_slice.copy()))

    summary = pd.DataFrame(
        {
            "open_time": df["open_time"].reset_index(drop=True),
            "est_long_liq_usd": est_long_liq,
            "est_short_liq_usd": est_short_liq,
            "est_total_liq_usd": est_long_liq + est_short_liq,
            "open_long_liq_usd": open_long_liq,
            "open_short_liq_usd": open_short_liq,
            "nearest_short_liq_dist_pct": nearest_short_dist,
            "nearest_short_liq_usd": nearest_short_usd,
            "nearest_long_liq_dist_pct": nearest_long_dist,
            "nearest_long_liq_usd": nearest_long_usd,
            "short_liq_within_5pct_usd": short_5pct_usd,
            "long_liq_within_5pct_usd": long_5pct_usd,
            "short_liq_within_5pct_share": short_5pct_share,
            "long_liq_within_5pct_share": long_5pct_share,
        }
    )

    last_close = float(close_p[-1])
    profile_df, _ = _extract_profile_and_clusters(long_book, short_book, base_idx, last_close, cfg)
    if n_oi_obs < min_oi_obs:
        profile_df = profile_df.iloc[0:0]
    heatmap_df = _build_heatmap_frame(hm_snapshots, step=step, win_half_bins=win_half_bins)

    return LiqMapResult(
        summary=summary,
        price_profile=profile_df,
        heatmap=heatmap_df,
        as_of_price=last_close,
    )


def _build_heatmap_frame(
    snapshots: list[tuple[pd.Timestamp, int, np.ndarray, np.ndarray]],
    *,
    step: float,
    win_half_bins: int,
    n_price_bins: int = 45,
) -> pd.DataFrame:
    """Convert trailing per-bar liquidation slices into a regular time x price heatmap grid."""
    if not snapshots:
        return pd.DataFrame(columns=["open_time", "price_bin", "long_liq_usd", "short_liq_usd", "total_liq_usd"])

    k_vals = [s[1] for s in snapshots]
    p_min = _bin_center(min(k_vals) - win_half_bins, step)
    p_max = _bin_center(max(k_vals) + win_half_bins, step)
    if p_max <= p_min:
        return pd.DataFrame(columns=["open_time", "price_bin", "long_liq_usd", "short_liq_usd", "total_liq_usd"])

    edges = np.linspace(p_min, p_max, n_price_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rows: list[dict[str, object]] = []
    for ts, k_c, l_slice, s_slice in snapshots:
        l_agg = np.zeros(n_price_bins, dtype=float)
        s_agg = np.zeros(n_price_bins, dtype=float)

        l_bins = np.arange(k_c - win_half_bins, k_c)
        l_prices = np.exp((l_bins + 0.5) * step)
        l_idx = np.clip(np.searchsorted(edges, l_prices, side="right") - 1, 0, n_price_bins - 1)
        np.add.at(l_agg, l_idx, l_slice)

        s_bins = np.arange(k_c + 1, k_c + 1 + win_half_bins)
        s_prices = np.exp((s_bins + 0.5) * step)
        s_idx = np.clip(np.searchsorted(edges, s_prices, side="right") - 1, 0, n_price_bins - 1)
        np.add.at(s_agg, s_idx, s_slice)

        for p_c, lv, sv in zip(centers, l_agg, s_agg):
            rows.append(
                {
                    "open_time": ts,
                    "price_bin": float(p_c),
                    "long_liq_usd": float(lv),
                    "short_liq_usd": float(sv),
                    "total_liq_usd": float(lv + sv),
                }
            )
    return pd.DataFrame(rows)

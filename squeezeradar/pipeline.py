"""High-level analysis pipeline uniting providers, features, liqmap, scores, and scanner table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from squeezeradar.config import (
    DEFAULT_SAMPLE_DIR,
    DEFAULT_SYMBOLS,
    DataModeLiteral,
    LiqMapConfig,
    ScoreConfig,
    StateConfig,
)
from squeezeradar.data import SymbolDataset
from squeezeradar.features import compute_features
from squeezeradar.liqmap import LiqMapResult, build_liquidation_map
from squeezeradar.providers.base import ProviderBadge
from squeezeradar.providers.chain import ProviderChain
from squeezeradar.scores import ScoreBreakdown, compute_scores_and_states, explain_score_row


@dataclass(frozen=True)
class SymbolAnalysis:
    """Complete analytical output for a single perpetual symbol."""

    symbol: str
    dataset: SymbolDataset
    scored_df: pd.DataFrame
    liq_map: LiqMapResult
    breakdown_short: ScoreBreakdown
    breakdown_long: ScoreBreakdown


@dataclass(frozen=True)
class RadarScanResult:
    """Scanner output across symbols plus badge metadata."""

    scanner_table: pd.DataFrame
    analyses: dict[str, SymbolAnalysis]
    badge: ProviderBadge
    as_of_label: str
    errors: dict[str, str]


def _fmt_usd_compact(usd: float) -> str:
    if usd >= 1e9:
        return f"${usd/1e9:,.2f}B"
    if usd >= 1e6:
        return f"${usd/1e6:,.1f}M"
    return f"${usd/1e3:,.0f}K"


def _format_cluster_label(row: pd.Series) -> tuple[str, float | None, float]:
    """Select the nearest relevant liquidation cluster label for the dominant side."""
    side = str(row.get("active_side", "short_squeeze"))
    s_dist = row.get("nearest_short_liq_dist_pct")
    s_usd = float(row.get("nearest_short_liq_usd") or 0.0)
    l_dist = row.get("nearest_long_liq_dist_pct")
    l_usd = float(row.get("nearest_long_liq_usd") or 0.0)

    if side == "short_squeeze":
        if pd.notna(s_dist) and s_usd > 0:
            return f"+{float(s_dist):.1f}% ({_fmt_usd_compact(s_usd)} Short)", float(s_dist), s_usd
        if pd.notna(l_dist) and l_usd > 0:
            return f"-{float(l_dist):.1f}% ({_fmt_usd_compact(l_usd)} Long)", -float(l_dist), l_usd
    else:
        if pd.notna(l_dist) and l_usd > 0:
            return f"-{float(l_dist):.1f}% ({_fmt_usd_compact(l_usd)} Long)", -float(l_dist), l_usd
        if pd.notna(s_dist) and s_usd > 0:
            return f"+{float(s_dist):.1f}% ({_fmt_usd_compact(s_usd)} Short)", float(s_dist), s_usd
    return "None", None, 0.0


def analyze_datasets(
    datasets: dict[str, SymbolDataset],
    badge: ProviderBadge,
    *,
    errors: dict[str, str] | None = None,
    liq_config: LiqMapConfig | None = None,
    score_config: ScoreConfig | None = None,
    state_config: StateConfig | None = None,
    heatmap_bars: int = 168,
    ml_probs: dict[str, dict[str, float]] | None = None,
) -> RadarScanResult:
    """Compute features, estimated liqmaps, scores, states, and scanner table."""
    analyses: dict[str, SymbolAnalysis] = {}
    rows: list[dict[str, Any]] = []
    latest_times: list[pd.Timestamp] = []

    for sym, ds in datasets.items():
        if ds.aligned.empty:
            continue
        feat_df = compute_features(ds.aligned)
        lm = build_liquidation_map(feat_df, config=liq_config, heatmap_bars=heatmap_bars)
        scored_df = compute_scores_and_states(
            feat_df,
            lm.summary,
            score_config=score_config,
            state_config=state_config,
        )
        last = scored_df.iloc[-1]
        bd_short = explain_score_row(last, symbol=sym, side="short_squeeze")
        bd_long = explain_score_row(last, symbol=sym, side="long_squeeze")
        analyses[sym] = SymbolAnalysis(
            symbol=sym,
            dataset=ds,
            scored_df=scored_df,
            liq_map=lm,
            breakdown_short=bd_short,
            breakdown_long=bd_long,
        )

        c_time = last.get("close_time") or last.get("open_time")
        if pd.notna(c_time):
            latest_times.append(pd.Timestamp(c_time))

        cluster_lbl, cluster_dist, cluster_usd = _format_cluster_label(last)
        probs = (ml_probs or {}).get(sym, {})
        rows.append(
            {
                "symbol": sym,
                "price": float(last["close"]),
                "change_24h_pct": (float(last["ret_24h"]) * 100.0) if pd.notna(last.get("ret_24h")) else np.nan,
                "funding_rate_pct": (
                    float(last["funding_rate_last"]) * 100.0
                    if pd.notna(last.get("funding_rate_last"))
                    else np.nan
                ),
                "oi_24h_pct": (float(last["oi_chg_24h"]) * 100.0) if pd.notna(last.get("oi_chg_24h")) else np.nan,
                "global_ls_ratio": (
                    float(last["global_ls_ratio"]) if pd.notna(last.get("global_ls_ratio")) else np.nan
                ),
                "top_ls_pos_ratio": (
                    float(last["top_ls_pos_ratio"]) if pd.notna(last.get("top_ls_pos_ratio")) else np.nan
                ),
                "short_squeeze_score": float(last["short_squeeze_score"]),
                "long_squeeze_score": float(last["long_squeeze_score"]),
                "max_squeeze_score": float(last["max_squeeze_score"]),
                "state": str(last["squeeze_state"]),
                "active_side": str(last["active_side"]),
                "nearest_cluster": cluster_lbl,
                "nearest_cluster_dist_pct": cluster_dist if cluster_dist is not None else np.nan,
                "nearest_cluster_usd": cluster_usd,
                "ml_prob_up": probs.get("prob_squeeze_up", np.nan),
                "ml_prob_down": probs.get("prob_squeeze_down", np.nan),
            }
        )

    scanner_df = pd.DataFrame(rows)
    if not scanner_df.empty:
        scanner_df = scanner_df.sort_values("max_squeeze_score", ascending=False).reset_index(drop=True)

    as_of_label = (
        max(latest_times).strftime("%Y-%m-%d %H:%M UTC")
        if latest_times
        else "N/A"
    )
    return RadarScanResult(
        scanner_table=scanner_df,
        analyses=analyses,
        badge=badge,
        as_of_label=as_of_label,
        errors=errors or {},
    )


def run_radar_scan(
    *,
    symbols: list[str] | tuple[str, ...] | None = None,
    mode: DataModeLiteral = "replay",
    as_of: str | None = None,
    limit: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    chain: ProviderChain | None = None,
    sample_dir: str | None = None,
    heatmap_bars: int = 168,
) -> RadarScanResult:
    """Fetch data via ProviderChain and return a full RadarScanResult."""
    prov_chain = chain or ProviderChain(sample_dir=sample_dir or DEFAULT_SAMPLE_DIR)
    target_syms = list(symbols) if symbols else list(DEFAULT_SYMBOLS)
    res = prov_chain.load_symbols(
        target_syms,
        mode=mode,
        as_of=as_of,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
    )
    return analyze_datasets(
        res.datasets,
        res.badge,
        errors=res.errors,
        heatmap_bars=heatmap_bars,
    )


def find_notable_liquidation_events(
    analyses: dict[str, SymbolAnalysis],
    *,
    top_n: int = 6,
    min_separation_days: int = 14,
) -> list[dict[str, Any]]:
    """Scan full history for the largest distinct 24h estimated liquidation events."""
    parts: list[pd.DataFrame] = []
    for sym, item in analyses.items():
        df = item.scored_df
        if df.empty:
            continue
        sub = pd.DataFrame(
            {
                "open_time": df["open_time"],
                "symbol": sym,
                "liq_24h": df["est_total_liq_usd"].rolling(24).sum(),
                "long_liq_24h": df["est_long_liq_usd"].rolling(24).sum(),
                "short_liq_24h": df["est_short_liq_usd"].rolling(24).sum(),
                "ret_24h": df["ret_24h"],
            }
        )
        parts.append(sub)

    if not parts:
        return []

    cat = pd.concat(parts, ignore_index=True).dropna(subset=["liq_24h"])
    mkt = (
        cat.groupby("open_time", as_index=False)
        .agg(
            total_liq_24h=("liq_24h", "sum"),
            long_liq_24h=("long_liq_24h", "sum"),
            short_liq_24h=("short_liq_24h", "sum"),
            avg_abs_ret_24h=("ret_24h", lambda x: float(np.abs(x).mean())),  # NaN-skipping mean
        )
        .sort_values("total_liq_24h", ascending=False)
    )

    events: list[dict[str, Any]] = []
    min_sec = min_separation_days * 86400
    for _, row in mkt.iterrows():
        ts = pd.Timestamp(row["open_time"])
        if any(abs((ts - e["peak_time"]).total_seconds()) < min_sec for e in events):
            continue
        dom = "Long Cascade" if row["long_liq_24h"] >= row["short_liq_24h"] else "Short Squeeze"
        # The 24h sum at bar `ts` covers bars opening ts-23h .. ts. Replaying "as of" ts-23h
        # (a close-time cutoff) shows the market just BEFORE the first bar of that window, so the
        # whole event lies inside the +48h reveal.
        preview_ts = ts - pd.Timedelta(hours=23)
        events.append(
            {
                "peak_time": ts,
                "preview_time": preview_ts,
                "label": f"{preview_ts.strftime('%Y-%m-%d %H:%M')} · Pre-{dom} (${row['total_liq_24h']/1e6:,.0f}M 24h est.)",
                "peak_label": f"{ts.strftime('%Y-%m-%d %H:%M')} · {dom} Peak (${row['total_liq_24h']/1e6:,.0f}M)",
                "total_liq_24h_usd": float(row["total_liq_24h"]),
                "long_liq_24h_usd": float(row["long_liq_24h"]),
                "short_liq_24h_usd": float(row["short_liq_24h"]),
                "event_type": dom,
            }
        )
        if len(events) >= top_n:
            break
    return events

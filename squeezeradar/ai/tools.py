"""Anthropic @beta_tool definitions closed over the current SqueezeRadar market context."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from anthropic import beta_tool

from squeezeradar.config import format_price
from squeezeradar.liqmap import LiqMapResult
from squeezeradar.model import EventStudyResult, WalkForwardTargetResult
from squeezeradar.scores import explain_score_row


@dataclass
class AnalystContext:
    """Market state and empirical evidence exposed to AI analyst tools."""

    scored_frames: dict[str, pd.DataFrame]
    liq_maps: dict[str, LiqMapResult]
    as_of_label: str
    data_source_label: str
    event_study: EventStudyResult | None = None
    walk_forward: dict[str, WalkForwardTargetResult] | None = None
    ml_probs: dict[str, dict[str, float]] = field(default_factory=dict)


def _clean_json_val(val: Any, digits: int = 4) -> Any:
    """Convert numpy/pandas scalars to JSON-safe primitives, replacing NaN/Inf with None."""
    if val is None:
        return None
    if isinstance(val, (bool, str)):
        return val
    if isinstance(val, (int,)):
        return int(val)
    if isinstance(val, float):
        if not math.isfinite(val):
            return None
        return round(val, digits)
    if hasattr(val, "item"):
        return _clean_json_val(val.item(), digits=digits)
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, dict):
        return {str(k): _clean_json_val(v, digits=digits) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_clean_json_val(v, digits=digits) for v in val]
    return str(val)


def _compact_json(payload: Any, max_chars: int = 2900) -> str:
    """Serialize compact JSON strictly under 3000 characters."""
    cleaned = _clean_json_val(payload)
    text = json.dumps(cleaned, separators=(",", ":"), allow_nan=False)
    if len(text) <= max_chars:
        return text
    # Iteratively trim list items if payload exceeded budget
    if isinstance(cleaned, dict):
        trimmed = dict(cleaned)
        for k, v in list(trimmed.items()):
            if isinstance(v, list) and len(v) > 3:
                trimmed[k] = v[: max(3, len(v) // 2)]
                candidate = json.dumps(trimmed, separators=(",", ":"), allow_nan=False)
                if len(candidate) <= max_chars:
                    return candidate
        # Final fallback truncation notice inside JSON
        return json.dumps({"truncated": True, "preview": text[: max_chars - 60]}, separators=(",", ":"))
    return text[:max_chars]


def _resolve_symbol(ctx: AnalystContext, symbol: str) -> str | None:
    sym = symbol.upper().strip()
    if sym in ctx.scored_frames:
        return sym
    for key in ctx.scored_frames:
        if key.startswith(sym):
            return key
    return None


def build_analyst_tools(ctx: AnalystContext) -> tuple[list[Any], dict[str, Any]]:
    """Build the 7 @beta_tool closures bound to `ctx` and a captured brief dict."""
    captured_brief: dict[str, Any] = {}

    @beta_tool
    def scan_market() -> str:
        """Return the ranked squeeze scanner table across all loaded symbols."""
        rows: list[dict[str, Any]] = []
        for sym, df in ctx.scored_frames.items():
            if df.empty:
                continue
            r = df.iloc[-1]
            probs = ctx.ml_probs.get(sym, {})
            rows.append(
                {
                    "sym": sym,
                    "px": r.get("close"),
                    "px_fmt": format_price(r.get("close")),
                    "ret_24h_pct": (r.get("ret_24h") * 100.0) if pd.notna(r.get("ret_24h")) else None,
                    "fund_bps": (r.get("funding_rate_last") * 10000.0) if pd.notna(r.get("funding_rate_last")) else None,
                    "oi_24h_pct": (r.get("oi_chg_24h") * 100.0) if pd.notna(r.get("oi_chg_24h")) else None,
                    "g_ls": r.get("global_ls_ratio"),
                    "short_sc": r.get("short_squeeze_score"),
                    "long_sc": r.get("long_squeeze_score"),
                    "state": r.get("squeeze_state"),
                    "side": r.get("active_side"),
                    "near_s_pct": r.get("nearest_short_liq_dist_pct"),
                    "near_l_pct": r.get("nearest_long_liq_dist_pct"),
                    "ml_up": probs.get("prob_squeeze_up"),
                    "ml_dn": probs.get("prob_squeeze_down"),
                }
            )
        rows.sort(
            key=lambda x: max(x.get("short_sc") or 0.0, x.get("long_sc") or 0.0),
            reverse=True,
        )
        return _compact_json(
            {
                "as_of": ctx.as_of_label,
                "source": ctx.data_source_label,
                "liq_note": "ESTIMATE",
                "symbols": rows,
            }
        )

    @beta_tool
    def get_symbol_snapshot(symbol: str) -> str:
        """Return latest derivatives, price action, flow, and ML probabilities for a symbol.

        Args:
            symbol: Perpetual symbol ticker, e.g. BTCUSDT.
        """
        sym = _resolve_symbol(ctx, symbol)
        if sym is None or ctx.scored_frames[sym].empty:
            return _compact_json({"error": f"Unknown or empty symbol: {symbol}"})
        r = ctx.scored_frames[sym].iloc[-1]
        probs = ctx.ml_probs.get(sym, {})
        payload = {
            "symbol": sym,
            "open_time": r.get("open_time"),
            "close": r.get("close"),
            "close_fmt": format_price(r.get("close")),
            "ret_1h_pct": (r.get("ret_1h") * 100.0) if pd.notna(r.get("ret_1h")) else None,
            "ret_4h_pct": (r.get("ret_4h") * 100.0) if pd.notna(r.get("ret_4h")) else None,
            "ret_24h_pct": (r.get("ret_24h") * 100.0) if pd.notna(r.get("ret_24h")) else None,
            "atr_14": r.get("atr_14"),
            "atr_14_fmt": format_price(r.get("atr_14")),
            "atr_pct": (r.get("atr_pct") * 100.0) if pd.notna(r.get("atr_pct")) else None,
            "bb_bw_pct_30d": r.get("bb_bw_pct_30d"),
            "dist_7d_high_pct": (r.get("dist_7d_high_pct") * 100.0) if pd.notna(r.get("dist_7d_high_pct")) else None,
            "dist_7d_low_pct": (r.get("dist_7d_low_pct") * 100.0) if pd.notna(r.get("dist_7d_low_pct")) else None,
            "funding_rate_pct_8h": (r.get("funding_rate_last") * 100.0) if pd.notna(r.get("funding_rate_last")) else None,
            "funding_z_30d": r.get("funding_z_30d"),
            "oi_usd_m": (r.get("oi_usd") / 1e6) if pd.notna(r.get("oi_usd")) else None,
            "oi_chg_1h_pct": (r.get("oi_chg_1h") * 100.0) if pd.notna(r.get("oi_chg_1h")) else None,
            "oi_chg_4h_pct": (r.get("oi_chg_4h") * 100.0) if pd.notna(r.get("oi_chg_4h")) else None,
            "oi_chg_24h_pct": (r.get("oi_chg_24h") * 100.0) if pd.notna(r.get("oi_chg_24h")) else None,
            "oi_to_vol_24h": r.get("oi_to_vol_24h"),
            "global_ls_ratio": r.get("global_ls_ratio"),
            "top_ls_pos_ratio": r.get("top_ls_pos_ratio"),
            "taker_buy_sell_ratio": r.get("taker_buy_sell_ratio_eff"),
            "cvd_24h_usd_m": (r.get("cvd_24h_usd") / 1e6) if pd.notna(r.get("cvd_24h_usd")) else None,
            "short_squeeze_score": r.get("short_squeeze_score"),
            "long_squeeze_score": r.get("long_squeeze_score"),
            "short_state": r.get("short_squeeze_state"),
            "long_state": r.get("long_squeeze_state"),
            "ml_prob_squeeze_up": probs.get("prob_squeeze_up"),
            "ml_prob_squeeze_down": probs.get("prob_squeeze_down"),
        }
        return _compact_json(payload)

    @beta_tool
    def get_liquidation_map(symbol: str) -> str:
        """Return the ESTIMATED liquidation cluster summary and top price buckets for a symbol.

        Args:
            symbol: Perpetual symbol ticker, e.g. BTCUSDT.
        """
        sym = _resolve_symbol(ctx, symbol)
        if sym is None or sym not in ctx.liq_maps:
            return _compact_json({"error": f"Liquidation map unavailable for {symbol}"})
        lm = ctx.liq_maps[sym]
        df = ctx.scored_frames[sym]
        r = df.iloc[-1] if not df.empty else pd.Series(dtype=float)

        prof = lm.price_profile
        top_short_bins: list[dict[str, Any]] = []
        top_long_bins: list[dict[str, Any]] = []
        if not prof.empty:
            s_top = prof[prof["short_liq_usd"] > 0].nlargest(4, "short_liq_usd")
            top_short_bins = [
                {
                    "pct": row["pct_from_spot"],
                    "px": row["price_level"],
                    "px_fmt": format_price(row["price_level"]),
                    "usd_m": row["short_liq_usd"] / 1e6,
                }
                for _, row in s_top.iterrows()
            ]
            l_top = prof[prof["long_liq_usd"] > 0].nlargest(4, "long_liq_usd")
            top_long_bins = [
                {
                    "pct": row["pct_from_spot"],
                    "px": row["price_level"],
                    "px_fmt": format_price(row["price_level"]),
                    "usd_m": row["long_liq_usd"] / 1e6,
                }
                for _, row in l_top.iterrows()
            ]

        s_24h = lm.summary["est_short_liq_usd"].tail(24).sum() if not lm.summary.empty else 0.0
        l_24h = lm.summary["est_long_liq_usd"].tail(24).sum() if not lm.summary.empty else 0.0

        payload = {
            "symbol": sym,
            "is_estimate": True,
            "spot_price": lm.as_of_price,
            "spot_price_fmt": format_price(lm.as_of_price),
            "open_short_liq_usd_m": (r.get("open_short_liq_usd") / 1e6) if pd.notna(r.get("open_short_liq_usd")) else None,
            "open_long_liq_usd_m": (r.get("open_long_liq_usd") / 1e6) if pd.notna(r.get("open_long_liq_usd")) else None,
            "nearest_short_cluster_dist_pct": r.get("nearest_short_liq_dist_pct"),
            "nearest_short_cluster_usd_m": (r.get("nearest_short_liq_usd") / 1e6) if pd.notna(r.get("nearest_short_liq_usd")) else None,
            "nearest_long_cluster_dist_pct": r.get("nearest_long_liq_dist_pct"),
            "nearest_long_cluster_usd_m": (r.get("nearest_long_liq_usd") / 1e6) if pd.notna(r.get("nearest_long_liq_usd")) else None,
            "short_within_5pct_usd_m": (r.get("short_liq_within_5pct_usd") / 1e6) if pd.notna(r.get("short_liq_within_5pct_usd")) else None,
            "long_within_5pct_usd_m": (r.get("long_liq_within_5pct_usd") / 1e6) if pd.notna(r.get("long_liq_within_5pct_usd")) else None,
            "est_24h_short_liq_usd_m": s_24h / 1e6,
            "est_24h_long_liq_usd_m": l_24h / 1e6,
            "top_short_clusters": top_short_bins,
            "top_long_clusters": top_long_bins,
        }
        return _compact_json(payload)

    @beta_tool
    def get_score_breakdown(symbol: str) -> str:
        """Return rule-score component breakdown and plain-language drivers for both sides.

        Args:
            symbol: Perpetual symbol ticker, e.g. BTCUSDT.
        """
        sym = _resolve_symbol(ctx, symbol)
        if sym is None or ctx.scored_frames[sym].empty:
            return _compact_json({"error": f"Unknown or empty symbol: {symbol}"})
        row = ctx.scored_frames[sym].iloc[-1]
        short_bd = explain_score_row(row, symbol=sym, side="short_squeeze")
        long_bd = explain_score_row(row, symbol=sym, side="long_squeeze")
        return _compact_json(
            {
                "symbol": sym,
                "active_side": row.get("active_side"),
                "overall_state": row.get("squeeze_state"),
                "short_squeeze": {
                    "score": short_bd.score,
                    "state": short_bd.state,
                    "components": short_bd.components,
                    "reasons": short_bd.reasons,
                },
                "long_squeeze": {
                    "score": long_bd.score,
                    "state": long_bd.state,
                    "components": long_bd.components,
                    "reasons": long_bd.reasons,
                },
            }
        )

    @beta_tool
    def get_event_study(signal: str) -> str:
        """Return historical 0-48h event study hit rates and forward returns for a signal.

        Args:
            signal: Signal filter substring, e.g. 'short', 'long', 'baseline', or 'all'.
        """
        if ctx.event_study is None or ctx.event_study.summary_df.empty:
            return _compact_json({"status": "Event study not yet computed for this view."})
        df_es = ctx.event_study.summary_df
        q = signal.strip().lower()
        if q and q != "all":
            matched = df_es[df_es["signal"].str.lower().str.contains(q, regex=False)]
            if matched.empty:
                matched = df_es
        else:
            matched = df_es
        return _compact_json({"signals": matched.to_dict(orient="records")})

    @beta_tool
    def get_model_evidence() -> str:
        """Return purged walk-forward ML out-of-sample metrics (AUC, top-5% precision, lift)."""
        if not ctx.walk_forward:
            return _compact_json({"status": "Walk-forward ML evidence not yet computed for this view."})
        out: dict[str, Any] = {}
        for target, res in ctx.walk_forward.items():
            out[target] = {
                "n_oos": res.n_oos,
                "n_folds": res.n_folds,
                "base_rate": res.base_rate,
                "auc": res.auc,
                "brier_score": res.brier_score,
                "precision_top_5pct": res.precision_top_5pct,
                "lift_top_5pct": res.lift_top_5pct,
            }
        return _compact_json(out)

    @beta_tool
    def submit_brief(
        market_summary: str,
        setups: list[str],
        risks: list[str],
        watch_levels: list[str],
    ) -> str:
        """Submit the final structured market intelligence brief.

        Args:
            market_summary: Concise executive synthesis grounded in tool metrics.
            setups: Specific squeeze setups with trigger levels and invalidation levels.
            risks: Key invalidation conditions, model limitations, and crowding risks.
            watch_levels: Actionable price and estimated liquidation cluster watch levels.
        """
        captured_brief["market_summary"] = market_summary
        captured_brief["setups"] = list(setups)
        captured_brief["risks"] = list(risks)
        captured_brief["watch_levels"] = list(watch_levels)
        return _compact_json({"status": "submitted", "setups_count": len(setups)})

    tools = [
        scan_market,
        get_symbol_snapshot,
        get_liquidation_map,
        get_score_breakdown,
        get_event_study,
        get_model_evidence,
        submit_brief,
    ]
    return tools, captured_brief

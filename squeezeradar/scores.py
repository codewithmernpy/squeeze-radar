"""Rule-based squeeze risk scores (0-100), component attribution, and causal state machine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from squeezeradar.config import (
    ACTIVE_SCORE_THRESHOLD,
    ELEVATED_SCORE_THRESHOLD,
    ScoreConfig,
    StateConfig,
)

STATE_PRIORITY: dict[str, int] = {
    "NONE": 0,
    "EXHAUSTION": 1,
    "SETUP": 2,
    "IN_PROGRESS": 3,
    "TRIGGER": 4,
}


def state_for_score(
    score: float | Any,
    *,
    elevated_min: float = ELEVATED_SCORE_THRESHOLD,
    active_min: float = ACTIVE_SCORE_THRESHOLD,
) -> Literal["Calm", "Elevated", "Active"]:
    """Classify a 0-100 squeeze risk score into 'Calm', 'Elevated', or 'Active'."""
    try:
        val = float(score)
    except (TypeError, ValueError):
        return "Calm"
    if not math.isfinite(val):
        return "Calm"
    if val >= active_min:
        return "Active"
    if val >= elevated_min:
        return "Elevated"
    return "Calm"


@dataclass(frozen=True)
class ScoreBreakdown:
    """Structured explanation of a symbol's squeeze risk scores and drivers."""

    symbol: str
    side: str
    score: float
    state: str
    components: dict[str, float | None]
    reasons: list[str]


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def _weighted_nan_mean(components: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Compute weighted mean across finite components per bar (0.0 if all NaN)."""
    n = len(components[0][0])
    num = np.zeros(n, dtype=float)
    den = np.zeros(n, dtype=float)
    for arr, weight in components:
        valid = np.isfinite(arr)
        num[valid] += arr[valid] * weight
        den[valid] += weight
    out = np.zeros(n, dtype=float)
    pos = den > 0.0
    out[pos] = num[pos] / den[pos]
    return np.clip(out, 0.0, 100.0)


def _blend_two_optional(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Average two arrays row-wise using whichever entries are finite."""
    out = np.full(len(a), np.nan, dtype=float)
    fa = np.isfinite(a)
    fb = np.isfinite(b)
    both = fa & fb
    only_a = fa & ~fb
    only_b = (~fa) & fb
    out[both] = 0.5 * (a[both] + b[both])
    out[only_a] = a[only_a]
    out[only_b] = b[only_b]
    return out


def _run_side_state_machine(
    *,
    side: str,
    score: np.ndarray,
    ret_1h: np.ndarray,
    ret_4h: np.ndarray,
    oi_chg_1h: np.ndarray,
    oi_chg_4h: np.ndarray,
    taker_ratio: np.ndarray,
    taker_z: np.ndarray,
    est_liq_usd: np.ndarray,
    cfg: StateConfig,
) -> list[str]:
    """Run causal bar-by-bar squeeze state machine for one side."""
    n = len(score)
    states: list[str] = ["NONE"] * n
    cur_state = "NONE"
    bars_in_progress = 0
    cooldown_bars = 0
    is_short_sq = side == "short_squeeze"

    score_l = score.tolist()
    r1_l = ret_1h.tolist()
    r4_l = ret_4h.tolist()
    o1_l = oi_chg_1h.tolist()
    o4_l = oi_chg_4h.tolist()
    tr_l = taker_ratio.tolist()
    tz_l = taker_z.tolist()
    liq_l = est_liq_usd.tolist()

    for t in range(n):
        sc = 0.0 if math.isnan(score_l[t]) else score_l[t]
        r1 = 0.0 if math.isnan(r1_l[t]) else r1_l[t]
        r4 = 0.0 if math.isnan(r4_l[t]) else r4_l[t]
        o1 = 0.0 if math.isnan(o1_l[t]) else o1_l[t]
        o4 = 0.0 if math.isnan(o4_l[t]) else o4_l[t]
        tr = 1.0 if math.isnan(tr_l[t]) else tr_l[t]
        tz = 0.0 if math.isnan(tz_l[t]) else tz_l[t]
        liq = 0.0 if math.isnan(liq_l[t]) else liq_l[t]

        setup_cond = sc >= cfg.setup_score_min

        if is_short_sq:
            price_against = (r1 >= cfg.trigger_ret_1h) or (r4 >= cfg.trigger_ret_4h)
            flow_spike = (tr >= cfg.trigger_taker_high) or (tz >= cfg.trigger_taker_z)
            cont_cond = (r1 > 0.0) or (r4 >= 0.5 * cfg.trigger_ret_4h) or (o1 < 0.0) or (liq > 0.0)
            stall_cond = r1 <= cfg.exhaustion_ret_stall
        else:
            price_against = (r1 <= -cfg.trigger_ret_1h) or (r4 <= -cfg.trigger_ret_4h)
            flow_spike = (tr <= cfg.trigger_taker_low) or (tz <= -cfg.trigger_taker_z)
            cont_cond = (r1 < 0.0) or (r4 <= -0.5 * cfg.trigger_ret_4h) or (o1 < 0.0) or (liq > 0.0)
            stall_cond = r1 >= -cfg.exhaustion_ret_stall

        oi_drop = (o1 <= cfg.trigger_oi_drop_1h) or (o4 <= cfg.trigger_oi_drop_4h)
        trigger_cond = price_against and oi_drop and flow_spike
        exhaustion_cond = ((o1 >= 0.0) and stall_cond and (not flow_spike)) or (
            bars_in_progress >= cfg.in_progress_max_bars
        )

        if cur_state == "NONE":
            if setup_cond:
                cur_state = "SETUP"
        elif cur_state == "SETUP":
            if trigger_cond and sc >= cfg.trigger_score_min:
                cur_state = "TRIGGER"
                bars_in_progress = 0
            elif sc < cfg.setup_exit_score:
                cur_state = "NONE"
        elif cur_state == "TRIGGER":
            if exhaustion_cond:
                cur_state = "EXHAUSTION"
                cooldown_bars = 1
            elif cont_cond:
                cur_state = "IN_PROGRESS"
                bars_in_progress = 1
            else:
                cur_state = "EXHAUSTION"
                cooldown_bars = 1
        elif cur_state == "IN_PROGRESS":
            if exhaustion_cond:
                cur_state = "EXHAUSTION"
                cooldown_bars = 1
                bars_in_progress = 0
            else:
                bars_in_progress += 1
        elif cur_state == "EXHAUSTION":
            if cooldown_bars >= cfg.exhaustion_cooldown_bars:
                cur_state = "SETUP" if setup_cond else "NONE"
                cooldown_bars = 0
            else:
                cooldown_bars += 1

        states[t] = cur_state

    return states


def compute_scores_and_states(
    features_df: pd.DataFrame,
    liq_summary_df: pd.DataFrame | None = None,
    *,
    score_config: ScoreConfig | None = None,
    state_config: StateConfig | None = None,
) -> pd.DataFrame:
    """Compute causal short/long squeeze risk scores, components, and state machines."""
    sc_cfg = score_config or ScoreConfig()
    st_cfg = state_config or StateConfig()
    df = features_df.copy()
    if df.empty:
        return df

    if liq_summary_df is not None and not liq_summary_df.empty:
        liq_cols = [
            c
            for c in liq_summary_df.columns
            if c != "open_time" and c not in df.columns
        ]
        if liq_cols:
            df = df.merge(liq_summary_df[["open_time", *liq_cols]], on="open_time", how="left")

    n = len(df)
    fz = df["funding_z_30d"].to_numpy(dtype=float) if "funding_z_30d" in df else np.full(n, np.nan)
    fr = df["funding_rate_last"].to_numpy(dtype=float) if "funding_rate_last" in df else np.full(n, np.nan)
    sb24 = df["shorts_building_24h"].to_numpy(dtype=float) if "shorts_building_24h" in df else np.full(n, np.nan)
    sb4 = df["shorts_building_4h"].to_numpy(dtype=float) if "shorts_building_4h" in df else np.full(n, np.nan)
    lb24 = df["longs_building_24h"].to_numpy(dtype=float) if "longs_building_24h" in df else np.full(n, np.nan)
    lb4 = df["longs_building_4h"].to_numpy(dtype=float) if "longs_building_4h" in df else np.full(n, np.nan)

    g_ls = df["global_ls_ratio"].to_numpy(dtype=float) if "global_ls_ratio" in df else np.full(n, np.nan)
    g_ls_z = df["global_ls_z_30d"].to_numpy(dtype=float) if "global_ls_z_30d" in df else np.full(n, np.nan)
    t_ls = df["top_ls_pos_ratio"].to_numpy(dtype=float) if "top_ls_pos_ratio" in df else np.full(n, np.nan)
    t_ls_z = df["top_ls_pos_z_30d"].to_numpy(dtype=float) if "top_ls_pos_z_30d" in df else np.full(n, np.nan)
    blend_ls = _blend_two_optional(g_ls, t_ls)
    blend_ls_z = _blend_two_optional(g_ls_z, t_ls_z)

    bb_pct = df["bb_bw_pct_30d"].to_numpy(dtype=float) if "bb_bw_pct_30d" in df else np.full(n, np.nan)

    s5_share = (
        df["short_liq_within_5pct_share"].to_numpy(dtype=float)
        if "short_liq_within_5pct_share" in df
        else np.full(n, np.nan)
    )
    s_dist = (
        df["nearest_short_liq_dist_pct"].to_numpy(dtype=float)
        if "nearest_short_liq_dist_pct" in df
        else np.full(n, np.nan)
    )
    l5_share = (
        df["long_liq_within_5pct_share"].to_numpy(dtype=float)
        if "long_liq_within_5pct_share" in df
        else np.full(n, np.nan)
    )
    l_dist = (
        df["nearest_long_liq_dist_pct"].to_numpy(dtype=float)
        if "nearest_long_liq_dist_pct" in df
        else np.full(n, np.nan)
    )

    # --- Short-squeeze components [0..100] ---
    short_comp_funding = np.where(
        np.isfinite(fz) | np.isfinite(fr),
        100.0
        * (
            0.70 * _clip01(np.where(np.isfinite(fz), -fz, 0.0) / sc_cfg.funding_z_cap)
            + 0.30 * _clip01((0.00005 - np.where(np.isfinite(fr), fr, 0.0001)) / 0.00025)
        ),
        np.nan,
    )
    short_comp_oi_div = np.where(
        np.isfinite(sb24) | np.isfinite(sb4),
        100.0
        * (
            0.65 * _clip01(np.where(np.isfinite(sb24), sb24, 0.0) / sc_cfg.short_oi_div_cap)
            + 0.35 * _clip01(np.where(np.isfinite(sb4), sb4, 0.0) / (0.5 * sc_cfg.short_oi_div_cap))
        ),
        np.nan,
    )
    short_comp_positioning = np.where(
        np.isfinite(blend_ls) | np.isfinite(blend_ls_z),
        100.0
        * (
            0.30 * _clip01((1.10 - np.where(np.isfinite(blend_ls), blend_ls, 1.10)) / 0.35)
            + 0.70 * _clip01(np.where(np.isfinite(blend_ls_z), -blend_ls_z, 0.0) / sc_cfg.positioning_z_cap)
        ),
        np.nan,
    )
    short_comp_liq = np.where(
        np.isfinite(s5_share),
        np.where(
            np.isfinite(s_dist) & (s5_share > 0.0),
            100.0
            * (
                0.55 * _clip01(s5_share / sc_cfg.liq_cluster_oi_share_cap)
                + 0.45 * _clip01(1.0 - s_dist / sc_cfg.liq_cluster_dist_scale_pct)
            ),
            0.0,
        ),
        np.nan,
    )
    comp_compression = np.where(np.isfinite(bb_pct), 100.0 * _clip01(1.0 - bb_pct), np.nan)

    # --- Long-squeeze (liquidation cascade) components [0..100] ---
    long_comp_funding = np.where(
        np.isfinite(fz) | np.isfinite(fr),
        100.0
        * (
            0.70 * _clip01(np.where(np.isfinite(fz), fz, 0.0) / sc_cfg.funding_z_cap)
            + 0.30 * _clip01((np.where(np.isfinite(fr), fr, 0.0001) - 0.00012) / 0.00045)
        ),
        np.nan,
    )
    long_comp_oi_div = np.where(
        np.isfinite(lb24) | np.isfinite(lb4),
        100.0
        * (
            0.65 * _clip01(np.where(np.isfinite(lb24), lb24, 0.0) / sc_cfg.long_oi_div_cap)
            + 0.35 * _clip01(np.where(np.isfinite(lb4), lb4, 0.0) / (0.5 * sc_cfg.long_oi_div_cap))
        ),
        np.nan,
    )
    long_comp_positioning = np.where(
        np.isfinite(blend_ls) | np.isfinite(blend_ls_z),
        100.0
        * (
            0.30 * _clip01((np.where(np.isfinite(blend_ls), blend_ls, 1.0) - 1.60) / 1.20)
            + 0.70 * _clip01(np.where(np.isfinite(blend_ls_z), blend_ls_z, 0.0) / sc_cfg.positioning_z_cap)
        ),
        np.nan,
    )
    long_comp_liq = np.where(
        np.isfinite(l5_share),
        np.where(
            np.isfinite(l_dist) & (l5_share > 0.0),
            100.0
            * (
                0.55 * _clip01(l5_share / sc_cfg.liq_cluster_oi_share_cap)
                + 0.45 * _clip01(1.0 - l_dist / sc_cfg.liq_cluster_dist_scale_pct)
            ),
            0.0,
        ),
        np.nan,
    )

    short_score = _weighted_nan_mean(
        [
            (short_comp_funding, sc_cfg.weight_funding),
            (short_comp_oi_div, sc_cfg.weight_oi_divergence),
            (short_comp_positioning, sc_cfg.weight_positioning),
            (short_comp_liq, sc_cfg.weight_liq_cluster),
            (comp_compression, sc_cfg.weight_compression),
        ]
    )
    long_score = _weighted_nan_mean(
        [
            (long_comp_funding, sc_cfg.weight_funding),
            (long_comp_oi_div, sc_cfg.weight_oi_divergence),
            (long_comp_positioning, sc_cfg.weight_positioning),
            (long_comp_liq, sc_cfg.weight_liq_cluster),
            (comp_compression, sc_cfg.weight_compression),
        ]
    )

    df["short_comp_funding"] = short_comp_funding
    df["short_comp_oi_div"] = short_comp_oi_div
    df["short_comp_positioning"] = short_comp_positioning
    df["short_comp_liq_cluster"] = short_comp_liq
    df["short_comp_compression"] = comp_compression
    df["short_squeeze_score"] = short_score

    df["long_comp_funding"] = long_comp_funding
    df["long_comp_oi_div"] = long_comp_oi_div
    df["long_comp_positioning"] = long_comp_positioning
    df["long_comp_liq_cluster"] = long_comp_liq
    df["long_comp_compression"] = comp_compression
    df["long_squeeze_score"] = long_score

    ret_1h = df["ret_1h"].to_numpy(dtype=float) if "ret_1h" in df else np.zeros(n)
    ret_4h = df["ret_4h"].to_numpy(dtype=float) if "ret_4h" in df else np.zeros(n)
    oi_1h = df["oi_chg_1h"].to_numpy(dtype=float) if "oi_chg_1h" in df else np.zeros(n)
    oi_4h = df["oi_chg_4h"].to_numpy(dtype=float) if "oi_chg_4h" in df else np.zeros(n)
    taker_r = (
        df["taker_buy_sell_ratio_eff"].to_numpy(dtype=float)
        if "taker_buy_sell_ratio_eff" in df
        else np.ones(n)
    )
    taker_z = df["taker_ratio_z_30d"].to_numpy(dtype=float) if "taker_ratio_z_30d" in df else np.zeros(n)
    est_s_liq = df["est_short_liq_usd"].to_numpy(dtype=float) if "est_short_liq_usd" in df else np.zeros(n)
    est_l_liq = df["est_long_liq_usd"].to_numpy(dtype=float) if "est_long_liq_usd" in df else np.zeros(n)

    short_states = _run_side_state_machine(
        side="short_squeeze",
        score=short_score,
        ret_1h=ret_1h,
        ret_4h=ret_4h,
        oi_chg_1h=oi_1h,
        oi_chg_4h=oi_4h,
        taker_ratio=taker_r,
        taker_z=taker_z,
        est_liq_usd=est_s_liq,
        cfg=st_cfg,
    )
    long_states = _run_side_state_machine(
        side="long_squeeze",
        score=long_score,
        ret_1h=ret_1h,
        ret_4h=ret_4h,
        oi_chg_1h=oi_1h,
        oi_chg_4h=oi_4h,
        taker_ratio=taker_r,
        taker_z=taker_z,
        est_liq_usd=est_l_liq,
        cfg=st_cfg,
    )

    df["short_squeeze_state"] = short_states
    df["long_squeeze_state"] = long_states

    active_side: list[str] = []
    squeeze_state: list[str] = []
    for ss, ls, sc_s, sc_l in zip(short_states, long_states, short_score.tolist(), long_score.tolist()):
        p_s = STATE_PRIORITY[ss]
        p_l = STATE_PRIORITY[ls]
        if p_s > p_l or (p_s == p_l and sc_s >= sc_l):
            active_side.append("short_squeeze")
            squeeze_state.append(ss)
        else:
            active_side.append("long_squeeze")
            squeeze_state.append(ls)

    df["active_side"] = active_side
    df["squeeze_state"] = squeeze_state
    df["max_squeeze_score"] = np.maximum(short_score, long_score)
    return df


def explain_score_row(
    row: pd.Series | dict[str, Any],
    symbol: str = "",
    side: str | None = None,
) -> ScoreBreakdown:
    """Produce structured score breakdown and plain-language drivers for UI and AI."""
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    short_sc = float(data.get("short_squeeze_score", 0.0) or 0.0)
    long_sc = float(data.get("long_squeeze_score", 0.0) or 0.0)
    chosen_side = side or str(
        data.get("active_side")
        or ("short_squeeze" if short_sc >= long_sc else "long_squeeze")
    )

    def _val(key: str) -> float | None:
        v = data.get(key)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        try:
            fv = float(v)
            return fv if np.isfinite(fv) else None
        except (TypeError, ValueError):
            return None

    if chosen_side == "short_squeeze":
        comps = {
            "funding": _val("short_comp_funding"),
            "oi_divergence": _val("short_comp_oi_div"),
            "positioning": _val("short_comp_positioning"),
            "liq_cluster": _val("short_comp_liq_cluster"),
            "compression": _val("short_comp_compression"),
        }
        score_val = short_sc
        state_val = str(data.get("short_squeeze_state", "NONE"))
    else:
        comps = {
            "funding": _val("long_comp_funding"),
            "oi_divergence": _val("long_comp_oi_div"),
            "positioning": _val("long_comp_positioning"),
            "liq_cluster": _val("long_comp_liq_cluster"),
            "compression": _val("long_comp_compression"),
        }
        score_val = long_sc
        state_val = str(data.get("long_squeeze_state", "NONE"))

    reasons: list[str] = []
    fz = _val("funding_z_30d")
    fr = _val("funding_rate_last")
    oi24 = _val("oi_chg_24h")
    r24 = _val("ret_24h")
    gls = _val("global_ls_ratio")
    tls = _val("top_ls_pos_ratio")
    bbp = _val("bb_bw_pct_30d")

    if chosen_side == "short_squeeze":
        if comps["funding"] is not None and comps["funding"] >= 40 and fz is not None and fr is not None:
            reasons.append(
                f"Negative funding pressure (rate {fr*100:+.4f}%/8h, 30d z={fz:+.2f}): shorts pay longs."
            )
        if comps["oi_divergence"] is not None and comps["oi_divergence"] >= 35 and oi24 is not None and r24 is not None:
            reasons.append(
                f"Short buildup divergence: OI climbed {oi24*100:+.1f}% over 24h while price returned {r24*100:+.1f}%."
            )
        if comps["positioning"] is not None and comps["positioning"] >= 40:
            reasons.append(
                f"Short-skewed positioning (global L/S={gls:.2f}, top pos L/S={tls:.2f})."
                if gls is not None and tls is not None
                else "Positioning z-score skewed toward shorts."
            )
        s_dist = _val("nearest_short_liq_dist_pct")
        s_usd = _val("nearest_short_liq_usd")
        if comps["liq_cluster"] is not None and comps["liq_cluster"] >= 30 and s_dist is not None:
            reasons.append(
                f"Estimated short-liq cluster {s_dist:+.2f}% above spot (~${(s_usd or 0)/1e6:,.1f}M in nearest major bin)."
            )
    else:
        if comps["funding"] is not None and comps["funding"] >= 40 and fz is not None and fr is not None:
            reasons.append(
                f"Elevated positive funding (rate {fr*100:+.4f}%/8h, 30d z={fz:+.2f}): longs crowded and paying shorts."
            )
        if comps["oi_divergence"] is not None and comps["oi_divergence"] >= 35 and oi24 is not None and r24 is not None:
            reasons.append(
                f"Aggressive long buildup: OI surged {oi24*100:+.1f}% over 24h alongside {r24*100:+.1f}% price gain."
            )
        if comps["positioning"] is not None and comps["positioning"] >= 40:
            reasons.append(
                f"Long-heavy positioning (global L/S={gls:.2f}, top pos L/S={tls:.2f})."
                if gls is not None and tls is not None
                else "Positioning z-score skewed toward longs."
            )
        l_dist = _val("nearest_long_liq_dist_pct")
        l_usd = _val("nearest_long_liq_usd")
        if comps["liq_cluster"] is not None and comps["liq_cluster"] >= 30 and l_dist is not None:
            reasons.append(
                f"Estimated long-liq cluster -{l_dist:.2f}% below spot (~${(l_usd or 0)/1e6:,.1f}M in nearest major bin)."
            )

    if comps["compression"] is not None and comps["compression"] >= 60 and bbp is not None:
        reasons.append(
            f"Volatility compression: 20h Bollinger bandwidth sits at the {bbp*100:.0f}th 30d percentile."
        )

    if not reasons:
        reasons.append(
            f"Balanced derivatives regime (score {score_val:.1f}/100, state {state_val}); no extreme crowding."
        )

    return ScoreBreakdown(
        symbol=symbol,
        side=chosen_side,
        score=round(score_val, 2),
        state=state_val,
        components={k: (round(v, 2) if v is not None else None) for k, v in comps.items()},
        reasons=reasons,
    )

"""Page 2: Single-Contract Deep Dive (Subplots, Estimated Liq Map, Score History & Why)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from squeezeradar.config import format_price
from squeezeradar.ui import (
    get_replay_scan_params,
    init_page,
    load_radar_cached,
    make_deep_dive_subplots,
    make_liq_heatmap_figure,
    make_liq_profile_figure,
    make_score_history_figure,
    render_footer,
    render_sidebar,
    show_sidebar_badge,
)

init_page("Deep Dive")
mode, symbols, cache_bust = render_sidebar(allow_live=True)

as_of_arg, limit_bars, r_start, r_end, is_custom_range = get_replay_scan_params(mode)
with st.spinner("Loading contract telemetry..."):
    scan_res = load_radar_cached(symbols, mode, as_of=as_of_arg, limit=limit_bars, cache_bust=cache_bust)

show_sidebar_badge(scan_res.badge, scan_res.as_of_label)

if scan_res.errors:
    for sym, err in scan_res.errors.items():
        st.warning(f"{sym}: {err}")

avail = list(scan_res.analyses.keys())
if not avail:
    st.error("No contract data available.")
    render_footer()
    st.stop()

st.title("🔬 Contract Deep Dive")

col_sym, col_win = st.columns([2, 2])
with col_sym:
    default_sym = st.session_state.get("sr_focus_symbol", avail[0])
    if default_sym not in avail:
        default_sym = avail[0]
    if "deep_dive_symbol" not in st.session_state or st.session_state["deep_dive_symbol"] not in avail:
        st.session_state["deep_dive_symbol"] = default_sym
    selected_sym = st.selectbox(
        "Contract",
        options=avail,
        key="deep_dive_symbol",
    )
    st.session_state["sr_focus_symbol"] = selected_sym

with col_win:
    win_options = ["72h (3d)", "168h (7d)", "336h (14d)", "720h (30d)", "Selected Range"]
    window_label = st.selectbox(
        "Chart Lookback Window",
        options=win_options,
        index=4 if is_custom_range else 1,
        key="deep_dive_window",
    )
    window_bars = None if window_label == "Selected Range" else int(window_label.split("h")[0])

analysis = scan_res.analyses[selected_sym]
df = analysis.scored_df
if mode == "replay" and is_custom_range and r_start is not None:
    start_ts = pd.Timestamp(r_start, tz="UTC")
    df_range = df[df["open_time"] >= start_ts]
    if df_range.empty:
        st.warning(
            f"{selected_sym} has no bars in the selected range ({r_start} → {r_end}); "
            "showing available history up to end date."
        )
        df_range = df
else:
    df_range = df
last = df_range.iloc[-1]

# Top telemetry row
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Spot Price", format_price(last["close"]))
m2.metric("24h Return", f"{last['ret_24h']*100:+.2f}%" if pd.notna(last.get("ret_24h")) else "N/A")
m3.metric(
    "Funding (8h)",
    f"{last['funding_rate_last']*100:+.4f}%" if pd.notna(last.get("funding_rate_last")) else "N/A",
)
m4.metric(
    "Open Interest",
    f"${last['oi_usd']/1e6:,.1f}M" if pd.notna(last.get("oi_usd")) else "N/A",
    delta=f"{last['oi_chg_24h']*100:+.2f}% (24h)" if pd.notna(last.get("oi_chg_24h")) else None,
    delta_color="off",
)
m5.metric(
    "Short Squeeze",
    f"{last['short_squeeze_score']:.1f} / 100",
    delta=str(last["short_squeeze_state"]),
    delta_color="off",
    delta_arrow="off",
)
m6.metric(
    "Long Squeeze",
    f"{last['long_squeeze_score']:.1f} / 100",
    delta=str(last["long_squeeze_state"]),
    delta_color="off",
    delta_arrow="off",
)

# Plain-language Why & component gauges
st.subheader("Why This Score? — Component Breakdown & Drivers")
c_why_s, c_why_l = st.columns(2)

for col, title, bd, bar_color in (
    (c_why_s, "Short-Squeeze Drivers (Upside Forced Buying)", analysis.breakdown_short, "#10B981"),
    (c_why_l, "Long-Squeeze / Cascade Drivers (Downside Forced Selling)", analysis.breakdown_long, "#EF4444"),
):
    with col:
        st.markdown(f"**{title}** — Score `{bd.score:.1f}/100` | State `{bd.state}`")
        for r in bd.reasons:
            st.markdown(f"- {r}")
        comp_names = list(bd.components.keys())
        comp_vals = [bd.components[k] if bd.components[k] is not None else 0.0 for k in comp_names]
        fig_c = go.Figure(
            go.Bar(
                x=comp_vals,
                y=[k.replace("_", " ").title() for k in comp_names],
                orientation="h",
                marker_color=bar_color,
                text=[f"{v:.0f}" for v in comp_vals],
                textposition="auto",
            )
        )
        fig_c.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1020",
            plot_bgcolor="#131A2E",
            xaxis_range=[0, 100],
            height=210,
            margin={"l": 10, "r": 15, "t": 10, "b": 20},
        )
        st.plotly_chart(fig_c, width="stretch")

# 4-panel market structure chart
st.subheader("Market Structure & Derivatives Flow")
df_win = df_range if window_bars is None else df_range.tail(window_bars)
if is_custom_range and r_start is not None and r_end is not None:
    st.caption(f"Date range filter: {r_start} → {r_end} UTC ({len(df_win):,} bars shown).")
st.plotly_chart(make_deep_dive_subplots(df_win, selected_sym), width="stretch")

# Estimated Liquidation Map section
st.subheader("Estimated Liquidation Map (Model Approximation)")
st.info(analysis.liq_map.methodology_note)
if analysis.liq_map.price_profile.empty:
    st.warning(
        "Not enough hourly open-interest history for an estimated liquidation map "
        "(needs at least 24 OI observations; the Hyperliquid fallback only exposes the current OI). "
        "The liquidation-cluster score component is skipped and the score re-weights the others."
    )

c_hm, c_prof = st.columns([3, 2])
with c_hm:
    st.plotly_chart(
        make_liq_heatmap_figure(analysis.liq_map.heatmap, df_win[["open_time", "close"]]),
        width="stretch",
        theme=None,
    )
with c_prof:
    st.plotly_chart(
        make_liq_profile_figure(analysis.liq_map.price_profile, analysis.liq_map.as_of_price),
        width="stretch",
    )

# Score history with state shading
st.subheader("Squeeze Score History & State Transitions")
st.plotly_chart(make_score_history_figure(df_win), width="stretch")

render_footer()

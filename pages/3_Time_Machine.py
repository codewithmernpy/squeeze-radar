"""Page 3: Time Machine — Causal Historical Squeeze Replay & Post-Event Reveal."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from squeezeradar.config import DEFAULT_SAMPLE_DIR
from squeezeradar.data import load_sample_symbol
from squeezeradar.liqmap import build_liquidation_map
from squeezeradar.ui import (
    CURATED_REPLAY_PRESETS,
    format_scanner_display_df,
    get_sidebar_replay_range,
    init_page,
    load_radar_cached,
    load_top_historical_presets_cached,
    make_liq_profile_figure,
    render_footer,
    render_sidebar,
    show_sidebar_badge,
    style_scanner_table,
)

init_page("Time Machine")
_, symbols, cache_bust = render_sidebar(allow_live=False)

r_start, r_end, avail_min, avail_max = get_sidebar_replay_range()
tm_min_date = r_start or avail_min or dt.date(2023, 2, 1)
tm_max_date = r_end or avail_max or dt.date(2025, 10, 31)
if tm_min_date > tm_max_date:
    tm_min_date, tm_max_date = tm_max_date, tm_min_date

st.title("⏪ Time Machine — Causal Historical Replay")
st.caption(
    f"Pick any historical timestamp within the active replay range ({tm_min_date} → {tm_max_date} UTC). "
    "Everything shown above the fold uses strictly data known at or before that timestamp — zero future leakage."
)

if "tm_dynamic_presets" not in st.session_state:
    st.session_state["tm_dynamic_presets"] = []

presets = list(CURATED_REPLAY_PRESETS) + list(st.session_state["tm_dynamic_presets"])
preset_labels = [p["label"] for p in presets]

col_pre, col_scan = st.columns([3, 1])
with col_pre:
    chosen_preset_label = st.selectbox(
        "Notable Historical Liquidation / Squeeze Episodes",
        options=preset_labels,
        index=0,
        key="tm_preset_select",
    )
with col_scan:
    st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
    if st.button("🔍 Scan Full Archive Peaks", width="stretch", key="tm_scan_btn"):
        with st.spinner("Scanning replay history for largest 24h liquidation events..."):
            top_events = load_top_historical_presets_cached(
                symbols,
                start_date=tm_min_date.strftime("%Y-%m-%d"),
                end_date=tm_max_date.strftime("%Y-%m-%d"),
                cache_bust=cache_bust,
            )
        st.session_state["tm_dynamic_presets"] = [
            {
                "label": f"[Scanned] {ev['label']}",
                "as_of": ev["preview_time"].strftime("%Y-%m-%d %H:%M"),
            }
            for ev in top_events
        ]
        st.rerun()

chosen_preset = next(p for p in presets if p["label"] == chosen_preset_label)
preset_dt = pd.Timestamp(chosen_preset["as_of"], tz="UTC")

# Sync custom date/hour inputs when preset changes, and clamp inside selected coin/range bounds
if st.session_state.get("tm_last_preset") != chosen_preset_label:
    st.session_state["tm_last_preset"] = chosen_preset_label
    if preset_dt.date() < tm_min_date:
        st.session_state["tm_date"] = min(tm_max_date, tm_min_date + dt.timedelta(days=1))
    elif preset_dt.date() > tm_max_date:
        st.session_state["tm_date"] = tm_max_date
    else:
        st.session_state["tm_date"] = preset_dt.date()
    st.session_state["tm_hour"] = int(preset_dt.hour)

if "tm_date" in st.session_state:
    if st.session_state["tm_date"] < tm_min_date:
        st.session_state["tm_date"] = min(tm_max_date, tm_min_date + dt.timedelta(days=1))
    elif st.session_state["tm_date"] > tm_max_date:
        st.session_state["tm_date"] = tm_max_date
if "tm_focus_sym" in st.session_state and st.session_state["tm_focus_sym"] not in symbols:
    st.session_state["tm_focus_sym"] = symbols[0]

col_d, col_h, col_sym = st.columns([1.5, 1.5, 1.5])
with col_d:
    sel_date = st.date_input(
        "Replay Date (UTC)",
        min_value=tm_min_date,
        max_value=tm_max_date,
        key="tm_date",
    )
with col_h:
    sel_hour = st.slider(
        "Replay Hour (UTC)",
        min_value=0,
        max_value=23,
        key="tm_hour",
    )
with col_sym:
    focus_sym = st.selectbox(
        "Focus Contract",
        options=list(symbols),
        index=0,
        key="tm_focus_sym",
    )

as_of_str = f"{sel_date.strftime('%Y-%m-%d')} {sel_hour:02d}:00"
as_of_ts = pd.Timestamp(as_of_str, tz="UTC")

with st.spinner(f"Replaying market causally up to {as_of_str} UTC..."):
    replay_res = load_radar_cached(
        symbols,
        "replay",
        as_of=as_of_str,
        limit=2160,
        cache_bust=cache_bust,
    )

show_sidebar_badge(replay_res.badge, replay_res.as_of_label)

st.subheader(f"Radar Snapshot Known At {as_of_str} UTC (Strictly Causal)")
if replay_res.scanner_table.empty:
    st.warning("No bars available at this timestamp.")
    render_footer()
    st.stop()

disp = format_scanner_display_df(replay_res.scanner_table)
st.dataframe(style_scanner_table(disp), width="stretch", hide_index=True)

if focus_sym in replay_res.analyses:
    an = replay_res.analyses[focus_sym]
    c_left, c_right = st.columns([1.2, 1.3])
    with c_left:
        st.markdown(f"#### {focus_sym} Pre-Move State & Drivers")
        st.markdown(
            f"- **Short Squeeze**: `{an.breakdown_short.score:.1f}/100` (`{an.breakdown_short.state}`)\n"
            f"- **Long Squeeze**: `{an.breakdown_long.score:.1f}/100` (`{an.breakdown_long.state}`)"
        )
        dom_bd = (
            an.breakdown_short
            if an.breakdown_short.score >= an.breakdown_long.score
            else an.breakdown_long
        )
        for reason in dom_bd.reasons:
            st.markdown(f"- {reason}")
    with c_right:
        st.plotly_chart(
            make_liq_profile_figure(an.liq_map.price_profile, an.liq_map.as_of_price),
            width="stretch",
        )
else:
    st.info(f"{focus_sym} has no closed bars at or before {as_of_str} UTC.")

st.divider()
reveal = st.checkbox(
    "🔓 Reveal What Happened Next (+48h Post-Snapshot Outcome)",
    value=False,
    key="tm_reveal_toggle",
)

if reveal:
    st.subheader(f"Post-Snapshot Outcome: {as_of_str} UTC → +48h")
    end_reveal_str = (as_of_ts + pd.Timedelta(hours=48)).strftime("%Y-%m-%d %H:%M")
    # Same window start as the snapshot (2160 bars before as_of) + the 48 revealed bars, so the
    # estimated liquidation book is identical to the snapshot's up to as_of and simply continues.
    full_ds = load_sample_symbol(
        focus_sym,
        sample_dir=DEFAULT_SAMPLE_DIR,
        as_of=end_reveal_str,
        tail_bars=2160 + 48,
    )
    if full_ds.aligned.empty:
        pre_win = pd.DataFrame()
        post_win = pd.DataFrame()
    else:
        lm_full = build_liquidation_map(full_ds.aligned, heatmap_bars=1)
        df_all = full_ds.aligned.merge(
            lm_full.summary[["open_time", "est_long_liq_usd", "est_short_liq_usd", "est_total_liq_usd"]],
            on="open_time",
            how="left",
        )

        pre_win = df_all[
            (df_all["close_time"] <= as_of_ts)
            & (df_all["close_time"] > as_of_ts - pd.Timedelta(hours=48))
        ]
        post_win = df_all[
            (df_all["close_time"] > as_of_ts)
            & (df_all["close_time"] <= as_of_ts + pd.Timedelta(hours=48))
        ]

    if post_win.empty or pre_win.empty:
        st.info("Selected timestamp is too close to the end of the bundled sample for a full 48h reveal.")
    else:
        if len(post_win) < 48:
            st.caption(f"Only {len(post_win)}h of bundled data exist after this snapshot; metrics cover that span.")
        entry_px = float(pre_win.iloc[-1]["close"])
        max_high = float(post_win["high"].max())
        min_low = float(post_win["low"].min())
        end_px = float(post_win.iloc[-1]["close"])
        ret_48h = (end_px / entry_px - 1.0) * 100.0
        max_up = (max_high / entry_px - 1.0) * 100.0
        max_dn = (min_low / entry_px - 1.0) * 100.0
        est_l_48 = float(post_win["est_long_liq_usd"].sum())
        est_s_48 = float(post_win["est_short_liq_usd"].sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("48h Close Return", f"{ret_48h:+.2f}%")
        k2.metric("Max Upside / Drawdown", f"{max_up:+.2f}% / {max_dn:+.2f}%")
        k3.metric("Est. Long Liqs (+48h)", f"${est_l_48/1e6:,.1f}M")
        k4.metric("Est. Short Liqs (+48h)", f"${est_s_48/1e6:,.1f}M")

        fig_rev = go.Figure()
        fig_rev.add_trace(
            go.Candlestick(
                x=pre_win["open_time"],
                open=pre_win["open"],
                high=pre_win["high"],
                low=pre_win["low"],
                close=pre_win["close"],
                name="Known (Pre)",
                increasing_line_color="#64748B",
                decreasing_line_color="#475569",
            )
        )
        fig_rev.add_trace(
            go.Candlestick(
                x=post_win["open_time"],
                open=post_win["open"],
                high=post_win["high"],
                low=post_win["low"],
                close=post_win["close"],
                name="Outcome (+48h)",
                increasing_line_color="#10B981",
                decreasing_line_color="#EF4444",
            )
        )
        fig_rev.add_vline(
            x=as_of_ts.isoformat(),
            line_dash="dash",
            line_color="#6366F1",
        )
        fig_rev.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1020",
            plot_bgcolor="#131A2E",
            title=f"{focus_sym}: 48h Before vs 48h After {as_of_str} UTC",
            xaxis_rangeslider_visible=False,
            height=430,
            margin={"l": 50, "r": 20, "t": 45, "b": 30},
        )
        st.plotly_chart(fig_rev, width="stretch")

render_footer()

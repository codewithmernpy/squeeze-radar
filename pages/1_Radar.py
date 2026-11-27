"""Page 1: SqueezeRadar Live / Replay Scanner Table."""

from __future__ import annotations

import streamlit as st

from squeezeradar.scores import state_for_score
from squeezeradar.ui import (
    format_scanner_display_df,
    get_replay_scan_params,
    init_page,
    load_radar_cached,
    render_footer,
    render_scanner_hero,
    render_sidebar,
    show_sidebar_badge,
    style_scanner_table,
)

init_page("Radar")
mode, symbols, cache_bust = render_sidebar(allow_live=True)

render_scanner_hero()

as_of_arg, limit_bars, _, _, _ = get_replay_scan_params(mode)
with st.spinner("Scanning perpetual futures markets..."):
    scan_res = load_radar_cached(symbols, mode, as_of=as_of_arg, limit=limit_bars, cache_bust=cache_bust)

show_sidebar_badge(scan_res.badge, scan_res.as_of_label)

if scan_res.errors:
    for sym, err in scan_res.errors.items():
        st.warning(f"{sym}: {err}")

df = scan_res.scanner_table
if df.empty:
    st.error("No contract data loaded. Check symbols or switch to Replay mode.")
    render_footer()
    st.stop()

# Top summary metrics
elevated_count = int(df["max_squeeze_score"].map(lambda s: state_for_score(s) != "Calm").sum())
active_states = int((df["state"] != "NONE").sum())
top_short_row = df.loc[df["short_squeeze_score"].idxmax()]
top_long_row = df.loc[df["long_squeeze_score"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Contracts Scanned", f"{len(df)}")
c2.metric("Elevated / Active State", f"{elevated_count} / {active_states}")
c3.metric(
    "Peak Short-Squeeze Risk",
    f"{top_short_row['symbol']} · {top_short_row['short_squeeze_score']:.1f}",
    delta=f"{top_short_row['state']}",
    delta_color="off",
    delta_arrow="off",
)
c4.metric(
    "Peak Long-Squeeze Risk",
    f"{top_long_row['symbol']} · {top_long_row['long_squeeze_score']:.1f}",
    delta=f"{top_long_row['state']}",
    delta_color="off",
    delta_arrow="off",
)

st.subheader("Perpetual Squeeze Scanner (Sorted by Max Risk Score)")
st.caption(
    "Liquidation cluster columns are analytical ESTIMATES inferred from delta-OI and modeled leverage tiers. "
    "Click any row or choose a contract below to inspect in Deep Dive."
)

disp_df = format_scanner_display_df(df)
styled = style_scanner_table(disp_df)

selection = st.dataframe(
    styled,
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="radar_table",
)

if selection and selection.selection and selection.selection.rows:
    picked_idx = selection.selection.rows[0]
    if 0 <= picked_idx < len(df):
        st.session_state["sr_focus_symbol"] = str(df.iloc[picked_idx]["symbol"])
        try:
            st.switch_page("pages/2_Deep_Dive.py")
        except Exception:
            pass

col_sel, col_btn, _ = st.columns([2, 1, 3])
with col_sel:
    default_sym = st.session_state.get("sr_focus_symbol", str(df.iloc[0]["symbol"]))
    if default_sym not in df["symbol"].tolist():
        default_sym = str(df.iloc[0]["symbol"])
    chosen = st.selectbox(
        "Quick Jump to Deep Dive",
        options=df["symbol"].tolist(),
        index=df["symbol"].tolist().index(default_sym),
        key="radar_focus_select",
    )
    st.session_state["sr_focus_symbol"] = chosen
with col_btn:
    st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
    if st.button("Open Deep Dive →", type="primary", width="stretch", key="radar_open_deep_dive"):
        try:
            st.switch_page("pages/2_Deep_Dive.py")
        except Exception:
            st.info(f"Selected {chosen} — open Deep Dive from the sidebar.")

render_footer()

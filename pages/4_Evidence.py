"""Page 4: Walk-Forward ML Evidence, Calibration, Event Studies & Honest Discussion."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from squeezeradar.ui import (
    compute_evidence_cached,
    get_replay_scan_params,
    get_sidebar_replay_range,
    init_page,
    load_precomputed_evidence,
    load_radar_cached,
    render_footer,
    render_sidebar,
    show_sidebar_badge,
)

init_page("Evidence")
mode, symbols, cache_bust = render_sidebar(allow_live=True, show_date_range=True)

# Resolve badge from current mode quickly
as_of_arg, limit_bars, _, _, _ = get_replay_scan_params(mode)
scan_res = load_radar_cached(symbols, mode, as_of=as_of_arg, limit=limit_bars, cache_bust=cache_bust)
show_sidebar_badge(scan_res.badge, scan_res.as_of_label)

st.title("📊 Empirical Evidence — Walk-Forward ML & Event Studies")
st.caption(
    "All ML results use expanding monthly retraining across pooled symbols with symbol-agnostic features "
    "and a strict 24-bar purge before every test month. Evidence uses local replay/cached Binance history. "
    "Numbers are shown exactly as observed."
)

r_start, r_end, avail_min, avail_max = get_sidebar_replay_range()
start_str = r_start.strftime("%Y-%m-%d") if r_start else None
end_str = r_end.strftime("%Y-%m-%d") if r_end else None

precomputed = load_precomputed_evidence()
col_src, col_btn = st.columns([3, 1])
with col_btn:
    if st.button(
        "🔁 Recompute now",
        width="stretch",
        key="ev_recompute_btn",
        help="Re-run the full purged walk-forward + event study on the selected contracts and date range (~1–2 min).",
    ):
        st.session_state["ev_live_symbols"] = list(symbols)
        st.session_state["ev_live_range"] = (start_str, end_str)

live_syms = st.session_state.get("ev_live_symbols")
if live_syms:
    live_start, live_end = st.session_state.get("ev_live_range", (start_str, end_str))
    with st.spinner(
        f"Recomputing purged walk-forward ML & 0-48h event study for {', '.join(live_syms)} "
        f"({live_start or '...'} → {live_end or '...'})..."
    ):
        bundle = compute_evidence_cached(
            tuple(live_syms),
            start_date=live_start,
            end_date=live_end,
            cache_bust=cache_bust,
        )
    shown_syms = ", ".join(bundle.meta["symbols"]) if bundle.meta.get("symbols") else ", ".join(live_syms)
    range_txt = f"{live_start} → {live_end}" if (live_start or live_end) else "full available range"
    data_span = (
        f"{bundle.meta['data_start_utc']} → {bundle.meta['data_end_utc']} UTC"
        if bundle.meta.get("data_start_utc")
        else "no bars in range"
    )
    source_note = (
        f"**Source:** recomputed live in this session for {shown_syms} "
        f"(selected range {range_txt}; data {data_span}, {bundle.meta['runtime_sec']}s)."
    )
elif precomputed is not None:
    bundle = precomputed
    source_note = (
        f"**Source:** precomputed file `data/evidence/evidence_results.json` — a real run of "
        f"`python -m squeezeradar evidence --build` generated {bundle.meta['generated_at_utc']} over "
        f"{', '.join(bundle.meta['symbols'])} ({bundle.meta['data_start_utc']} → {bundle.meta['data_end_utc']} UTC, "
        f"{bundle.meta['runtime_sec']}s)."
    )
else:
    with st.spinner("No precomputed evidence file found — running purged walk-forward ML & event study (~1–2 min)..."):
        bundle = compute_evidence_cached(
            tuple(symbols),
            start_date=start_str,
            end_date=end_str,
            cache_bust=cache_bust,
        )
    source_note = (
        f"**Source:** no precomputed file found, computed live for {', '.join(bundle.meta['symbols'])} "
        f"({bundle.meta['data_start_utc']} → {bundle.meta['data_end_utc']} UTC). "
        "Run `python -m squeezeradar evidence --build` to make this page load instantly."
    )
with col_src:
    st.markdown(source_note)
    pre_start = str(precomputed.meta.get("data_start_utc") or "")[:10] if precomputed else None
    pre_end = str(precomputed.meta.get("data_end_utc") or "")[:10] if precomputed else None
    if not live_syms and precomputed is not None and (
        set(symbols) != set(precomputed.meta.get("symbols", []))
        or (start_str and pre_start and start_str != pre_start)
        or (end_str and pre_end and end_str != pre_end)
    ):
        st.caption(
            "Your sidebar contracts or date range differ from the precomputed dataset — "
            "click **Recompute now** to run it on your selection."
        )
    elif live_syms and (
        list(live_syms) != list(symbols)
        or st.session_state.get("ev_live_range") != (start_str, end_str)
    ):
        st.caption(
            "Sidebar contracts or date range changed since the last recompute — "
            "click **Recompute now** to update."
        )

wf_results = bundle.walk_forward
event_study = bundle.event_study
latest_probs = bundle.latest_probs

# 1. Walk-Forward Summary Cards
st.subheader("1. Purged Expanding Walk-Forward ML (`HistGradientBoostingClassifier`)")
up_res = wf_results.get("squeeze_up")
dn_res = wf_results.get("squeeze_down")

c_up, c_dn = st.columns(2)
for col, label, res in (
    (c_up, "Target: `squeeze_up` (+2 ATR before -1 ATR & OI -3% in 24h)", up_res),
    (c_dn, "Target: `squeeze_down` (-2 ATR before +1 ATR & OI -3% in 24h)", dn_res),
):
    with col:
        st.markdown(f"**{label}**")
        if res is None or res.n_oos == 0:
            st.info(
                "Insufficient OOS folds for the selected symbols and date range "
                "(purged monthly walk-forward requires > 6 months of history)."
            )
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("OOS ROC AUC", f"{res.auc:.3f}")
            m2.metric("Base Rate", f"{res.base_rate*100:.2f}%")
            m3.metric("Precision @ Top 5%", f"{res.precision_top_5pct*100:.2f}%")
            m4.metric("Lift vs Base Rate", f"{res.lift_top_5pct:.2f}x")
            st.caption(
                f"OOS Samples: {res.n_oos:,} across {res.n_folds} monthly purged folds | "
                f"Brier Score: {res.brier_score:.4f}"
            )

# Calibration + Monthly Stability Charts
c_cal, c_mon = st.columns(2)
with c_cal:
    fig_cal = go.Figure()
    cal_max = max(
        [0.05]
        + [
            float(r.calibration_df[c].max())
            for r in (up_res, dn_res)
            if r is not None and not r.calibration_df.empty
            for c in ("pred_mean", "obs_rate")
        ]
    )
    fig_cal.add_trace(
        go.Scatter(
            x=[0, cal_max],
            y=[0, cal_max],
            mode="lines",
            name="Perfect Calibration",
            line={"color": "#64748B", "dash": "dash"},
        )
    )
    if up_res and not up_res.calibration_df.empty:
        fig_cal.add_trace(
            go.Scatter(
                x=up_res.calibration_df["pred_mean"],
                y=up_res.calibration_df["obs_rate"],
                mode="lines+markers",
                name="squeeze_up",
                line={"color": "#10B981", "width": 2},
            )
        )
    if dn_res and not dn_res.calibration_df.empty:
        fig_cal.add_trace(
            go.Scatter(
                x=dn_res.calibration_df["pred_mean"],
                y=dn_res.calibration_df["obs_rate"],
                mode="lines+markers",
                name="squeeze_down",
                line={"color": "#EF4444", "width": 2},
            )
        )
    fig_cal.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1020",
        plot_bgcolor="#131A2E",
        title="Out-of-Sample Calibration Curve (Quantile Deciles)",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Observed Squeeze Rate",
        height=370,
        margin={"l": 50, "r": 20, "t": 45, "b": 35},
    )
    st.plotly_chart(fig_cal, width="stretch")

with c_mon:
    fig_mon = go.Figure()
    if up_res and not up_res.monthly_df.empty:
        fig_mon.add_trace(
            go.Scatter(
                x=up_res.monthly_df["month"],
                y=up_res.monthly_df["auc"],
                mode="lines+markers",
                name="squeeze_up AUC",
                line={"color": "#10B981", "width": 2},
            )
        )
    if dn_res and not dn_res.monthly_df.empty:
        fig_mon.add_trace(
            go.Scatter(
                x=dn_res.monthly_df["month"],
                y=dn_res.monthly_df["auc"],
                mode="lines+markers",
                name="squeeze_down AUC",
                line={"color": "#EF4444", "width": 2},
            )
        )
    fig_mon.add_hline(y=0.5, line_dash="dot", line_color="#64748B", annotation_text="Random (0.50)")
    fig_mon.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1020",
        plot_bgcolor="#131A2E",
        title="Monthly Walk-Forward OOS AUC Stability",
        yaxis_title="OOS ROC AUC",
        height=370,
        margin={"l": 50, "r": 20, "t": 45, "b": 35},
    )
    st.plotly_chart(fig_mon, width="stretch")

# 2. Event Study (0-48h)
st.subheader("2. Event Study — 0–48h Forward Return Paths vs Unconditional Baseline")
if not event_study.summary_df.empty:
    disp_es = event_study.summary_df.copy()
    if "n_events" in disp_es.columns:
        disp_es["n_events"] = pd.to_numeric(disp_es["n_events"], errors="coerce").fillna(0).astype(int)
    st.dataframe(
        disp_es.style.format(
            {
                "n_events": "{:d}",
                "mean_ret_24h_pct": "{:+.2f}%",
                "median_ret_24h_pct": "{:+.2f}%",
                "mean_ret_48h_pct": "{:+.2f}%",
                "pos_hit_rate_24h": "{:.1%}",
                "neg_hit_rate_24h": "{:.1%}",
                "squeeze_up_label_rate": "{:.1%}",
                "squeeze_down_label_rate": "{:.1%}",
            },
            na_rep="N/A",
        ),
        column_config={
            "n_events": st.column_config.NumberColumn("n_events", format="%d"),
        },
        width="stretch",
        hide_index=True,
    )

if not event_study.paths_df.empty:
    fig_es = go.Figure()
    for sig_name, grp in event_study.paths_df.groupby("signal"):
        dash = "dash" if "Baseline" in str(sig_name) else "solid"
        fig_es.add_trace(
            go.Scatter(
                x=grp["hour"],
                y=grp["mean_ret_pct"],
                mode="lines",
                name=f"{sig_name} (n={int(grp['n_events'].iloc[0])})",
                line={"dash": dash, "width": 2.2},
            )
        )
    fig_es.add_hline(y=0.0, line_dash="dot", line_color="#64748B")
    fig_es.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1020",
        plot_bgcolor="#131A2E",
        title="Average Forward Return Trajectory (0h to 48h, Declustered >= 12h Apart)",
        xaxis_title="Hours After Signal",
        yaxis_title="Mean Forward Return (%)",
        height=400,
        margin={"l": 50, "r": 20, "t": 45, "b": 35},
    )
    st.plotly_chart(fig_es, width="stretch")

# 3. Current Purged Latest Probabilities
if latest_probs:
    st.subheader("3. Latest Snapshot Purged ML Probabilities")
    st.caption(
        f"Model trained on bars with open_time < (last bar − 24h) and applied to the last bundled bar "
        f"(known at {bundle.meta.get('latest_probs_as_of_utc', 'N/A')})."
    )
    prob_rows = [
        {
            "Symbol": sym,
            "P(Squeeze Up, 24h)": f"{vals.get('prob_squeeze_up', float('nan'))*100:.2f}%",
            "P(Squeeze Down, 24h)": f"{vals.get('prob_squeeze_down', float('nan'))*100:.2f}%",
        }
        for sym, vals in latest_probs.items()
    ]
    st.dataframe(pd.DataFrame(prob_rows), width="stretch", hide_index=True)

# 4. Honest Discussion
st.subheader("4. Honest Empirical Assessment & Limitations")
stability_lines = []
for res in (up_res, dn_res):
    if res is not None and not res.monthly_df.empty and res.monthly_df["auc"].notna().any():
        aucs = res.monthly_df["auc"].dropna()
        stability_lines.append(
            f"`{res.target}`: monthly OOS AUC ranged {aucs.min():.3f}–{aucs.max():.3f} "
            f"(median {aucs.median():.3f}; {int((aucs < 0.5).sum())} of {len(aucs)} months below 0.50)"
        )
stability_text = "; ".join(stability_lines) if stability_lines else "not available for this run"
st.markdown(
    f"""
- **Rare Events, Ranking not Forecasting**: Squeeze labels (`+2 ATR before -1 ATR` *with* a `>= 3%` OI drop within 24h, and the mirror) are rare (base rates above). AUC and top-5% lift measure how well the model *ranks* bars; the raw probabilities stay small, so read the calibration curve before treating them as odds.
- **Monthly Stability**: {stability_text}.
- **Event Study Scope**: Rule scores are fixed formulas (not fitted), and the event study runs them over the whole bundled history with signals declustered >= 12h apart; it measures association, not a tradeable edge.
- **No Profitability Claims**: SqueezeRadar does **not** simulate trading fees, funding carry, slippage during liquidation cascades, or execution latency, and makes **no** claim of positive trading P&L.
- **Estimated Liquidation Book**: Exchanges do not publish account-level liquidation prices, so the cluster features are model estimates rather than ground-truth orderbook snapshots.
    """
)

render_footer()

"""Page 5: AI Analyst — Claude Tool Runner + Deterministic Offline Brief."""

from __future__ import annotations

import streamlit as st

from squeezeradar.ai import AnalystContext, build_offline_brief, generate_analyst_brief
from squeezeradar.ui import (
    get_replay_scan_params,
    get_session_api_key,
    init_page,
    load_precomputed_evidence,
    load_radar_cached,
    render_footer,
    render_sidebar,
    show_sidebar_badge,
)

init_page("AI Analyst")
mode, symbols, cache_bust = render_sidebar(allow_live=True)

as_of_arg, limit_bars, _, _, _ = get_replay_scan_params(mode)
scan_res = load_radar_cached(symbols, mode, as_of=as_of_arg, limit=limit_bars, cache_bust=cache_bust)
show_sidebar_badge(scan_res.badge, scan_res.as_of_label)

st.title("🤖 AI Squeeze Analyst")
st.caption(
    "Uses Anthropic's `tool_runner` with 7 data-grounded tools (`scan_market`, `get_symbol_snapshot`, "
    "`get_liquidation_map`, `get_score_breakdown`, `get_event_study`, `get_model_evidence`, `submit_brief`) "
    "when an API key is provided, or runs a deterministic offline template brief with zero API key."
)

avail = list(scan_res.analyses.keys())
if not avail:
    st.error("No symbol data loaded.")
    render_footer()
    st.stop()

c_focus, c_btn = st.columns([2, 1])
with c_focus:
    focus_choice = st.selectbox(
        "Analysis Focus",
        options=["Market-Wide (All)"] + avail,
        index=0,
        key="ai_focus_symbol",
    )
    focus_sym = None if focus_choice.startswith("Market-Wide") else focus_choice
with c_btn:
    st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
    run_clicked = st.button("✨ Generate Market Brief", type="primary", width="stretch", key="ai_generate_btn")

api_key = get_session_api_key()
evidence = load_precomputed_evidence()
ml_probs: dict[str, dict[str, float]] = {}
if evidence is not None and evidence.meta.get("latest_probs_as_of_utc") == scan_res.as_of_label:
    # The evidence run's purged probabilities refer to the last bundled bar; use them only when
    # this scan is a snapshot of exactly that bar.
    ml_probs = evidence.latest_probs
ctx = AnalystContext(
    scored_frames={s: a.scored_df for s, a in scan_res.analyses.items()},
    liq_maps={s: a.liq_map for s, a in scan_res.analyses.items()},
    as_of_label=scan_res.as_of_label,
    data_source_label=scan_res.badge.label,
    event_study=evidence.event_study if evidence is not None else None,
    walk_forward=evidence.walk_forward if evidence is not None else None,
    ml_probs=ml_probs,
)
if evidence is None:
    st.caption("No precomputed evidence file: `get_event_study` / `get_model_evidence` report it as unavailable.")

brief_key = (focus_sym, scan_res.as_of_label, scan_res.badge.label, tuple(avail))
if run_clicked:
    with st.spinner("Generating evidence-grounded squeeze brief..."):
        brief = generate_analyst_brief(
            ctx,
            api_key=api_key or None,
            focus_symbol=focus_sym,
        )
    st.session_state["sr_last_brief"] = brief
    st.session_state["sr_last_brief_key"] = brief_key
elif "sr_last_brief" not in st.session_state or st.session_state.get("sr_last_brief_key") != brief_key:
    # Offline template brief for the current focus/snapshot so the page is immediately informative
    st.session_state["sr_last_brief"] = build_offline_brief(
        ctx,
        focus_symbol=focus_sym,
        fallback_reason=(
            "Showing offline template brief. Click 'Generate Market Brief' "
            + ("to run Claude with your key." if api_key else "or add an Anthropic API key in the sidebar for Claude.")
        ),
    )
    st.session_state["sr_last_brief_key"] = brief_key

brief = st.session_state["sr_last_brief"]

mode_label = f"Claude ({brief.model})" if brief.mode == "claude" else "Offline Deterministic Brief"
st.markdown(f"**Analyst Mode:** `{mode_label}` · **Generated:** `{brief.generated_at}`")

st.markdown(brief.narrative_markdown)

if brief.setups:
    with st.expander("Structured Scenarios & Watch Levels", expanded=True):
        st.markdown("**Setups**")
        for s in brief.setups:
            st.markdown(f"- {s}")
        if brief.watch_levels:
            st.markdown("**Watch Levels**")
            for w in brief.watch_levels:
                st.markdown(f"- {w}")
        if brief.risks:
            st.markdown("**Risks**")
            for r in brief.risks:
                st.markdown(f"- {r}")

with st.expander(f"Tool-Call Audit Log ({len(brief.tool_calls)} calls)", expanded=False):
    if brief.tool_calls:
        st.json(brief.tool_calls)
    else:
        st.caption("No external LLM tool calls executed in offline template mode.")

render_footer()

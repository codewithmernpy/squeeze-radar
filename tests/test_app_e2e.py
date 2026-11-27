"""End-to-end Streamlit AppTest suite covering every page and main widget interactions."""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path

from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def _app(rel_path: str, timeout: float = 60.0) -> AppTest:
    return AppTest.from_file(str(ROOT / rel_path), default_timeout=timeout)


def test_app_home_and_radar_page() -> None:
    """Verify streamlit_app.py and pages/1_Radar.py render and interact cleanly in replay mode."""
    at_home = _app("streamlit_app.py", timeout=60.0)
    at_home.run()
    assert not at_home.exception
    assert len(at_home.dataframe) >= 1
    assert len(at_home.metric) >= 4
    assert any("REPLAY" in m.value for m in at_home.sidebar.markdown)

    at = _app("pages/1_Radar.py", timeout=60.0)
    at.run()
    assert not at.exception
    at.selectbox(key="radar_focus_select").select("ETHUSDT").run()
    assert not at.exception
    assert at.session_state["sr_focus_symbol"] == "ETHUSDT"
    at.button(key="radar_open_deep_dive").click().run()
    assert not at.exception


def test_deep_dive_page_interactions() -> None:
    """Verify pages/2_Deep_Dive.py renders telemetry, liqmap charts, and symbol switching."""
    at = _app("pages/2_Deep_Dive.py", timeout=60.0)
    at.run()
    assert not at.exception

    at.selectbox(key="deep_dive_symbol").select("SOLUSDT").run()
    at.selectbox(key="deep_dive_window").select("336h (14d)").run()
    assert not at.exception
    assert len(at.metric) >= 6
    assert any("Why This Score?" in sh.value for sh in at.subheader)
    assert any("ESTIMATE ONLY" in inf.value for inf in at.info)


def test_time_machine_page_causal_replay_and_reveal() -> None:
    """Verify pages/3_Time_Machine.py changes replay date/hour and reveals +48h outcome."""
    at = _app("pages/3_Time_Machine.py", timeout=60.0)
    at.run()
    assert not at.exception

    at.date_input(key="tm_date").set_value(dt.date(2024, 8, 5)).run()
    at.slider(key="tm_hour").set_value(0).run()
    at.selectbox(key="tm_focus_sym").select("BTCUSDT").run()
    assert not at.exception
    assert any("2024-08-05 00:00 UTC" in sh.value for sh in at.subheader)

    at.checkbox(key="tm_reveal_toggle").set_value(True).run()
    assert not at.exception
    assert any("Post-Snapshot Outcome" in sh.value for sh in at.subheader)


def test_evidence_page_renders_walk_forward_and_event_study() -> None:
    """Verify pages/4_Evidence.py renders walk-forward ML, calibration, and event study."""
    at = _app("pages/4_Evidence.py", timeout=120.0)
    at.session_state["sr_symbols"] = ["BTCUSDT"]
    at.run()
    assert not at.exception
    assert any("Empirical Evidence" in t.value for t in at.title)
    assert any("Purged Expanding Walk-Forward" in sh.value for sh in at.subheader)
    assert any("Event Study" in sh.value for sh in at.subheader)
    assert len(at.metric) >= 8


def test_ai_analyst_page_offline_brief_generation() -> None:
    """Verify pages/5_AI_Analyst.py generates an offline brief cleanly on button click."""
    at = _app("pages/5_AI_Analyst.py", timeout=60.0)
    at.run()
    assert not at.exception

    at.selectbox(key="ai_focus_symbol").select("BTCUSDT").run()
    at.button(key="ai_generate_btn").click().run()
    assert not at.exception
    assert any("not financial advice" in m.value.lower() for m in at.markdown)
    assert any("BTCUSDT" in m.value for m in at.markdown)


def test_about_page_renders_methodology_and_limitations() -> None:
    """Verify pages/6_About.py renders cleanly with methodology and limitations."""
    at = _app("pages/6_About.py", timeout=60.0)
    at.run()
    assert not at.exception
    assert any("Methodology" in t.value for t in at.title)
    assert any("Limitations" in m.value for m in at.markdown)


def test_navigation_config_metric_deltas_and_single_disclaimer() -> None:
    """Verify viewer toolbarMode, st.navigation switch_page, informational metric delta_color='off', and single AI disclaimer."""
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["client"]["toolbarMode"] == "viewer"

    # streamlit_app.py navigation + Open Deep Dive switch_page
    at_nav = _app("streamlit_app.py", timeout=60.0)
    at_nav.run()
    assert not at_nav.exception
    radar_metrics = {m.label: m for m in at_nav.metric}
    for lbl in ("Peak Short-Squeeze Risk", "Peak Long-Squeeze Risk"):
        assert radar_metrics[lbl].proto.color == MetricProto.MetricColor.GRAY
        assert radar_metrics[lbl].proto.direction == MetricProto.MetricDirection.NONE

    at_nav.button(key="radar_open_deep_dive").click().run()
    assert not at_nav.exception
    assert any("Contract Deep Dive" in t.value for t in at_nav.title)

    dd_metrics = {m.label: m for m in at_nav.metric}
    assert dd_metrics["Open Interest"].proto.color == MetricProto.MetricColor.GRAY
    for lbl in ("Short Squeeze", "Long Squeeze"):
        assert dd_metrics[lbl].proto.color == MetricProto.MetricColor.GRAY
        assert dd_metrics[lbl].proto.direction == MetricProto.MetricDirection.NONE

    # AI Analyst offline brief shows "not financial advice" once (no duplicate sentence) and no scientific notation
    at_ai = _app("pages/5_AI_Analyst.py", timeout=60.0)
    at_ai.run()
    assert not at_ai.exception
    brief = at_ai.session_state["sr_last_brief"]
    assert brief.narrative_markdown.lower().count("not financial advice") == 1
    assert "e+0" not in brief.narrative_markdown.lower()

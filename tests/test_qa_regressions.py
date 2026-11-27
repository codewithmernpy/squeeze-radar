"""Regression tests for SqueezeRadar QA fixes (AppTest multipage state, thresholds, formats, CSS)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from squeezeradar.config import ELEVATED_SCORE_THRESHOLD, StateConfig
from squeezeradar.evidence import DEFAULT_EVIDENCE_PATH, load_evidence
from squeezeradar.pipeline import RadarScanResult
from squeezeradar.providers.base import ProviderBadge
from squeezeradar.scores import state_for_score
from squeezeradar.ui import (
    _CUSTOM_CSS,
    format_scanner_display_df,
    score_badge_style,
    style_scanner_table,
)

ROOT = Path(__file__).resolve().parent.parent


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}
        self.text = "" if payload is None else str(payload)

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _MockSession:
    def __init__(self, handler: Any) -> None:
        self._handler = handler

    def request(self, method: str, url: str, params=None, json=None, headers=None, timeout=None):
        return self._handler(method, url, params=params, json_body=json)

    def get(self, url: str, timeout=None):
        return self._handler("GET", url, params=None, json_body=None)

    def close(self) -> None:
        return None


def _mock_live_binance_session() -> _MockSession:
    now_ms = 1_722_816_000_000
    base_ms = now_ms - 36 * 3_600_000
    listed = (
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "BNBUSDT",
        "1000PEPEUSDT",
        "SUIUSDT",
    )

    def _handler(method: str, url: str, params=None, json_body=None):
        if url.endswith("/fapi/v1/exchangeInfo"):
            return _FakeResponse(
                200,
                {
                    "symbols": [
                        {
                            "symbol": s,
                            "quoteAsset": "USDT",
                            "contractType": "PERPETUAL",
                            "status": "TRADING",
                        }
                        for s in listed
                    ]
                },
            )
        if url.endswith("/fapi/v1/ticker/24hr"):
            return _FakeResponse(
                200,
                [{"symbol": s, "quoteVolume": str(10_000_000 - i * 100_000)} for i, s in enumerate(listed)],
            )
        if url.endswith("/fapi/v1/klines"):
            sym = (params or {}).get("symbol", "BTCUSDT")
            px = 2.5 if sym == "SUIUSDT" else (0.012 if sym == "1000PEPEUSDT" else 100.0)
            rows = [
                [
                    base_ms + i * 3_600_000,
                    f"{px}",
                    f"{px * 1.01}",
                    f"{px * 0.99}",
                    f"{px * 1.002}",
                    "1000",
                    base_ms + i * 3_600_000 + 3_599_999,
                    f"{1000 * px}",
                    100,
                    "550",
                    f"{550 * px}",
                    "0",
                ]
                for i in range(30)
            ]
            return _FakeResponse(200, rows)
        if url.endswith("/fapi/v1/fundingRate"):
            return _FakeResponse(200, [{"fundingTime": base_ms, "fundingRate": "-0.0001"}])
        if url.endswith("/fapi/v1/openInterest"):
            return _FakeResponse(
                200,
                {
                    "symbol": (params or {}).get("symbol"),
                    "openInterest": "50000",
                    "time": base_ms + 29 * 3_600_000,
                },
            )
        if "/futures/data/openInterestHist" in url:
            return _FakeResponse(
                200,
                [
                    {
                        "timestamp": base_ms + i * 3_600_000,
                        "sumOpenInterest": str(45000 + i * 100),
                        "sumOpenInterestValue": str((45000 + i * 100) * 100),
                    }
                    for i in range(30)
                ],
            )
        if (
            "/futures/data/globalLongShortAccountRatio" in url
            or "/futures/data/topLongShortPositionRatio" in url
            or "/futures/data/topLongShortAccountRatio" in url
        ):
            return _FakeResponse(
                200,
                [{"timestamp": base_ms + i * 3_600_000, "longShortRatio": "1.05"} for i in range(30)],
            )
        if "/futures/data/takerlongshortRatio" in url:
            return _FakeResponse(
                200,
                [{"timestamp": base_ms + i * 3_600_000, "buySellRatio": "1.15"} for i in range(30)],
            )
        return _FakeResponse(404)

    return _MockSession(_handler)


def test_apptest_live_mode_custom_coins_and_focus_persist_to_deep_dive(monkeypatch) -> None:
    """Radar(Live) + custom symbols + focus coin persist across st.switch_page to Deep Dive."""
    st.cache_data.clear()
    monkeypatch.setattr("requests.Session", _mock_live_binance_session)

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60.0)
    at.run()
    assert not at.exception

    # Switch to Live Auto, add custom symbols & choose focus coin on Radar
    at.radio(key="sr_mode_choice").set_value("Live Auto").run()
    assert not at.exception
    at.multiselect(key="sr_symbols").select("suiusdt").select("pepe").run()
    assert not at.exception
    at.text_input(key="sr_add_coins").set_value("sui").run()
    assert not at.exception
    at.text_input(key="sr_api_key").set_value("sk-session-only-test").run()
    at.selectbox(key="radar_focus_select").select("SUIUSDT").run()
    assert not at.exception
    assert at.session_state["sr_focus_symbol"] == "SUIUSDT"

    # Click "Open Deep Dive ->" (calls st.switch_page("pages/2_Deep_Dive.py"))
    at.button(key="radar_open_deep_dive").click().run()
    assert not at.exception
    assert any("Contract Deep Dive" in t.value for t in at.title)
    assert at.radio(key="sr_mode_choice").value == "Live Auto"
    assert "SUIUSDT" in at.multiselect(key="sr_symbols").value
    assert "1000PEPEUSDT" in at.multiselect(key="sr_symbols").value
    assert "SUIUSDT" in at.selectbox(key="deep_dive_symbol").options
    assert "1000PEPEUSDT" in at.selectbox(key="deep_dive_symbol").options
    assert at.selectbox(key="deep_dive_symbol").value == "SUIUSDT"
    assert at.text_input(key="sr_add_coins").value == "sui"
    assert at.text_input(key="sr_api_key").value == "sk-session-only-test"
    assert any("LIVE" in m.value for m in at.sidebar.markdown)


def test_apptest_replay_date_range_persists_across_pages() -> None:
    """Custom Replay date range selected on Radar survives switch_page to Deep Dive."""
    st.cache_data.clear()
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60.0)
    at.run()
    assert not at.exception

    custom_start = dt.date(2025, 1, 15)
    custom_end = dt.date(2025, 6, 15)
    at.date_input(key="sr_range_start").set_value(custom_start).run()
    at.date_input(key="sr_range_end").set_value(custom_end).run()
    assert not at.exception

    at.switch_page("pages/2_Deep_Dive.py").run()
    assert not at.exception
    assert at.date_input(key="sr_range_start").value == custom_start
    assert at.date_input(key="sr_range_end").value == custom_end
    assert any("2025-01-15" in c.value for c in at.caption)


def test_apptest_radar_live_time_machine_deep_dive_keeps_live(monkeypatch) -> None:
    """Radar(Live) -> Time Machine (allow_live=False) -> Deep Dive restores Live Auto & custom coins."""
    st.cache_data.clear()
    monkeypatch.setattr("requests.Session", _mock_live_binance_session)

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60.0)
    at.run()
    at.radio(key="sr_mode_choice").set_value("Live Auto").run()
    at.multiselect(key="sr_symbols").select("suiusdt").select("pepe").run()
    at.selectbox(key="radar_focus_select").select("SUIUSDT").run()
    assert not at.exception

    # Visit Time Machine (replay-only page)
    at.switch_page("pages/3_Time_Machine.py").run()
    assert not at.exception
    assert any("Time Machine" in t.value for t in at.title)
    assert at.radio(key="sr_mode_choice_replay").value == "Replay (Offline)"
    assert at.session_state["_sr_mode_store"] == "Live Auto"

    # Visit Deep Dive -> Live Auto, custom coins, and focus symbol must all return
    at.switch_page("pages/2_Deep_Dive.py").run()
    assert not at.exception
    assert any("Contract Deep Dive" in t.value for t in at.title)
    assert at.radio(key="sr_mode_choice").value == "Live Auto"
    assert "SUIUSDT" in at.multiselect(key="sr_symbols").value
    assert "1000PEPEUSDT" in at.multiselect(key="sr_symbols").value
    assert "SUIUSDT" in at.selectbox(key="deep_dive_symbol").options
    assert at.selectbox(key="deep_dive_symbol").value == "SUIUSDT"


def test_threshold_consistency_between_kpi_and_table_badges(monkeypatch) -> None:
    """KPI count and table score badges share the documented 50.0 threshold via state_for_score."""
    assert ELEVATED_SCORE_THRESHOLD == StateConfig().setup_score_min == 50.0
    assert state_for_score(47.0) == "Calm"
    assert state_for_score(52.0) == "Elevated"
    assert state_for_score(62.0) == "Active"

    # Score 47 vs 52 classified identically by KPI count and table badge styler
    calm_style = score_badge_style(47.0)
    elevated_style = score_badge_style(52.0)
    assert "background-color" not in calm_style
    assert "245, 158, 11" in elevated_style

    scanner_df = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "price": 90000.0,
                "change_24h_pct": 1.2,
                "funding_rate_pct": 0.01,
                "oi_24h_pct": 2.0,
                "global_ls_ratio": 1.1,
                "short_squeeze_score": 52.0,
                "long_squeeze_score": 20.0,
                "max_squeeze_score": 52.0,
                "state": "SETUP",
                "nearest_cluster": "+2.0% ($10.0M Short)",
            },
            {
                "symbol": "ETHUSDT",
                "price": 3000.0,
                "change_24h_pct": -0.5,
                "funding_rate_pct": 0.01,
                "oi_24h_pct": 1.0,
                "global_ls_ratio": 1.0,
                "short_squeeze_score": 47.0,
                "long_squeeze_score": 15.0,
                "max_squeeze_score": 47.0,
                "state": "NONE",
                "nearest_cluster": "None",
            },
        ]
    )
    disp = format_scanner_display_df(scanner_df)
    styled_html = style_scanner_table(disp).to_html()
    assert "rgba(245, 158, 11, 0.25)" in styled_html

    fake_scan = RadarScanResult(
        scanner_table=scanner_df,
        analyses={},
        badge=ProviderBadge(source="replay", mode="replay", label="REPLAY (Offline)", detail="Test"),
        as_of_label="2025-10-31 23:00 UTC",
        errors={},
    )
    monkeypatch.setattr("squeezeradar.ui.load_radar_cached", lambda *args, **kwargs: fake_scan)

    at = AppTest.from_file(str(ROOT / "pages/1_Radar.py"), default_timeout=60.0)
    at.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    # Only BTCUSDT (52.0) is Elevated; ETHUSDT (47.0) is Calm -> "1 / 1"
    assert metrics["Elevated / Active State"] == "1 / 1"


def test_evidence_n_events_column_is_integer_dtype_and_format() -> None:
    """Evidence bundle and Evidence page render n_events as integer dtype without float decimals."""
    bundle = load_evidence(DEFAULT_EVIDENCE_PATH)
    assert bundle is not None
    assert pd.api.types.is_integer_dtype(bundle.event_study.summary_df["n_events"])
    for res in bundle.walk_forward.values():
        for col in ("n_train", "n_purged", "n_test"):
            assert pd.api.types.is_integer_dtype(res.monthly_df[col])

    at = AppTest.from_file(str(ROOT / "pages/4_Evidence.py"), default_timeout=60.0)
    at.run()
    assert not at.exception
    es_df = at.dataframe[0].value
    assert pd.api.types.is_integer_dtype(es_df["n_events"])
    assert "492.000000" not in es_df.to_string()


def test_shared_css_injects_metric_label_and_value_rules_on_pages() -> None:
    """Shared CSS injected by init_page prevents st.metric label ellipsis truncation."""
    assert '[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *' in _CUSTOM_CSS
    assert "white-space: normal !important;" in _CUSTOM_CSS
    assert "overflow: visible !important;" in _CUSTOM_CSS
    assert "text-overflow: clip !important;" in _CUSTOM_CSS
    assert '[data-testid="stMetricValue"]' in _CUSTOM_CSS
    assert "font-size: clamp(1.1rem, 2.2vw, 1.8rem);" in _CUSTOM_CSS

    for page_rel in ("pages/1_Radar.py", "pages/4_Evidence.py", "pages/6_About.py"):
        at = AppTest.from_file(str(ROOT / page_rel), default_timeout=60.0)
        at.run()
        assert not at.exception
        assert any('[data-testid="stMetricLabel"]' in m.value for m in at.markdown)

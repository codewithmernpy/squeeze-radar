"""Tests for the estimated liquidation map engine (hand-built 3-bar verification & edge cases)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from squeezeradar.config import LiqMapConfig
from squeezeradar.liqmap import build_liquidation_map
from squeezeradar.ui import make_liq_heatmap_figure


def test_liqmap_hand_built_three_bar_math() -> None:
    """Verify exact long/short placement, liquidation sweeps, and proportional negative delta-OI."""
    cfg = LiqMapConfig(
        leverage_weights=((10, 0.5), (100, 0.5)),
        mmr=0.005,
        log_step=0.0005,  # fine 0.05% grid so exact boundary math matches cleanly
        major_cluster_rel_threshold=0.25,
        min_oi_history_bars=1,  # 3-bar toy series: report cluster metrics from the first bar
    )
    # Bar 0: O=H=L=C=100 -> vwap=100, taker_ratio=1.0, body=0 -> 50% long ($500), 50% short ($500)
    #   Long liq prices:  10x -> 100*(1 - 0.1 + 0.005) = 90.5 ($250); 100x -> 100*(1 - 0.01 + 0.005) = 99.5 ($250)
    #   Short liq prices: 10x -> 100*(1 + 0.1 - 0.005) = 109.5 ($250); 100x -> 100*(1 + 0.01 - 0.005) = 100.5 ($250)
    # Bar 1: H=101.0, L=100.0, C=100.8, O=100.0 -> sweeps 100x short at 100.5 ($250)!
    #   oi 10 -> 12 contracts: +2 contracts valued at VWAP=(101+100+100.8)/3 = 100.6 -> +$201.2
    # Bar 2: H=100.5, L=99.0, C=99.2, O=100.5 -> sweeps bar-0 100x long at 99.5 ($250)
    #   and bar-1 100x long at 100.6*0.995 = 100.097!
    #   oi 12 -> 7 contracts; the drop beyond the swept longs closes the same fraction of every survivor.
    times = pd.date_range("2024-01-01 00:00", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open_time": times,
            "open": [100.0, 100.0, 100.5],
            "high": [100.0, 101.0, 100.5],
            "low": [100.0, 100.0, 99.0],
            "close": [100.0, 100.8, 99.2],
            "volume": [100.0, 100.0, 100.0],
            "quote_volume": [10000.0, 10080.0, 9920.0],
            "trades": [10.0, 10.0, 10.0],
            "taker_buy_base": [50.0, 50.0, 50.0],
            "taker_buy_sell_ratio": [1.0, 1.0, 1.0],
            "oi": [10.0, 12.0, 7.0],
            "oi_value": [1000.0, 1209.6, 694.4],
        }
    )

    res = build_liquidation_map(df, config=cfg, seed_initial_oi=True, heatmap_bars=3)
    s = res.summary

    assert res.is_estimate is True
    assert "ESTIMATE" in res.methodology_note

    # Bar 0: initial seeding, no sweeps yet
    assert s.loc[0, "est_long_liq_usd"] == pytest.approx(0.0)
    assert s.loc[0, "est_short_liq_usd"] == pytest.approx(0.0)
    assert s.loc[0, "open_long_liq_usd"] == pytest.approx(500.0, rel=1e-6)
    assert s.loc[0, "open_short_liq_usd"] == pytest.approx(500.0, rel=1e-6)

    # Bar 1: H=101.0 crossed short 100x at 100.5 ($250), missed short 10x at 109.5
    assert s.loc[1, "est_short_liq_usd"] == pytest.approx(250.0, rel=1e-6)
    assert s.loc[1, "est_long_liq_usd"] == pytest.approx(0.0)

    # Bar 2: L=99.0 crossed both 100x long buckets (99.5 and ~100.097)
    # Bar 1 added +$201.2 split with long_share = 0.7*0.5 + 0.3*(0.5 + 0.5*0.8) = 0.35 + 0.27 = 0.62
    #   -> $124.744 long ($62.372 at 100x)
    expected_bar2_long_liq = 250.0 + 0.62 * 201.2 * 0.5
    assert s.loc[2, "est_long_liq_usd"] == pytest.approx(expected_bar2_long_liq, rel=1e-3)

    # Total surviving open exposure after bar 2 must not exceed current OI ($694.4)
    total_open_2 = s.loc[2, "open_long_liq_usd"] + s.loc[2, "open_short_liq_usd"]
    assert total_open_2 <= 694.4 + 1e-6
    assert total_open_2 > 0.0

    # Price profile should have positive short liqs above spot and positive long liqs below spot
    assert not res.price_profile.empty
    assert res.price_profile["short_liq_usd"].sum() > 0.0
    assert res.price_profile["long_liq_usd"].sum() > 0.0
    assert not res.heatmap.empty


def test_liqmap_handles_nan_oi_and_empty_df() -> None:
    """Verify graceful handling of empty frames and NaN OI stretches."""
    empty_res = build_liquidation_map(pd.DataFrame())
    assert empty_res.summary.empty
    assert np.isnan(empty_res.as_of_price)

    times = pd.date_range("2024-01-01 00:00", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open_time": times,
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [10.0, 10.0, 10.0, 10.0],
            "quote_volume": [1000.0, 1010.0, 1020.0, 1030.0],
            "trades": [5.0, 5.0, 5.0, 5.0],
            "taker_buy_base": [5.0, 5.0, 5.0, 5.0],
            "taker_buy_sell_ratio": [np.nan, np.nan, np.nan, np.nan],
            "oi_value": [1000.0, np.nan, np.nan, 1100.0],
        }
    )
    res = build_liquidation_map(df)
    assert len(res.summary) == 4
    assert np.all(np.isfinite(res.summary["est_total_liq_usd"]))


def test_make_liq_heatmap_figure_nan_for_zero_and_dark_colorscale() -> None:
    """Zero and empty heatmap cells must become NaN (transparent) with a dark-starting colorscale."""
    t0 = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    t1 = pd.Timestamp("2025-01-01 01:00", tz="UTC")
    heatmap_df = pd.DataFrame(
        {
            "open_time": [t0, t0, t1],
            "price_bin": [100.0, 105.0, 105.0],
            "long_liq_usd": [0.0, 2_500_000.0, 0.0],
            "short_liq_usd": [0.0, 0.0, 1_500_000.0],
            "total_liq_usd": [0.0, 2_500_000.0, 1_500_000.0],
        }
    )
    fig = make_liq_heatmap_figure(heatmap_df)
    trace = fig.data[0]
    z = np.asarray(trace.z, dtype=float)
    # t0 @ 100 is 0.0 -> NaN; t1 @ 100 is missing -> NaN
    assert np.isnan(z[0, 0])
    assert np.isnan(z[0, 1])
    # Real positive clusters remain intact in $M
    assert z[1, 0] == pytest.approx(2.5)
    assert z[1, 1] == pytest.approx(1.5)
    assert trace.hoverongaps is False
    assert trace.autocolorscale is False
    # First stop of colorscale must be dark (not bright red)
    first_stop, first_hex = trace.colorscale[0]
    assert first_stop == 0.0
    r = int(first_hex[1:3], 16)
    g = int(first_hex[3:5], 16)
    b = int(first_hex[5:7], 16)
    assert max(r, g, b) < 80

"""Tests for rule-based squeeze risk scores and causal state machine transitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from squeezeradar.config import ScoreConfig, StateConfig
from squeezeradar.scores import compute_scores_and_states, explain_score_row


def _make_base_feature_frame(n: int) -> pd.DataFrame:
    times = pd.date_range("2024-01-01 00:00", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": times,
            "close": np.full(n, 100.0),
            "funding_z_30d": np.zeros(n),
            "funding_rate_last": np.full(n, 0.0001),
            "shorts_building_24h": np.zeros(n),
            "shorts_building_4h": np.zeros(n),
            "longs_building_24h": np.zeros(n),
            "longs_building_4h": np.zeros(n),
            "global_ls_ratio": np.full(n, 1.15),
            "global_ls_z_30d": np.zeros(n),
            "top_ls_pos_ratio": np.full(n, 1.15),
            "top_ls_pos_z_30d": np.zeros(n),
            "bb_bw_pct_30d": np.full(n, 0.50),
            "short_liq_within_5pct_share": np.zeros(n),
            "nearest_short_liq_dist_pct": np.full(n, np.nan),
            "nearest_short_liq_usd": np.zeros(n),
            "long_liq_within_5pct_share": np.zeros(n),
            "nearest_long_liq_dist_pct": np.full(n, np.nan),
            "nearest_long_liq_usd": np.zeros(n),
            "ret_1h": np.zeros(n),
            "ret_4h": np.zeros(n),
            "ret_24h": np.zeros(n),
            "oi_chg_1h": np.zeros(n),
            "oi_chg_4h": np.zeros(n),
            "oi_chg_24h": np.zeros(n),
            "taker_buy_sell_ratio_eff": np.ones(n),
            "taker_ratio_z_30d": np.zeros(n),
            "est_short_liq_usd": np.zeros(n),
            "est_long_liq_usd": np.zeros(n),
        }
    )


def test_short_squeeze_state_machine_full_lifecycle() -> None:
    """Verify NONE -> SETUP -> TRIGGER -> IN_PROGRESS -> EXHAUSTION -> NONE on crafted series."""
    df = _make_base_feature_frame(7)

    # Bar 1: strong short crowding -> SETUP
    df.loc[1, ["funding_z_30d", "funding_rate_last", "shorts_building_24h", "global_ls_z_30d", "bb_bw_pct_30d"]] = [
        -2.5,
        -0.0003,
        0.08,
        -2.0,
        0.05,
    ]
    df.loc[1, ["short_liq_within_5pct_share", "nearest_short_liq_dist_pct", "nearest_short_liq_usd"]] = [
        0.07,
        1.5,
        5e7,
    ]

    # Bar 2: keep elevated score + trigger conditions (price moves up, OI drops, taker buy spike) -> TRIGGER
    df.loc[2] = df.loc[1]
    df.loc[2, ["ret_1h", "ret_4h", "oi_chg_1h", "taker_buy_sell_ratio_eff", "taker_ratio_z_30d"]] = [
        0.020,
        0.025,
        -0.025,
        1.45,
        2.0,
    ]

    # Bar 3: continuation (price still rising, short liquidations firing) -> IN_PROGRESS
    df.loc[3, ["ret_1h", "ret_4h", "oi_chg_1h", "est_short_liq_usd", "taker_buy_sell_ratio_eff"]] = [
        0.012,
        0.035,
        -0.010,
        1e7,
        1.25,
    ]

    # Bar 4: OI drop halts, price stalls, flow normalizes -> EXHAUSTION
    df.loc[4, ["ret_1h", "oi_chg_1h", "taker_buy_sell_ratio_eff", "taker_ratio_z_30d"]] = [
        -0.001,
        0.004,
        0.95,
        0.0,
    ]

    # Bar 5: cooldown elapsed and crowding low -> resets to NONE
    # Bar 6: remains NONE
    scored = compute_scores_and_states(df, score_config=ScoreConfig(), state_config=StateConfig())
    states = scored["short_squeeze_state"].tolist()
    assert states == ["NONE", "SETUP", "TRIGGER", "IN_PROGRESS", "EXHAUSTION", "NONE", "NONE"]

    bd = explain_score_row(scored.iloc[1], symbol="BTCUSDT", side="short_squeeze")
    assert set(bd.components.keys()) == {
        "funding",
        "oi_divergence",
        "positioning",
        "liq_cluster",
        "compression",
    }
    assert bd.score >= 50.0
    assert len(bd.reasons) >= 2


def test_long_squeeze_state_machine_and_setup_reset() -> None:
    """Verify long-squeeze lifecycle AND SETUP -> NONE reset when crowding cools without trigger."""
    df = _make_base_feature_frame(8)

    # Bar 1: heavy long crowding -> SETUP
    df.loc[1, ["funding_z_30d", "funding_rate_last", "longs_building_24h", "global_ls_z_30d", "bb_bw_pct_30d"]] = [
        2.5,
        0.0006,
        0.15,
        2.2,
        0.05,
    ]
    df.loc[1, ["long_liq_within_5pct_share", "nearest_long_liq_dist_pct", "nearest_long_liq_usd"]] = [
        0.07,
        1.5,
        5e7,
    ]

    # Bar 2: trigger downside cascade -> TRIGGER
    df.loc[2] = df.loc[1]
    df.loc[2, ["ret_1h", "ret_4h", "oi_chg_1h", "taker_buy_sell_ratio_eff", "taker_ratio_z_30d"]] = [
        -0.025,
        -0.030,
        -0.030,
        0.65,
        -2.0,
    ]

    # Bar 3: cascade continues -> IN_PROGRESS
    df.loc[3, ["ret_1h", "ret_4h", "oi_chg_1h", "est_long_liq_usd", "taker_buy_sell_ratio_eff"]] = [
        -0.015,
        -0.045,
        -0.020,
        2e7,
        0.75,
    ]

    # Bar 4: cascade stalls -> EXHAUSTION
    df.loc[4, ["ret_1h", "oi_chg_1h", "taker_buy_sell_ratio_eff", "taker_ratio_z_30d"]] = [
        0.001,
        0.005,
        1.02,
        0.1,
    ]

    # Bar 5: resets to NONE
    # Bar 6: re-enters SETUP without triggering
    df.loc[6] = df.loc[1]
    # Bar 7: cools back below setup_exit_score -> resets to NONE
    scored = compute_scores_and_states(df)
    assert scored["long_squeeze_state"].tolist() == [
        "NONE",
        "SETUP",
        "TRIGGER",
        "IN_PROGRESS",
        "EXHAUSTION",
        "NONE",
        "SETUP",
        "NONE",
    ]
    assert scored["long_squeeze_score"].between(0.0, 100.0).all()
    assert scored["short_squeeze_score"].between(0.0, 100.0).all()

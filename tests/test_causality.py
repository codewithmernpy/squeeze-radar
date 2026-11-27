"""Strict causality, truncation invariance, label isolation, and walk-forward purge tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from squeezeradar.config import DEFAULT_SYMBOLS, MLConfig
from squeezeradar.data import align_symbol_frames, load_sample_symbol
from squeezeradar.features import MODEL_FEATURE_COLUMNS, compute_features
from squeezeradar.liqmap import build_liquidation_map
from squeezeradar.model import (
    ML_EXTENDED_FEATURES,
    build_squeeze_labels,
    compute_event_study,
    run_walk_forward,
    split_purged_walk_forward,
)
from squeezeradar.scores import compute_scores_and_states


def test_alignment_causality_and_metrics_ffill_limit() -> None:
    """Verify metrics forward-fill at most 2 bars and as_of blocks future bars/funding."""
    times = pd.date_range("2024-01-01 00:00", periods=8, freq="1h", tz="UTC")
    klines = pd.DataFrame(
        {
            "open_time": times,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "quote_volume": 1005.0,
            "trades": 5.0,
            "taker_buy_base": 5.0,
        }
    )
    # Funding at 00:00 and 08:00 (08:00 settles at bar open_time=07:00 close_time=08:00)
    funding = pd.DataFrame(
        {
            "funding_time": [times[0], times[7] + pd.Timedelta(hours=1)],  # 00:00 and 08:00
            "funding_rate": [0.0001, -0.0005],
        }
    )
    # Metrics present at bar 0, missing at bars 1, 2, 3, 4, present at bar 5
    metrics = pd.DataFrame(
        {
            "open_time": [times[0], times[5]],
            "oi": [1000.0, 1200.0],
            "oi_value": [100_000.0, 120_000.0],
            "top_trader_ls_accounts": [1.1, 1.2],
            "top_trader_ls_positions": [1.1, 1.2],
            "global_ls_accounts": [1.1, 1.2],
            "taker_buy_sell_ratio": [1.0, 1.3],
            "n_snapshots": [12, 12],
        }
    )

    aligned = align_symbol_frames(klines, funding, metrics, max_metrics_ffill_bars=2)
    # Bars 1 & 2 forward-filled from bar 0; bars 3 & 4 must stay NaN (>2 bars missing)
    assert aligned.loc[0, "oi_value"] == 100_000.0
    assert aligned.loc[1, "oi_value"] == 100_000.0
    assert aligned.loc[2, "oi_value"] == 100_000.0
    assert np.isnan(aligned.loc[3, "oi_value"])
    assert np.isnan(aligned.loc[4, "oi_value"])
    assert aligned.loc[5, "oi_value"] == 120_000.0

    # Cutoff at 03:00 UTC -> only bars with close_time <= 03:00 (open_time <= 02:00)
    cut = align_symbol_frames(klines, funding, metrics, as_of="2024-01-01 03:00")
    assert len(cut) == 3
    assert cut["funding_rate"].iloc[-1] == 0.0001  # -0.0005 at 08:00 not visible


@pytest.mark.parametrize("symbol", DEFAULT_SYMBOLS)
def test_truncation_causality_20_random_points_per_symbol(symbol: str) -> None:
    """Features, liqmap, and scores on data[:t+1] must equal full-data values at bar t."""
    ds = load_sample_symbol(symbol)
    full_aligned = ds.aligned

    f_full = compute_features(full_aligned)
    lm_full = build_liquidation_map(f_full, heatmap_bars=1)
    s_full = compute_scores_and_states(f_full, lm_full.summary)

    rng = np.random.default_rng(1000 + DEFAULT_SYMBOLS.index(symbol))
    # Sample 20 random points across [200, 2200] and verify exact equality against full 24,840-bar series
    sample_ts = sorted(int(x) for x in rng.choice(np.arange(200, 2200), size=20, replace=False))

    numeric_cols = [
        "funding_rate_last",
        "funding_z_30d",
        "oi_chg_24h",
        "oi_price_div_24h",
        "global_ls_z_30d",
        "taker_ratio_z_30d",
        "cvd_24h_norm",
        "atr_14",
        "bb_bw_pct_30d",
        "est_long_liq_usd",
        "est_short_liq_usd",
        "open_long_liq_usd",
        "open_short_liq_usd",
        "nearest_short_liq_dist_pct",
        "nearest_long_liq_dist_pct",
        "short_squeeze_score",
        "long_squeeze_score",
    ]
    state_cols = ["short_squeeze_state", "long_squeeze_state", "squeeze_state"]

    for t in sample_ts:
        trunc_aligned = full_aligned.iloc[: t + 1].copy()
        ft = compute_features(trunc_aligned)
        lmt = build_liquidation_map(ft, heatmap_bars=1)
        st = compute_scores_and_states(ft, lmt.summary)

        for col in numeric_cols:
            vf = s_full[col].iloc[t]
            vt = st[col].iloc[t]
            if np.isnan(vf):
                assert np.isnan(vt), f"{symbol} t={t} col={col}: full NaN vs trunc {vt}"
            else:
                assert vt == pytest.approx(vf, abs=1e-10), f"{symbol} t={t} col={col}: {vt} != {vf}"

        for scol in state_cols:
            assert st[scol].iloc[t] == s_full[scol].iloc[t], f"{symbol} t={t} state mismatch on {scol}"


def test_labels_use_strictly_future_bars_and_excluded_from_features() -> None:
    """Verify squeeze labels are excluded from ML features and depend only on [t+1..t+24]."""
    assert "squeeze_up" not in MODEL_FEATURE_COLUMNS
    assert "squeeze_down" not in MODEL_FEATURE_COLUMNS
    assert "squeeze_up" not in ML_EXTENDED_FEATURES
    assert "squeeze_down" not in ML_EXTENDED_FEATURES
    for forbidden in ("open", "high", "low", "close", "volume", "oi_usd"):
        assert forbidden not in ML_EXTENDED_FEATURES

    ds = load_sample_symbol("BTCUSDT")
    sub = ds.aligned.iloc[:300].copy()
    feat = compute_features(sub)
    labels_orig = build_squeeze_labels(feat)

    # Mutate bars t+25..end -> must not affect bar t=100 features or labels
    t = 100
    sub_far = sub.copy()
    sub_far.loc[t + 25 :, "high"] *= 1.5
    sub_far.loc[t + 25 :, ["oi", "oi_value"]] *= 0.5
    feat_far = compute_features(sub_far)
    labels_far = build_squeeze_labels(feat_far)

    for col in MODEL_FEATURE_COLUMNS:
        v1 = feat[col].iloc[t]
        v2 = feat_far[col].iloc[t]
        if np.isnan(v1):
            assert np.isnan(v2)
        else:
            assert v1 == pytest.approx(v2, abs=1e-12)
    assert labels_orig["squeeze_up"].iloc[t] == labels_far["squeeze_up"].iloc[t]
    assert labels_orig["squeeze_down"].iloc[t] == labels_far["squeeze_down"].iloc[t]

    # Mutate bars t+1..t+24 to force a squeeze_up at t -> features at t remain unchanged, label flips to 1
    sub_near = sub.copy()
    sub_near.loc[t + 1 : t + 5, "high"] = sub_near.loc[t, "close"] + 5.0 * feat.loc[t, "atr_14"]
    sub_near.loc[t + 1 : t + 5, "low"] = sub_near.loc[t, "close"]
    sub_near.loc[t + 1 : t + 5, "oi"] = sub_near.loc[t, "oi"] * 0.90  # labels use contract OI
    sub_near.loc[t + 1 : t + 5, "oi_value"] = sub_near.loc[t, "oi_value"] * 0.90
    feat_near = compute_features(sub_near)
    labels_near = build_squeeze_labels(feat_near)

    for col in MODEL_FEATURE_COLUMNS:
        v1 = feat[col].iloc[t]
        v2 = feat_near[col].iloc[t]
        if np.isnan(v1):
            assert np.isnan(v2)
        else:
            assert v1 == pytest.approx(v2, abs=1e-12)
    assert labels_near["squeeze_up"].iloc[t] == 1.0


def test_walk_forward_purge_and_event_study_end_to_end() -> None:
    """Verify strict 24-bar purge before each test month and valid ML/event-study outputs."""
    ds_btc = load_sample_symbol("BTCUSDT")
    ds_eth = load_sample_symbol("ETHUSDT")
    # Use trailing 9 months (~6500 bars per symbol) for fast realistic multi-fold check
    frames = {}
    for sym, ds in (("BTCUSDT", ds_btc), ("ETHUSDT", ds_eth)):
        sub = ds.aligned.iloc[-6500:].reset_index(drop=True)
        f = compute_features(sub)
        lm = build_liquidation_map(f, heatmap_bars=24)
        frames[sym] = compute_scores_and_states(f, lm.summary)

    pooled = pd.concat(
        [df.assign(symbol=sym) for sym, df in frames.items()],
        ignore_index=True,
    )
    splits = split_purged_walk_forward(pooled, purge_bars=24, min_train_months=6)
    assert len(splits) >= 2
    for month_str, tr_idx, te_idx, purged_idx in splits:
        assert len(purged_idx) == 24 * 2, f"Expected 24 purged bars per symbol in {month_str}"
        max_tr_time = pooled.iloc[tr_idx]["open_time"].max()
        min_te_time = pooled.iloc[te_idx]["open_time"].min()
        assert max_tr_time <= min_te_time - pd.Timedelta(hours=25)
        assert set(tr_idx).isdisjoint(set(te_idx))
        assert set(tr_idx).isdisjoint(set(purged_idx))
        assert set(te_idx).isdisjoint(set(purged_idx))

    cfg = MLConfig(min_train_months=6, max_iter=20, max_bins=64)
    wf = run_walk_forward(frames, config=cfg)
    for target in ("squeeze_up", "squeeze_down"):
        res = wf[target]
        assert res.n_oos > 0
        assert res.n_folds >= 2
        assert 0.0 <= res.auc <= 1.0
        assert 0.0 <= res.precision_top_5pct <= 1.0
        assert not res.calibration_df.empty
        assert not res.monthly_df.empty

    es = compute_event_study(frames, ml_config=cfg)
    assert not es.summary_df.empty
    assert not es.paths_df.empty
    assert "Unconditional Baseline" in es.summary_df["signal"].tolist()

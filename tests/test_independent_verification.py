"""Independent verification suite (written separately from the original tests).

Covers: truncation causality on real bundled data (features, liqmap, scores, states, profile),
raw-level future-perturbation invariance, a hand-computed liquidation-map example, crafted
labels, purge / walk-forward boundaries and future-invariance of OOS predictions, provider
parsing of REAL recorded Binance / Hyperliquid JSON (tests/fixtures/live_recorded_btc.json.gz)
plus simulated 451 / timeout fallback, the AI analyst through the real anthropic SDK
tool_runner over a mocked HTTP transport, the precomputed evidence file, and AppTest runs of
every Streamlit page. No network access.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import numpy as np
import pandas as pd
import pytest
import requests
from streamlit.testing.v1 import AppTest

from squeezeradar.ai import AnalystContext, build_analyst_tools, generate_analyst_brief
from squeezeradar.config import LiqMapConfig, MLConfig, ProviderConfig
from squeezeradar.data import TTLCache, align_symbol_frames, load_sample_symbol, load_sample_symbol_raw
from squeezeradar.evidence import (
    DEFAULT_EVIDENCE_PATH,
    evidence_from_dict,
    evidence_to_dict,
    load_evidence,
)
from squeezeradar.features import compute_features
from squeezeradar.liqmap import build_liquidation_map
from squeezeradar.model import (
    _decluster_mask,
    build_squeeze_labels,
    predict_latest_probabilities,
    run_walk_forward,
    split_purged_walk_forward,
)
from squeezeradar.pipeline import run_radar_scan
from squeezeradar.providers.base import HttpJsonClient
from squeezeradar.providers.binance import BinanceProvider
from squeezeradar.providers.chain import ProviderChain
from squeezeradar.providers.hyperliquid import HL_FUNDING_TO_8H, HyperliquidProvider
from squeezeradar.scores import compute_scores_and_states

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "live_recorded_btc.json.gz"
H = pd.Timedelta(hours=1)

CHECK_COLS = [
    "funding_rate_last",
    "funding_mean_24h",
    "funding_z_30d",
    "oi_chg_1h",
    "oi_chg_24h",
    "oi_z_30d",
    "oi_price_div_4h",
    "oi_price_div_24h",
    "oi_to_vol_24h",
    "global_ls_z_30d",
    "top_ls_pos_z_30d",
    "taker_ratio_z_30d",
    "volume_z_30d",
    "cvd_z_30d",
    "atr_14",
    "bb_bw_pct_30d",
    "dist_30d_high_pct",
    "est_long_liq_usd",
    "est_short_liq_usd",
    "open_long_liq_usd",
    "open_short_liq_usd",
    "nearest_short_liq_dist_pct",
    "nearest_long_liq_dist_pct",
    "short_liq_within_5pct_share",
    "long_liq_within_5pct_share",
    "short_squeeze_score",
    "long_squeeze_score",
]
STATE_COLS = ["short_squeeze_state", "long_squeeze_state", "squeeze_state", "active_side"]


def _score(aligned: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    feat = compute_features(aligned)
    lm = build_liquidation_map(feat, heatmap_bars=1)
    return compute_scores_and_states(feat, lm.summary), lm


def _assert_row_equal(a: pd.Series, b: pd.Series, cols: list[str], ctx: str) -> None:
    for c in cols:
        va, vb = a[c], b[c]
        if isinstance(va, str) or isinstance(vb, str):
            assert va == vb, f"{ctx} {c}: {va!r} != {vb!r}"
        elif pd.isna(va):
            assert pd.isna(vb), f"{ctx} {c}: NaN vs {vb}"
        else:
            assert vb == pytest.approx(va, rel=1e-9, abs=1e-12), f"{ctx} {c}: {va} != {vb}"


@pytest.fixture(scope="module")
def full_scored() -> dict[str, pd.DataFrame]:
    out = {}
    for sym in ("BTCUSDT", "DOGEUSDT", "SOLUSDT"):
        out[sym] = _score(load_sample_symbol(sym).aligned)[0]
    return out


# --------------------------------------------------------------------------------------
# A. Causality on real data
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("symbol", ["BTCUSDT", "DOGEUSDT", "SOLUSDT"])
def test_truncation_full_range_random_points(symbol: str, full_scored) -> None:
    """Every feature/liqmap/score/state at t on aligned[:t+1] equals the full-history value."""
    aligned = load_sample_symbol(symbol).aligned
    full = full_scored[symbol]
    rng = np.random.default_rng(20261125 + len(symbol))
    ts = sorted(int(x) for x in rng.choice(np.arange(750, len(aligned)), size=6, replace=False))
    for t in ts:
        trunc, _ = _score(aligned.iloc[: t + 1].copy())
        assert len(trunc) == t + 1
        _assert_row_equal(full.iloc[t], trunc.iloc[t], CHECK_COLS + STATE_COLS, f"{symbol} t={t}")
        # whole state path up to t must match, not only the last bar
        assert trunc["squeeze_state"].tolist() == full["squeeze_state"].iloc[: t + 1].tolist()


@pytest.mark.parametrize(
    "symbol,as_of",
    [("BTCUSDT", "2024-08-05 00:00"), ("DOGEUSDT", "2025-04-22 12:00"), ("SOLUSDT", "2023-11-23 05:00")],
)
def test_raw_as_of_loader_matches_full_history_row(symbol: str, as_of: str, full_scored) -> None:
    """Loading raw CSVs cut at `as_of` (klines/metrics by close, funding by time) reproduces the
    full-history row of the last bar known at `as_of`, including the liqmap price profile."""
    ds = load_sample_symbol(symbol, as_of=as_of)
    cut, lm_cut = _score(ds.aligned)
    t_open = pd.Timestamp(as_of, tz="UTC") - H
    assert cut["open_time"].iloc[-1] == t_open
    assert (cut["close_time"] <= pd.Timestamp(as_of, tz="UTC")).all()
    full = full_scored[symbol]
    idx = int(np.flatnonzero(full["open_time"] == t_open)[0])
    _assert_row_equal(full.iloc[idx], cut.iloc[-1], CHECK_COLS + STATE_COLS, f"{symbol}@{as_of}")
    # profile computed from data[:t+1] equals the one from plain row truncation of the full frame
    lm_rows = build_liquidation_map(compute_features(load_sample_symbol(symbol).aligned.iloc[: idx + 1]), heatmap_bars=1)
    pd.testing.assert_frame_equal(lm_cut.price_profile, lm_rows.price_profile)
    assert not lm_cut.price_profile.empty
    # the profile's nearest major short cluster is above spot, long cluster below
    prof = lm_cut.price_profile
    assert (prof.loc[prof["short_liq_usd"] > 0, "pct_from_spot"] > 0).all()
    assert (prof.loc[prof["long_liq_usd"] > 0, "pct_from_spot"] < 0).all()


def test_future_raw_perturbation_cannot_change_past_rows() -> None:
    """Scramble every raw kline/metric/funding record after T: rows known at or before T are
    bit-identical (checks merge_asof direction, ffill limits, rolling windows, liqmap, states)."""
    k, f, m = load_sample_symbol_raw("BTCUSDT")
    T = pd.Timestamp("2024-06-15 12:00", tz="UTC")
    base, _ = _score(align_symbol_frames(k, f, m))

    k2, f2, m2 = k.copy(), f.copy(), m.copy()
    fut_k = k2["open_time"] + H > T
    for c in ("open", "high", "low", "close"):
        k2.loc[fut_k, c] *= 3.0
    k2.loc[fut_k, ["volume", "quote_volume", "taker_buy_base"]] *= 0.1
    f2.loc[f2["funding_time"] > T, "funding_rate"] = 0.01
    fut_m = m2["open_time"] + H > T
    m2.loc[fut_m, ["oi", "oi_value"]] *= 0.2
    m2.loc[fut_m, ["global_ls_accounts", "top_trader_ls_positions", "taker_buy_sell_ratio"]] = 9.0
    pert, _ = _score(align_symbol_frames(k2, f2, m2))

    known = base["close_time"] <= T
    assert known.sum() > 12_000
    for c in CHECK_COLS:
        np.testing.assert_allclose(
            base.loc[known, c].to_numpy(float), pert.loc[known, c].to_numpy(float), rtol=0, atol=0, equal_nan=True, err_msg=c
        )
    for c in STATE_COLS:
        assert base.loc[known, c].tolist() == pert.loc[known, c].tolist()
    # ... and the perturbation is visible right after T (the test really perturbs something)
    assert not np.isclose(base.loc[~known, "close"].iloc[0], pert.loc[~known, "close"].iloc[0])


def test_alignment_semantics_by_hand() -> None:
    """Hand checks on BTCUSDT raw rows: metrics row h sits on bar h; funding settled at 00:00 is
    visible on the bar closing at 00:00 but not on the bar closing at 23:00; zero-OI glitches
    become NaN (never a -100% OI change)."""
    k, f, m = load_sample_symbol_raw("BTCUSDT")
    al = align_symbol_frames(k, f, m)
    row = al.loc[al["open_time"] == pd.Timestamp("2024-08-04 23:00", tz="UTC")].iloc[0]
    mrow = m.loc[m["open_time"] == pd.Timestamp("2024-08-04 23:00", tz="UTC")].iloc[0]
    assert row["oi"] == mrow["oi"] and row["global_ls_accounts"] == mrow["global_ls_accounts"]
    assert row["funding_time"] == pd.Timestamp("2024-08-05 00:00", tz="UTC")
    prev = al.loc[al["open_time"] == pd.Timestamp("2024-08-04 22:00", tz="UTC")].iloc[0]
    assert prev["funding_time"] == pd.Timestamp("2024-08-04 16:00", tz="UTC")
    # raw file has 30 zero-OI hours for BTCUSDT; aligned/featured data must not carry them
    raw = pd.read_csv(ROOT / "data" / "sample" / "BTCUSDT_metrics_1h.csv.gz")
    assert int((raw["oi"] <= 0).sum()) == 30
    assert not (al["oi"] <= 0).any()
    feat = compute_features(al)
    assert feat["oi_chg_1h"].min() > -0.5
    # OI 24h change is a contract-OI change: hand-compute it
    t, t24 = pd.Timestamp("2024-08-04 23:00", tz="UTC"), pd.Timestamp("2024-08-03 23:00", tz="UTC")
    oi_t = m.set_index("open_time").loc[t, "oi"]
    oi_24 = m.set_index("open_time").loc[t24, "oi"]
    got = feat.set_index("open_time").loc[t, "oi_chg_24h"]
    assert got == pytest.approx(oi_t / oi_24 - 1.0, rel=1e-12)


def test_event_study_declustering_is_causal() -> None:
    rng = np.random.default_rng(7)
    mask = rng.random(3000) < 0.08
    full = _decluster_mask(mask, min_gap_bars=12)
    for i in rng.choice(np.arange(10, 3000), size=40, replace=False):
        assert np.array_equal(_decluster_mask(mask[: i + 1], min_gap_bars=12), full[: i + 1])


# --------------------------------------------------------------------------------------
# Liquidation map: hand-computed 4-bar example
# --------------------------------------------------------------------------------------
def test_liqmap_hand_computed_four_bars() -> None:
    cfg = LiqMapConfig(
        leverage_weights=((10, 0.5), (50, 0.5)),
        mmr=0.005,
        log_step=0.0002,  # 0.02% bins: no level sits within a bin of any bar high/low
        taker_direction_weight=1.0,
        price_direction_weight=0.0,  # long share == taker buy share exactly
        min_oi_history_bars=1,
    )
    # Bar0 O=H=L=C=100, oi 10 contracts, taker ratio 1 (share 0.5).
    #   Seed $1000 (10 x VWAP 100): longs $500, shorts $500, $250 per leverage tier.
    #   long liq  10x 100*(1-0.10+0.005)=90.5   50x 100*(1-0.02+0.005)=98.5
    #   short liq 10x 100*(1+0.10-0.005)=109.5  50x 100*(1+0.02-0.005)=101.5
    # Bar1 O=100 H=102 L=100 C=101 -> VWAP 101. Sweep: short 101.5 <= 102 -> $250 short liq.
    #   oi 10->12: +2 x 101 = $202; ratio 1.5 -> share 0.6 -> longs $121.2, shorts $80.8.
    #   long 10x 101*0.905=91.405 ($60.6), 50x 101*0.985=99.485 ($60.6)
    #   short 10x 101*1.095=110.595 ($40.4), 50x 101*1.015=102.515 ($40.4)
    #   book: longs 500+121.2=621.2, shorts 250+80.8=330.8
    # Bar2 O=101 H=101 L=98 C=98.5 -> VWAP 99.1667. Sweep longs >= 98: 98.5 ($250) + 99.485 ($60.6)
    #   = $310.6 long liq. oi 12->9: -3 x 99.1667 = -$297.5 < swept 310.6 -> no extra closing.
    #   book: longs 310.6 (90.5:$250, 91.405:$60.6), shorts 330.8
    # Bar3 O=98.5 H=99 L=98 C=98.8 -> VWAP 98.6. No level crossed.
    #   oi 9->6: -3 x 98.6 = -$295.8; open before = 9 x 98.6 = $887.4 -> closed fraction 1/3
    #   -> every level x 2/3: longs 207.0667, shorts 220.5333 (cap $592.8 not binding).
    df = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC"),
            "open": [100.0, 100.0, 101.0, 98.5],
            "high": [100.0, 102.0, 101.0, 99.0],
            "low": [100.0, 100.0, 98.0, 98.0],
            "close": [100.0, 101.0, 98.5, 98.8],
            "volume": [10.0] * 4,
            "taker_buy_base": [5.0] * 4,
            "taker_buy_sell_ratio": [1.0, 1.5, 1.0, 1.0],
            "oi": [10.0, 12.0, 9.0, 6.0],
            "oi_value": [1000.0, 1212.0, 886.5, 592.8],
        }
    )
    s = build_liquidation_map(df, config=cfg, heatmap_bars=4).summary
    assert s["est_short_liq_usd"].tolist() == pytest.approx([0.0, 250.0, 0.0, 0.0])
    assert s["est_long_liq_usd"].tolist() == pytest.approx([0.0, 0.0, 310.6, 0.0])
    assert s.loc[0, "open_long_liq_usd"] == pytest.approx(500.0)
    assert s.loc[1, "open_long_liq_usd"] == pytest.approx(621.2)
    assert s.loc[1, "open_short_liq_usd"] == pytest.approx(330.8)
    assert s.loc[2, "open_long_liq_usd"] == pytest.approx(310.6)
    assert s.loc[3, "open_long_liq_usd"] == pytest.approx(310.6 * 2 / 3)
    assert s.loc[3, "open_short_liq_usd"] == pytest.approx(330.8 * 2 / 3)
    # nearest MAJOR short cluster at bar 3: levels 102.515 ($26.93), 109.5 ($166.67), 110.595
    # ($26.93); 25% of the max is $41.67, so the nearest major level is 109.5 = +10.83% vs 98.8
    assert s.loc[3, "nearest_short_liq_dist_pct"] == pytest.approx((109.5 / 98.8 - 1) * 100, abs=0.05)
    assert s.loc[3, "nearest_short_liq_usd"] == pytest.approx(250.0 * 2 / 3)
    # nearest major long cluster: 91.405 ($40.4) vs 90.5 ($166.67): threshold $41.67 -> 90.5
    assert s.loc[3, "nearest_long_liq_dist_pct"] == pytest.approx((1 - 90.5 / 98.8) * 100, abs=0.05)
    # share of OI within 5% above spot: only 102.515 ($26.93) / OI $592.8
    assert s.loc[3, "short_liq_within_5pct_share"] == pytest.approx((40.4 * 2 / 3) / 592.8, rel=1e-6)

    # with the default 24-observation warm-up the cluster metrics are not reported yet
    s_warm = build_liquidation_map(df, config=cfg.model_copy(update={"min_oi_history_bars": 24})).summary
    assert s_warm["nearest_short_liq_dist_pct"].isna().all()
    assert s_warm["short_liq_within_5pct_share"].isna().all()


def test_liqmap_price_move_alone_does_not_create_positions() -> None:
    """Constant contract OI with a rising price (USD OI rising) must not add or remove levels."""
    n = 30
    close = 100.0 * (1.01 ** np.arange(n))
    df = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
            "taker_buy_base": 0.5,
            "taker_buy_sell_ratio": 1.0,
            "oi": 10.0,
            "oi_value": 10.0 * close,
        }
    )
    s = build_liquidation_map(df, config=LiqMapConfig(min_oi_history_bars=1)).summary
    book = s["open_long_liq_usd"] + s["open_short_liq_usd"] + s["est_total_liq_usd"].cumsum()
    assert book.iloc[-1] == pytest.approx(1000.0)  # only the $1000 seed ever existed


# --------------------------------------------------------------------------------------
# Labels, purge, walk-forward
# --------------------------------------------------------------------------------------
def _label_frame(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": np.full(n, 100.0),
            "high": np.full(n, 100.5),
            "low": np.full(n, 99.5),
            "atr_14": np.full(n, 1.0),
            "oi": np.full(n, 1000.0),
            "oi_value": np.full(n, 100_000.0),
        }
    )


def test_labels_on_crafted_series() -> None:
    df = _label_frame()
    # t=0: +2 ATR (102) hit at bar 3 before -1 ATR (99), contract OI 1000 -> 960 (-4%) -> up=1
    df.loc[3, "high"] = 102.0
    df.loc[5, "oi"] = 960.0
    # t=10: -2 ATR (98) at bar 12, OI -4% at bar 13 -> down=1 (bar 12's high 100.5 < 101)
    df.loc[12, "low"] = 98.0
    df.loc[13, "oi"] = 960.0
    lab = build_squeeze_labels(df)
    assert lab.loc[0, "squeeze_up"] == 1.0 and lab.loc[0, "squeeze_down"] == 0.0
    assert lab.loc[10, "squeeze_down"] == 1.0 and lab.loc[10, "squeeze_up"] == 0.0
    assert lab["squeeze_up"].iloc[-24:].isna().all() and lab["squeeze_down"].iloc[-24:].isna().all()
    assert lab["squeeze_up"].iloc[:-24].notna().all()

    # tie: TP and SL touched inside the same future bar -> not a squeeze
    tie = _label_frame()
    tie.loc[2, ["high", "low"]] = [102.5, 98.5]
    tie.loc[4, "oi"] = 900.0
    assert build_squeeze_labels(tie).loc[0, "squeeze_up"] == 0.0
    # OI drop only 2% -> not a squeeze
    small = _label_frame()
    small.loc[3, "high"] = 102.0
    small.loc[5, "oi"] = 980.0
    assert build_squeeze_labels(small).loc[0, "squeeze_up"] == 0.0
    # USD OI falls 10% (price effect) but contract OI flat -> not a squeeze
    usd_only = _label_frame()
    usd_only.loc[12, "low"] = 98.0
    usd_only.loc[11:, "oi_value"] = 90_000.0
    assert build_squeeze_labels(usd_only).loc[10, "squeeze_down"] == 0.0
    # label at t ignores bar t itself and everything after t+24
    far = _label_frame()
    far.loc[25, "high"] = 105.0  # t=0 horizon is bars 1..24
    far.loc[25, "oi"] = 500.0
    assert build_squeeze_labels(far).loc[0, "squeeze_up"] == 0.0
    assert build_squeeze_labels(far).loc[1, "squeeze_up"] == 1.0


def _synthetic_pool(months: int = 9, seed: int = 3) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01", periods=months * 30 * 24, freq="1h", tz="UTC")
    out = {}
    for k, sym in enumerate(("AAAUSDT", "BBBUSDT")):
        n = len(t)
        ret = rng.normal(0, 0.01, n)
        close = 100 * np.exp(np.cumsum(ret))
        df = pd.DataFrame(
            {
                "open_time": t,
                "close": close,
                "high": close * (1 + np.abs(rng.normal(0, 0.006, n))),
                "low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
                "atr_14": close * 0.008,
                "oi": 1e6 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                "funding_z_30d": rng.normal(size=n),
                "oi_chg_24h": rng.normal(size=n),
                "ret_24h": rng.normal(size=n),
                "short_squeeze_score": rng.uniform(0, 100, n),
            }
        )
        df["oi_value"] = df["oi"] * df["close"]
        out[sym] = df
    return out


def test_purged_split_boundaries() -> None:
    frames = _synthetic_pool()
    pooled = pd.concat([d.assign(symbol=s) for s, d in frames.items()], ignore_index=True).sort_values(
        ["open_time", "symbol"]
    ).reset_index(drop=True)
    splits = split_purged_walk_forward(pooled, purge_bars=24, min_train_months=6)
    months = sorted(pooled["open_time"].dt.tz_convert(None).dt.to_period("M").unique())
    assert [s[0] for s in splits] == [str(m) for m in months[6:]]
    prev_train = 0
    for month, tr, te, pg in splits:
        ot = pooled["open_time"]
        test_start = ot.iloc[te].min()
        assert str(test_start.tz_convert(None).to_period("M")) == month
        assert (ot.iloc[te].dt.tz_convert(None).dt.to_period("M").astype(str) == month).all()
        # every training row's 24-bar label window (bars t+1..t+24) ends before the test month
        assert (ot.iloc[tr] + 24 * H < test_start).all()
        assert ot.iloc[tr].max() == test_start - 25 * H
        # purge = exactly the 24 hours before the test month, for each symbol
        assert len(pg) == 24 * len(frames)
        assert (ot.iloc[pg] >= test_start - 24 * H).all() and (ot.iloc[pg] < test_start).all()
        assert len(set(tr) & set(te)) == 0 and len(set(tr) & set(pg)) == 0
        assert len(tr) > prev_train  # expanding window
        prev_train = len(tr)


def test_walk_forward_oos_predictions_ignore_future_data() -> None:
    """Scrambling the last month (features, prices, OI -> labels) must not change any OOS
    prediction of earlier test months."""
    cfg = MLConfig(min_train_months=6, max_iter=15, min_samples_leaf=50)
    base = run_walk_forward(_synthetic_pool(), config=cfg)
    pert_frames = _synthetic_pool()
    for df in pert_frames.values():
        late = df["open_time"] >= pd.Timestamp("2024-09-01", tz="UTC")  # last test month
        df.loc[late, ["funding_z_30d", "oi_chg_24h", "ret_24h", "short_squeeze_score"]] = 7.0
        df.loc[late, ["close", "high", "low"]] *= 1.5
        df.loc[late, "oi"] *= 0.5
    pert = run_walk_forward(pert_frames, config=cfg)
    for target in ("squeeze_up", "squeeze_down"):
        a = base[target].oos_predictions
        b = pert[target].oos_predictions
        early_a = a[a["month"] < "2024-09"]
        early_b = b[b["month"] < "2024-09"]
        assert set(early_a["month"]) == {"2024-07", "2024-08"}
        np.testing.assert_array_equal(early_a["y_prob"].to_numpy(), early_b["y_prob"].to_numpy())


def test_latest_probabilities_train_only_on_purged_past() -> None:
    cfg = MLConfig(max_iter=15, min_samples_leaf=50)
    frames = _synthetic_pool(months=3)
    base = predict_latest_probabilities(frames, config=cfg)
    pert = _synthetic_pool(months=3)
    for df in pert.values():
        last = df["open_time"].max()
        recent = (df["open_time"] >= last - 24 * H) & (df["open_time"] < last)
        df.loc[recent, ["funding_z_30d", "oi_chg_24h", "ret_24h", "short_squeeze_score"]] = -9.0
    assert predict_latest_probabilities(pert, config=cfg) == base


# --------------------------------------------------------------------------------------
# Providers: real recorded JSON + simulated failures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as fh:
        return json.load(fh)


class _Resp:
    def __init__(self, status: int, payload: Any = None) -> None:
        self.status_code = status
        self._payload = payload
        self.headers: dict[str, str] = {}
        self.text = json.dumps(payload)[:200]

    def json(self) -> Any:
        return self._payload


class _Session:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[tuple[str, Any, Any]] = []

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append((url, params, json))
        return self.handler(url, params, json)


def _binance_handler(rec: dict[str, Any]):
    b = rec["binance"]

    def handle(url, params, body):
        for key, path in (
            ("klines", "/fapi/v1/klines"),
            ("fundingRate", "/fapi/v1/fundingRate"),
            ("openInterest", "/fapi/v1/openInterest"),
        ):
            if url.endswith(path):
                return _Resp(200, b[key])
        for ep in ("openInterestHist", "globalLongShortAccountRatio", "topLongShortPositionRatio",
                   "topLongShortAccountRatio", "takerlongshortRatio"):
            if url.endswith(f"/futures/data/{ep}"):
                return _Resp(200, b[ep])
        return _Resp(404, {"msg": "unknown"})

    return handle


def _hl_handler(rec: dict[str, Any]):
    h = rec["hyperliquid"]
    p1, p2 = h["fundingHistory_pages"]

    def handle(url, params, body):
        kind = body.get("type")
        if kind == "metaAndAssetCtxs":
            return _Resp(200, h["metaAndAssetCtxs"])
        if kind == "candleSnapshot":
            return _Resp(200, h["candleSnapshot"])
        if kind == "fundingHistory":
            if body["startTime"] == h["funding_start_ms"]:
                return _Resp(200, p1)
            if body["startTime"] == max(int(x["time"]) for x in p1) + 1:
                return _Resp(200, p2)
            return _Resp(200, [])
        return _Resp(400, {})

    return handle


def _client(handler, **cfg) -> HttpJsonClient:
    return HttpJsonClient(session=_Session(handler), config=ProviderConfig(**cfg), sleep_fn=lambda _: None)  # type: ignore[arg-type]


def test_binance_recorded_payload_parsing_and_alignment(recorded) -> None:
    now = recorded["recorded_at_ms"]
    b = recorded["binance"]
    ds = BinanceProvider(client=_client(_binance_handler(recorded))).fetch_symbol("BTCUSDT", limit=30, now_ms=now)
    al = ds.aligned
    now_ts = pd.to_datetime(now, unit="ms", utc=True)
    # only closed bars: the recorded payload ends with the still-open bar
    assert int(b["klines"][-1][6]) > now
    assert len(al) == len(b["klines"]) - 1
    assert (al["close_time"] <= now_ts).all()
    assert al["close"].iloc[-1] == float(b["klines"][-2][4])
    # snapshot series: row at timestamp T belongs to the bar that CLOSES at T (open T-1h)
    for item in b["openInterestHist"]:
        bar = pd.to_datetime(int(item["timestamp"]), unit="ms", utc=True) - H
        hit = al.loc[al["open_time"] == bar]
        if len(hit):
            assert hit["oi"].iloc[0] == float(item["sumOpenInterest"])
            assert hit["oi_value"].iloc[0] == float(item["sumOpenInterestValue"])
    for item in b["globalLongShortAccountRatio"]:
        bar = pd.to_datetime(int(item["timestamp"]), unit="ms", utc=True) - H
        hit = al.loc[al["open_time"] == bar]
        if len(hit):
            assert hit["global_ls_accounts"].iloc[0] == float(item["longShortRatio"])
    # taker rows aggregate [T, T+1h): they must equal the same bar's kline taker split
    k_share = al["taker_buy_base"] / al["volume"]
    k_ratio = k_share / (1 - k_share)
    ok = al["taker_buy_sell_ratio"].notna()
    assert ok.sum() >= 20
    np.testing.assert_allclose(al.loc[ok, "taker_buy_sell_ratio"], k_ratio[ok], rtol=2e-3)
    # funding: last settlement with funding_time <= bar close
    last_close = al["close_time"].iloc[-1]
    settled = [x for x in b["fundingRate"] if pd.to_datetime(x["fundingTime"], unit="ms", utc=True) <= last_close]
    assert al["funding_rate"].iloc[-1] == float(settled[-1]["fundingRate"])
    assert al["oi"].iloc[-1] > 0


def test_hyperliquid_recorded_payload_parsing(recorded) -> None:
    now = recorded["recorded_at_ms"]
    h = recorded["hyperliquid"]
    sess_handler = _hl_handler(recorded)
    client = _client(sess_handler)
    hp = HyperliquidProvider(client=client)
    ctx = hp.fetch_meta_and_asset_ctxs()
    assert set(ctx) == {"BTC", "ETH", "SOL"}
    assert ctx["BTC"]["openInterest"] == h["metaAndAssetCtxs"][1][0]["openInterest"]
    # funding: 2 pages (500 + 100 rows), converted from per-hour to 8h-equivalent
    fund = hp.fetch_funding_history("BTCUSDT", limit=600, now_ms=now)
    raw = h["fundingHistory_pages"][0] + h["fundingHistory_pages"][1]
    assert len(fund) == len(raw) == 600
    assert fund["funding_rate"].iloc[-1] == pytest.approx(float(raw[-1]["fundingRate"]) * HL_FUNDING_TO_8H)
    assert HL_FUNDING_TO_8H == 8.0
    starts = [c[2]["startTime"] for c in client.session.calls if c[2].get("type") == "fundingHistory"]
    assert starts == [h["funding_start_ms"], max(int(x["time"]) for x in h["fundingHistory_pages"][0]) + 1]
    ds = hp.fetch_symbol("BTCUSDT", limit=30, now_ms=now)
    al = ds.aligned
    assert (al["close_time"] <= pd.to_datetime(now, unit="ms", utc=True)).all()
    assert len(al) == len(h["candleSnapshot"]) - 1
    assert al["close"].iloc[-1] == float(h["candleSnapshot"][-2]["c"])
    # only the current OI snapshot exists -> only the last bar has OI; no liquidation map
    assert al["oi"].notna().sum() == 1
    assert al["oi"].iloc[-1] == float(h["metaAndAssetCtxs"][1][0]["openInterest"])
    lm = build_liquidation_map(compute_features(al))
    assert lm.price_profile.empty and lm.summary["nearest_short_liq_dist_pct"].isna().all()


def test_chain_fallbacks_451_timeout_partial(recorded) -> None:
    now = recorded["recorded_at_ms"]

    def blocked(url, params, body):
        return _Resp(451, {"code": 0, "msg": "Service unavailable from a restricted location"})

    def timeout(url, params, body):
        raise requests.Timeout("simulated timeout")

    class _HL(HyperliquidProvider):
        def fetch_symbol(self, symbol, limit=720, as_of=None, now_ms=None):  # pin recorded "now"
            return super().fetch_symbol(symbol, limit=30, as_of=as_of, now_ms=now)

    class _BN(BinanceProvider):
        def fetch_symbol(self, symbol, limit=720, as_of=None, now_ms=None):
            return super().fetch_symbol(symbol, limit=30, as_of=as_of, now_ms=now)

    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    hl_ok = _HL(client=_client(_hl_handler(recorded)))

    # 451 -> Hyperliquid, Binance tried only once (short-circuit, no per-symbol retries)
    bn_451 = _BN(client=_client(blocked))
    res = ProviderChain(binance=bn_451, hyperliquid=hl_ok, cache=TTLCache(0)).load_symbols(syms, mode="live")
    assert res.badge.source == "hyperliquid" and "451" in (res.badge.fallback_reason or "")
    assert set(res.datasets) == set(syms)
    assert len(bn_451.client.session.calls) == 1

    # timeouts -> retried max_retries times on ONE request, then skip to Hyperliquid
    bn_to = _BN(client=_client(timeout, max_retries=2))
    res = ProviderChain(binance=bn_to, hyperliquid=hl_ok, cache=TTLCache(0)).load_symbols(syms, mode="live")
    assert res.badge.source == "hyperliquid" and "timeout" in (res.badge.fallback_reason or "")
    assert len(bn_to.client.session.calls) == 3

    # both down -> bundled replay, with both reasons in the badge
    hl_to = _HL(client=_client(timeout, max_retries=0))
    res = ProviderChain(binance=_BN(client=_client(blocked)), hyperliquid=hl_to, cache=TTLCache(0)).load_symbols(
        ["BTCUSDT"], mode="live", limit=48
    )
    assert res.badge.source == "replay" and res.badge.mode == "replay"
    assert "Binance" in res.badge.fallback_reason and "Hyperliquid" in res.badge.fallback_reason
    assert len(res.datasets["BTCUSDT"].aligned) == 48

    # Binance reachable but one symbol rejected (HTTP 400) -> keep Binance, report the symbol
    good = _binance_handler(recorded)

    def partial(url, params, body):
        if params and params.get("symbol") == "FAKEUSDT":
            return _Resp(400, {"code": -1121, "msg": "Invalid symbol."})
        return good(url, params, body)

    res = ProviderChain(binance=_BN(client=_client(partial)), hyperliquid=hl_ok, cache=TTLCache(0)).load_symbols(
        ["BTCUSDT", "FAKEUSDT"], mode="live"
    )
    assert res.badge.source == "binance"
    assert set(res.datasets) == {"BTCUSDT"} and "FAKEUSDT" in res.errors


# --------------------------------------------------------------------------------------
# Evidence file
# --------------------------------------------------------------------------------------
def test_evidence_file_roundtrip_and_readme_numbers() -> None:
    bundle = load_evidence(DEFAULT_EVIDENCE_PATH)
    assert bundle is not None, "run: python -m squeezeradar evidence --build"
    assert DEFAULT_EVIDENCE_PATH.stat().st_size < 64_000
    meta = bundle.meta
    assert meta["symbols"] == sorted(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"])
    assert meta["data_start_utc"] == "2023-01-01 00:00" and meta["data_end_utc"] == "2025-10-31 23:00"
    # dict -> bundle -> dict is lossless
    again = evidence_to_dict(evidence_from_dict(evidence_to_dict(bundle)))
    assert again == evidence_to_dict(bundle)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in ("squeeze_up", "squeeze_down"):
        r = bundle.walk_forward[target]
        assert r.n_folds == 28 and r.n_oos > 100_000
        assert 0.5 < r.auc < 1.0 and 0.0 < r.base_rate < r.precision_top_5pct < 1.0
        assert r.lift_top_5pct == pytest.approx(r.precision_top_5pct / r.base_rate, rel=1e-4)
        row = next(line for line in readme.splitlines() if line.startswith(f"| `{target}`"))
        for txt in (
            f"{r.auc:.3f}",
            f"{r.base_rate * 100:.2f}%",
            f"{r.precision_top_5pct * 100:.2f}%",
            f"{r.lift_top_5pct:.2f}x",
            f"{r.brier_score:.4f}",
            f"{r.monthly_df['auc'].min():.3f} – {r.monthly_df['auc'].max():.3f}",
        ):
            assert txt in row, (target, txt, row)
    es = bundle.event_study.summary_df.set_index("signal")
    base = es.loc["Unconditional Baseline"]
    assert f"| {int(base['n_events']):,} | {base['mean_ret_24h_pct']:+.2f}% |" in readme
    sq = es.loc["Short Squeeze Score >= 50"]
    assert f"| Short Squeeze Score ≥ 50 | {int(sq['n_events'])} | {sq['mean_ret_24h_pct']:+.2f}% | {sq['squeeze_up_label_rate'] * 100:.1f}% |" in readme
    # README must not reference images that do not exist, nor placeholder links/text
    for ref in [w for w in readme.split("(") if w.startswith("docs/") and ".png" in w]:
        assert (ROOT / ref.split(")")[0]).exists(), ref
    assert "Placeholder" not in readme and "](#)" not in readme and "&lt;name&gt;" not in readme


def test_cli_evidence_prints_stored_results(capsys) -> None:
    from squeezeradar.cli import main

    assert main(["evidence"]) == 0
    out = capsys.readouterr().out
    r = load_evidence().walk_forward["squeeze_up"]
    assert f"OOS AUC {r.auc:.3f}" in out and "monthly folds" in out


# --------------------------------------------------------------------------------------
# AI analyst: tools + real SDK tool_runner over a mocked HTTP transport
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ai_ctx() -> AnalystContext:
    scan = run_radar_scan(symbols=["BTCUSDT", "ETHUSDT"], mode="replay", limit=2160)
    ev = load_evidence()
    return AnalystContext(
        scored_frames={s: a.scored_df for s, a in scan.analyses.items()},
        liq_maps={s: a.liq_map for s, a in scan.analyses.items()},
        as_of_label=scan.as_of_label,
        data_source_label=scan.badge.label,
        event_study=ev.event_study,
        walk_forward=ev.walk_forward,
        ml_probs=ev.latest_probs if ev.meta["latest_probs_as_of_utc"] == scan.as_of_label else {},
    )


def test_ai_tools_return_correct_compact_json(ai_ctx: AnalystContext) -> None:
    tools, captured = build_analyst_tools(ai_ctx)
    tm = {t.name: t for t in tools}
    btc = ai_ctx.scored_frames["BTCUSDT"].iloc[-1]
    calls = {
        "scan_market": tm["scan_market"].call({}),
        "get_symbol_snapshot": tm["get_symbol_snapshot"].call({"symbol": "btc"}),
        "get_liquidation_map": tm["get_liquidation_map"].call({"symbol": "BTCUSDT"}),
        "get_score_breakdown": tm["get_score_breakdown"].call({"symbol": "BTCUSDT"}),
        "get_event_study": tm["get_event_study"].call({"signal": "short"}),
        "get_model_evidence": tm["get_model_evidence"].call({}),
        "submit_brief": tm["submit_brief"].call(
            {"market_summary": "S", "setups": ["a", "b"], "risks": ["r"], "watch_levels": ["w"]}
        ),
    }
    parsed = {}
    for name, text in calls.items():
        assert isinstance(text, str) and len(text) < 3000, name
        parsed[name] = json.loads(text)
        assert "truncated" not in parsed[name], name
    rows = {r["sym"]: r for r in parsed["scan_market"]["symbols"]}
    assert rows["BTCUSDT"]["px"] == pytest.approx(btc["close"])
    assert rows["BTCUSDT"]["fund_bps"] == pytest.approx(btc["funding_rate_last"] * 1e4, abs=1e-4)
    assert rows["BTCUSDT"]["ml_up"] == pytest.approx(load_evidence().latest_probs["BTCUSDT"]["prob_squeeze_up"], abs=1e-4)
    snap = parsed["get_symbol_snapshot"]
    assert snap["symbol"] == "BTCUSDT"
    assert snap["oi_chg_24h_pct"] == pytest.approx(btc["oi_chg_24h"] * 100, abs=1e-4)
    assert snap["funding_rate_pct_8h"] == pytest.approx(btc["funding_rate_last"] * 100, abs=1e-4)
    lm = parsed["get_liquidation_map"]
    assert lm["is_estimate"] is True
    assert lm["nearest_short_cluster_dist_pct"] == pytest.approx(btc["nearest_short_liq_dist_pct"], abs=1e-4)
    assert all(c["pct"] > 0 for c in lm["top_short_clusters"]) and all(c["pct"] < 0 for c in lm["top_long_clusters"])
    assert parsed["get_score_breakdown"]["short_squeeze"]["score"] == pytest.approx(btc["short_squeeze_score"], abs=0.01)
    ev = load_evidence()
    assert parsed["get_model_evidence"]["squeeze_up"]["auc"] == pytest.approx(ev.walk_forward["squeeze_up"].auc, abs=1e-4)
    sig = {s["signal"]: s for s in parsed["get_event_study"]["signals"]}
    assert set(sig) == {"Short Squeeze Score >= 50", "Short Squeeze State (SETUP/TRIGGER)"}
    assert captured["setups"] == ["a", "b"]


def _msg(content: list[dict[str, Any]], stop: str) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def _factory(script: list[Any], seen: list[dict[str, Any]]):
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append({"url": str(request.url), "headers": dict(request.headers), "body": json.loads(request.content)})
        item = script[len(seen) - 1]
        if isinstance(item, Exception):
            raise item
        status, payload = item
        return httpx2.Response(status, json=payload)

    def make(api_key: str) -> anthropic.Anthropic:
        return anthropic.Anthropic(
            api_key=api_key, max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
        )

    return make


def test_real_sdk_tool_runner_request_shape_and_tool_loop(ai_ctx: AnalystContext, monkeypatch) -> None:
    monkeypatch.delenv("SR_MODEL", raising=False)
    seen: list[dict[str, Any]] = []
    script = [
        (200, _msg([
            {"type": "tool_use", "id": "toolu_1", "name": "scan_market", "input": {}},
            {"type": "tool_use", "id": "toolu_2", "name": "get_model_evidence", "input": {}},
        ], "tool_use")),
        (200, _msg([{"type": "tool_use", "id": "toolu_3", "name": "submit_brief", "input": {
            "market_summary": "BTC funding flat.", "setups": ["BTC above cluster"], "risks": ["estimate"],
            "watch_levels": ["$110k"]}}], "tool_use")),
        (200, _msg([{"type": "text", "text": "Brief submitted. Not financial advice."}], "end_turn")),
    ]
    brief = generate_analyst_brief(ai_ctx, api_key="sk-test", client_factory=_factory(script, seen))
    assert brief.mode == "claude" and brief.model == "claude-opus-5-5"
    assert [c["tool"] for c in brief.tool_calls] == ["scan_market", "get_model_evidence", "submit_brief"]
    assert brief.market_summary == "BTC funding flat." and brief.watch_levels == ["$110k"]
    assert len(seen) == 3
    first = seen[0]
    assert "/v1/messages" in first["url"]
    assert "server-side-fallback-2026-07-01" in first["headers"].get("anthropic-beta", "")
    body = first["body"]
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 16000
    assert body["output_config"] == {"effort": "high"} and body["fallbacks"] == "default"
    assert body["system"].startswith("You are the SqueezeRadar")
    assert [t["name"] for t in body["tools"]] == [
        "scan_market", "get_symbol_snapshot", "get_liquidation_map", "get_score_breakdown",
        "get_event_study", "get_model_evidence", "submit_brief",
    ]
    for banned in ("max_iterations", "temperature", "top_p", "thinking", "tool_choice"):
        assert banned not in body
    # second request carries both tool results (one user turn) with our real tool JSON
    results = [b for b in seen[1]["body"]["messages"][-1]["content"] if b["type"] == "tool_result"]
    assert [r["tool_use_id"] for r in results] == ["toolu_1", "toolu_2"]
    text = json.dumps(results)
    assert "ESTIMATE" in text and "auc" in text


@pytest.mark.parametrize(
    "item,fragment",
    [
        ((200, _msg([], "refusal")), "refusal"),
        ((529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}), "Anthropic API error"),
        ((500, {"type": "error", "error": {"type": "api_error", "message": "boom"}}), "InternalServerError"),
        ((401, {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}), "AuthenticationError"),
        (httpx2.ConnectError("offline"), "APIConnectionError"),
    ],
)
def test_runner_failures_fall_back_to_offline_brief(ai_ctx: AnalystContext, item, fragment) -> None:
    brief = generate_analyst_brief(ai_ctx, api_key="sk-test", client_factory=_factory([item], []))
    assert brief.mode == "offline"
    assert fragment.lower() in (brief.fallback_reason or "").lower()
    assert "not financial advice" in brief.narrative_markdown.lower()
    assert f"squeeze_up={ai_ctx.walk_forward['squeeze_up'].auc:.3f}" in brief.market_summary


def test_offline_brief_without_key(ai_ctx: AnalystContext, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    brief = generate_analyst_brief(ai_ctx, api_key=None, focus_symbol="ETHUSDT")
    assert brief.mode == "offline" and "ETHUSDT" in brief.market_summary
    eth = ai_ctx.scored_frames["ETHUSDT"].iloc[-1]
    assert f"{eth['funding_rate_last'] * 100:+.4f}%/8h" in " ".join(brief.setups)


# --------------------------------------------------------------------------------------
# Streamlit pages (AppTest): every page, main widgets, real content
# --------------------------------------------------------------------------------------
def _app(rel: str, timeout: float = 120.0) -> AppTest:
    return AppTest.from_file(str(ROOT / rel), default_timeout=timeout)


def _footer_ok(at: AppTest) -> None:
    foot = [m.value for m in at.markdown if "Crypto Quant Suite" in m.value]
    assert foot, "footer missing"
    for f in foot:
        assert "<a " not in f and "href" not in f
        for name in ("QuantCritic", "PropPilot", "SqueezeRadar", "RiskGuard"):
            assert name in f


def test_app_radar_and_deep_dive_pages() -> None:
    at = _app("streamlit_app.py")
    at.run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Contracts Scanned"] == "6"
    _footer_ok(at)

    at = _app("pages/1_Radar.py")
    at.run()
    at.selectbox(key="radar_focus_select").select("XRPUSDT").run()
    assert not at.exception and at.session_state["sr_focus_symbol"] == "XRPUSDT"

    at = _app("pages/2_Deep_Dive.py")
    at.run()
    at.selectbox(key="deep_dive_symbol").select("SOLUSDT").run()
    at.selectbox(key="deep_dive_window").select("72h (3d)").run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    sol_last = pd.read_csv(ROOT / "data/sample/SOLUSDT_1h.csv.gz")["close"].iloc[-1]
    assert metrics["Spot Price"] == f"${sol_last:,.2f}"
    assert len(at.get("plotly_chart")) >= 5
    assert not [w for w in at.warning if "open-interest history" in w.value]  # replay has a map
    _footer_ok(at)


def test_app_time_machine_replay_and_reveal_numbers() -> None:
    at = _app("pages/3_Time_Machine.py")
    at.run()
    at.selectbox(key="tm_preset_select").select_index(5).run()  # Oct 10 2025 preset
    assert at.date_input(key="tm_date").value == dt.date(2025, 10, 10)
    at.date_input(key="tm_date").set_value(dt.date(2024, 8, 5)).run()
    at.slider(key="tm_hour").set_value(0).run()
    at.selectbox(key="tm_focus_sym").select("BTCUSDT").run()
    assert not at.exception
    assert any("2024-08-05 00:00 UTC" in s.value for s in at.subheader)
    assert any("Snapshot: 2024-08-05 00:00 UTC" in c.value for c in at.sidebar.caption)
    assert not at.metric  # nothing about the future before the reveal
    at.checkbox(key="tm_reveal_toggle").check().run()
    assert not at.exception
    k = pd.read_csv(ROOT / "data/sample/BTCUSDT_1h.csv.gz", parse_dates=["open_time"]).set_index("open_time")
    entry = k.loc[pd.Timestamp("2024-08-04 23:00", tz="UTC"), "close"]
    post = k.loc[pd.Timestamp("2024-08-05 00:00", tz="UTC"): pd.Timestamp("2024-08-06 23:00", tz="UTC")]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["48h Close Return"] == f"{(post['close'].iloc[-1] / entry - 1) * 100:+.2f}%"
    assert metrics["Max Upside / Drawdown"] == (
        f"{(post['high'].max() / entry - 1) * 100:+.2f}% / {(post['low'].min() / entry - 1) * 100:+.2f}%"
    )
    _footer_ok(at)


def test_app_evidence_precomputed_then_recompute() -> None:
    ev = load_evidence()
    at = _app("pages/4_Evidence.py", timeout=240.0)
    at.run()
    assert not at.exception
    metrics = [(m.label, m.value) for m in at.metric]
    assert ("OOS ROC AUC", f"{ev.walk_forward['squeeze_up'].auc:.3f}") in metrics
    assert ("Lift vs Base Rate", f"{ev.walk_forward['squeeze_down'].lift_top_5pct:.2f}x") in metrics
    assert any("precomputed file" in m.value for m in at.markdown)
    # recompute live on a single contract
    at.session_state["sr_symbols"] = ["BTCUSDT"]
    at.run()
    at.button(key="ev_recompute_btn").click().run()
    assert not at.exception
    assert any("recomputed live in this session for BTCUSDT" in m.value for m in at.markdown)
    assert len([m for m in at.metric if m.label == "OOS ROC AUC"]) == 2
    _footer_ok(at)


def test_app_ai_analyst_offline_brief_and_about(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ev = load_evidence()
    at = _app("pages/5_AI_Analyst.py")
    at.run()
    at.selectbox(key="ai_focus_symbol").select("DOGEUSDT").run()
    at.button(key="ai_generate_btn").click().run()
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "Offline Deterministic Brief" in text and "DOGEUSDT" in text
    assert f"squeeze_up={ev.walk_forward['squeeze_up'].auc:.3f}" in text
    assert "not financial advice" in text.lower()
    _footer_ok(at)

    at = _app("pages/6_About.py")
    at.run()
    assert not at.exception
    assert any("Methodology" in t.value for t in at.title)
    _footer_ok(at)

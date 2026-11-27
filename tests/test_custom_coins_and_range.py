"""Tests for custom coin selection, symbol normalization, cached downloads, date ranges, and AppTest."""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from squeezeradar import cli, config as sr_config, data as sr_data, ui as sr_ui
from squeezeradar.data import (
    align_symbol_frames,
    discover_sample_symbols,
    get_replay_date_bounds,
    get_symbol_date_bounds,
    load_sample_symbol,
)
from squeezeradar.features import compute_features
from squeezeradar.liqmap import build_liquidation_map
from squeezeradar.providers.base import HttpJsonClient
from squeezeradar.providers.binance import (
    BinanceProvider,
    parse_symbol_list_text,
    resolve_binance_symbol,
    resolve_binance_symbols,
)
from squeezeradar.providers.chain import ProviderChain
from squeezeradar.providers.downloader import download_symbol_history
from squeezeradar.providers.replay import ReplayProvider
from squeezeradar.scores import compute_scores_and_states

ROOT = Path(__file__).resolve().parent.parent


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, *, content: bytes = b"") -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.headers: dict[str, str] = {}
        self.text = "" if payload is None else str(payload)

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _MockSession:
    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request(self, method: str, url: str, params=None, json=None, headers=None, timeout=None):
        self.calls.append((method, url, params))
        return self._handler(method, url, params=params, json_body=json)

    def get(self, url: str, timeout=None):
        self.calls.append(("GET", url, None))
        return self._handler("GET", url, params=None, json_body=None)

    def close(self) -> None:
        return None


def _zip_csv(name: str, body: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, body.encode("utf-8"))
    return buf.getvalue()


def _make_synthetic_vision_archive(
    symbol: str,
    start_date: str = "2025-08-01",
    days: int = 45,
) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, bytes]]:
    """Build realistic monthly klines/funding zips and daily 5-min metrics zips for a symbol."""
    start_ts = pd.Timestamp(f"{start_date} 00:00:00", tz="UTC")
    n_hours = days * 24
    hours = pd.date_range(start_ts, periods=n_hours, freq="1h")

    rng = np.random.default_rng(1234)
    rets = rng.normal(0.0, 0.008, n_hours)
    close = 15.0 * np.exp(np.cumsum(rets))
    open_p = np.roll(close, 1)
    open_p[0] = 15.0
    high_p = np.maximum(open_p, close) * (1.0 + np.abs(rng.normal(0.002, 0.002, n_hours)))
    low_p = np.minimum(open_p, close) * (1.0 - np.abs(rng.normal(0.002, 0.002, n_hours)))
    vol = rng.uniform(50_000, 150_000, n_hours)
    qvol = vol * close
    taker_buy = vol * rng.uniform(0.4, 0.6, n_hours)

    k_df = pd.DataFrame(
        {
            "open_time": (hours.astype("int64") // 10**6),
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close,
            "volume": vol,
            "close_time": ((hours + pd.Timedelta(hours=1)).astype("int64") // 10**6) - 1,
            "quote_volume": qvol,
            "trades": 1000,
            "taker_buy_base": taker_buy,
            "taker_buy_quote": taker_buy * close,
            "ignore": 0,
        }
    )
    k_df["ym"] = hours.strftime("%Y-%m")
    k_zips: dict[str, bytes] = {}
    for ym, grp in k_df.groupby("ym"):
        k_zips[ym] = _zip_csv(f"{symbol}-1h-{ym}.csv", grp.drop(columns=["ym"]).to_csv(index=False))

    f_times = pd.date_range(start_ts, periods=days * 3, freq="8h")
    f_df = pd.DataFrame(
        {
            "calc_time": (f_times.astype("int64") // 10**6),
            "funding_interval_hours": 8,
            "last_funding_rate": rng.normal(0.0001, 0.00015, len(f_times)),
        }
    )
    f_df["ym"] = f_times.strftime("%Y-%m")
    f_zips: dict[str, bytes] = {}
    for ym, grp in f_df.groupby("ym"):
        f_zips[ym] = _zip_csv(
            f"{symbol}-fundingRate-{ym}.csv", grp.drop(columns=["ym"]).to_csv(index=False)
        )

    # 5-minute metrics snapshots inside each UTC day
    m_times = pd.date_range(start_ts, periods=n_hours * 12, freq="5min")
    oi_series = 5_000_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(m_times))))
    px_interp = np.repeat(close, 12)
    m_df = pd.DataFrame(
        {
            "create_time": m_times.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "sum_open_interest": oi_series,
            "sum_open_interest_value": oi_series * px_interp,
            "count_toptrader_long_short_ratio": rng.uniform(0.9, 1.4, len(m_times)),
            "sum_toptrader_long_short_ratio": rng.uniform(0.9, 1.4, len(m_times)),
            "count_long_short_ratio": rng.uniform(0.85, 1.5, len(m_times)),
            "sum_taker_long_short_vol_ratio": rng.uniform(0.8, 1.3, len(m_times)),
        }
    )
    m_df["ds"] = m_times.strftime("%Y-%m-%d")
    m_zips: dict[str, bytes] = {}
    for ds, grp in m_df.groupby("ds"):
        m_zips[ds] = _zip_csv(
            f"{symbol}-metrics-{ds}.csv", grp.drop(columns=["ds"]).to_csv(index=False)
        )
    return k_zips, f_zips, m_zips


def _vision_handler(symbol: str, k_zips: dict[str, bytes], f_zips: dict[str, bytes], m_zips: dict[str, bytes]):
    def _handle(method: str, url: str, params=None, json_body=None):
        if f"/monthly/klines/{symbol}/1h/" in url:
            ym = url.rsplit("-", 2)[-2] + "-" + url.rsplit("-", 1)[-1].removesuffix(".zip")
            if ym in k_zips:
                return _FakeResponse(200, content=k_zips[ym])
        elif f"/monthly/fundingRate/{symbol}/" in url:
            ym = url.rsplit("-", 2)[-2] + "-" + url.rsplit("-", 1)[-1].removesuffix(".zip")
            if ym in f_zips:
                return _FakeResponse(200, content=f_zips[ym])
        elif f"/daily/metrics/{symbol}/" in url:
            tail = url.rsplit(f"{symbol}-metrics-", 1)[-1].removesuffix(".zip")
            if tail in m_zips:
                return _FakeResponse(200, content=m_zips[tail])
        return _FakeResponse(404)

    return _handle


# --------------------------------------------------------------------------------------
# 1. Symbol normalization & validation
# --------------------------------------------------------------------------------------
def test_symbol_normalization_and_validation_against_mocked_lists() -> None:
    """Validate 'pepe' -> 'PEPEUSDT' or '1000PEPEUSDT', unknown coin rejection, and ticker fallback."""
    tokens = parse_symbol_list_text("pepe, link, BTC-USDT, boguscoin")
    assert tokens == ["pepe", "link", "BTC-USDT", "boguscoin"]

    def _exchange_info_handler(method: str, url: str, params=None, json_body=None):
        if url.endswith("/fapi/v1/exchangeInfo"):
            return _FakeResponse(
                200,
                {
                    "symbols": [
                        {"symbol": "BTCUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"},
                        {"symbol": "1000PEPEUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"},
                        {"symbol": "LINKUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"},
                        {"symbol": "BTCUSDT_241227", "quoteAsset": "USDT", "contractType": "CURRENT_QUARTER", "status": "TRADING"},
                        {"symbol": "OLDUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "SETTLED"},
                    ]
                },
            )
        return _FakeResponse(404)

    bp = BinanceProvider(client=HttpJsonClient(session=_MockSession(_exchange_info_handler), sleep_fn=lambda _: None))  # type: ignore[arg-type]
    listed = bp.fetch_listed_symbols()
    assert listed == ["BTCUSDT", "1000PEPEUSDT", "LINKUSDT"]

    resolved, unknown = resolve_binance_symbols(tokens, listed)
    assert resolved == ["1000PEPEUSDT", "LINKUSDT", "BTCUSDT"]
    assert unknown == ["boguscoin"]

    # When PEPEUSDT itself is the listed contract, prefer PEPEUSDT
    assert resolve_binance_symbol("pepe", ["BTCUSDT", "PEPEUSDT"]) == "PEPEUSDT"
    assert resolve_binance_symbol("pepeusdt", ["BTCUSDT", "1000PEPEUSDT"]) == "1000PEPEUSDT"

    # Fallback to /fapi/v1/ticker/24hr when /fapi/v1/exchangeInfo is unavailable (404)
    def _ticker_only_handler(method: str, url: str, params=None, json_body=None):
        if url.endswith("/fapi/v1/ticker/24hr"):
            return _FakeResponse(
                200,
                [
                    {"symbol": "PEPEUSDT", "quoteVolume": "900000"},
                    {"symbol": "LINKUSDT", "quoteVolume": "800000"},
                    {"symbol": "ETHUSDT_250328", "quoteVolume": "500000"},
                ],
            )
        return _FakeResponse(404)

    bp_tick = BinanceProvider(client=HttpJsonClient(session=_MockSession(_ticker_only_handler), sleep_fn=lambda _: None))  # type: ignore[arg-type]
    listed_tick = bp_tick.fetch_listed_symbols()
    assert listed_tick == ["PEPEUSDT", "LINKUSDT"]
    assert resolve_binance_symbol("pepe", listed_tick) == "PEPEUSDT"
    assert resolve_binance_symbol("notreal", listed_tick) is None


# --------------------------------------------------------------------------------------
# 2. Download aggregation of daily 5m metrics & cache reuse
# --------------------------------------------------------------------------------------
def test_download_daily_5m_aggregation_and_cache_reuse(tmp_path: Path) -> None:
    """Mocked daily 5m metrics aggregate like bundled files and cached files are never re-downloaded."""
    sym = "LINKUSDT"
    k_csv = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,trades,taker_buy_base,taker_buy_quote,ignore\n"
        "1754006400000,15.0,15.5,14.8,15.2,1000,1754009999999,15200,100,600,9120,0\n"
        "1754010000000,15.2,15.6,15.1,15.4,1200,1754013599999,18480,120,700,10780,0\n"
    )
    f_csv = "calc_time,funding_interval_hours,last_funding_rate\n1754006400000,8,0.0001\n"
    # Hour 00: 3 snapshots inside [00:00, 01:00); Hour 01: 2 snapshots inside [01:00, 02:00)
    m_csv_day1 = (
        "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
        "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2025-08-01 00:05:00,LINKUSDT,1000,15000,1.10,1.20,1.30,0.80\n"
        "2025-08-01 00:30:00,LINKUSDT,1020,15300,1.12,1.22,1.32,1.00\n"
        "2025-08-01 00:55:00,LINKUSDT,1050,15960,1.15,1.25,1.35,1.20\n"
        "2025-08-01 01:05:00,LINKUSDT,1080,16632,1.18,1.28,1.40,1.40\n"
        "2025-08-01 01:55:00,LINKUSDT,1100,16940,1.20,1.30,1.42,1.60\n"
    )
    m_csv_day2 = (
        "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
        "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2025-08-02 00:05:00,LINKUSDT,1150,17710,1.22,1.32,1.45,1.10\n"
        "2025-08-02 00:55:00,LINKUSDT,1200,18480,1.25,1.35,1.50,1.30\n"
    )

    k_zips = {"2025-08": _zip_csv("k.csv", k_csv)}
    f_zips = {"2025-08": _zip_csv("f.csv", f_csv)}
    m_zips = {
        "2025-08-01": _zip_csv("m1.csv", m_csv_day1),
        "2025-08-02": _zip_csv("m2.csv", m_csv_day2),
    }

    sess = _MockSession(_vision_handler(sym, k_zips, f_zips, m_zips))
    progress_events: list[tuple[int, int, str]] = []
    written = download_symbol_history(
        sym,
        "2025-08-01",
        "2025-08-01",
        out_dir=tmp_path,
        session=sess,  # type: ignore[arg-type]
        progress_callback=lambda d, t, msg: progress_events.append((d, t, msg)),
    )
    assert set(written.keys()) == {"klines", "funding", "metrics"}
    assert len(progress_events) == 3
    assert progress_events[-1][0] == progress_events[-1][1] == 3

    m_out = pd.read_csv(written["metrics"])
    assert len(m_out) == 2
    # Row 0 (00:00): last OI/L-S from 00:55, mean taker ratio (0.8+1.0+1.2)/3 = 1.0, n_snapshots = 3
    assert m_out.loc[0, "oi"] == pytest.approx(1050.0)
    assert m_out.loc[0, "oi_value"] == pytest.approx(15960.0)
    assert m_out.loc[0, "global_ls_accounts"] == pytest.approx(1.35)
    assert m_out.loc[0, "taker_buy_sell_ratio"] == pytest.approx(1.0)
    assert int(m_out.loc[0, "n_snapshots"]) == 3
    # Row 1 (01:00): last OI from 01:55, mean taker ratio (1.4+1.6)/2 = 1.5, n_snapshots = 2
    assert m_out.loc[1, "oi"] == pytest.approx(1100.0)
    assert m_out.loc[1, "taker_buy_sell_ratio"] == pytest.approx(1.5)
    assert int(m_out.loc[1, "n_snapshots"]) == 2

    # Second download of same range must reuse cached files with ZERO HTTP requests
    calls_after_first = len(sess.calls)
    download_symbol_history(sym, "2025-08-01", "2025-08-01", out_dir=tmp_path, session=sess)  # type: ignore[arg-type]
    assert len(sess.calls) == calls_after_first

    # Expanding end date to 2025-08-02 must only fetch the new day's metrics file (no re-download of 08-01 or monthly zips)
    download_symbol_history(sym, "2025-08-01", "2025-08-02", out_dir=tmp_path, session=sess)  # type: ignore[arg-type]
    new_urls = [c[1] for c in sess.calls[calls_after_first:]]
    assert len(new_urls) == 1
    assert "LINKUSDT-metrics-2025-08-02.zip" in new_urls[0]
    m_expanded = pd.read_csv(written["metrics"])
    assert len(m_expanded) == 3
    assert m_expanded.loc[2, "oi"] == pytest.approx(1200.0)


# --------------------------------------------------------------------------------------
# 3. Date range bounds & causality truncation on a downloaded coin
# --------------------------------------------------------------------------------------
def test_range_bounds_and_causality_truncation_for_downloaded_coin(tmp_path: Path) -> None:
    """Verify date bounds, range slicing, and exact truncation causality for a downloaded coin."""
    sym = "LINKUSDT"
    cache_dir = tmp_path / "cache"
    k_zips, f_zips, m_zips = _make_synthetic_vision_archive(sym, start_date="2025-08-01", days=45)
    sess = _MockSession(_vision_handler(sym, k_zips, f_zips, m_zips))
    download_symbol_history(sym, "2025-08", "2025-09", out_dir=cache_dir, session=sess)  # type: ignore[arg-type]

    # Bounds & discovery
    avail = discover_sample_symbols(sample_dir=tmp_path / "empty_sample", cache_dir=cache_dir)
    assert avail == [sym]
    sym_bounds = get_symbol_date_bounds(sym, sample_dir=tmp_path / "empty_sample", cache_dir=cache_dir)
    assert sym_bounds == (dt.date(2025, 8, 1), dt.date(2025, 9, 14))
    assert get_replay_date_bounds([sym], sample_dir=tmp_path / "empty_sample", cache_dir=cache_dir) == sym_bounds

    # Date range slicing respects inclusive bounds and close_time causality
    ranged = load_sample_symbol(
        sym,
        sample_dir=tmp_path / "empty_sample",
        cache_dir=cache_dir,
        start_date="2025-08-10",
        end_date="2025-08-20",
    )
    assert ranged.aligned["open_time"].min() == pd.Timestamp("2025-08-10 00:00", tz="UTC")
    assert ranged.aligned["open_time"].max() == pd.Timestamp("2025-08-20 23:00", tz="UTC")
    assert (ranged.aligned["close_time"] <= pd.Timestamp("2025-08-21 00:00", tz="UTC")).all()

    # Truncation causality on the downloaded coin
    full_ds = load_sample_symbol(sym, sample_dir=tmp_path / "empty_sample", cache_dir=cache_dir)
    f_full = compute_features(full_ds.aligned)
    lm_full = build_liquidation_map(f_full, heatmap_bars=1)
    s_full = compute_scores_and_states(f_full, lm_full.summary)

    check_cols = [
        "funding_z_30d",
        "oi_chg_24h",
        "oi_price_div_24h",
        "nearest_short_liq_dist_pct",
        "nearest_long_liq_dist_pct",
        "short_squeeze_score",
        "long_squeeze_score",
    ]
    for t in (100, 400, 800, 1000):
        trunc_aligned = full_ds.aligned.iloc[: t + 1].copy()
        ft = compute_features(trunc_aligned)
        lmt = build_liquidation_map(ft, heatmap_bars=1)
        st_df = compute_scores_and_states(ft, lmt.summary)
        for col in check_cols:
            vf = s_full[col].iloc[t]
            vt = st_df[col].iloc[t]
            if np.isnan(vf):
                assert np.isnan(vt)
            else:
                assert vt == pytest.approx(vf, abs=1e-10)
        assert st_df["squeeze_state"].iloc[t] == s_full["squeeze_state"].iloc[t]

    # Raw as_of cutoff reproduces exact row
    cutoff_ds = load_sample_symbol(
        sym, sample_dir=tmp_path / "empty_sample", cache_dir=cache_dir, as_of="2025-08-20 12:00"
    )
    assert cutoff_ds.aligned["close_time"].max() == pd.Timestamp("2025-08-20 12:00", tz="UTC")
    idx_row = int(np.flatnonzero(s_full["open_time"] == pd.Timestamp("2025-08-20 11:00", tz="UTC"))[0])
    s_cut = compute_scores_and_states(
        compute_features(cutoff_ds.aligned),
        build_liquidation_map(compute_features(cutoff_ds.aligned), heatmap_bars=1).summary,
    )
    assert s_cut["short_squeeze_score"].iloc[-1] == pytest.approx(s_full["short_squeeze_score"].iloc[idx_row], abs=1e-10)


# --------------------------------------------------------------------------------------
# 4. CLI download (YYYY-MM), scan (live & downloaded replay), evidence (--symbols --start --end)
# --------------------------------------------------------------------------------------
def test_cli_download_scan_and_evidence_with_custom_coins_and_range(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Verify CLI download YYYY-MM, scan with downloaded/unknown coins, and evidence range filtering."""
    dl_dir = tmp_path / "downloaded"
    k_zips, f_zips, m_zips = _make_synthetic_vision_archive("LINKUSDT", start_date="2025-08-01", days=31)
    mock_sess = _MockSession(_vision_handler("LINKUSDT", k_zips, f_zips, m_zips))

    # 1) download --symbols LINKUSDT --start 2025-08 --end 2025-08
    monkeypatch.setattr(
        "squeezeradar.providers.downloader.requests.Session",
        lambda: mock_sess,
    )
    rc_dl = cli.main(
        ["download", "--symbols", "LINKUSDT", "--start", "2025-08", "--end", "2025-08", "--out-dir", str(dl_dir)]
    )
    assert rc_dl == 0
    assert (dl_dir / "LINKUSDT_1h.csv.gz").exists()

    # 2) scan --mode replay --symbols LINKUSDT (reads from patched cache dir)
    monkeypatch.setattr(sr_config, "DEFAULT_CACHE_DIR", dl_dir)
    monkeypatch.setattr(sr_data, "DEFAULT_CACHE_DIR", dl_dir)
    rc_scan_rep = cli.main(["scan", "--mode", "replay", "--symbols", "LINKUSDT"])
    assert rc_scan_rep == 0
    out_rep = capsys.readouterr().out
    assert "LINKUSDT" in out_rep

    # 3) scan --mode live with unknown coin -> warning on stderr, valid coin scanned, no crash
    monkeypatch.setattr(
        cli.BinanceProvider,
        "fetch_listed_symbols",
        lambda self: ["BTCUSDT", "ETHUSDT", "1000PEPEUSDT"],
    )

    orig_run_scan = cli.run_radar_scan

    def _fake_run_scan(*, symbols, mode="replay", as_of=None, limit=None, **kwargs):
        return orig_run_scan(
            symbols=["BTCUSDT"],
            mode="replay",
            as_of=as_of,
            limit=48,
        )

    monkeypatch.setattr(cli, "run_radar_scan", _fake_run_scan)
    rc_scan_live = cli.main(["scan", "--mode", "live", "--symbols", "BTCUSDT", "BOGUSCOIN"])
    assert rc_scan_live == 0
    captured = capsys.readouterr()
    assert "BOGUSCOIN" in captured.err
    assert "BTCUSDT" in captured.out

    # 4) evidence --symbols BTCUSDT --start 2024-01 --end 2024-08 (does not touch evidence_results.json)
    rc_ev = cli.main(["evidence", "--symbols", "BTCUSDT", "--start", "2024-01", "--end", "2024-08"])
    assert rc_ev == 0
    ev_out = capsys.readouterr().out
    assert "BTCUSDT" in ev_out
    assert "2024-01-01" in ev_out
    assert "2024-08-31" in ev_out
    assert "2 monthly folds" in ev_out


# --------------------------------------------------------------------------------------
# 5. Streamlit AppTest: live add coin + unknown coin + Top N; download + range across pages
# --------------------------------------------------------------------------------------
def _app(rel_path: str, timeout: float = 120.0) -> AppTest:
    return AppTest.from_file(str(ROOT / rel_path), default_timeout=timeout)


def test_apptest_live_add_coins_unknown_and_top_n(monkeypatch) -> None:
    """AppTest: live mode Add coins ('pepe' -> '1000PEPEUSDT'), unknown coin warning, and Top N."""
    st.cache_data.clear()
    now_ms = 1_722_816_000_000
    base_ms = now_ms - 36 * 3_600_000

    def _live_handler(method: str, url: str, params=None, json_body=None):
        if url.endswith("/fapi/v1/exchangeInfo"):
            return _FakeResponse(
                200,
                {
                    "symbols": [
                        {"symbol": s, "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"}
                        for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "1000PEPEUSDT", "SUIUSDT")
                    ]
                },
            )
        if url.endswith("/fapi/v1/ticker/24hr"):
            return _FakeResponse(
                200,
                [
                    {"symbol": "BTCUSDT", "quoteVolume": "9000000"},
                    {"symbol": "1000PEPEUSDT", "quoteVolume": "8000000"},
                    {"symbol": "SUIUSDT", "quoteVolume": "7000000"},
                ],
            )
        if url.endswith("/fapi/v1/klines"):
            rows = []
            px = 100.0 if params.get("symbol") != "1000PEPEUSDT" else 0.012
            for i in range(30):
                t0 = base_ms + i * 3_600_000
                rows.append(
                    [t0, f"{px}", f"{px*1.01}", f"{px*0.99}", f"{px*1.002}", "1000", t0 + 3_599_999, f"{1000*px}", 100, "550", f"{550*px}", "0"]
                )
            return _FakeResponse(200, rows)
        if url.endswith("/fapi/v1/fundingRate"):
            return _FakeResponse(200, [{"fundingTime": base_ms, "fundingRate": "-0.0001"}])
        if url.endswith("/fapi/v1/openInterest"):
            return _FakeResponse(200, {"symbol": params.get("symbol"), "openInterest": "50000", "time": base_ms + 29 * 3_600_000})
        if "/futures/data/openInterestHist" in url:
            return _FakeResponse(
                200,
                [
                    {"timestamp": base_ms + i * 3_600_000, "sumOpenInterest": str(45000 + i * 100), "sumOpenInterestValue": str((45000 + i * 100) * 100)}
                    for i in range(30)
                ],
            )
        if "/futures/data/globalLongShortAccountRatio" in url or "/futures/data/topLongShortPositionRatio" in url or "/futures/data/topLongShortAccountRatio" in url:
            return _FakeResponse(200, [{"timestamp": base_ms + i * 3_600_000, "longShortRatio": "1.05"} for i in range(30)])
        if "/futures/data/takerlongshortRatio" in url:
            return _FakeResponse(200, [{"timestamp": base_ms + i * 3_600_000, "buySellRatio": "1.15"} for i in range(30)])
        return _FakeResponse(404)

    mock_sess = _MockSession(_live_handler)
    monkeypatch.setattr("requests.Session", lambda: mock_sess)

    # Radar in Live Auto mode: add 'pepe' and 'boguscoin'
    at = _app("pages/1_Radar.py")
    at.run()
    at.radio(key="sr_mode_choice").set_value("Live Auto").run()
    assert not at.exception
    at.text_input(key="sr_add_coins").set_value("pepe, boguscoin").run()
    assert not at.exception
    assert any("boguscoin" in w.value for w in at.sidebar.warning)
    radar_df = at.dataframe[0].value
    assert "1000PEPEUSDT" in radar_df["Symbol"].tolist()

    # Enable Top N by volume (N=3 -> adds SUIUSDT as well)
    at.checkbox(key="sr_top_n_enabled").check().run()
    at.slider(key="sr_top_n_count").set_value(3).run()
    assert not at.exception
    radar_df = at.dataframe[0].value
    assert "SUIUSDT" in radar_df["Symbol"].tolist()

    # Open Deep Dive in Live Auto with 1000PEPEUSDT selected
    at_dd = _app("pages/2_Deep_Dive.py")
    at_dd.session_state["sr_mode_choice"] = "Live Auto"
    at_dd.session_state["sr_add_coins"] = "pepe"
    at_dd.run()
    assert not at_dd.exception
    at_dd.selectbox(key="deep_dive_symbol").select("1000PEPEUSDT").run()
    assert not at_dd.exception
    assert len(at_dd.metric) >= 6


def test_apptest_downloaded_coin_and_date_range_across_pages(tmp_path: Path, monkeypatch) -> None:
    """AppTest: download coin to temp cache via sidebar, set range, open Radar/Deep Dive/Time Machine/Evidence."""
    st.cache_data.clear()
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(sr_config, "DEFAULT_CACHE_DIR", cache_dir)
    monkeypatch.setattr(sr_data, "DEFAULT_CACHE_DIR", cache_dir)
    monkeypatch.setattr(sr_ui, "DEFAULT_CACHE_DIR", cache_dir)

    sym = "LINKUSDT"
    k_zips, f_zips, m_zips = _make_synthetic_vision_archive(sym, start_date="2025-08-01", days=61)
    vision_sess = _MockSession(_vision_handler(sym, k_zips, f_zips, m_zips))
    monkeypatch.setattr("squeezeradar.providers.downloader.requests.Session", lambda: vision_sess)
    monkeypatch.setattr(
        BinanceProvider,
        "fetch_listed_symbols",
        lambda self: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"],
    )

    # 1) Trigger download from the sidebar expander on Radar page
    at = _app("pages/1_Radar.py")
    at.run()
    assert not at.exception
    at.text_input(key="sr_dl_symbol").set_value("link").run()
    at.text_input(key="sr_dl_start").set_value("2025-08").run()
    at.text_input(key="sr_dl_end").set_value("2025-09").run()
    at.button(key="sr_dl_btn").click().run()
    assert not at.exception
    assert (cache_dir / "LINKUSDT_1h.csv.gz").exists()

    # Select only LINKUSDT and set custom date range 2025-08-10 .. 2025-09-20
    at.multiselect(key="sr_symbols").set_value(["LINKUSDT"]).run()
    at.date_input(key="sr_range_start").set_value(dt.date(2025, 8, 10)).run()
    at.date_input(key="sr_range_end").set_value(dt.date(2025, 9, 20)).run()
    assert not at.exception
    assert at.dataframe[0].value["Symbol"].tolist() == ["LINKUSDT"]

    # 2) Open Deep Dive with LINKUSDT + date range
    at_dd = _app("pages/2_Deep_Dive.py")
    at_dd.session_state["sr_symbols"] = ["LINKUSDT"]
    at_dd.session_state["sr_range_start"] = dt.date(2025, 8, 10)
    at_dd.session_state["sr_range_end"] = dt.date(2025, 9, 20)
    at_dd.session_state["sr_last_avail_bounds"] = (dt.date(2025, 8, 1), dt.date(2025, 9, 30))
    at_dd.run()
    assert not at_dd.exception
    assert len(at_dd.metric) >= 6
    assert any("2025-08-10" in c.value for c in at_dd.caption)

    # 3) Open Time Machine with LINKUSDT + date range (verify picker bounds & content)
    at_tm = _app("pages/3_Time_Machine.py")
    at_tm.session_state["sr_symbols"] = ["LINKUSDT"]
    at_tm.session_state["sr_range_start"] = dt.date(2025, 8, 10)
    at_tm.session_state["sr_range_end"] = dt.date(2025, 9, 20)
    at_tm.session_state["sr_last_avail_bounds"] = (dt.date(2025, 8, 1), dt.date(2025, 9, 30))
    at_tm.run()
    assert not at_tm.exception
    assert at_tm.date_input(key="tm_date").min == dt.date(2025, 8, 10)
    assert at_tm.date_input(key="tm_date").max == dt.date(2025, 9, 20)
    assert len(at_tm.dataframe) >= 1

    # 4) Open Evidence & Recompute now for LINKUSDT + range
    at_ev = _app("pages/4_Evidence.py", timeout=120.0)
    at_ev.session_state["sr_symbols"] = ["LINKUSDT"]
    at_ev.session_state["sr_range_start"] = dt.date(2025, 8, 10)
    at_ev.session_state["sr_range_end"] = dt.date(2025, 9, 20)
    at_ev.session_state["sr_last_avail_bounds"] = (dt.date(2025, 8, 1), dt.date(2025, 9, 30))
    at_ev.run()
    assert not at_ev.exception
    at_ev.button(key="ev_recompute_btn").click().run()
    assert not at_ev.exception
    assert any("recomputed live in this session for LINKUSDT" in m.value for m in at_ev.markdown)
    assert any("2025-08-10 → 2025-09-20" in m.value for m in at_ev.markdown)
    assert len(at_ev.dataframe) >= 1


def test_apptest_multiselect_accept_new_options_live_and_replay(tmp_path: Path, monkeypatch) -> None:
    """AppTest: typing new coins into Perpetual Contracts multiselect in Live and Replay modes."""
    st.cache_data.clear()
    now_ms = 1_722_816_000_000
    base_ms = now_ms - 36 * 3_600_000

    def _live_handler(method: str, url: str, params=None, json_body=None):
        if url.endswith("/fapi/v1/exchangeInfo"):
            return _FakeResponse(
                200,
                {
                    "symbols": [
                        {"symbol": s, "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"}
                        for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "1000PEPEUSDT", "SUIUSDT")
                    ]
                },
            )
        if url.endswith("/fapi/v1/klines"):
            px = 2.5 if params.get("symbol") == "SUIUSDT" else 0.012
            rows = [
                [
                    base_ms + i * 3_600_000,
                    f"{px}",
                    f"{px*1.01}",
                    f"{px*0.99}",
                    f"{px*1.002}",
                    "1000",
                    base_ms + i * 3_600_000 + 3_599_999,
                    f"{1000*px}",
                    100,
                    "550",
                    f"{550*px}",
                    "0",
                ]
                for i in range(30)
            ]
            return _FakeResponse(200, rows)
        if url.endswith("/fapi/v1/fundingRate"):
            return _FakeResponse(200, [{"fundingTime": base_ms, "fundingRate": "-0.0001"}])
        if url.endswith("/fapi/v1/openInterest"):
            return _FakeResponse(200, {"symbol": params.get("symbol"), "openInterest": "50000", "time": base_ms + 29 * 3_600_000})
        if "/futures/data/openInterestHist" in url:
            return _FakeResponse(
                200,
                [
                    {"timestamp": base_ms + i * 3_600_000, "sumOpenInterest": str(45000 + i * 100), "sumOpenInterestValue": str((45000 + i * 100) * 2.5)}
                    for i in range(30)
                ],
            )
        if (
            "/futures/data/globalLongShortAccountRatio" in url
            or "/futures/data/topLongShortPositionRatio" in url
            or "/futures/data/topLongShortAccountRatio" in url
        ):
            return _FakeResponse(200, [{"timestamp": base_ms + i * 3_600_000, "longShortRatio": "1.05"} for i in range(30)])
        if "/futures/data/takerlongshortRatio" in url:
            return _FakeResponse(200, [{"timestamp": base_ms + i * 3_600_000, "buySellRatio": "1.15"} for i in range(30)])
        return _FakeResponse(404)

    mock_sess = _MockSession(_live_handler)
    monkeypatch.setattr("requests.Session", lambda: mock_sess)

    # 1) Replay mode: typing an un-downloaded symbol removes it and points to "Download history for a coin"
    cache_dir = tmp_path / "empty_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sr_config, "DEFAULT_CACHE_DIR", cache_dir)
    monkeypatch.setattr(sr_data, "DEFAULT_CACHE_DIR", cache_dir)
    monkeypatch.setattr(sr_ui, "DEFAULT_CACHE_DIR", cache_dir)

    at = _app("pages/1_Radar.py")
    at.run()
    at.multiselect(key="sr_symbols").select("suiusdt").run()
    assert not at.exception
    assert "suiusdt" not in at.multiselect(key="sr_symbols").value
    assert "SUIUSDT" not in at.multiselect(key="sr_symbols").value
    assert any("Download history for a coin" in w.value for w in at.sidebar.warning)

    # Typing a bundled/downloaded coin abbreviation normalizes and scans it in Replay mode
    at.multiselect(key="sr_symbols").set_value(["btc"]).run()
    assert not at.exception
    assert at.multiselect(key="sr_symbols").value == ["BTCUSDT"]
    assert at.dataframe[0].value["Symbol"].tolist() == ["BTCUSDT"]

    # 2) Live Auto mode: typing 'suiusdt' & 'pepe' resolves and scans SUIUSDT & 1000PEPEUSDT; 'boguscoin' removed with warning
    at.radio(key="sr_mode_choice").set_value("Live Auto").run()
    assert not at.exception
    at.multiselect(key="sr_symbols").select("suiusdt").select("pepe").run()
    assert not at.exception
    ms_vals = at.multiselect(key="sr_symbols").value
    assert "SUIUSDT" in ms_vals and "1000PEPEUSDT" in ms_vals
    radar_syms = at.dataframe[0].value["Symbol"].tolist()
    assert "SUIUSDT" in radar_syms and "1000PEPEUSDT" in radar_syms

    at.multiselect(key="sr_symbols").select("boguscoin").run()
    assert not at.exception
    assert "boguscoin" not in at.multiselect(key="sr_symbols").value
    assert any("boguscoin" in w.value for w in at.sidebar.warning)

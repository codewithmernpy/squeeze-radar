"""Tests for market data providers, JSON fixture parsing, 429 retries, and fallback chain."""

from __future__ import annotations

import io
import zipfile
from typing import Any

import numpy as np
import requests

from squeezeradar.config import ProviderConfig
from squeezeradar.data import TTLCache
from squeezeradar.providers.base import HttpJsonClient
from squeezeradar.providers.binance import BinanceProvider
from squeezeradar.providers.chain import ProviderChain
from squeezeradar.providers.downloader import (
    _parse_funding_csv_bytes,
    _parse_klines_csv_bytes,
    _parse_metrics_csv_bytes,
    download_symbol_history,
)
from squeezeradar.providers.hyperliquid import HyperliquidProvider


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any = None,
        *,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.headers = headers or {}
        self.text = "" if payload is None else str(payload)

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _MockSession:
    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls: list[tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]] = []

    def request(self, method: str, url: str, params=None, json=None, headers=None, timeout=None):
        self.calls.append((method, url, params, json))
        return self._handler(method, url, params=params, json_body=json)

    def get(self, url: str, timeout=None):
        self.calls.append(("GET", url, None, None))
        return self._handler("GET", url, params=None, json_body=None)


def _binance_handler(now_ms: int):
    base_ms = now_ms - 3 * 3_600_000

    def _handle(method: str, url: str, params=None, json_body=None):
        if url.endswith("/fapi/v1/ticker/24hr"):
            return _FakeResponse(
                200,
                [
                    {"symbol": "ETHUSDT", "quoteVolume": "2000000"},
                    {"symbol": "BTCUSDT", "quoteVolume": "5000000"},
                    {"symbol": "BTCUSDT_241227", "quoteVolume": "9000000"},
                ],
            )
        if url.endswith("/fapi/v1/klines"):
            return _FakeResponse(
                200,
                [
                    [base_ms, "60000", "60500", "59800", "60200", "100", base_ms + 3_599_999, "6020000", 500, "55", "3311000", "0"],
                    [base_ms + 3_600_000, "60200", "61000", "60100", "60900", "120", base_ms + 7_199_999, "7308000", 620, "72", "4384800", "0"],
                    # Open bar (close_ms > now_ms) must be excluded
                    [now_ms - 1000, "60900", "61100", "60800", "61000", "10", now_ms + 3_598_000, "610000", 50, "6", "366000", "0"],
                ],
            )
        if url.endswith("/fapi/v1/fundingRate"):
            return _FakeResponse(
                200,
                [{"fundingTime": base_ms, "fundingRate": "-0.00015"}],
            )
        if url.endswith("/fapi/v1/premiumIndex"):
            return _FakeResponse(
                200,
                {"symbol": "BTCUSDT", "lastFundingRate": "-0.0002", "time": base_ms + 3_600_000},
            )
        if url.endswith("/fapi/v1/openInterest"):
            return _FakeResponse(
                200,
                {"symbol": "BTCUSDT", "openInterest": "85000", "time": base_ms + 3_600_000},
            )
        if "/futures/data/openInterestHist" in url:
            return _FakeResponse(
                200,
                [
                    {"timestamp": base_ms, "sumOpenInterest": "80000", "sumOpenInterestValue": "4816000000"},
                    {"timestamp": base_ms + 3_600_000, "sumOpenInterest": "85000", "sumOpenInterestValue": "5176500000"},
                ],
            )
        if "/futures/data/globalLongShortAccountRatio" in url:
            return _FakeResponse(
                200,
                [
                    {"timestamp": base_ms, "longShortRatio": "1.10"},
                    {"timestamp": base_ms + 3_600_000, "longShortRatio": "0.95"},
                ],
            )
        if "/futures/data/topLongShortPositionRatio" in url:
            return _FakeResponse(
                200,
                [
                    {"timestamp": base_ms, "longShortRatio": "1.20"},
                    {"timestamp": base_ms + 3_600_000, "longShortRatio": "1.05"},
                ],
            )
        if "/futures/data/topLongShortAccountRatio" in url:
            return _FakeResponse(
                200,
                [
                    {"timestamp": base_ms, "longShortRatio": "1.15"},
                    {"timestamp": base_ms + 3_600_000, "longShortRatio": "0.98"},
                ],
            )
        if "/futures/data/takerlongshortRatio" in url:
            return _FakeResponse(
                200,
                [
                    {"timestamp": base_ms, "buySellRatio": "1.05"},
                    {"timestamp": base_ms + 3_600_000, "buySellRatio": "1.35"},
                ],
            )
        return _FakeResponse(404, {"msg": "not found"})

    return _handle


def _hyperliquid_handler(now_ms: int):
    base_ms = now_ms - 3 * 3_600_000

    def _handle(method: str, url: str, params=None, json_body=None):
        req_type = (json_body or {}).get("type")
        if req_type == "metaAndAssetCtxs":
            return _FakeResponse(
                200,
                [
                    {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
                    [
                        {"funding": "-0.0001", "openInterest": "45000", "markPx": "60900", "dayNtlVlm": "1e9", "premium": "-0.0002"},
                        {"funding": "0.00005", "openInterest": "250000", "markPx": "3000", "dayNtlVlm": "5e8", "premium": "0.0001"},
                    ],
                ],
            )
        if req_type == "candleSnapshot":
            return _FakeResponse(
                200,
                [
                    {"t": base_ms, "T": base_ms + 3_599_999, "o": "60000", "h": "60500", "l": "59800", "c": "60200", "v": "100", "n": 500},
                    {"t": base_ms + 3_600_000, "T": base_ms + 7_199_999, "o": "60200", "h": "61000", "l": "60100", "c": "60900", "v": "120", "n": 620},
                ],
            )
        if req_type == "fundingHistory":
            return _FakeResponse(
                200,
                [{"time": base_ms, "fundingRate": "-0.0001"}],
            )
        return _FakeResponse(404, {})

    return _handle


def test_binance_provider_fixture_parsing() -> None:
    """Verify BinanceProvider parses REST JSON fixtures and excludes unclosed bars."""
    now_ms = 1_722_816_000_000
    client = HttpJsonClient(session=_MockSession(_binance_handler(now_ms)), sleep_fn=lambda _: None)  # type: ignore[arg-type]
    bp = BinanceProvider(client=client)

    ranked = bp.fetch_top_universe(top_n=2)
    assert ranked == ["BTCUSDT", "ETHUSDT"]

    ds = bp.fetch_symbol("BTCUSDT", now_ms=now_ms)
    assert ds.source == "binance"
    assert len(ds.aligned) == 2
    assert ds.aligned["oi_value"].iloc[-1] == 5_176_500_000.0
    assert ds.aligned["global_ls_accounts"].iloc[-1] == 0.95


def test_hyperliquid_provider_fixture_parsing() -> None:
    """Verify HyperliquidProvider parses /info JSON fixtures cleanly."""
    now_ms = 1_722_816_000_000
    client = HttpJsonClient(session=_MockSession(_hyperliquid_handler(now_ms)), sleep_fn=lambda _: None)  # type: ignore[arg-type]
    hp = HyperliquidProvider(client=client)

    ds = hp.fetch_symbol("BTCUSDT", now_ms=now_ms)
    assert ds.source == "hyperliquid"
    assert len(ds.aligned) == 2
    assert ds.aligned["oi"].iloc[-1] == 45000.0
    assert np.isnan(ds.aligned["global_ls_accounts"].iloc[-1])


def test_http_429_backoff_retry() -> None:
    """Verify exponential backoff retries on HTTP 429 and succeeds."""
    attempts = {"count": 0}
    sleeps: list[float] = []

    def _handler(method, url, params=None, json_body=None):
        attempts["count"] += 1
        if attempts["count"] < 3:
            return _FakeResponse(429, {"msg": "rate limit"})
        return _FakeResponse(200, {"ok": True})

    cfg = ProviderConfig(max_retries=3, backoff_base_sec=0.1)
    client = HttpJsonClient(session=_MockSession(_handler), config=cfg, sleep_fn=sleeps.append)  # type: ignore[arg-type]
    out = client.request_json("GET", "https://example.test/ping")
    assert out == {"ok": True}
    assert attempts["count"] == 3
    assert sleeps == [0.1, 0.2]


def test_provider_chain_fallback_451_and_timeout() -> None:
    """Verify Binance 451 -> Hyperliquid fallback, and Hyperliquid timeout -> Replay fallback."""
    now_ms = 1_722_816_000_000

    def _451_handler(method, url, params=None, json_body=None):
        return _FakeResponse(451, {"msg": "Restricted location"})

    def _timeout_handler(method, url, params=None, json_body=None):
        raise requests.Timeout("simulated timeout")

    binance_451 = BinanceProvider(
        client=HttpJsonClient(session=_MockSession(_451_handler), sleep_fn=lambda _: None)  # type: ignore[arg-type]
    )
    hl_ok = HyperliquidProvider(
        client=HttpJsonClient(session=_MockSession(_hyperliquid_handler(now_ms)), sleep_fn=lambda _: None)  # type: ignore[arg-type]
    )
    hl_timeout = HyperliquidProvider(
        client=HttpJsonClient(
            session=_MockSession(_timeout_handler),  # type: ignore[arg-type]
            config=ProviderConfig(max_retries=0),
            sleep_fn=lambda _: None,
        )
    )

    # 1) Binance 451 -> falls back to Hyperliquid
    chain_hl = ProviderChain(binance=binance_451, hyperliquid=hl_ok, cache=TTLCache(60))
    res_hl = chain_hl.load_symbols(["BTCUSDT"], mode="live")
    assert res_hl.badge.source == "hyperliquid"
    assert "451" in (res_hl.badge.fallback_reason or "")
    assert "BTCUSDT" in res_hl.datasets

    # 2) Binance 451 + Hyperliquid timeout -> falls back to Replay
    chain_rep = ProviderChain(binance=binance_451, hyperliquid=hl_timeout, cache=TTLCache(60))
    res_rep = chain_rep.load_symbols(["BTCUSDT"], mode="live", limit=100)
    assert res_rep.badge.source == "replay"
    assert "451" in (res_rep.badge.fallback_reason or "")
    assert len(res_rep.datasets["BTCUSDT"].aligned) == 100


def test_downloader_csv_variants_ms_us_and_headers(tmp_path) -> None:
    """Verify downloader parsers handle ms/us timestamps and header/headerless CSVs."""
    # Headerless klines with microseconds (2025 style on data.binance.vision)
    k_csv = b"1704067200000000,42000,42500,41900,42300,100,1704070799999000,4230000,250,55,2326500,0\n"
    k_df = _parse_klines_csv_bytes(k_csv)
    assert len(k_df) == 1
    assert k_df["open_time"].iloc[0].year == 2024

    # Headered funding CSV with milliseconds
    f_csv = b"calc_time,funding_interval_hours,last_funding_rate\n1704067200000,8,0.0001\n"
    f_df = _parse_funding_csv_bytes(f_csv)
    assert len(f_df) == 1
    assert f_df["funding_rate"].iloc[0] == 0.0001

    # Metrics CSV with two 5-min snapshots in same hour
    m_csv = (
        b"create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        b"2024-01-01 00:05:00,BTCUSDT,1000,42000000,1.1,1.2,1.3,0.9\n"
        b"2024-01-01 00:55:00,BTCUSDT,1050,44100000,1.2,1.3,1.4,1.1\n"
    )
    m_df = _parse_metrics_csv_bytes(m_csv)
    assert len(m_df) == 1
    assert m_df["oi"].iloc[0] == 1050.0
    assert abs(m_df["taker_buy_sell_ratio"].iloc[0] - 1.0) < 1e-9
    assert m_df["n_snapshots"].iloc[0] == 2

    # Mocked download_symbol_history writing gzipped outputs
    def _zip_bytes(name: str, body: bytes) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(name, body)
        return buf.getvalue()

    def _dl_handler(method, url, params=None, json_body=None):
        if "klines" in url:
            return _FakeResponse(200, content=_zip_bytes("k.csv", k_csv))
        if "fundingRate" in url:
            return _FakeResponse(200, content=_zip_bytes("f.csv", f_csv))
        if "metrics" in url:
            return _FakeResponse(200, content=_zip_bytes("m.csv", m_csv))
        return _FakeResponse(404)

    written = download_symbol_history(
        "BTCUSDT",
        "2024-01-01",
        "2024-01-01",
        out_dir=tmp_path,
        session=_MockSession(_dl_handler),  # type: ignore[arg-type]
    )
    assert set(written.keys()) == {"klines", "funding", "metrics"}
    assert all(p.exists() for p in written.values())

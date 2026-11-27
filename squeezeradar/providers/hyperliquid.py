"""Hyperliquid public /info market data fallback provider."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from squeezeradar.config import ProviderConfig
from squeezeradar.data import (
    FUNDING_COLUMNS,
    KLINE_COLUMNS,
    METRICS_COLUMNS,
    SymbolDataset,
    align_symbol_frames,
)
from squeezeradar.providers.base import HttpJsonClient, ProviderError

logger = logging.getLogger(__name__)

# Hyperliquid funding is paid hourly; x8 gives the 8h-equivalent rate used by the rest of the app.
HL_FUNDING_TO_8H = 8.0


def _to_hyperliquid_coin(symbol: str) -> str:
    """Convert canonical Binance-style symbol (e.g. BTCUSDT) to Hyperliquid coin name."""
    sym = symbol.upper().strip()
    for suffix in ("USDT", "USDC", "USD", "-PERP"):
        if sym.endswith(suffix) and len(sym) > len(suffix):
            return sym[: -len(suffix)]
    return sym


class HyperliquidProvider:
    """Fallback live provider using Hyperliquid's public POST /info API."""

    name = "hyperliquid"

    def __init__(
        self,
        client: HttpJsonClient | None = None,
        config: ProviderConfig | None = None,
    ) -> None:
        self.config = config or ProviderConfig()
        self.client = client or HttpJsonClient(config=self.config)
        self.url = self.config.hyperliquid_url

    def fetch_meta_and_asset_ctxs(self) -> dict[str, dict[str, Any]]:
        """Fetch current per-coin funding, OI, markPx, dayNtlVlm, and premium."""
        payload = self.client.request_json(
            "POST",
            self.url,
            json_body={"type": "metaAndAssetCtxs"},
        )
        if not isinstance(payload, list) or len(payload) < 2:
            raise ProviderError("Unexpected Hyperliquid metaAndAssetCtxs response")

        meta, ctxs = payload[0], payload[1]
        universe = meta.get("universe", []) if isinstance(meta, dict) else []
        if not isinstance(ctxs, list) or len(universe) != len(ctxs):
            raise ProviderError("Mismatched universe/assetCtxs lengths from Hyperliquid")

        mapping: dict[str, dict[str, Any]] = {}
        for asset_meta, ctx in zip(universe, ctxs):
            coin = str(asset_meta.get("name", "")).upper()
            if coin:
                mapping[coin] = dict(ctx)
        return mapping

    def fetch_candles(
        self,
        symbol: str,
        limit: int = 720,
        now_ms: int | None = None,
    ) -> pd.DataFrame:
        """Fetch 1h closed candles via candleSnapshot."""
        coin = _to_hyperliquid_coin(symbol)
        end_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        start_ms = end_ms - int(limit) * 3_600_000
        payload = self.client.request_json(
            "POST",
            self.url,
            json_body={
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": "1h",
                    "startTime": start_ms,
                    "endTime": end_ms,
                },
            },
        )
        if not isinstance(payload, list):
            raise ProviderError(f"Unexpected Hyperliquid candleSnapshot for {symbol}")

        rows: list[dict[str, Any]] = []
        for item in payload:
            open_ms = int(item["t"])
            close_ms = int(item.get("T", open_ms + 3_599_999))
            if close_ms > end_ms:
                continue
            o = float(item["o"])
            h = float(item["h"])
            l = float(item["l"])
            c = float(item["c"])
            v = float(item["v"])
            trades = float(item.get("n", 0))
            rows.append(
                {
                    "open_time": pd.to_datetime(open_ms, unit="ms", utc=True),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "quote_volume": v * c,
                    "trades": trades,
                    # Hyperliquid candleSnapshot lacks taker buy volume; leave NaN cleanly
                    "taker_buy_base": np.nan,
                }
            )
        return pd.DataFrame(rows, columns=list(KLINE_COLUMNS))

    def fetch_funding_history(
        self,
        symbol: str,
        limit: int = 720,
        now_ms: int | None = None,
    ) -> pd.DataFrame:
        """Fetch hourly funding history via fundingHistory (paginated, 500 rows per page).

        Hyperliquid settles funding every hour; `fundingRate` is a per-hour rate. It is
        converted to an 8h-equivalent rate (x8) so it has the same units as Binance 8h
        funding everywhere downstream (scores, z-scores, "%/8h" displays).
        """
        coin = _to_hyperliquid_coin(symbol)
        end_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        start_ms = end_ms - int(limit) * 3_600_000
        items: list[dict[str, Any]] = []
        cursor = start_ms
        for _ in range(max(1, int(limit) // 500 + 2)):
            payload = self.client.request_json(
                "POST",
                self.url,
                json_body={
                    "type": "fundingHistory",
                    "coin": coin,
                    "startTime": cursor,
                    "endTime": end_ms,
                },
            )
            if not isinstance(payload, list):
                raise ProviderError(f"Unexpected Hyperliquid fundingHistory for {symbol}")
            page = [it for it in payload if "time" in it and "fundingRate" in it]
            items.extend(page)
            if len(payload) < 500 or not page:
                break
            cursor = max(int(it["time"]) for it in page) + 1
            if cursor > end_ms:
                break
        rows = [
            {
                "funding_time": pd.to_datetime(int(item["time"]), unit="ms", utc=True),
                "funding_rate": float(item["fundingRate"]) * HL_FUNDING_TO_8H,
            }
            for item in items
            if int(item["time"]) <= end_ms
        ]
        return pd.DataFrame(rows, columns=list(FUNDING_COLUMNS))

    def fetch_symbol(
        self,
        symbol: str,
        limit: int = 720,
        as_of: str | None = None,
        now_ms: int | None = None,
    ) -> SymbolDataset:
        """Fetch and align candle, funding, and current OI snapshot from Hyperliquid."""
        sym = symbol.upper()
        coin = _to_hyperliquid_coin(sym)

        ctxs = self.fetch_meta_and_asset_ctxs()
        if coin not in ctxs:
            raise ProviderError(f"Symbol {sym} ({coin}) not found in Hyperliquid universe")
        ctx = ctxs[coin]

        klines = self.fetch_candles(sym, limit=limit, now_ms=now_ms)
        funding = self.fetch_funding_history(sym, limit=limit, now_ms=now_ms)

        # Hyperliquid exposes current OI snapshot in metaAndAssetCtxs (no historical OI endpoint)
        metrics_rows: list[dict[str, Any]] = []
        if not klines.empty:
            latest_open = klines["open_time"].max()
            oi_coins = float(ctx.get("openInterest", np.nan))
            mark_px = float(ctx.get("markPx", klines.loc[klines["open_time"] == latest_open, "close"].iloc[-1]))
            oi_usd = oi_coins * mark_px if np.isfinite(oi_coins) and np.isfinite(mark_px) else np.nan
            metrics_rows.append(
                {
                    "open_time": latest_open,
                    "oi": oi_coins,
                    "oi_value": oi_usd,
                    "top_trader_ls_accounts": np.nan,
                    "top_trader_ls_positions": np.nan,
                    "global_ls_accounts": np.nan,
                    "taker_buy_sell_ratio": np.nan,
                    "n_snapshots": 1,
                }
            )
        metrics = pd.DataFrame(metrics_rows, columns=list(METRICS_COLUMNS))
        aligned = align_symbol_frames(klines, funding, metrics, as_of=as_of)
        return SymbolDataset(
            symbol=sym,
            klines=klines,
            funding=funding,
            metrics=metrics,
            aligned=aligned,
            source=self.name,
        )

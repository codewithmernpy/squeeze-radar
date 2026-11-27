"""Binance USDT-M Futures public REST market data provider."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from squeezeradar.config import DEFAULT_SYMBOLS, ProviderConfig
from squeezeradar.data import (
    FUNDING_COLUMNS,
    KLINE_COLUMNS,
    METRICS_COLUMNS,
    SymbolDataset,
    align_symbol_frames,
)
from squeezeradar.providers.base import (
    HttpJsonClient,
    ProviderBlockedError,
    ProviderError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


def normalize_symbol_token(raw: str) -> str:
    """Normalize a user coin/symbol token into uppercase alphanumeric ticker text."""
    s = raw.strip().upper()
    if not s:
        return ""
    if s.endswith("-PERP"):
        s = s[:-5]
    if s.endswith("/USDT:USDT"):
        s = s[:-10] + "USDT"
    s = re.sub(r"[\s/_-]+", "", s)
    return s


def fallback_normalize_symbol(raw: str) -> str:
    """Normalize a user coin input when the live exchange directory is unavailable."""
    token = normalize_symbol_token(raw)
    if not token:
        return ""
    if token.endswith(("USDT", "USDC", "USD")):
        return token
    return f"{token}USDT"


def parse_symbol_list_text(text: str) -> list[str]:
    """Split a comma/whitespace-separated user symbol input into non-empty tokens."""
    return [tok.strip() for tok in re.split(r"[,;\s]+", text or "") if tok.strip()]


def resolve_binance_symbol(raw: str, listed_symbols: Iterable[str]) -> str | None:
    """Resolve a user coin input (e.g. 'pepe') against listed Binance USDT-M contracts."""
    token = normalize_symbol_token(raw)
    if not token:
        return None
    listed_set = {str(s).upper() for s in listed_symbols}
    candidates: list[str] = [token]
    if not token.endswith("USDT"):
        candidates.append(f"{token}USDT")
        if not token.startswith("1000"):
            candidates.append(f"1000{token}USDT")
        candidates.append(f"1000000{token}USDT")
        candidates.append(f"1M{token}USDT")
    else:
        base = token[:-4]
        if base and not base.startswith("1000"):
            candidates.append(f"1000{base}USDT")
    for cand in candidates:
        if cand in listed_set:
            return cand
    return None


def resolve_binance_symbols(
    inputs: Iterable[str],
    listed_symbols: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Resolve multiple coin inputs against listed symbols, returning `(resolved, unknown)`."""
    resolved: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    listed = list(listed_symbols)
    for item in inputs:
        tok = item.strip()
        if not tok:
            continue
        match = resolve_binance_symbol(tok, listed)
        if match is None:
            unknown.append(tok)
        elif match not in seen:
            seen.add(match)
            resolved.append(match)
    return resolved, unknown


class BinanceProvider:
    """Public REST provider for Binance USDT-M Perpetual Futures (no API keys)."""

    name = "binance"

    def __init__(
        self,
        client: HttpJsonClient | None = None,
        config: ProviderConfig | None = None,
    ) -> None:
        self.config = config or ProviderConfig()
        self.client = client or HttpJsonClient(config=self.config)
        self.base_url = self.config.binance_base_url.rstrip("/")

    def fetch_listed_symbols(self) -> list[str]:
        """Fetch active Binance USDT-M perpetual symbols via exchangeInfo or 24hr ticker."""
        payload: Any = None
        try:
            payload = self.client.request_json("GET", f"{self.base_url}/fapi/v1/exchangeInfo")
        except (ProviderBlockedError, ProviderUnavailableError):
            raise
        except ProviderError:
            payload = None

        if isinstance(payload, dict) and isinstance(payload.get("symbols"), list):
            syms: list[str] = []
            seen: set[str] = set()
            for item in payload["symbols"]:
                if not isinstance(item, dict):
                    continue
                sym = str(item.get("symbol", "")).upper()
                if not sym.endswith("USDT") or "_" in sym:
                    continue
                quote = item.get("quoteAsset")
                if quote is not None and str(quote).upper() != "USDT":
                    continue
                ctype = item.get("contractType")
                if ctype is not None and str(ctype).upper() != "PERPETUAL":
                    continue
                status = item.get("status")
                if status is not None and str(status).upper() != "TRADING":
                    continue
                if sym not in seen:
                    seen.add(sym)
                    syms.append(sym)
            if syms:
                return syms

        ticker_payload = self.client.request_json("GET", f"{self.base_url}/fapi/v1/ticker/24hr")
        if not isinstance(ticker_payload, list):
            raise ProviderError("Unexpected response format from Binance 24hr ticker")
        syms = []
        seen = set()
        for item in ticker_payload:
            sym = str(item.get("symbol", "")).upper()
            if not sym.endswith("USDT") or "_" in sym:
                continue
            if sym not in seen:
                seen.add(sym)
                syms.append(sym)
        return syms

    def fetch_top_universe(self, top_n: int | None = None) -> list[str]:
        """Fetch top N USDT-M perpetual symbols ranked by 24h quote volume."""
        n = top_n or self.config.top_n_universe
        payload = self.client.request_json("GET", f"{self.base_url}/fapi/v1/ticker/24hr")
        if not isinstance(payload, list):
            raise ProviderError("Unexpected response format from Binance 24hr ticker")

        candidates: list[tuple[str, float]] = []
        for item in payload:
            sym = str(item.get("symbol", ""))
            if not sym.endswith("USDT"):
                continue
            if "_" in sym:  # Skip quarterly delivery contracts (e.g. BTCUSDT_241227)
                continue
            try:
                qvol = float(item.get("quoteVolume", 0.0))
            except (TypeError, ValueError):
                qvol = 0.0
            candidates.append((sym, qvol))

        candidates.sort(key=lambda x: x[1], reverse=True)
        ranked = [sym for sym, _ in candidates[:n]]
        return ranked or list(DEFAULT_SYMBOLS[:n])

    def fetch_klines(self, symbol: str, limit: int = 720, now_ms: int | None = None) -> pd.DataFrame:
        """Fetch closed 1h klines from /fapi/v1/klines."""
        limit = max(1, min(int(limit), 1500))
        payload = self.client.request_json(
            "GET",
            f"{self.base_url}/fapi/v1/klines",
            params={"symbol": symbol.upper(), "interval": "1h", "limit": limit},
        )
        if not isinstance(payload, list):
            raise ProviderError(f"Unexpected klines response for {symbol}")
        if not payload:
            return pd.DataFrame(columns=list(KLINE_COLUMNS))

        current_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        rows: list[dict[str, Any]] = []
        for row in payload:
            open_ms = int(row[0])
            close_ms = int(row[6])
            # Strictly keep closed bars only (avoid lookahead / partial bar noise)
            if close_ms > current_ms:
                continue
            rows.append(
                {
                    "open_time": pd.to_datetime(open_ms, unit="ms", utc=True),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "quote_volume": float(row[7]),
                    "trades": float(row[8]),
                    "taker_buy_base": float(row[9]),
                }
            )
        return pd.DataFrame(rows, columns=list(KLINE_COLUMNS))

    def fetch_premium_index(self, symbol: str) -> dict[str, Any]:
        """Fetch current mark price, index price, and predicted funding from /fapi/v1/premiumIndex."""
        payload = self.client.request_json(
            "GET",
            f"{self.base_url}/fapi/v1/premiumIndex",
            params={"symbol": symbol.upper()},
        )
        if not isinstance(payload, dict):
            raise ProviderError(f"Unexpected premiumIndex response for {symbol}")
        return payload

    def fetch_funding_history(self, symbol: str, limit: int = 250) -> pd.DataFrame:
        """Fetch settled funding rates from /fapi/v1/fundingRate."""
        limit = max(1, min(int(limit), 1000))
        payload = self.client.request_json(
            "GET",
            f"{self.base_url}/fapi/v1/fundingRate",
            params={"symbol": symbol.upper(), "limit": limit},
        )
        if not isinstance(payload, list):
            raise ProviderError(f"Unexpected fundingRate response for {symbol}")
        rows = [
            {
                "funding_time": pd.to_datetime(int(item["fundingTime"]), unit="ms", utc=True),
                "funding_rate": float(item["fundingRate"]),
            }
            for item in payload
            if "fundingTime" in item and "fundingRate" in item
        ]
        return pd.DataFrame(rows, columns=list(FUNDING_COLUMNS))

    def fetch_open_interest(self, symbol: str) -> tuple[pd.Timestamp, float]:
        """Fetch latest open interest snapshot from /fapi/v1/openInterest."""
        payload = self.client.request_json(
            "GET",
            f"{self.base_url}/fapi/v1/openInterest",
            params={"symbol": symbol.upper()},
        )
        if not isinstance(payload, dict) or "openInterest" not in payload:
            raise ProviderError(f"Unexpected openInterest response for {symbol}")
        ts_ms = int(payload.get("time", int(time.time() * 1000)))
        return pd.to_datetime(ts_ms, unit="ms", utc=True), float(payload["openInterest"])

    def _fetch_futures_data_series(
        self,
        endpoint: str,
        symbol: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch 1h history from Binance /futures/data/* (max 30 days / 500 rows)."""
        lim = max(1, min(int(limit), 500))
        payload = self.client.request_json(
            "GET",
            f"{self.base_url}/futures/data/{endpoint}",
            params={"symbol": symbol.upper(), "period": "1h", "limit": lim},
        )
        if not isinstance(payload, list):
            raise ProviderError(f"Unexpected {endpoint} response for {symbol}")
        return payload

    def fetch_metrics_1h(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """Assemble 1h OI and positioning metrics from Binance /futures/data/*."""
        oi_rows = self._fetch_futures_data_series("openInterestHist", symbol, limit=limit)
        global_ls_rows = self._fetch_futures_data_series(
            "globalLongShortAccountRatio", symbol, limit=limit
        )
        top_pos_rows = self._fetch_futures_data_series(
            "topLongShortPositionRatio", symbol, limit=limit
        )
        taker_rows = self._fetch_futures_data_series(
            "takerlongshortRatio", symbol, limit=limit
        )

        # Optional top trader account ratio (same family, graceful if omitted)
        try:
            top_acc_rows = self._fetch_futures_data_series(
                "topLongShortAccountRatio", symbol, limit=limit
            )
        except ProviderError:
            top_acc_rows = []

        merged: dict[int, dict[str, Any]] = {}

        def _hour_key(ts_ms: int) -> int:
            return (int(ts_ms) // 3_600_000) * 3_600_000

        def _snapshot_bar_key(ts_ms: int) -> int:
            # OI and long/short ratio rows (period=1h) are point-in-time snapshots taken AT
            # `timestamp` T (identical to the 5m snapshot at T). A snapshot at T is the state at the
            # close of the bar [T-1h, T), so it belongs to bar open_time T-1h -- the same meaning as
            # the bundled metrics (last snapshot inside [h, h+1h), known at h+1h).
            return _hour_key(ts_ms) - 3_600_000

        for item in oi_rows:
            k = _snapshot_bar_key(int(item["timestamp"]))
            slot = merged.setdefault(k, {"open_time": pd.to_datetime(k, unit="ms", utc=True)})
            slot["oi"] = float(item["sumOpenInterest"])
            slot["oi_value"] = float(item["sumOpenInterestValue"])
            slot["n_snapshots"] = 1

        for item in global_ls_rows:
            k = _snapshot_bar_key(int(item["timestamp"]))
            slot = merged.setdefault(k, {"open_time": pd.to_datetime(k, unit="ms", utc=True)})
            slot["global_ls_accounts"] = float(item["longShortRatio"])

        for item in top_pos_rows:
            k = _snapshot_bar_key(int(item["timestamp"]))
            slot = merged.setdefault(k, {"open_time": pd.to_datetime(k, unit="ms", utc=True)})
            slot["top_trader_ls_positions"] = float(item["longShortRatio"])

        for item in top_acc_rows:
            k = _snapshot_bar_key(int(item["timestamp"]))
            slot = merged.setdefault(k, {"open_time": pd.to_datetime(k, unit="ms", utc=True)})
            slot["top_trader_ls_accounts"] = float(item["longShortRatio"])

        for item in taker_rows:
            # Taker buy/sell volume rows aggregate the period [T, T+1h) (buyVol equals the kline
            # taker-buy volume of the bar opening at T), so they map to bar open_time T.
            k = _hour_key(int(item["timestamp"]))
            slot = merged.setdefault(k, {"open_time": pd.to_datetime(k, unit="ms", utc=True)})
            slot["taker_buy_sell_ratio"] = float(item["buySellRatio"])

        if not merged:
            return pd.DataFrame(columns=list(METRICS_COLUMNS))

        df = pd.DataFrame(list(merged.values()))
        for col in METRICS_COLUMNS:
            if col not in df.columns:
                df[col] = 0 if col == "n_snapshots" else np.nan
        return df[list(METRICS_COLUMNS)].sort_values("open_time").reset_index(drop=True)

    def fetch_symbol(
        self,
        symbol: str,
        limit: int = 720,
        as_of: str | None = None,
        now_ms: int | None = None,
    ) -> SymbolDataset:
        """Fetch and causally align full dataset for one perpetual symbol."""
        sym = symbol.upper()
        klines = self.fetch_klines(sym, limit=limit, now_ms=now_ms)
        funding = self.fetch_funding_history(sym, limit=max(100, limit // 4))
        metrics = self.fetch_metrics_1h(sym, limit=min(500, limit))

        # If the hourly snapshot closing the latest bar is not published yet, use the current
        # /fapi/v1/openInterest value, but only when it was taken at/after that bar's close.
        try:
            oi_ts, oi_qty = self.fetch_open_interest(sym)
            if not klines.empty and not metrics.empty:
                latest_k = klines["open_time"].max()
                at_latest = metrics["open_time"] == latest_k
                has_latest_oi = bool((at_latest & metrics["oi_value"].notna()).any())
                if not has_latest_oi and oi_ts >= latest_k + pd.Timedelta(hours=1):
                    latest_close = float(klines.loc[klines["open_time"] == latest_k, "close"].iloc[-1])
                    if bool(at_latest.any()):
                        metrics.loc[at_latest, "oi"] = oi_qty
                        metrics.loc[at_latest, "oi_value"] = oi_qty * latest_close
                        metrics.loc[at_latest, "n_snapshots"] = 1
                    else:
                        extra_m = pd.DataFrame(
                            [
                                {
                                    "open_time": latest_k,
                                    "oi": oi_qty,
                                    "oi_value": oi_qty * latest_close,
                                    "top_trader_ls_accounts": np.nan,
                                    "top_trader_ls_positions": np.nan,
                                    "global_ls_accounts": np.nan,
                                    "taker_buy_sell_ratio": np.nan,
                                    "n_snapshots": 1,
                                }
                            ]
                        )
                        metrics = pd.concat([metrics, extra_m], ignore_index=True)
        except ProviderError as exc:
            logger.debug("Optional openInterest fetch skipped for %s: %s", sym, exc)

        aligned = align_symbol_frames(klines, funding, metrics, as_of=as_of)
        return SymbolDataset(
            symbol=sym,
            klines=klines,
            funding=funding,
            metrics=metrics,
            aligned=aligned,
            source=self.name,
        )

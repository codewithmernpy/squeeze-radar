"""Provider chain orchestrator with automatic fallback and TTL caching."""

from __future__ import annotations

import logging
from pathlib import Path

from squeezeradar.config import DEFAULT_SAMPLE_DIR, DEFAULT_SYMBOLS, DataModeLiteral, ProviderConfig
from squeezeradar.data import SymbolDataset, TTLCache
from squeezeradar.providers.base import (
    ProviderBadge,
    ProviderBlockedError,
    ProviderResult,
    ProviderUnavailableError,
)
from squeezeradar.providers.binance import BinanceProvider
from squeezeradar.providers.hyperliquid import HyperliquidProvider
from squeezeradar.providers.replay import ReplayProvider

logger = logging.getLogger(__name__)


class ProviderChain:
    """Market data loader with Binance -> Hyperliquid -> Replay fallback and TTL cache."""

    def __init__(
        self,
        *,
        binance: BinanceProvider | None = None,
        hyperliquid: HyperliquidProvider | None = None,
        replay: ReplayProvider | None = None,
        sample_dir: Path | str = DEFAULT_SAMPLE_DIR,
        config: ProviderConfig | None = None,
        cache: TTLCache | None = None,
    ) -> None:
        self.config = config or ProviderConfig()
        self.binance = binance or BinanceProvider(config=self.config)
        self.hyperliquid = hyperliquid or HyperliquidProvider(config=self.config)
        self.replay = replay or ReplayProvider(sample_dir=sample_dir)
        self.cache = cache or TTLCache(ttl_sec=self.config.cache_ttl_sec)

    def load_symbols(
        self,
        symbols: list[str] | tuple[str, ...] | None = None,
        *,
        mode: DataModeLiteral = "live",
        as_of: str | None = None,
        limit: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        use_cache: bool = True,
    ) -> ProviderResult:
        """Load datasets using either Replay mode or Live Auto fallback chain."""
        sym_list = [s.upper() for s in (symbols or DEFAULT_SYMBOLS)]
        eff_limit = limit if limit is not None else (
            None if mode == "replay" else self.config.history_bars
        )
        cache_key = (
            f"{mode}|{','.join(sym_list)}|{as_of or ''}|{start_date or ''}|{end_date or ''}|{eff_limit or 'all'}"
        )
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        if mode == "replay":
            result = self._load_from_replay(
                sym_list,
                as_of=as_of,
                limit=eff_limit,
                start_date=start_date,
                end_date=end_date,
                reason=None,
            )
        else:
            result = self._load_live_with_fallback(sym_list, as_of=as_of, limit=eff_limit)

        if use_cache:
            self.cache.set(cache_key, result)
        return result

    def _load_from_replay(
        self,
        symbols: list[str],
        *,
        as_of: str | None,
        limit: int | None,
        start_date: str | None = None,
        end_date: str | None = None,
        reason: str | None,
    ) -> ProviderResult:
        datasets: dict[str, SymbolDataset] = {}
        errors: dict[str, str] = {}
        for sym in symbols:
            try:
                datasets[sym] = self.replay.fetch_symbol(
                    sym,
                    limit=limit,
                    as_of=as_of,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as exc:
                errors[sym] = str(exc)

        label = "REPLAY (Offline)" if reason is None else "REPLAY (Fallback)"
        span_note = (
            f" ({start_date or '...'} → {end_date or '...'})"
            if (start_date or end_date)
            else ""
        )
        detail = (
            f"Local sample/cached replay{f' up to {as_of}' if as_of else span_note}"
            if reason is None
            else f"Fell back to local replay data ({reason})"
        )
        badge = ProviderBadge(
            source="replay",
            mode="replay",
            label=label,
            detail=detail,
            fallback_reason=reason,
        )
        return ProviderResult(datasets=datasets, badge=badge, errors=errors)

    def _fetch_all(
        self,
        provider: BinanceProvider | HyperliquidProvider,
        symbols: list[str],
        limit: int | None,
    ) -> tuple[dict[str, SymbolDataset], dict[str, str], bool]:
        """Fetch every symbol from one live provider.

        Returns `(datasets, errors_by_symbol, provider_down)`. A geo-block (HTTP 451/403) or an
        unreachable endpoint (timeouts/connection errors after retries) marks the provider as down
        and skips the remaining symbols, so the chain falls back quickly instead of re-trying
        every symbol against a dead endpoint.
        """
        datasets: dict[str, SymbolDataset] = {}
        errors: dict[str, str] = {}
        for sym in symbols:
            try:
                ds = provider.fetch_symbol(sym, limit=limit or self.config.history_bars)
                if ds.aligned.empty:
                    errors[sym] = f"{provider.name} returned no closed bars"
                else:
                    datasets[sym] = ds
            except (ProviderBlockedError, ProviderUnavailableError) as exc:
                logger.warning("%s unavailable (%s); skipping remaining symbols", provider.name, exc)
                errors[sym] = str(exc)
                return datasets, errors, True
            except Exception as exc:
                logger.warning("%s fetch failed for %s: %s", provider.name, sym, exc)
                errors[sym] = str(exc)
        return datasets, errors, False

    def _load_live_with_fallback(
        self,
        symbols: list[str],
        *,
        as_of: str | None,
        limit: int | None,
    ) -> ProviderResult:
        if as_of is not None:
            return self._load_from_replay(
                symbols,
                as_of=as_of,
                limit=limit,
                reason="Historical as_of requested -> using offline replay archive",
            )

        # 1) Try Binance USDT-M REST
        binance_data, binance_errs, binance_down = self._fetch_all(self.binance, symbols, limit)

        if binance_data and (len(binance_data) == len(symbols) or not binance_down):
            # Every symbol loaded, or Binance is reachable and only some symbols failed
            # (e.g. not listed): keep Binance and report the per-symbol errors.
            return ProviderResult(
                datasets=binance_data,
                badge=ProviderBadge(
                    source="binance",
                    mode="live",
                    label="LIVE · Binance USDT-M",
                    detail="Real-time Binance USDT-M public REST",
                ),
                errors=binance_errs,
            )

        reason_binance = (
            "{}: {}".format(*next(iter(binance_errs.items())))
            if binance_errs
            else "Binance returned empty data"
        )

        # 2) Fallback to Hyperliquid /info
        hl_data, hl_errs, _ = self._fetch_all(self.hyperliquid, symbols, limit)

        if hl_data and len(hl_data) >= max(1, len(symbols) // 2):
            return ProviderResult(
                datasets=hl_data,
                badge=ProviderBadge(
                    source="hyperliquid",
                    mode="live",
                    label="LIVE · Hyperliquid Fallback",
                    detail=f"Fallback after Binance error ({reason_binance})",
                    fallback_reason=reason_binance,
                ),
                errors={s: hl_errs.get(s, "No Hyperliquid data") for s in symbols if s not in hl_data},
            )

        # 3) Final fallback to bundled Replay
        reason_hl = (
            "{}: {}".format(*next(iter(hl_errs.items())))
            if hl_errs
            else "Hyperliquid returned empty data"
        )
        combined_reason = f"Binance: {reason_binance}; Hyperliquid: {reason_hl}"
        return self._load_from_replay(
            symbols,
            as_of=None,
            limit=limit,
            reason=combined_reason,
        )

"""Offline replay provider backed by bundled sample CSV.GZ files."""

from __future__ import annotations

from pathlib import Path

from squeezeradar.config import DEFAULT_SAMPLE_DIR
from squeezeradar.data import (
    SymbolDataset,
    discover_sample_symbols,
    load_sample_symbol,
)


class ReplayProvider:
    """Offline provider reading bundled and cached historical CSV.GZ snapshots."""

    name = "replay"

    def __init__(
        self,
        sample_dir: Path | str = DEFAULT_SAMPLE_DIR,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.sample_dir = Path(sample_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def available_symbols(self) -> list[str]:
        """List symbols present in the replay and cached data directories."""
        return discover_sample_symbols(self.sample_dir, cache_dir=self.cache_dir)

    def fetch_symbol(
        self,
        symbol: str,
        limit: int | None = None,
        as_of: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> SymbolDataset:
        """Load bundled/cached history up to `as_of` (or `[start_date, end_date]`) and optionally tail `limit` bars."""
        return load_sample_symbol(
            symbol,
            sample_dir=self.sample_dir,
            as_of=as_of,
            tail_bars=limit,
            start_date=start_date,
            end_date=end_date,
            cache_dir=self.cache_dir,
        )

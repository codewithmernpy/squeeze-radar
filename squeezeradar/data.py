"""Data loading, normalization, and strict causal alignment for SqueezeRadar."""

from __future__ import annotations

import calendar
import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from squeezeradar.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DOWNLOADED_DIR,
    DEFAULT_SAMPLE_DIR,
    DEFAULT_SYMBOLS,
)

logger = logging.getLogger(__name__)

KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
)

FUNDING_COLUMNS = ("funding_time", "funding_rate")

METRICS_VALUE_COLUMNS = (
    "oi",
    "oi_value",
    "top_trader_ls_accounts",
    "top_trader_ls_positions",
    "global_ls_accounts",
    "taker_buy_sell_ratio",
)

METRICS_COLUMNS = ("open_time", *METRICS_VALUE_COLUMNS, "n_snapshots")


@dataclass(frozen=True)
class SymbolDataset:
    """Canonical raw and causally aligned dataset for one perpetual contract."""

    symbol: str
    klines: pd.DataFrame
    funding: pd.DataFrame
    metrics: pd.DataFrame
    aligned: pd.DataFrame
    source: str = "replay"


class TTLCache:
    """Thread-safe in-memory TTL cache for library calls."""

    def __init__(self, ttl_sec: float = 60.0) -> None:
        self.ttl_sec = ttl_sec
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Return cached value if present and unexpired, else None."""
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if now >= expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_sec: float | None = None) -> None:
        """Store value in cache with explicit or default TTL."""
        ttl = self.ttl_sec if ttl_sec is None else ttl_sec
        expires_at = time.monotonic() + max(0.0, ttl)
        with self._lock:
            self._store[key] = (expires_at, value)

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._store.clear()


def _parse_utc_series(series: pd.Series) -> pd.Series:
    """Parse timestamps (ISO strings or epoch ms/us/s) into canonical UTC ns Timestamps."""
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        median_abs = float(vals.dropna().abs().median()) if not vals.dropna().empty else 0.0
        if median_abs > 1e14:
            unit = "us"
        elif median_abs > 1e11:
            unit = "ms"
        else:
            unit = "s"
        parsed = pd.to_datetime(vals, unit=unit, utc=True)
        return parsed.dt.as_unit("ns")
    try:
        parsed = pd.to_datetime(series, format="ISO8601", utc=True)
    except (ValueError, TypeError):
        parsed = pd.to_datetime(series, utc=True)
    return parsed.dt.as_unit("ns")


def parse_timestamp_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    """Parse a user/CLI datetime string or Timestamp into UTC."""
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def parse_range_start_utc(value: str | date | pd.Timestamp) -> pd.Timestamp:
    """Parse a range start (`YYYY-MM`, `YYYY-MM-DD`, date, or timestamp) as inclusive UTC start."""
    if isinstance(value, str):
        s = value.strip()
        if len(s) == 7 and s[4] == "-":
            return pd.Timestamp(f"{s}-01 00:00:00", tz="UTC")
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return pd.Timestamp(f"{s} 00:00:00", tz="UTC")
        return parse_timestamp_utc(s)
    if isinstance(value, date) and not isinstance(value, datetime):
        return pd.Timestamp(value, tz="UTC")
    return parse_timestamp_utc(value)


def parse_range_end_cutoff_utc(value: str | date | pd.Timestamp) -> pd.Timestamp:
    """Parse a range end (`YYYY-MM`, `YYYY-MM-DD`, date, or timestamp) as close_time cutoff."""
    if isinstance(value, str):
        s = value.strip()
        if len(s) == 7 and s[4] == "-":
            yr, mo = map(int, s.split("-"))
            last_d = date(yr, mo, calendar.monthrange(yr, mo)[1])
            return pd.Timestamp(last_d, tz="UTC") + pd.Timedelta(days=1)
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return pd.Timestamp(s, tz="UTC") + pd.Timedelta(days=1)
        return parse_timestamp_utc(s)
    if isinstance(value, date) and not isinstance(value, datetime):
        return pd.Timestamp(value, tz="UTC") + pd.Timedelta(days=1)
    return parse_timestamp_utc(value)


def normalize_klines_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize kline dataframe types, columns, sorting, and deduplication."""
    if df.empty:
        return pd.DataFrame(columns=list(KLINE_COLUMNS))
    out = df.copy()
    out["open_time"] = _parse_utc_series(out["open_time"])
    for col in KLINE_COLUMNS[1:]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        else:
            out[col] = np.nan
    out = (
        out[list(KLINE_COLUMNS)]
        .dropna(subset=["open_time"])
        .drop_duplicates(subset=["open_time"], keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    return out


def normalize_funding_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize funding dataframe types, columns, sorting, and deduplication."""
    if df.empty:
        return pd.DataFrame(columns=list(FUNDING_COLUMNS))
    out = df.copy()
    out["funding_time"] = _parse_utc_series(out["funding_time"])
    out["funding_rate"] = pd.to_numeric(out["funding_rate"], errors="coerce").astype(float)
    out = (
        out[list(FUNDING_COLUMNS)]
        .dropna(subset=["funding_time"])
        .drop_duplicates(subset=["funding_time"], keep="last")
        .sort_values("funding_time")
        .reset_index(drop=True)
    )
    return out


def normalize_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize 1h metrics dataframe types, columns, sorting, and deduplication.

    Open interest and long/short / taker ratios are strictly positive for a live perpetual; the
    Binance metrics archive contains occasional 0 values (recording glitches, e.g. 30 hours of
    BTCUSDT), which would read as a -100% OI crash. Non-positive values are treated as missing.
    """
    if df.empty:
        return pd.DataFrame(columns=list(METRICS_COLUMNS))
    out = df.copy()
    out["open_time"] = _parse_utc_series(out["open_time"])
    for col in METRICS_VALUE_COLUMNS:
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce").astype(float)
            out[col] = vals.where(vals > 0.0)
        else:
            out[col] = np.nan
    if "n_snapshots" in out.columns:
        out["n_snapshots"] = pd.to_numeric(out["n_snapshots"], errors="coerce").fillna(0).astype(int)
    else:
        out["n_snapshots"] = 0
    out = (
        out[list(METRICS_COLUMNS)]
        .dropna(subset=["open_time"])
        .drop_duplicates(subset=["open_time"], keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    return out


def align_symbol_frames(
    klines: pd.DataFrame,
    funding: pd.DataFrame,
    metrics: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
    max_metrics_ffill_bars: int = 2,
) -> pd.DataFrame:
    """Align klines, funding, and metrics causally at bar close (open_time + 1h).

    Causality guarantees:
    - Bar open_time=h represents [h, h+1h), known at close_time = h + 1h.
    - If `as_of` is given, only bars whose `close_time <= as_of` are included,
      and only funding settlements with `funding_time <= as_of` are visible.
    - Metrics row `open_time=h` aggregates 5m snapshots inside `[h, h+1h)`,
      known at bar close `h+1h`. Missing hours are forward-filled at most
      `max_metrics_ffill_bars` (default 2) bars, otherwise left as NaN.
    - Funding rate at bar `t` uses backward `merge_asof` on `close_time` vs
      `funding_time` (only settlements at or before bar close `t`).
    """
    k_df = normalize_klines_df(klines)
    f_df = normalize_funding_df(funding)
    m_df = normalize_metrics_df(metrics)

    if k_df.empty:
        return pd.DataFrame(
            columns=[
                *KLINE_COLUMNS,
                "close_time",
                "funding_rate",
                "funding_time",
                *METRICS_VALUE_COLUMNS,
                "n_snapshots",
            ]
        )

    k_df["close_time"] = (k_df["open_time"] + pd.Timedelta(hours=1)).dt.as_unit("ns")
    if as_of is not None:
        cutoff = parse_timestamp_utc(as_of)
        k_df = k_df[k_df["close_time"] <= cutoff].reset_index(drop=True)
        if not f_df.empty:
            f_df = f_df[f_df["funding_time"] <= cutoff].reset_index(drop=True)
        if not m_df.empty:
            m_df = m_df[m_df["open_time"] + pd.Timedelta(hours=1) <= cutoff].reset_index(drop=True)

    if k_df.empty:
        return pd.DataFrame(
            columns=[
                *KLINE_COLUMNS,
                "close_time",
                "funding_rate",
                "funding_time",
                *METRICS_VALUE_COLUMNS,
                "n_snapshots",
            ]
        )

    # Align metrics onto exact hourly bar open_time and causally ffill up to limit=2
    if m_df.empty:
        merged = k_df.copy()
        for col in METRICS_VALUE_COLUMNS:
            merged[col] = np.nan
        merged["n_snapshots"] = 0
    else:
        # Build regular hourly grid from earliest kline/metric up to max bar open_time
        # so multi-hour missing blocks in klines don't falsely stretch ffill limits.
        min_t = min(k_df["open_time"].min(), m_df["open_time"].min())
        max_t = k_df["open_time"].max()
        hourly_idx = pd.date_range(start=min_t, end=max_t, freq="1h", tz="UTC").as_unit("ns")
        m_grid = pd.DataFrame({"open_time": hourly_idx}).merge(m_df, on="open_time", how="left")
        m_grid["n_snapshots"] = m_grid["n_snapshots"].fillna(0).astype(int)
        for col in METRICS_VALUE_COLUMNS:
            m_grid[col] = m_grid[col].ffill(limit=max_metrics_ffill_bars)
        merged = k_df.merge(m_grid, on="open_time", how="left")

    # Align funding settlements causally at close_time = open_time + 1h
    if f_df.empty:
        merged["funding_time"] = pd.NaT
        merged["funding_rate"] = np.nan
    else:
        merged = pd.merge_asof(
            merged.sort_values("close_time"),
            f_df.sort_values("funding_time"),
            left_on="close_time",
            right_on="funding_time",
            direction="backward",
            tolerance=pd.Timedelta(hours=24),
        )

    # Ensure taker_buy_base sanity & return sorted frame
    merged = merged.sort_values("open_time").reset_index(drop=True)
    return merged


def _resolve_replay_dirs(
    sample_dir: Path | str = DEFAULT_SAMPLE_DIR,
    cache_dir: Path | str | None = None,
) -> list[Path]:
    """Return replay directories in priority order (bundled sample + local cache/downloads)."""
    primary = Path(sample_dir)
    dirs = [primary]
    if cache_dir is not None:
        extra = Path(cache_dir)
        if extra not in dirs:
            dirs.append(extra)
    elif primary.resolve() == DEFAULT_SAMPLE_DIR.resolve():
        for extra in (DEFAULT_CACHE_DIR, DEFAULT_DOWNLOADED_DIR):
            if extra not in dirs:
                dirs.append(extra)
    return dirs


def discover_sample_symbols(
    sample_dir: Path | str = DEFAULT_SAMPLE_DIR,
    cache_dir: Path | str | None = None,
) -> list[str]:
    """List symbols available in the bundled sample and cached replay directories."""
    dirs = [d for d in _resolve_replay_dirs(sample_dir, cache_dir=cache_dir) if d.exists()]
    if not dirs:
        return []
    found: set[str] = set()
    for directory in dirs:
        for path in directory.glob("*_1h.csv.gz"):
            if path.name.endswith("_metrics_1h.csv.gz"):
                continue
            found.add(path.name.removesuffix("_1h.csv.gz").upper())

    available: list[str] = [sym for sym in DEFAULT_SYMBOLS if sym in found]
    for sym in sorted(found):
        if sym not in available:
            available.append(sym)
    return available


def load_sample_symbol_raw(
    symbol: str,
    sample_dir: Path | str = DEFAULT_SAMPLE_DIR,
    cache_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw klines, funding, and metrics CSV.GZ files for a symbol."""
    sym = symbol.upper()
    dirs = _resolve_replay_dirs(sample_dir, cache_dir=cache_dir)
    matching_dirs = [d for d in dirs if (d / f"{sym}_1h.csv.gz").exists()]
    if not matching_dirs:
        raise FileNotFoundError(f"Sample klines file not found: {Path(sample_dir) / f'{sym}_1h.csv.gz'}")

    k_parts: list[pd.DataFrame] = []
    f_parts: list[pd.DataFrame] = []
    m_parts: list[pd.DataFrame] = []
    for directory in matching_dirs:
        k_path = directory / f"{sym}_1h.csv.gz"
        f_path = directory / f"{sym}_funding.csv.gz"
        m_path = directory / f"{sym}_metrics_1h.csv.gz"
        k_parts.append(pd.read_csv(k_path))
        if f_path.exists():
            f_parts.append(pd.read_csv(f_path))
        if m_path.exists():
            m_parts.append(pd.read_csv(m_path))

    klines = pd.concat(k_parts, ignore_index=True) if len(k_parts) > 1 else k_parts[0]
    funding = (
        (pd.concat(f_parts, ignore_index=True) if len(f_parts) > 1 else f_parts[0])
        if f_parts
        else pd.DataFrame(columns=list(FUNDING_COLUMNS))
    )
    metrics = (
        (pd.concat(m_parts, ignore_index=True) if len(m_parts) > 1 else m_parts[0])
        if m_parts
        else pd.DataFrame(columns=list(METRICS_COLUMNS))
    )
    return (
        normalize_klines_df(klines),
        normalize_funding_df(funding),
        normalize_metrics_df(metrics),
    )


def get_symbol_date_bounds(
    symbol: str,
    sample_dir: Path | str = DEFAULT_SAMPLE_DIR,
    cache_dir: Path | str | None = None,
) -> tuple[date, date] | None:
    """Return available `(min_date, max_date)` UTC for a symbol in replay/cache dirs."""
    sym = symbol.upper()
    dirs = _resolve_replay_dirs(sample_dir, cache_dir=cache_dir)
    min_ts: pd.Timestamp | None = None
    max_ts: pd.Timestamp | None = None
    for directory in dirs:
        k_path = directory / f"{sym}_1h.csv.gz"
        if not k_path.exists():
            continue
        df = pd.read_csv(k_path, usecols=["open_time"])
        if df.empty:
            continue
        ts = _parse_utc_series(df["open_time"]).dropna()
        if ts.empty:
            continue
        lo, hi = ts.min(), ts.max()
        min_ts = lo if min_ts is None or lo < min_ts else min_ts
        max_ts = hi if max_ts is None or hi > max_ts else max_ts
    if min_ts is None or max_ts is None:
        return None
    return min_ts.date(), max_ts.date()


def get_replay_date_bounds(
    symbols: list[str] | tuple[str, ...] | None = None,
    sample_dir: Path | str = DEFAULT_SAMPLE_DIR,
    cache_dir: Path | str | None = None,
) -> tuple[date, date] | None:
    """Return `(min_date, max_date)` UTC spanned across the selected replay symbols."""
    target = list(symbols) if symbols else discover_sample_symbols(sample_dir, cache_dir=cache_dir)
    bounds = [
        b
        for sym in target
        if (b := get_symbol_date_bounds(sym, sample_dir=sample_dir, cache_dir=cache_dir)) is not None
    ]
    if not bounds:
        return None
    return min(b[0] for b in bounds), max(b[1] for b in bounds)


def load_sample_symbol(
    symbol: str,
    sample_dir: Path | str = DEFAULT_SAMPLE_DIR,
    as_of: str | pd.Timestamp | None = None,
    tail_bars: int | None = None,
    start_date: str | date | pd.Timestamp | None = None,
    end_date: str | date | pd.Timestamp | None = None,
    cache_dir: Path | str | None = None,
) -> SymbolDataset:
    """Load and causally align a bundled sample or cached replay symbol dataset."""
    sym = symbol.upper()
    klines, funding, metrics = load_sample_symbol_raw(sym, sample_dir=sample_dir, cache_dir=cache_dir)

    eff_as_of: str | pd.Timestamp | None = as_of
    if eff_as_of is None and end_date is not None:
        eff_as_of = parse_range_end_cutoff_utc(end_date)

    if eff_as_of is not None:
        cutoff = parse_timestamp_utc(eff_as_of)
        if not klines.empty:
            klines = klines[klines["open_time"] + pd.Timedelta(hours=1) <= cutoff].reset_index(drop=True)
        if not funding.empty:
            funding = funding[funding["funding_time"] <= cutoff].reset_index(drop=True)
        if not metrics.empty:
            metrics = metrics[metrics["open_time"] + pd.Timedelta(hours=1) <= cutoff].reset_index(drop=True)

    if tail_bars is not None and tail_bars > 0 and len(klines) > tail_bars + 48:
        klines = klines.iloc[-(tail_bars + 48) :].reset_index(drop=True)
        min_t = klines["open_time"].iloc[0] - pd.Timedelta(hours=48)
        if not funding.empty:
            funding = funding[funding["funding_time"] >= min_t].reset_index(drop=True)
        if not metrics.empty:
            metrics = metrics[metrics["open_time"] >= min_t].reset_index(drop=True)

    aligned = align_symbol_frames(klines, funding, metrics, as_of=eff_as_of)
    if start_date is not None and not aligned.empty:
        start_ts = parse_range_start_utc(start_date)
        aligned = aligned[aligned["open_time"] >= start_ts].reset_index(drop=True)
        klines = klines[klines["open_time"] >= start_ts].reset_index(drop=True)

    if tail_bars is not None and tail_bars > 0 and len(aligned) > tail_bars:
        aligned = aligned.iloc[-tail_bars:].reset_index(drop=True)

    return SymbolDataset(
        symbol=sym,
        klines=klines,
        funding=funding,
        metrics=metrics,
        aligned=aligned,
        source="replay",
    )


def load_all_sample_symbols(
    symbols: list[str] | tuple[str, ...] | None = None,
    sample_dir: Path | str = DEFAULT_SAMPLE_DIR,
    as_of: str | pd.Timestamp | None = None,
    start_date: str | date | pd.Timestamp | None = None,
    end_date: str | date | pd.Timestamp | None = None,
    cache_dir: Path | str | None = None,
) -> dict[str, SymbolDataset]:
    """Load and align multiple bundled or cached replay symbols."""
    target = list(symbols) if symbols else discover_sample_symbols(sample_dir, cache_dir=cache_dir)
    return {
        sym: load_sample_symbol(
            sym,
            sample_dir=sample_dir,
            as_of=as_of,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
        )
        for sym in target
    }

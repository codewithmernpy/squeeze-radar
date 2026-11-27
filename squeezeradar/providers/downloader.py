"""Historical data downloader CLI for https://data.binance.vision (futures/um).

Supports daily 5m/1h metrics archives and monthly 1h klines & fundingRate archives,
automatically normalizing millisecond vs microsecond timestamps and CSVs with or
without header rows.
"""

from __future__ import annotations

import argparse
import calendar
import gzip
import io
import logging
import threading
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from squeezeradar.config import DEFAULT_CACHE_DIR, DEFAULT_SYMBOLS, ProviderConfig
from squeezeradar.data import (
    FUNDING_COLUMNS,
    KLINE_COLUMNS,
    METRICS_COLUMNS,
    normalize_funding_df,
    normalize_klines_df,
    normalize_metrics_df,
)

logger = logging.getLogger(__name__)


def _has_csv_header(first_line: str) -> bool:
    """Detect whether a CSV first line is a header (contains alphabetic text)."""
    first_token = first_line.strip().split(",")[0].strip().strip('"')
    return not first_token.replace("-", "").replace(".", "").isdigit()


def _parse_klines_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    """Parse Binance Vision klines CSV bytes (header or headerless, ms or us)."""
    text = raw_bytes.decode("utf-8", errors="replace").splitlines()
    if not text:
        return pd.DataFrame(columns=list(KLINE_COLUMNS))
    has_header = _has_csv_header(text[0])
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        header=0 if has_header else None,
        names=None if has_header else cols,
    )
    rename_map = {
        "count": "trades",
        "taker_buy_volume": "taker_buy_base",
    }
    df = df.rename(columns=rename_map)
    return normalize_klines_df(df)


def _parse_funding_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    """Parse Binance Vision fundingRate CSV bytes (header or headerless, ms or us)."""
    text = raw_bytes.decode("utf-8", errors="replace").splitlines()
    if not text:
        return pd.DataFrame(columns=list(FUNDING_COLUMNS))
    has_header = _has_csv_header(text[0])
    cols = ["calc_time", "funding_interval_hours", "last_funding_rate"]
    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        header=0 if has_header else None,
        names=None if has_header else cols,
    )
    df = df.rename(
        columns={
            "calc_time": "funding_time",
            "last_funding_rate": "funding_rate",
        }
    )
    return normalize_funding_df(df)


def _parse_metrics_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    """Parse Binance Vision daily metrics CSV bytes and resample 5m rows to 1h."""
    text = raw_bytes.decode("utf-8", errors="replace").splitlines()
    if not text:
        return pd.DataFrame(columns=list(METRICS_COLUMNS))
    has_header = _has_csv_header(text[0])
    cols = [
        "create_time",
        "symbol",
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        header=0 if has_header else None,
        names=None if has_header else cols,
    )
    df = df.rename(
        columns={
            "create_time": "open_time",
            "sum_open_interest": "oi",
            "sum_open_interest_value": "oi_value",
            "count_toptrader_long_short_ratio": "top_trader_ls_accounts",
            "sum_toptrader_long_short_ratio": "top_trader_ls_positions",
            "count_long_short_ratio": "global_ls_accounts",
            "sum_taker_long_short_vol_ratio": "taker_buy_sell_ratio",
        }
    )
    if "open_time" not in df.columns:
        return pd.DataFrame(columns=list(METRICS_COLUMNS))

    # Resample 5m snapshots inside [h, h+1h): last value for OI/LS, mean for taker ratio
    df["open_time"] = normalize_metrics_df(
        df.assign(n_snapshots=1)
    )["open_time"]
    df["hour_open"] = df["open_time"].dt.floor("1h")
    agg = df.sort_values("open_time").groupby("hour_open", as_index=False).agg(
        oi=("oi", "last"),
        oi_value=("oi_value", "last"),
        top_trader_ls_accounts=("top_trader_ls_accounts", "last"),
        top_trader_ls_positions=("top_trader_ls_positions", "last"),
        global_ls_accounts=("global_ls_accounts", "last"),
        taker_buy_sell_ratio=("taker_buy_sell_ratio", "mean"),
        n_snapshots=("open_time", "count"),
    )
    agg = agg.rename(columns={"hour_open": "open_time"})
    return normalize_metrics_df(agg)


def _download_zip_csv(
    url: str,
    session: requests.Session,
    timeout: float = 15.0,
) -> bytes | None:
    """Download a Binance Vision ZIP archive and return the first CSV file's bytes."""
    resp = session.get(url, timeout=timeout)
    if resp.status_code == 404:
        logger.debug("Archive not found (404): %s", url)
        return None
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            return None
        return zf.read(csv_names[0])


def _iter_months(start_d: date, end_d: date) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    cur = date(start_d.year, start_d.month, 1)
    stop = date(end_d.year, end_d.month, 1)
    while cur <= stop:
        months.append((cur.year, cur.month))
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return months


def _iter_days(start_d: date, end_d: date) -> list[date]:
    days: list[date] = []
    cur = start_d
    while cur <= end_d:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _parse_date_bound(value: str, *, is_end: bool) -> date:
    """Parse `YYYY-MM` or `YYYY-MM-DD` into an inclusive calendar date."""
    s = value.strip()
    if len(s) == 7 and s[4] == "-":
        yr, mo = map(int, s.split("-"))
        return date(yr, mo, calendar.monthrange(yr, mo)[1]) if is_end else date(yr, mo, 1)
    return datetime.strptime(s, "%Y-%m-%d").date()


def _write_gzip_bytes_atomic(path: Path, raw_bytes: bytes) -> None:
    """Atomically write compressed bytes to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(gzip.compress(raw_bytes))
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _fetch_cached_or_download_csv(
    url: str,
    cache_path: Path,
    get_session: Callable[[], requests.Session],
    *,
    timeout: float,
    use_cache: bool,
) -> tuple[bytes | None, bool]:
    """Return `(csv_bytes, from_cache)` without re-downloading files already cached."""
    if use_cache and cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            return gzip.decompress(cache_path.read_bytes()), True
        except Exception as exc:
            logger.warning("Corrupt cache file %s (%s), re-downloading", cache_path, exc)

    raw_bytes = _download_zip_csv(url, get_session(), timeout=timeout)
    if raw_bytes and use_cache:
        _write_gzip_bytes_atomic(cache_path, raw_bytes)
    return raw_bytes, False


def download_symbol_history(
    symbol: str,
    start_date: str,
    end_date: str,
    out_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    session: requests.Session | None = None,
    config: ProviderConfig | None = None,
    use_cache: bool = True,
    max_workers: int = 8,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Path]:
    """Download klines, funding, and metrics from data.binance.vision into `out_dir`."""
    cfg = config or ProviderConfig()
    base = cfg.data_vision_base_url.rstrip("/")
    sym = symbol.upper()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    archive_dir = out_path / "_archives" / sym

    start_d = _parse_date_bound(start_date, is_end=False)
    end_d = _parse_date_bound(end_date, is_end=True)
    if start_d > end_d:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")

    months = _iter_months(start_d, end_d)
    days = _iter_days(start_d, end_d)
    total_tasks = 2 * len(months) + len(days)
    completed = 0

    def _report(msg: str) -> None:
        if progress_callback is not None and total_tasks > 0:
            progress_callback(completed, total_tasks, msg)

    thread_local = threading.local()
    created_sessions: list[requests.Session] = []
    sessions_lock = threading.Lock()

    def _get_session() -> requests.Session:
        if session is not None:
            return session
        sess = getattr(thread_local, "session", None)
        if sess is None:
            sess = requests.Session()
            thread_local.session = sess
            with sessions_lock:
                created_sessions.append(sess)
        return sess

    k_frames: list[pd.DataFrame] = []
    f_frames: list[pd.DataFrame] = []
    m_by_day: dict[date, pd.DataFrame] = {}

    try:
        for yr, mo in months:
            ym = f"{yr:04d}-{mo:02d}"
            k_url = f"{base}/monthly/klines/{sym}/1h/{sym}-1h-{ym}.zip"
            k_cache = archive_dir / f"{sym}-1h-{ym}.csv.gz"
            k_bytes, k_hit = _fetch_cached_or_download_csv(
                k_url, k_cache, _get_session, timeout=cfg.timeout_sec, use_cache=use_cache
            )
            if k_bytes:
                k_frames.append(_parse_klines_csv_bytes(k_bytes))
            completed += 1
            _report(f"{sym} klines {ym}{' (cached)' if k_hit else ''}")

            f_url = f"{base}/monthly/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip"
            f_cache = archive_dir / f"{sym}-fundingRate-{ym}.csv.gz"
            f_bytes, f_hit = _fetch_cached_or_download_csv(
                f_url, f_cache, _get_session, timeout=cfg.timeout_sec, use_cache=use_cache
            )
            if f_bytes:
                f_frames.append(_parse_funding_csv_bytes(f_bytes))
            completed += 1
            _report(f"{sym} funding {ym}{' (cached)' if f_hit else ''}")

        if days:
            def _fetch_day_metrics(d: date) -> tuple[date, pd.DataFrame | None, bool]:
                ds = d.strftime("%Y-%m-%d")
                m_url = f"{base}/daily/metrics/{sym}/{sym}-metrics-{ds}.zip"
                m_cache = archive_dir / f"{sym}-metrics-{ds}.csv.gz"
                m_bytes, hit = _fetch_cached_or_download_csv(
                    m_url, m_cache, _get_session, timeout=cfg.timeout_sec, use_cache=use_cache
                )
                frame = _parse_metrics_csv_bytes(m_bytes) if m_bytes else None
                return d, frame, hit

            workers = max(1, min(int(max_workers), len(days)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_fetch_day_metrics, d): d for d in days}
                for fut in as_completed(futures):
                    d_done, frame, hit = fut.result()
                    if frame is not None and not frame.empty:
                        m_by_day[d_done] = frame
                    completed += 1
                    _report(f"{sym} metrics {d_done.strftime('%Y-%m-%d')}{' (cached)' if hit else ''}")
    finally:
        for sess in created_sessions:
            if hasattr(sess, "close"):
                sess.close()

    m_frames = [m_by_day[d] for d in sorted(m_by_day)]

    kp = out_path / f"{sym}_1h.csv.gz"
    fp = out_path / f"{sym}_funding.csv.gz"
    mp = out_path / f"{sym}_metrics_1h.csv.gz"

    written: dict[str, Path] = {}
    if k_frames:
        if use_cache and kp.exists():
            k_frames.insert(0, pd.read_csv(kp))
        k_all = normalize_klines_df(pd.concat(k_frames, ignore_index=True))
        if not k_all.empty:
            k_all.to_csv(kp, index=False, compression="gzip")
            written["klines"] = kp
    if f_frames:
        if use_cache and fp.exists():
            f_frames.insert(0, pd.read_csv(fp))
        f_all = normalize_funding_df(pd.concat(f_frames, ignore_index=True))
        if not f_all.empty:
            f_all.to_csv(fp, index=False, compression="gzip")
            written["funding"] = fp
    if m_frames:
        if use_cache and mp.exists():
            m_frames.insert(0, pd.read_csv(mp))
        m_all = normalize_metrics_df(pd.concat(m_frames, ignore_index=True))
        if not m_all.empty:
            m_all.to_csv(mp, index=False, compression="gzip")
            written["metrics"] = mp
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for Binance Vision historical downloader."""
    parser = argparse.ArgumentParser(
        description="Download Binance USDT-M historical 1h klines, funding, and metrics."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Symbols to download (default: bundled 6).",
    )
    parser.add_argument("--start", required=True, help="Start month/date (YYYY-MM or YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="End month/date (YYYY-MM or YYYY-MM-DD).")
    parser.add_argument(
        "--out-dir",
        default="data/downloaded",
        help="Destination directory (default: data/downloaded).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for data.binance.vision downloader."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    for sym in args.symbols:
        written = download_symbol_history(
            sym,
            args.start,
            args.end,
            out_dir=args.out_dir,
        )
        print(f"{sym}: wrote { {k: str(v) for k, v in written.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

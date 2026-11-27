"""Command-line interface for SqueezeRadar (`scan`, `watch`, `download`, `evidence`)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any

import pandas as pd
import requests

from squeezeradar.config import DEFAULT_SYMBOLS, format_price
from squeezeradar.data import discover_sample_symbols
from squeezeradar.evidence import (
    DEFAULT_EVIDENCE_PATH,
    build_evidence,
    format_evidence_summary,
    load_evidence,
    save_evidence,
)
from squeezeradar.pipeline import RadarScanResult, run_radar_scan
from squeezeradar.providers.base import ProviderError
from squeezeradar.providers.binance import (
    BinanceProvider,
    fallback_normalize_symbol,
    normalize_symbol_token,
    resolve_binance_symbol,
    resolve_binance_symbols,
)
from squeezeradar.providers.downloader import download_symbol_history

logger = logging.getLogger(__name__)


def _fmt_price(x: float) -> str:
    return format_price(x)


def format_radar_table(scan_result: RadarScanResult) -> str:
    """Format a clean ASCII/UTF-8 radar table for terminal output."""
    df = scan_result.scanner_table
    if df.empty:
        return "No symbol data available."

    header = (
        f"SqueezeRadar Scan | Source: {scan_result.badge.label} | "
        f"As of: {scan_result.as_of_label}"
    )
    disp = pd.DataFrame(
        {
            "Symbol": df["symbol"],
            "Price": df["price"].map(_fmt_price),
            "24h %": df["change_24h_pct"].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"),
            "Funding": df["funding_rate_pct"].map(lambda x: f"{x:+.4f}%" if pd.notna(x) else "N/A"),
            "OI 24h %": df["oi_24h_pct"].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"),
            "L/S Ratio": df["global_ls_ratio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"),
            "Short Sq": df["short_squeeze_score"].map(lambda x: f"{x:.1f}"),
            "Long Sq": df["long_squeeze_score"].map(lambda x: f"{x:.1f}"),
            "State": df["state"],
            "Nearest Cluster (Est.)": df["nearest_cluster"],
        }
    )
    table_str = disp.to_string(index=False)
    return f"{header}\n{'-' * len(header)}\n{table_str}"


def send_telegram_alert(
    message: str,
    *,
    bot_token: str | None = None,
    chat_id: str | None = None,
    session: Any = None,
) -> bool:
    """Send a Telegram alert when TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are configured."""
    token = (bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    cid = (chat_id or os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not cid:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    http = session or requests
    try:
        resp = http.post(url, json={"chat_id": cid, "text": message}, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Failed to send Telegram alert: %s", exc)
        return False


def watch_loop(
    *,
    symbols: list[str],
    mode: str = "live",
    interval_sec: int = 60,
    max_iterations: int | None = None,
    sleep_fn: Any = time.sleep,
) -> list[str]:
    """Poll radar and emit notifications whenever a symbol's state changes."""
    prev_states: dict[str, tuple[str, str]] = {}
    alerts_emitted: list[str] = []
    iteration = 0

    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        scan_res = run_radar_scan(symbols=symbols, mode=mode)  # type: ignore[arg-type]
        for sym, analysis in scan_res.analyses.items():
            last = analysis.scored_df.iloc[-1]
            cur = (str(last["short_squeeze_state"]), str(last["long_squeeze_state"]))
            prev = prev_states.get(sym)
            if prev is not None and prev != cur:
                msg = (
                    f"[SqueezeRadar {scan_res.as_of_label}] {sym} state transition: "
                    f"short={prev[0]}->{cur[0]} (score {last['short_squeeze_score']:.1f}), "
                    f"long={prev[1]}->{cur[1]} (score {last['long_squeeze_score']:.1f})"
                )
                print(msg, flush=True)
                alerts_emitted.append(msg)
                send_telegram_alert(msg)
            prev_states[sym] = cur

        if iteration == 1 and not alerts_emitted:
            print(format_radar_table(scan_res), flush=True)

        if max_iterations is not None and iteration >= max_iterations:
            break
        sleep_fn(interval_sec)

    return alerts_emitted


def build_parser() -> argparse.ArgumentParser:
    """Build top-level CLI parser for `python -m squeezeradar`."""
    parser = argparse.ArgumentParser(
        prog="squeezeradar",
        description="SqueezeRadar — See the squeeze before it happens.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Scan perpetual symbols and print the radar table.")
    p_scan.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p_scan.add_argument("--mode", choices=["live", "replay"], default="replay")
    p_scan.add_argument("--at", dest="as_of", default=None, help="UTC timestamp for replay.")

    p_watch = sub.add_parser("watch", help="Monitor state transitions and optionally alert via Telegram.")
    p_watch.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p_watch.add_argument("--mode", choices=["live", "replay"], default="live")
    p_watch.add_argument("--interval", type=int, default=60, help="Polling interval in seconds.")
    p_watch.add_argument("--iterations", type=int, default=None, help="Optional max polling cycles.")

    p_dl = sub.add_parser("download", help="Download historical data from data.binance.vision.")
    p_dl.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p_dl.add_argument("--start", required=True, help="Start month/date (YYYY-MM or YYYY-MM-DD).")
    p_dl.add_argument("--end", required=True, help="End month/date (YYYY-MM or YYYY-MM-DD).")
    p_dl.add_argument("--out-dir", default="data/downloaded")

    p_ev = sub.add_parser(
        "evidence",
        help="Show (or --build) the precomputed walk-forward ML + event-study evidence file.",
    )
    p_ev.add_argument(
        "--build",
        action="store_true",
        help="Run the full purged walk-forward + event study and write the output JSON file.",
    )
    p_ev.add_argument("--symbols", nargs="+", default=None, help="Symbols to include (default: bundled 6).")
    p_ev.add_argument("--start", default=None, help="Optional start month/date (YYYY-MM or YYYY-MM-DD).")
    p_ev.add_argument("--end", default=None, help="Optional end month/date (YYYY-MM or YYYY-MM-DD).")
    p_ev.add_argument("--out", default=str(DEFAULT_EVIDENCE_PATH), help="Evidence JSON path.")

    return parser


def _resolve_cli_symbols(symbols: list[str], *, validate_live: bool = False) -> tuple[list[str], list[str]]:
    """Normalize CLI symbol arguments, validating against Binance live list when requested."""
    if not validate_live:
        out: list[str] = []
        for s in symbols:
            norm = fallback_normalize_symbol(s)
            if norm and norm not in out:
                out.append(norm)
        return out, []
    try:
        listed = BinanceProvider().fetch_listed_symbols()
        return resolve_binance_symbols(symbols, listed)
    except ProviderError:
        out = []
        for s in symbols:
            norm = fallback_normalize_symbol(s)
            if norm and norm not in out:
                out.append(norm)
        return out, []


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        if args.mode == "live":
            target_syms, unknown = _resolve_cli_symbols(list(args.symbols), validate_live=True)
            for unk in unknown:
                print(f"Warning: unknown Binance USDT-M symbol '{unk}' — skipped.", file=sys.stderr)
        else:
            replay_avail = discover_sample_symbols()
            target_syms = [
                resolve_binance_symbol(s, replay_avail) or fallback_normalize_symbol(s)
                for s in args.symbols
            ]
            target_syms = list(dict.fromkeys(s for s in target_syms if s))

        if not target_syms:
            print("No valid symbols to scan.")
            return 0

        res = run_radar_scan(
            symbols=target_syms,
            mode=args.mode,
            as_of=args.as_of,
            limit=2160 if args.mode == "replay" else None,
        )
        for sym, err in res.errors.items():
            print(f"Warning: {sym}: {err}", file=sys.stderr)
        print(format_radar_table(res))
        return 0
    if args.command == "watch":
        watch_loop(
            symbols=args.symbols,
            mode=args.mode,
            interval_sec=args.interval,
            max_iterations=args.iterations,
        )
        return 0
    if args.command == "download":
        for raw_sym in args.symbols:
            tok = normalize_symbol_token(raw_sym)
            sym = tok
            if sym and not sym.endswith("USDT"):
                try:
                    listed = BinanceProvider().fetch_listed_symbols()
                    sym = resolve_binance_symbol(raw_sym, listed) or fallback_normalize_symbol(raw_sym)
                except ProviderError:
                    sym = fallback_normalize_symbol(raw_sym)
            written = download_symbol_history(
                sym, args.start, args.end, out_dir=args.out_dir
            )
            print(f"{sym}: { {k: str(v) for k, v in written.items()} }")
        return 0
    if args.command == "evidence":
        target_syms = list(args.symbols) if args.symbols is not None else list(DEFAULT_SYMBOLS)
        target_syms = [fallback_normalize_symbol(s) for s in target_syms]
        if args.build:
            bundle = build_evidence(
                symbols=target_syms,
                start_date=args.start,
                end_date=args.end,
            )
            path = save_evidence(bundle, args.out)
            print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
        elif args.symbols is not None or args.start is not None or args.end is not None:
            bundle = build_evidence(
                symbols=target_syms,
                start_date=args.start,
                end_date=args.end,
            )
        else:
            bundle = load_evidence(args.out)
            if bundle is None:
                print(f"No evidence file at {args.out}. Run: python -m squeezeradar evidence --build")
                return 1
        print(format_evidence_summary(bundle))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

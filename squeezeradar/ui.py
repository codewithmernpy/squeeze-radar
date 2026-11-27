"""Shared Streamlit UI layout, theme styling, caching, and Plotly chart builders."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from squeezeradar.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SAMPLE_DIR,
    DEFAULT_SYMBOLS,
    ELEVATED_SCORE_THRESHOLD,
    SUITE_PROJECTS,
    format_price,
)
from squeezeradar.data import discover_sample_symbols, get_replay_date_bounds
from squeezeradar.evidence import (
    DEFAULT_EVIDENCE_PATH,
    EvidenceBundle,
    build_evidence,
    load_evidence,
)
from squeezeradar.pipeline import RadarScanResult, find_notable_liquidation_events, run_radar_scan
from squeezeradar.providers.base import ProviderBadge, ProviderError
from squeezeradar.providers.binance import (
    BinanceProvider,
    fallback_normalize_symbol,
    parse_symbol_list_text,
    resolve_binance_symbol,
    resolve_binance_symbols,
)
from squeezeradar.providers.downloader import download_symbol_history
from squeezeradar.scores import state_for_score

STATE_COLORS: dict[str, str] = {
    "NONE": "#64748B",
    "SETUP": "#F59E0B",
    "TRIGGER": "#EF4444",
    "IN_PROGRESS": "#8B5CF6",
    "EXHAUSTION": "#10B981",
}

# Each snapshot time is BEFORE the named move (checked against the bundled data: the move
# happens inside the following 48h reveal window).
CURATED_REPLAY_PRESETS: list[dict[str, str]] = [
    {
        "label": "2024-08-05 00:00 UTC · Pre-Yen Carry Crash (Long Cascade)",
        "as_of": "2024-08-05 00:00",
    },
    {
        "label": "2024-07-14 20:00 UTC · Pre-July 2024 Rally (Short Squeeze)",
        "as_of": "2024-07-14 20:00",
    },
    {
        "label": "2024-12-05 12:00 UTC · Pre-Dec 2024 $100K Volatility Flush",
        "as_of": "2024-12-05 12:00",
    },
    {
        "label": "2025-04-22 12:00 UTC · Pre-April 2025 Short Squeeze Wave",
        "as_of": "2025-04-22 12:00",
    },
    {
        "label": "2025-05-08 12:00 UTC · Pre-May 2025 Short Covering Impulse",
        "as_of": "2025-05-08 12:00",
    },
    {
        "label": "2025-10-10 12:00 UTC · Pre-Oct 10 2025 Long Liquidation Cascade",
        "as_of": "2025-10-10 12:00",
    },
]

_CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }
    .sr-hero {
        background: linear-gradient(135deg, #131A2E 0%, #1B2440 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 12px;
        padding: 1.15rem 1.4rem;
        margin-bottom: 1rem;
    }
    .sr-hero h1 { margin: 0; font-size: 1.75rem; color: #E6E9F2; }
    .sr-hero p { margin: 0.25rem 0 0 0; color: #A5B4FC; font-size: 0.95rem; }
    .sr-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .sr-badge-binance { background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid #10B981; }
    .sr-badge-hyperliquid { background: rgba(245, 158, 11, 0.18); color: #FBBF24; border: 1px solid #F59E0B; }
    .sr-badge-replay { background: rgba(99, 102, 241, 0.22); color: #A5B4FC; border: 1px solid #6366F1; }
    .sr-card {
        background-color: #131A2E;
        border: 1px solid rgba(230, 233, 242, 0.08);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
    }
    .sr-footer {
        margin-top: 2.5rem;
        padding-top: 0.85rem;
        border-top: 1px solid rgba(230, 233, 242, 0.1);
        text-align: center;
        color: #94A3B8;
        font-size: 0.85rem;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    [data-testid="stMetricValue"] {
        font-size: clamp(1.1rem, 2.2vw, 1.8rem);
    }
</style>
"""


def init_page(title: str) -> None:
    """Configure page layout and inject consistent suite styling."""
    try:
        st.set_page_config(
            page_title=f"SqueezeRadar · {title}",
            page_icon="📡",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def get_session_api_key() -> str:
    """Resolve Anthropic API key from session input, st.secrets, or env (never persisted)."""
    key = str(
        st.session_state.get("sr_api_key") or st.session_state.get("_sr_api_key_store") or ""
    ).strip()
    if key:
        return key
    try:
        secret_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass
    return str(os.getenv("ANTHROPIC_API_KEY", "") or "").strip()


@st.cache_data(ttl=60, show_spinner=False)
def load_radar_cached(
    symbols: tuple[str, ...],
    mode: str,
    as_of: str | None = None,
    limit: int | None = 2160,
    start_date: str | None = None,
    end_date: str | None = None,
    cache_bust: int = 0,
) -> RadarScanResult:
    """Cached scanner pipeline run for fast page switches and <10s first load."""
    _ = cache_bust
    return run_radar_scan(
        symbols=symbols,
        mode=mode,  # type: ignore[arg-type]
        as_of=as_of,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        sample_dir=str(DEFAULT_SAMPLE_DIR),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_evidence_file_cached(path: str, mtime_ns: int) -> EvidenceBundle | None:
    """Load the precomputed evidence JSON (cache keyed on file path + modification time)."""
    _ = mtime_ns
    return load_evidence(path)


def load_precomputed_evidence(path: str | os.PathLike[str] = DEFAULT_EVIDENCE_PATH) -> EvidenceBundle | None:
    """Return the precomputed evidence bundle if the file exists (instant), else None."""
    p = os.fspath(path)
    if not os.path.exists(p):
        return None
    return load_evidence_file_cached(p, os.stat(p).st_mtime_ns)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_evidence_cached(
    symbols: tuple[str, ...],
    start_date: str | None = None,
    end_date: str | None = None,
    cache_bust: int = 0,
) -> EvidenceBundle:
    """Recompute walk-forward ML, event study and latest purged probabilities live (1-2 min)."""
    _ = cache_bust
    return build_evidence(
        symbols=symbols,
        sample_dir=DEFAULT_SAMPLE_DIR,
        start_date=start_date,
        end_date=end_date,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_top_historical_presets_cached(
    symbols: tuple[str, ...],
    start_date: str | None = None,
    end_date: str | None = None,
    cache_bust: int = 0,
) -> list[dict[str, Any]]:
    """Scan bundled/cached replay data for top distinct 24h liquidation-estimate episodes."""
    _ = cache_bust
    scan_res = run_radar_scan(
        symbols=symbols,
        mode="replay",
        limit=None,
        start_date=start_date,
        end_date=end_date,
    )
    return find_notable_liquidation_events(scan_res.analyses, top_n=6)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_binance_symbols_cached(cache_bust: int = 0) -> tuple[str, ...]:
    """Fetch active Binance USDT-M perpetual contracts (cached 5 min)."""
    _ = cache_bust
    return tuple(BinanceProvider().fetch_listed_symbols())


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_top_universe_cached(top_n: int, cache_bust: int = 0) -> tuple[str, ...]:
    """Fetch top N Binance USDT-M perpetual contracts by 24h quote volume."""
    _ = cache_bust
    return tuple(BinanceProvider().fetch_top_universe(top_n=top_n))


@st.cache_data(ttl=3600, show_spinner=False)
def get_replay_date_bounds_cached(
    symbols: tuple[str, ...],
    cache_bust: int = 0,
) -> tuple[dt.date, dt.date] | None:
    """Cached `(min_date, max_date)` UTC across selected replay symbols."""
    _ = cache_bust
    return get_replay_date_bounds(symbols=symbols, sample_dir=DEFAULT_SAMPLE_DIR)


def _clear_ui_data_caches() -> None:
    """Clear Streamlit data caches after refresh or downloading new history."""
    load_radar_cached.clear()
    compute_evidence_cached.clear()
    load_top_historical_presets_cached.clear()
    fetch_live_binance_symbols_cached.clear()
    fetch_live_top_universe_cached.clear()
    get_replay_date_bounds_cached.clear()


def _parse_month_or_date_str(value: str, *, is_end: bool) -> dt.date:
    """Parse `YYYY-MM` or `YYYY-MM-DD` input string into a calendar date."""
    import calendar as _cal

    s = value.strip()
    if len(s) == 7 and s[4] == "-":
        yr, mo = map(int, s.split("-"))
        return dt.date(yr, mo, _cal.monthrange(yr, mo)[1]) if is_end else dt.date(yr, mo, 1)
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def get_sidebar_replay_range() -> tuple[dt.date | None, dt.date | None, dt.date | None, dt.date | None]:
    """Return `(range_start, range_end, avail_min, avail_max)` from sidebar state."""
    bounds = st.session_state.get("sr_avail_bounds")
    eff = st.session_state.get("sr_effective_range")
    avail_min, avail_max = bounds if bounds else (None, None)
    r_start, r_end = eff if eff else (avail_min, avail_max)
    return r_start, r_end, avail_min, avail_max


def get_replay_scan_params(mode: str) -> tuple[str | None, int | None, dt.date | None, dt.date | None, bool]:
    """Return `(as_of, limit_bars, range_start, range_end, is_custom_range)` for Radar/Deep Dive."""
    if mode != "replay":
        return None, 720, None, None, False
    r_start, r_end, avail_min, avail_max = get_sidebar_replay_range()
    if r_start is None or r_end is None or avail_min is None or avail_max is None:
        return None, 2160, None, None, False
    if r_start == avail_min and r_end == avail_max:
        return None, 2160, r_start, r_end, False
    end_cutoff = (r_end + dt.timedelta(days=1)).strftime("%Y-%m-%d 00:00")
    n_hours = max(24, ((r_end - r_start).days + 1) * 24)
    return end_cutoff, max(2160, n_hours + 720), r_start, r_end, True


def render_badge_html(badge: ProviderBadge) -> str:
    """Render an HTML status pill for the active data source."""
    cls = {
        "binance": "sr-badge-binance",
        "hyperliquid": "sr-badge-hyperliquid",
    }.get(badge.source, "sr-badge-replay")
    return f'<span class="sr-badge {cls}">{badge.label}</span>'


def _resolve_live_tokens(
    tokens: list[str],
    cache_bust: int,
    *,
    listed_syms: tuple[str, ...] | None = None,
) -> tuple[list[str], list[str], tuple[str, ...], bool]:
    """Normalize and validate live symbol tokens against listed Binance perpetuals."""
    if not tokens:
        return [], [], listed_syms or (), False
    used_fallback = False
    if listed_syms is None:
        try:
            listed_syms = fetch_live_binance_symbols_cached(cache_bust)
        except ProviderError:
            listed_syms = ()
            used_fallback = True
    if listed_syms:
        resolved, unknown = resolve_binance_symbols(tokens, listed_syms)
        return resolved, unknown, listed_syms, used_fallback
    resolved = list(
        dict.fromkeys(norm for t in tokens if (norm := fallback_normalize_symbol(t)))
    )
    return resolved, [], (), True


def _copy_store_value(val: Any) -> Any:
    return list(val) if isinstance(val, list) else val


def _sync_persisted_widget(
    widget_key: str,
    store_key: str,
    default: Any,
    *,
    force_restore: bool = False,
) -> Any:
    """Re-seed a widget key from its non-widget store on EVERY run (before the widget is created).

    Streamlit gives a keyed widget a new identity on another page, so a value merely left in session_state can
    render empty after a page switch (seen in the browser: Radar -> Deep Dive showed no contracts). The store is
    the single source of truth: user edits reach it through each widget's on_change callback (which runs before
    the script), and code paths that change a widget programmatically update the store too."""
    last = st.session_state.setdefault("_sr_last_rendered", {})
    if widget_key in st.session_state and store_key in st.session_state and widget_key in last:
        if st.session_state[widget_key] != last[widget_key]:      # changed since it was last rendered (user or code)
            st.session_state[store_key] = _copy_store_value(st.session_state[widget_key])
    if store_key in st.session_state:
        st.session_state[widget_key] = _copy_store_value(st.session_state[store_key])
    elif widget_key in st.session_state:
        st.session_state[store_key] = _copy_store_value(st.session_state[widget_key])
    else:
        st.session_state[widget_key] = _copy_store_value(default)
        st.session_state[store_key] = _copy_store_value(default)
    last[widget_key] = _copy_store_value(st.session_state[widget_key])
    return st.session_state[widget_key]


def _make_store_callback(widget_key: str, store_key: str) -> Callable[[], None]:
    """Return an on_change callback copying widget state into its persistent non-widget store."""

    def _cb() -> None:
        if widget_key in st.session_state:
            st.session_state[store_key] = _copy_store_value(st.session_state[widget_key])

    return _cb


def render_sidebar(
    *,
    allow_live: bool = True,
    show_date_range: bool | None = None,
) -> tuple[str, tuple[str, ...], int]:
    """Render the shared sidebar controls and return `(mode, symbols, cache_bust)`."""
    with st.sidebar:
        st.markdown("### 📡 SqueezeRadar")
        st.caption("See the squeeze before it happens.")

        if "sr_cache_bust" not in st.session_state:
            st.session_state["sr_cache_bust"] = 0
        cache_bust = int(st.session_state["sr_cache_bust"])

        returning_from_replay_only = bool(
            allow_live and st.session_state.pop("_sr_from_replay_only", False)
        )
        if not allow_live:
            st.session_state["_sr_from_replay_only"] = True

        if allow_live:
            _sync_persisted_widget(
                "sr_mode_choice",
                "_sr_mode_store",
                "Replay (Offline)",
                force_restore=returning_from_replay_only,
            )
            choice = st.radio(
                "Data Mode",
                options=["Replay (Offline)", "Live Auto"],
                key="sr_mode_choice",
                on_change=_make_store_callback("sr_mode_choice", "_sr_mode_store"),
                help="Replay uses bundled 2023–2025 + cached Binance history (offline). Live Auto tries Binance -> Hyperliquid -> Replay.",
            )
            mode = "live" if choice == "Live Auto" else "replay"
        else:
            st.radio(
                "Data Mode",
                options=["Replay (Offline)"],
                index=0,
                key="sr_mode_choice_replay",
                help="Time Machine operates strictly on offline historical Replay data.",
            )
            mode = "replay"

        replay_available = discover_sample_symbols(DEFAULT_SAMPLE_DIR)
        pending_dl = st.session_state.pop("sr_pending_add_symbol", None)
        # Every page (live or replay-only) re-seeds the contract list from the store; replay-only pages then show the
        # replay-valid subset below without touching the store, so live-only coins come back on live pages.
        _sync_persisted_widget("sr_symbols", "_sr_symbols_store", list(DEFAULT_SYMBOLS))

        if pending_dl:
            cur = list(st.session_state["sr_symbols"])
            if pending_dl not in cur:
                cur.append(pending_dl)
            st.session_state["sr_symbols"] = cur
            stored = list(st.session_state.get("_sr_symbols_store", list(DEFAULT_SYMBOLS)))
            if pending_dl not in stored:
                stored.append(pending_dl)
            st.session_state["_sr_symbols_store"] = stored

        skipped_non_replay: list[str] = []
        unknown_selected: list[str] = []
        live_dir_fallback = False
        cached_listed_syms: tuple[str, ...] | None = None

        cur_selected = list(st.session_state.get("sr_symbols", []))
        flat_tokens = [
            tok
            for raw_item in cur_selected
            for tok in (parse_symbol_list_text(str(raw_item)) or [str(raw_item)])
            if str(tok).strip()
        ]

        if mode == "replay":
            sym_options = list(replay_available)
            valid_selected: list[str] = []
            for tok in flat_tokens:
                resolved = resolve_binance_symbol(tok, replay_available)
                if resolved is not None:
                    if resolved not in valid_selected:
                        valid_selected.append(resolved)
                elif tok not in skipped_non_replay:
                    skipped_non_replay.append(tok)
            if not valid_selected and cur_selected:
                valid_selected = [s for s in DEFAULT_SYMBOLS if s in sym_options] or sym_options[:6]
            if valid_selected != cur_selected:
                st.session_state["sr_symbols"] = valid_selected
                if allow_live:
                    st.session_state["_sr_symbols_store"] = list(valid_selected)
        else:
            known_local = set(DEFAULT_SYMBOLS) | set(replay_available)
            needs_live_lookup = any(tok not in known_local for tok in flat_tokens)
            if needs_live_lookup:
                valid_selected, unknown_selected, fetched_syms, live_dir_fallback = _resolve_live_tokens(
                    flat_tokens, cache_bust
                )
                if fetched_syms:
                    cached_listed_syms = fetched_syms
            else:
                valid_selected = list(dict.fromkeys(flat_tokens))
            if not valid_selected and cur_selected:
                valid_selected = list(DEFAULT_SYMBOLS)
            if valid_selected != cur_selected:
                st.session_state["sr_symbols"] = valid_selected
                st.session_state["_sr_symbols_store"] = list(valid_selected)
            sym_options = list(
                dict.fromkeys([*DEFAULT_SYMBOLS, *replay_available, *valid_selected])
            )

        def _on_symbols_change() -> None:
            cur_vals = list(st.session_state.get("sr_symbols", []))
            if allow_live:
                st.session_state["_sr_symbols_store"] = cur_vals
            else:
                replay_set = set(replay_available)
                prev_store = list(st.session_state.get("_sr_symbols_store", []))
                live_only = [s for s in prev_store if s not in replay_set]
                st.session_state["_sr_symbols_store"] = list(dict.fromkeys([*cur_vals, *live_only]))

        st.session_state.setdefault("_sr_last_rendered", {})["sr_symbols"] = list(st.session_state.get("sr_symbols", []))
        selected = st.multiselect(
            "Perpetual Contracts",
            options=sym_options,
            key="sr_symbols",
            on_change=_on_symbols_change,
            accept_new_options=True,
            placeholder="Select or type a coin (e.g. suiusdt, pepe)...",
            help=(
                "Select or type coins/pairs (e.g. suiusdt, sui, pepe) and press Enter. "
                "Live Auto normalizes and validates against active Binance USDT-M perpetuals. "
                "Replay (Offline) accepts only coins with local/downloaded history — use "
                "'Download history for a coin' below to add more offline."
            ),
        )
        if skipped_non_replay:
            st.warning(
                f"Not available in Replay mode (no downloaded history for: {', '.join(skipped_non_replay)}). "
                "Use **Download history for a coin** below or switch to **Live Auto**."
            )
        if unknown_selected:
            st.warning(f"Unknown Binance USDT-M symbol(s) removed: {', '.join(unknown_selected)}")
        if live_dir_fallback:
            st.caption("Live Binance symbol directory unavailable; using fallback normalization.")

        extra_live_syms: list[str] = []
        top_n_syms: list[str] = []
        if mode == "live":
            _sync_persisted_widget(
                "sr_add_coins",
                "_sr_add_coins_store",
                "",
                force_restore=returning_from_replay_only,
            )
            add_coins_raw = st.text_input(
                "Add coins",
                key="sr_add_coins",
                on_change=_make_store_callback("sr_add_coins", "_sr_add_coins_store"),
                placeholder="e.g. pepe, link, sui",
                help="Optional comma-separated Binance USDT-M coins or pairs (you can also type directly into Perpetual Contracts above).",
            )
            tokens = parse_symbol_list_text(add_coins_raw)
            if tokens:
                extra_live_syms, unknown_tokens, fetched_syms, add_fallback = _resolve_live_tokens(
                    tokens, cache_bust, listed_syms=cached_listed_syms
                )
                if fetched_syms:
                    cached_listed_syms = fetched_syms
                if add_fallback and not live_dir_fallback:
                    st.caption("Live Binance symbol directory unavailable; using fallback normalization.")
                if unknown_tokens:
                    st.warning(f"Unknown Binance USDT-M symbol(s): {', '.join(unknown_tokens)}")
                if extra_live_syms:
                    st.caption(f"Added live: {', '.join(extra_live_syms)}")

            _sync_persisted_widget(
                "sr_top_n_enabled",
                "_sr_top_n_enabled_store",
                False,
                force_restore=returning_from_replay_only,
            )
            use_top_n = st.checkbox(
                "Top N by volume",
                key="sr_top_n_enabled",
                on_change=_make_store_callback("sr_top_n_enabled", "_sr_top_n_enabled_store"),
                help="Include the top N USDT-M perpetuals ranked by 24h quote volume.",
            )
            if use_top_n:
                _sync_persisted_widget(
                    "sr_top_n_count",
                    "_sr_top_n_count_store",
                    10,
                    force_restore=returning_from_replay_only,
                )
                top_n_val = st.slider(
                    "Top N coins",
                    min_value=1,
                    max_value=30,
                    key="sr_top_n_count",
                    on_change=_make_store_callback("sr_top_n_count", "_sr_top_n_count_store"),
                )
                try:
                    top_n_syms = list(fetch_live_top_universe_cached(int(top_n_val), cache_bust))
                    if top_n_syms:
                        st.caption(f"Top {len(top_n_syms)} by 24h volume active.")
                except ProviderError:
                    st.warning("Could not fetch Top N by volume from Binance; using selected coins.")

        combined_syms = list(dict.fromkeys([*selected, *top_n_syms, *extra_live_syms]))
        symbols = tuple(combined_syms) if combined_syms else DEFAULT_SYMBOLS

        # Replay / Evidence Date Range bounded by data available for selected coins
        eff_show_range = (mode == "replay") if show_date_range is None else bool(show_date_range)
        range_syms = tuple(s for s in symbols if s in replay_available) or tuple(replay_available)
        bounds = get_replay_date_bounds_cached(range_syms, cache_bust) if range_syms else None
        if bounds is not None:
            avail_min, avail_max = bounds
            prev_bounds = st.session_state.get("sr_last_avail_bounds")
            if "sr_range_start" not in st.session_state and "_sr_range_start_store" in st.session_state:
                st.session_state["sr_range_start"] = st.session_state["_sr_range_start_store"]
            if "sr_range_end" not in st.session_state and "_sr_range_end_store" in st.session_state:
                st.session_state["sr_range_end"] = st.session_state["_sr_range_end_store"]
            if "sr_range_start" not in st.session_state or "sr_range_end" not in st.session_state:
                st.session_state["sr_range_start"] = avail_min
                st.session_state["sr_range_end"] = avail_max
            elif prev_bounds is not None and prev_bounds != bounds:
                was_full = (
                    st.session_state["sr_range_start"] == prev_bounds[0]
                    and st.session_state["sr_range_end"] == prev_bounds[1]
                )
                if was_full:
                    st.session_state["sr_range_start"] = avail_min
                    st.session_state["sr_range_end"] = avail_max
                else:
                    st.session_state["sr_range_start"] = max(
                        avail_min, min(avail_max, st.session_state["sr_range_start"])
                    )
                    st.session_state["sr_range_end"] = max(
                        avail_min, min(avail_max, st.session_state["sr_range_end"])
                    )
            else:
                st.session_state["sr_range_start"] = max(
                    avail_min, min(avail_max, st.session_state["sr_range_start"])
                )
                st.session_state["sr_range_end"] = max(
                    avail_min, min(avail_max, st.session_state["sr_range_end"])
                )
            if st.session_state["sr_range_start"] > st.session_state["sr_range_end"]:
                st.session_state["sr_range_start"] = avail_min
                st.session_state["sr_range_end"] = avail_max
            st.session_state["_sr_range_start_store"] = st.session_state["sr_range_start"]
            st.session_state["_sr_range_end_store"] = st.session_state["sr_range_end"]
            st.session_state["sr_last_avail_bounds"] = bounds
            st.session_state["sr_avail_bounds"] = bounds

            if eff_show_range:
                st.markdown("**Date Range (UTC)**")
                c_rs = c_re = st                     # stacked: full-width date boxes in the narrow sidebar
                r_start = c_rs.date_input(
                    "Start date",
                    min_value=avail_min,
                    max_value=avail_max,
                    key="sr_range_start",
                    on_change=_make_store_callback("sr_range_start", "_sr_range_start_store"),
                )
                r_end = c_re.date_input(
                    "End date",
                    min_value=avail_min,
                    max_value=avail_max,
                    key="sr_range_end",
                    on_change=_make_store_callback("sr_range_end", "_sr_range_end_store"),
                )
                if r_start > r_end:
                    st.warning("Start date must be <= End date; clamping end date.")
                    r_end = r_start
                st.caption(f"Available: {avail_min} → {avail_max}")
                st.session_state["sr_effective_range"] = (r_start, r_end)
            else:
                st.session_state["sr_effective_range"] = (
                    st.session_state["sr_range_start"],
                    st.session_state["sr_range_end"],
                )
        else:
            st.session_state["sr_avail_bounds"] = None
            st.session_state["sr_effective_range"] = None

        with st.expander("Download history for a coin", expanded=False):
            dl_status = st.session_state.pop("sr_dl_status", None)
            if dl_status:
                kind, msg = dl_status
                if kind == "success":
                    st.success(msg)
                else:
                    st.warning(msg)

            _sync_persisted_widget("sr_dl_symbol", "_sr_dl_symbol_store", "")
            dl_sym_raw = st.text_input(
                "Symbol",
                placeholder="e.g. LINK, LINKUSDT, pepe",
                key="sr_dl_symbol",
                on_change=_make_store_callback("sr_dl_symbol", "_sr_dl_symbol_store"),
            )
            dc1, dc2 = st.columns(2)
            _sync_persisted_widget("sr_dl_start", "_sr_dl_start_store", "2025-08")
            _sync_persisted_widget("sr_dl_end", "_sr_dl_end_store", "2025-09")
            dl_start_raw = dc1.text_input(
                "Start month",
                key="sr_dl_start",
                on_change=_make_store_callback("sr_dl_start", "_sr_dl_start_store"),
            )
            dl_end_raw = dc2.text_input(
                "End month",
                key="sr_dl_end",
                on_change=_make_store_callback("sr_dl_end", "_sr_dl_end_store"),
            )
            if st.button("Download to Cache", width="stretch", key="sr_dl_btn"):
                tok = dl_sym_raw.strip()
                if not tok:
                    st.warning("Enter a Binance USDT-M symbol (e.g. LINKUSDT or link).")
                else:
                    try:
                        d_start = _parse_month_or_date_str(dl_start_raw, is_end=False)
                        d_end = _parse_month_or_date_str(dl_end_raw, is_end=True)
                    except ValueError:
                        st.warning("Use YYYY-MM (or YYYY-MM-DD) for start and end months.")
                    else:
                        if d_start > d_end:
                            st.warning("Start month must be <= End month.")
                        else:
                            resolved_dl: str | None = None
                            try:
                                listed_dl = fetch_live_binance_symbols_cached(cache_bust)
                            except ProviderError:
                                listed_dl = ()

                            if listed_dl:
                                resolved_dl = resolve_binance_symbol(tok, listed_dl)
                                if resolved_dl is None:
                                    st.warning(f"Unknown Binance USDT-M symbol: '{tok}'")
                            else:
                                resolved_dl = fallback_normalize_symbol(tok)

                            if resolved_dl:
                                prog = st.progress(0.0, text=f"Downloading {resolved_dl}...")

                                def _on_progress(done: int, total: int, text: str) -> None:
                                    prog.progress(min(1.0, done / max(1, total)), text=text)

                                try:
                                    written = download_symbol_history(
                                        resolved_dl,
                                        dl_start_raw.strip(),
                                        dl_end_raw.strip(),
                                        out_dir=DEFAULT_CACHE_DIR,
                                        progress_callback=_on_progress,
                                    )
                                except Exception as exc:
                                    prog.empty()
                                    st.warning(f"Download failed for {resolved_dl}: {exc}")
                                else:
                                    prog.empty()
                                    if "klines" in written:
                                        _clear_ui_data_caches()
                                        st.session_state["sr_cache_bust"] += 1
                                        st.session_state["sr_pending_add_symbol"] = resolved_dl
                                        st.session_state["sr_dl_status"] = (
                                            "success",
                                            f"Cached {resolved_dl} ({dl_start_raw.strip()} → {dl_end_raw.strip()}) in data/cache.",
                                        )
                                        st.rerun()
                                    else:
                                        st.warning(
                                            f"No archives found on data.binance.vision for {resolved_dl} "
                                            f"({dl_start_raw.strip()} → {dl_end_raw.strip()})."
                                        )

        if st.button("🔄 Refresh Data", width="stretch", key="sr_refresh_btn"):
            st.session_state["sr_cache_bust"] += 1
            _clear_ui_data_caches()

        st.divider()
        _sync_persisted_widget("sr_api_key", "_sr_api_key_store", "")
        st.text_input(
            "Anthropic API Key (session only)",
            type="password",
            key="sr_api_key",
            on_change=_make_store_callback("sr_api_key", "_sr_api_key_store"),
            placeholder="sk-ant-... (optional)",
            help="Used only for Claude AI Analyst in this browser session. Never written to disk.",
        )

    return mode, symbols, int(st.session_state["sr_cache_bust"])


def show_sidebar_badge(badge: ProviderBadge, as_of_label: str) -> None:
    """Display the resolved data-source badge in the sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Active Data Source**")
        st.markdown(render_badge_html(badge), unsafe_allow_html=True)
        st.caption(f"Snapshot: {as_of_label}")
        if badge.fallback_reason:
            st.warning(f"Fallback active: {badge.fallback_reason[:140]}")
        else:
            st.caption(badge.detail)


def render_footer() -> None:
    """Render the shared Crypto Quant Suite footer (sister projects as plain-text names)."""
    names = " | ".join(
        f"<strong>{name}</strong>" if name == "SqueezeRadar" else name for name in SUITE_PROJECTS
    )
    html = (
        '<div class="sr-footer">'
        f"Part of the Crypto Quant Suite: {names}"
        '<br/><small>Estimated liquidation maps and model probabilities are analytical approximations, not financial advice.</small>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_scanner_hero(subtitle: str = "See the squeeze before it happens — Real-time derivatives crowding, estimated liquidation clusters, and walk-forward ML.") -> None:
    """Render top brand banner matching the Crypto Quant Suite dark look."""
    st.markdown(
        f'<div class="sr-hero"><h1>📡 SqueezeRadar</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def format_scanner_display_df(scanner_df: pd.DataFrame) -> pd.DataFrame:
    """Format scanner dataframe cleanly for Streamlit dataframe / styling."""
    if scanner_df.empty:
        return scanner_df
    disp = pd.DataFrame(
        {
            "Symbol": scanner_df["symbol"],
            "Price ($)": scanner_df["price"],
            "24h %": scanner_df["change_24h_pct"].round(2),
            "Funding (bps)": (scanner_df["funding_rate_pct"] * 100.0).round(2),
            "OI 24h %": scanner_df["oi_24h_pct"].round(2),
            "Global L/S": scanner_df["global_ls_ratio"].round(2),
            "Short Squeeze": scanner_df["short_squeeze_score"].round(1),
            "Long Squeeze": scanner_df["long_squeeze_score"].round(1),
            "State": scanner_df["state"],
            "Nearest Liq Cluster (Est.)": scanner_df["nearest_cluster"],
        }
    )
    return disp


def score_badge_style(val: Any) -> str:
    """Apply consistent color-coding to score cells using `state_for_score`."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    tier = state_for_score(v)
    if tier == "Active":
        return "background-color: rgba(239, 68, 68, 0.30); color: #FCA5A5; font-weight: 700;"
    if tier == "Elevated":
        return "background-color: rgba(245, 158, 11, 0.25); color: #FCD34D; font-weight: 600;"
    return "color: #CBD5E1;"


def style_scanner_table(disp_df: pd.DataFrame) -> Any:
    """Apply subtle color-coding to scores and states."""

    def _state_color(val: Any) -> str:
        s = str(val)
        c = STATE_COLORS.get(s, "#64748B")
        if s == "NONE":
            return f"color: {c};"
        return f"color: {c}; font-weight: 700;"

    def _ret_color(val: Any) -> str:
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if pd.isna(v):
            return ""
        return "color: #34D399;" if v > 0 else ("color: #F87171;" if v < 0 else "")

    styler = disp_df.style
    styler = styler.map(score_badge_style, subset=["Short Squeeze", "Long Squeeze"])
    styler = styler.map(_state_color, subset=["State"])
    styler = styler.map(_ret_color, subset=["24h %", "OI 24h %"])
    styler = styler.format(
        {
            "Price ($)": lambda v: (f"{v:,.2f}" if abs(v) >= 100 else f"{v:,.4f}" if abs(v) >= 0.01 else f"{v:.8f}") if v == v else "n/a",
            "24h %": "{:+.2f}%",
            "Funding (bps)": "{:+.2f}",
            "OI 24h %": "{:+.2f}%",
            "Global L/S": "{:.2f}",
            "Short Squeeze": "{:.1f}",
            "Long Squeeze": "{:.1f}",
        },
        na_rep="N/A",
    )
    return styler


def make_deep_dive_subplots(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create 4-row synchronized Plotly chart: Candles, OI, Funding, Taker Ratio."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.46, 0.20, 0.17, 0.17],
        subplot_titles=(
            f"{symbol} Perpetual 1h Price",
            "Open Interest (USD M)",
            "Funding Rate (%/8h)",
            "Taker Buy/Sell Ratio",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=df["open_time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#10B981",
            decreasing_line_color="#EF4444",
        ),
        row=1,
        col=1,
    )

    oi_m = df["oi_usd"] / 1e6 if "oi_usd" in df else df["oi_value"] / 1e6
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=oi_m,
            mode="lines",
            name="OI ($M)",
            line={"color": "#6366F1", "width": 1.8},
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.12)",
        ),
        row=2,
        col=1,
    )

    fund_pct = df["funding_rate_last"] * 100.0 if "funding_rate_last" in df else df["funding_rate"] * 100.0
    fund_colors = ["#10B981" if (pd.notna(v) and v >= 0) else "#EF4444" for v in fund_pct]
    fig.add_trace(
        go.Bar(
            x=df["open_time"],
            y=fund_pct,
            name="Funding %",
            marker_color=fund_colors,
        ),
        row=3,
        col=1,
    )

    taker_s = df["taker_buy_sell_ratio_eff"] if "taker_buy_sell_ratio_eff" in df else df["taker_buy_sell_ratio"]
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=taker_s,
            mode="lines",
            name="Taker Ratio",
            line={"color": "#38BDF8", "width": 1.5},
        ),
        row=4,
        col=1,
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#64748B", row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1020",
        plot_bgcolor="#131A2E",
        height=760,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin={"l": 50, "r": 25, "t": 45, "b": 30},
    )
    return fig


_LIQ_HEATMAP_COLORSCALE: list[list[object]] = [
    [0.0, "#131A2E"],
    [0.15, "#1E1B4B"],
    [0.35, "#4C1D95"],
    [0.55, "#9D174D"],
    [0.75, "#EA580C"],
    [0.90, "#F59E0B"],
    [1.0, "#FEF08A"],
]


def make_liq_heatmap_figure(heatmap_df: pd.DataFrame, close_series: pd.DataFrame | None = None) -> go.Figure:
    """Build Plotly time x price liquidation heatmap (clearly marked ESTIMATE)."""
    fig = go.Figure()
    if heatmap_df.empty:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1020",
            plot_bgcolor="#131A2E",
            title="Estimated Liquidation Heatmap (No OI history available)",
        )
        return fig

    piv = heatmap_df.pivot(index="price_bin", columns="open_time", values="total_liq_usd")
    z_raw = piv.to_numpy(dtype=float) / 1e6
    z_m = np.where(np.isfinite(z_raw) & (z_raw > 0.0), z_raw, np.nan)
    max_p = float(piv.index.max()) if len(piv.index) else 100.0
    if max_p >= 10_000.0:
        y_fmt = ",.0f"
    elif max_p >= 1.0:
        y_fmt = ",.2f"
    elif max_p >= 0.01:
        y_fmt = ",.4f"
    else:
        y_fmt = ",.6f"
    hm_kwargs: dict[str, Any] = {
        "x": piv.columns,
        "y": piv.index,
        "z": z_m,
        "colorscale": _LIQ_HEATMAP_COLORSCALE,
        "autocolorscale": False,
        "hoverongaps": False,
        "colorbar": {"title": "Est. Liq ($M)"},
        "hovertemplate": f"Time: %{{x}}<br>Price: $%{{y:{y_fmt}}}<br>Est. Exposure: $%{{z:,.2f}}M<extra></extra>",
    }
    if np.isfinite(z_m).any():
        hm_kwargs["zmin"] = 0.0
    fig.add_trace(go.Heatmap(**hm_kwargs))
    if close_series is not None and not close_series.empty:
        t_min, t_max = piv.columns.min(), piv.columns.max()
        sub = close_series[(close_series["open_time"] >= t_min) & (close_series["open_time"] <= t_max)]
        fig.add_trace(
            go.Scatter(
                x=sub["open_time"],
                y=sub["close"],
                mode="lines",
                name="Spot Price",
                line={"color": "#E6E9F2", "width": 1.8},
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1020",
        plot_bgcolor="#131A2E",
        title="ESTIMATED Liquidation Heatmap (Time x Price)",
        yaxis_title="Price ($)",
        height=420,
        margin={"l": 55, "r": 20, "t": 45, "b": 35},
    )
    return fig


def make_liq_profile_figure(profile_df: pd.DataFrame, spot_price: float) -> go.Figure:
    """Build horizontal price-level histogram of open estimated long & short liq clusters."""
    fig = go.Figure()
    if profile_df.empty:
        return fig

    fig.add_trace(
        go.Bar(
            y=profile_df["pct_from_spot"],
            x=profile_df["short_liq_usd"] / 1e6,
            orientation="h",
            name="Short Liqs (Above Spot)",
            marker_color="#F59E0B",
        )
    )
    fig.add_trace(
        go.Bar(
            y=profile_df["pct_from_spot"],
            x=profile_df["long_liq_usd"] / 1e6,
            orientation="h",
            name="Long Liqs (Below Spot)",
            marker_color="#10B981",
        )
    )
    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="#E6E9F2",
        annotation_text=f"Spot {format_price(spot_price)}",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1020",
        plot_bgcolor="#131A2E",
        title="ESTIMATED Open Liquidation Profile (% From Spot)",
        xaxis_title="Estimated Open Liquidation Size ($M)",
        yaxis_title="% From Current Spot",
        barmode="overlay",
        height=420,
        margin={"l": 55, "r": 20, "t": 45, "b": 35},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
    )
    return fig


def make_score_history_figure(df: pd.DataFrame) -> go.Figure:
    """Plot short & long squeeze score curves with state shading bands."""
    fig = go.Figure()
    if df.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["short_squeeze_score"],
            mode="lines",
            name="Short Squeeze Risk",
            line={"color": "#10B981", "width": 2.0},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["long_squeeze_score"],
            mode="lines",
            name="Long Squeeze Risk",
            line={"color": "#EF4444", "width": 2.0},
        )
    )

    # Add subtle state background shading for non-NONE states
    times = df["open_time"].tolist()
    states = df["squeeze_state"].tolist()
    i = 0
    while i < len(states):
        st_val = states[i]
        if st_val == "NONE":
            i += 1
            continue
        j = i
        while j + 1 < len(states) and states[j + 1] == st_val:
            j += 1
        fig.add_vrect(
            x0=times[i],
            x1=times[j],
            fillcolor=STATE_COLORS.get(st_val, "#6366F1"),
            opacity=0.14,
            layer="below",
            line_width=0,
        )
        i = j + 1

    fig.add_hline(
        y=ELEVATED_SCORE_THRESHOLD,
        line_dash="dot",
        line_color="#6366F1",
        annotation_text=f"Setup Threshold ({int(ELEVATED_SCORE_THRESHOLD)})",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1020",
        plot_bgcolor="#131A2E",
        title="Squeeze Risk Scores (0–100) & State Machine Shading",
        yaxis_title="Score",
        yaxis_range=[0, 100],
        height=340,
        margin={"l": 50, "r": 20, "t": 45, "b": 30},
        legend={"orientation": "h", "y": 1.03, "x": 0.0},
    )
    return fig

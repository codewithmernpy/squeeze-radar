"""Configuration models, constants, and curated replay presets for SqueezeRadar."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DEFAULT_DOWNLOADED_DIR = PROJECT_ROOT / "data" / "downloaded"

DEFAULT_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "BNBUSDT",
)

SUITE_PROJECTS: tuple[str, ...] = ("QuantCritic", "PropPilot", "SqueezeRadar", "RiskGuard")
SUITE_FOOTER = "Part of the Crypto Quant Suite: " + " | ".join(SUITE_PROJECTS)

DISCLAIMER_TEXT = (
    "Estimated liquidation levels and model probabilities are analytical approximations, "
    "not financial advice."
)


def format_price(price: Any) -> str:
    """Format a USD price with thousands separators and magnitude-appropriate decimals."""
    try:
        val = float(price)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(val):
        return "N/A"
    sign = "-" if val < 0 else ""
    av = abs(val)
    if av == 0.0:
        return "$0.00"
    if av >= 10_000.0:
        return f"{sign}${av:,.0f}"
    if av >= 1.0:
        return f"{sign}${av:,.2f}"
    if av >= 0.01:
        return f"{sign}${av:,.4f}"
    if av >= 0.0001:
        return f"{sign}${av:,.6f}"
    return f"{sign}${av:,.8f}"


class LiqMapConfig(BaseModel):
    """Parameters for the estimated liquidation map engine."""

    leverage_weights: tuple[tuple[int, float], ...] = (
        (5, 0.25),
        (10, 0.35),
        (20, 0.20),
        (50, 0.15),
        (100, 0.05),
    )
    mmr: float = Field(default=0.005, gt=0.0, lt=0.1)
    log_step: float = Field(default=0.002, gt=0.0001, lt=0.05)
    profile_window_pct: float = Field(default=0.15, gt=0.02, lt=0.5)
    profile_bins: int = Field(default=60, ge=10, le=300)
    major_cluster_rel_threshold: float = Field(default=0.25, gt=0.0, lt=1.0)
    taker_direction_weight: float = Field(default=0.70, ge=0.0, le=1.0)
    price_direction_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    max_long_share: float = Field(default=0.85, ge=0.5, le=1.0)
    min_long_share: float = Field(default=0.15, ge=0.0, le=0.5)
    # Cluster metrics/profile need this many hourly OI observations (else NaN / empty)
    min_oi_history_bars: int = Field(default=24, ge=1, le=720)


class ScoreConfig(BaseModel):
    """Weights and scaling bounds for rule-based squeeze risk scores (0-100)."""

    weight_funding: float = 0.22
    weight_oi_divergence: float = 0.22
    weight_positioning: float = 0.20
    weight_liq_cluster: float = 0.20
    weight_compression: float = 0.16

    funding_z_cap: float = 2.5
    short_oi_div_cap: float = 0.06
    long_oi_div_cap: float = 0.12
    positioning_z_cap: float = 2.2
    liq_cluster_oi_share_cap: float = 0.06
    liq_cluster_dist_scale_pct: float = 6.0


ELEVATED_SCORE_THRESHOLD: float = 50.0
ACTIVE_SCORE_THRESHOLD: float = 60.0


class StateConfig(BaseModel):
    """Explicit state-machine thresholds per side (short squeeze & long squeeze)."""

    setup_score_min: float = ELEVATED_SCORE_THRESHOLD
    setup_exit_score: float = 38.0
    trigger_score_min: float = 42.0
    trigger_ret_1h: float = 0.005
    trigger_ret_4h: float = 0.012
    trigger_oi_drop_1h: float = -0.004
    trigger_oi_drop_4h: float = -0.010
    trigger_taker_high: float = 1.12
    trigger_taker_low: float = 0.88
    trigger_taker_z: float = 0.90
    in_progress_max_bars: int = 24
    exhaustion_cooldown_bars: int = 1
    exhaustion_ret_stall: float = 0.002


class MLConfig(BaseModel):
    """Configuration for labels and purged walk-forward validation."""

    horizon_bars: int = 24
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0
    min_oi_drop_pct: float = 0.03
    purge_bars: int = 24
    min_train_months: int = 6
    random_state: int = 42
    max_iter: int = 40
    max_bins: int = 64
    learning_rate: float = 0.08
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 100
    top_pct_precision: float = 0.05


class ProviderConfig(BaseModel):
    """Configuration for live REST providers, caching, and retries."""

    timeout_sec: float = 10.0
    max_retries: int = 3
    backoff_base_sec: float = 0.5
    cache_ttl_sec: float = 60.0
    history_bars: int = 720
    top_n_universe: int = 12
    binance_base_url: str = "https://fapi.binance.com"
    hyperliquid_url: str = "https://api.hyperliquid.xyz/info"
    data_vision_base_url: str = "https://data.binance.vision/data/futures/um"


SideLiteral = Literal["short_squeeze", "long_squeeze"]
StateLiteral = Literal["NONE", "SETUP", "TRIGGER", "IN_PROGRESS", "EXHAUSTION"]
DataModeLiteral = Literal["live", "replay"]

"""Purged walk-forward ML classifier, squeeze labeling, and event-study validation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from squeezeradar.config import ELEVATED_SCORE_THRESHOLD, MLConfig
from squeezeradar.features import MODEL_FEATURE_COLUMNS, open_interest_contracts

logger = logging.getLogger(__name__)

ML_EXTENDED_FEATURES: tuple[str, ...] = (
    *MODEL_FEATURE_COLUMNS,
    "nearest_short_liq_dist_pct",
    "nearest_long_liq_dist_pct",
    "short_liq_within_5pct_share",
    "long_liq_within_5pct_share",
    "short_squeeze_score",
    "long_squeeze_score",
)


@dataclass(frozen=True)
class WalkForwardTargetResult:
    """Out-of-sample walk-forward evaluation metrics for one target label."""

    target: str
    n_oos: int
    n_folds: int
    base_rate: float
    auc: float
    brier_score: float
    precision_top_5pct: float
    lift_top_5pct: float
    calibration_df: pd.DataFrame
    monthly_df: pd.DataFrame
    oos_predictions: pd.DataFrame


@dataclass(frozen=True)
class EventStudyResult:
    """Forward 0-48h return paths and hit rates for rule signals vs unconditional baseline."""

    summary_df: pd.DataFrame
    paths_df: pd.DataFrame


def build_squeeze_labels(
    df: pd.DataFrame,
    config: MLConfig | None = None,
) -> pd.DataFrame:
    """Compute future-looking squeeze_up and squeeze_down labels (STRICTLY FOR LABELS ONLY).

    `squeeze_up[t] = 1` iff over `k in [t+1 .. t+horizon]`:
      1) `high[k] >= close[t] + tp_atr_mult * atr_14[t]` strictly before
         `low[k] <= close[t] - sl_atr_mult * atr_14[t]`, AND
      2) `min(oi[t+1..t+horizon]) <= oi[t] * (1 - min_oi_drop_pct)` on CONTRACT open interest
         (`oi`; USD notional would drop mechanically with price and inflate `squeeze_down`).
    `squeeze_down[t]` is the exact mirror (`low` hits `-2*ATR` before `high` hits `+1*ATR`
    AND OI drops >= 3%).
    The final `horizon` bars are NaN because their future window is incomplete.
    """
    cfg = config or MLConfig()
    n = len(df)
    h = cfg.horizon_bars
    y_up = np.full(n, np.nan, dtype=float)
    y_dn = np.full(n, np.nan, dtype=float)

    if n <= h:
        return pd.DataFrame({"squeeze_up": y_up, "squeeze_down": y_dn}, index=df.index)

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr = df["atr_14"].to_numpy(dtype=float)
    oi = open_interest_contracts(df).to_numpy(dtype=float)

    tp_mult = cfg.tp_atr_mult
    sl_mult = cfg.sl_atr_mult
    oi_mult = 1.0 - cfg.min_oi_drop_pct

    # Build sliding future matrices of shape (n - h, h) cleanly via stride/slices
    n_valid = n - h
    fut_high = np.column_stack([high[k : k + n_valid] for k in range(1, h + 1)])
    fut_low = np.column_stack([low[k : k + n_valid] for k in range(1, h + 1)])
    fut_oi = np.column_stack([oi[k : k + n_valid] for k in range(1, h + 1)])

    c0 = close[:n_valid, None]
    a0 = atr[:n_valid, None]
    oi0 = oi[:n_valid]

    valid_base = (
        np.isfinite(close[:n_valid])
        & np.isfinite(atr[:n_valid])
        & (atr[:n_valid] > 0.0)
        & np.isfinite(oi0)
        & (oi0 > 0.0)
    )

    with np.errstate(all="ignore"):
        fut_oi_min = np.nanmin(fut_oi, axis=1)
    has_fut_oi = np.isfinite(fut_oi_min)
    oi_dropped = valid_base & has_fut_oi & (fut_oi_min <= oi0 * oi_mult)

    # First-hit step index (0..h-1), or h if not hit within horizon
    def _first_hit(mask: np.ndarray) -> np.ndarray:
        any_hit = np.any(mask, axis=1)
        idx = np.argmax(mask, axis=1)
        return np.where(any_hit, idx, h + 1)

    up_tp_mask = fut_high >= (c0 + tp_mult * a0)
    up_sl_mask = fut_low <= (c0 - sl_mult * a0)
    up_hit = _first_hit(up_tp_mask) < _first_hit(up_sl_mask)

    dn_tp_mask = fut_low <= (c0 - tp_mult * a0)
    dn_sl_mask = fut_high >= (c0 + sl_mult * a0)
    dn_hit = _first_hit(dn_tp_mask) < _first_hit(dn_sl_mask)

    usable = valid_base & has_fut_oi
    y_up[:n_valid] = np.where(usable, (up_hit & oi_dropped).astype(float), np.nan)
    y_dn[:n_valid] = np.where(usable, (dn_hit & oi_dropped).astype(float), np.nan)

    return pd.DataFrame({"squeeze_up": y_up, "squeeze_down": y_dn}, index=df.index)


def split_purged_walk_forward(
    df: pd.DataFrame,
    *,
    purge_bars: int = 24,
    min_train_months: int = 6,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Generate expanding monthly walk-forward splits with strict 24-bar purge.

    Returns a list of `(month_label, train_pos_idx, test_pos_idx, purged_pos_idx)`
    using positional integer indices into `df`.
    """
    if df.empty or "open_time" not in df.columns:
        return []

    open_times = pd.to_datetime(df["open_time"], utc=True)
    months = open_times.dt.tz_convert(None).dt.to_period("M")
    unique_months = sorted(months.unique())
    if len(unique_months) <= min_train_months:
        return []

    splits: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    purge_delta = pd.Timedelta(hours=purge_bars)

    for m_idx in range(min_train_months, len(unique_months)):
        test_month = unique_months[m_idx]
        test_mask = (months == test_month).to_numpy()
        test_idx = np.flatnonzero(test_mask)
        if len(test_idx) == 0:
            continue

        test_start = open_times.iloc[test_idx].min()
        purge_start = test_start - purge_delta

        train_mask = (open_times < purge_start).to_numpy()
        purge_mask = ((open_times >= purge_start) & (open_times < test_start)).to_numpy()

        train_idx = np.flatnonzero(train_mask)
        purged_idx = np.flatnonzero(purge_mask)
        if len(train_idx) == 0:
            continue

        splits.append((str(test_month), train_idx, test_idx, purged_idx))

    return splits


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return available symbol-agnostic causal ML feature columns present in `df`."""
    return [col for col in ML_EXTENDED_FEATURES if col in df.columns]


def _evaluate_target_walk_forward(
    pooled_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    splits: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    cfg: MLConfig,
) -> WalkForwardTargetResult:
    """Run purged expanding walk-forward training and OOS evaluation for one target."""
    y_all = pooled_df[target_col].to_numpy(dtype=float)
    X_all = pooled_df[feature_cols].to_numpy(dtype=np.float32)
    open_times = pooled_df["open_time"].to_numpy()
    symbols = pooled_df["symbol"].to_numpy()

    oos_frames: list[pd.DataFrame] = []
    monthly_rows: list[dict[str, object]] = []

    for month_str, train_idx, test_idx, purged_idx in splits:
        tr_valid = train_idx[np.isfinite(y_all[train_idx])]
        te_valid = test_idx[np.isfinite(y_all[test_idx])]
        if len(tr_valid) < 200 or len(te_valid) < 24:
            continue

        y_tr = y_all[tr_valid].astype(int)
        y_te = y_all[te_valid].astype(int)
        if len(np.unique(y_tr)) < 2:
            continue

        clf = HistGradientBoostingClassifier(
            max_iter=cfg.max_iter,
            max_bins=cfg.max_bins,
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.max_leaf_nodes,
            min_samples_leaf=cfg.min_samples_leaf,
            early_stopping=False,
            random_state=cfg.random_state,
        )
        clf.fit(X_all[tr_valid], y_tr)
        p_te = clf.predict_proba(X_all[te_valid])[:, 1]

        fold_base = float(np.mean(y_te))
        fold_auc = float(roc_auc_score(y_te, p_te)) if len(np.unique(y_te)) >= 2 else float("nan")
        n_top = max(1, int(math.ceil(cfg.top_pct_precision * len(y_te))))
        top_order = np.argsort(p_te)[::-1][:n_top]
        fold_p5 = float(np.mean(y_te[top_order]))
        fold_lift = (fold_p5 / fold_base) if fold_base > 0 else float("nan")

        monthly_rows.append(
            {
                "month": month_str,
                "n_train": int(len(tr_valid)),
                "n_purged": int(len(purged_idx)),
                "n_test": int(len(te_valid)),
                "base_rate": fold_base,
                "auc": fold_auc,
                "precision_top_5pct": fold_p5,
                "lift_top_5pct": fold_lift,
            }
        )
        oos_frames.append(
            pd.DataFrame(
                {
                    "open_time": open_times[te_valid],
                    "symbol": symbols[te_valid],
                    "month": month_str,
                    "y_true": y_te,
                    "y_prob": p_te,
                }
            )
        )

    if not oos_frames:
        return WalkForwardTargetResult(
            target=target_col,
            n_oos=0,
            n_folds=0,
            base_rate=float("nan"),
            auc=float("nan"),
            brier_score=float("nan"),
            precision_top_5pct=float("nan"),
            lift_top_5pct=float("nan"),
            calibration_df=pd.DataFrame(columns=["pred_mean", "obs_rate"]),
            monthly_df=pd.DataFrame(),
            oos_predictions=pd.DataFrame(),
        )

    oos_df = pd.concat(oos_frames, ignore_index=True)
    y_oos = oos_df["y_true"].to_numpy(dtype=int)
    p_oos = oos_df["y_prob"].to_numpy(dtype=float)

    base_rate = float(np.mean(y_oos))
    auc = float(roc_auc_score(y_oos, p_oos)) if len(np.unique(y_oos)) >= 2 else float("nan")
    brier = float(brier_score_loss(y_oos, p_oos))

    n_top_all = max(1, int(math.ceil(cfg.top_pct_precision * len(y_oos))))
    top_idx_all = np.argsort(p_oos)[::-1][:n_top_all]
    p5_all = float(np.mean(y_oos[top_idx_all]))
    lift_all = (p5_all / base_rate) if base_rate > 0 else float("nan")

    try:
        obs_rate, pred_mean = calibration_curve(y_oos, p_oos, n_bins=10, strategy="quantile")
        cal_df = pd.DataFrame({"pred_mean": pred_mean, "obs_rate": obs_rate})
    except ValueError:
        cal_df = pd.DataFrame(columns=["pred_mean", "obs_rate"])

    return WalkForwardTargetResult(
        target=target_col,
        n_oos=int(len(oos_df)),
        n_folds=len(monthly_rows),
        base_rate=base_rate,
        auc=auc,
        brier_score=brier,
        precision_top_5pct=p5_all,
        lift_top_5pct=lift_all,
        calibration_df=cal_df,
        monthly_df=pd.DataFrame(monthly_rows),
        oos_predictions=oos_df,
    )


def run_walk_forward(
    scored_frames_by_symbol: dict[str, pd.DataFrame],
    config: MLConfig | None = None,
) -> dict[str, WalkForwardTargetResult]:
    """Pool multi-symbol data and run purged expanding walk-forward validation."""
    cfg = config or MLConfig()
    pooled_parts: list[pd.DataFrame] = []
    for sym, df in scored_frames_by_symbol.items():
        if df.empty:
            continue
        labels = build_squeeze_labels(df, config=cfg)
        part = df.copy()
        part["symbol"] = sym
        part["squeeze_up"] = labels["squeeze_up"]
        part["squeeze_down"] = labels["squeeze_down"]
        pooled_parts.append(part)

    if not pooled_parts:
        empty = WalkForwardTargetResult(
            target="",
            n_oos=0,
            n_folds=0,
            base_rate=float("nan"),
            auc=float("nan"),
            brier_score=float("nan"),
            precision_top_5pct=float("nan"),
            lift_top_5pct=float("nan"),
            calibration_df=pd.DataFrame(),
            monthly_df=pd.DataFrame(),
            oos_predictions=pd.DataFrame(),
        )
        return {"squeeze_up": empty, "squeeze_down": empty}

    pooled = pd.concat(pooled_parts, ignore_index=True).sort_values(["open_time", "symbol"]).reset_index(drop=True)
    feature_cols = get_model_feature_columns(pooled)
    splits = split_purged_walk_forward(
        pooled,
        purge_bars=cfg.purge_bars,
        min_train_months=cfg.min_train_months,
    )

    return {
        "squeeze_up": _evaluate_target_walk_forward(pooled, "squeeze_up", feature_cols, splits, cfg),
        "squeeze_down": _evaluate_target_walk_forward(pooled, "squeeze_down", feature_cols, splits, cfg),
    }


def predict_latest_probabilities(
    scored_frames_by_symbol: dict[str, pd.DataFrame],
    config: MLConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Train causally on past purged bars and predict current-bar squeeze probabilities.

    For any latest evaluation timestamp `T_eval`, training uses strictly bars with
    `open_time < T_eval - purge_bars` so no future label leaks into current predictions.
    """
    cfg = config or MLConfig()
    pooled_parts: list[pd.DataFrame] = []
    latest_rows: list[pd.DataFrame] = []
    for sym, df in scored_frames_by_symbol.items():
        if df.empty:
            continue
        labels = build_squeeze_labels(df, config=cfg)
        part = df.copy()
        part["symbol"] = sym
        part["squeeze_up"] = labels["squeeze_up"]
        part["squeeze_down"] = labels["squeeze_down"]
        pooled_parts.append(part)
        latest_rows.append(part.iloc[[-1]])

    if not pooled_parts or not latest_rows:
        return {}

    pooled = pd.concat(pooled_parts, ignore_index=True)
    latest_df = pd.concat(latest_rows, ignore_index=True)
    feature_cols = get_model_feature_columns(pooled)

    min_latest_time = pd.to_datetime(latest_df["open_time"], utc=True).min()
    cutoff = min_latest_time - pd.Timedelta(hours=cfg.purge_bars)
    train_pool = pooled[pd.to_datetime(pooled["open_time"], utc=True) < cutoff]

    results: dict[str, dict[str, float]] = {sym: {} for sym in latest_df["symbol"].tolist()}
    X_latest = latest_df[feature_cols].to_numpy(dtype=np.float32)

    for target in ("squeeze_up", "squeeze_down"):
        valid_tr = train_pool[np.isfinite(train_pool[target].to_numpy(dtype=float))]
        if len(valid_tr) < 200 or valid_tr[target].nunique() < 2:
            for sym in results:
                results[sym][f"prob_{target}"] = float("nan")
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=cfg.max_iter,
            max_bins=cfg.max_bins,
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.max_leaf_nodes,
            min_samples_leaf=cfg.min_samples_leaf,
            early_stopping=False,
            random_state=cfg.random_state,
        )
        clf.fit(valid_tr[feature_cols].to_numpy(dtype=np.float32), valid_tr[target].to_numpy(dtype=int))
        probs = clf.predict_proba(X_latest)[:, 1]
        for sym, p in zip(latest_df["symbol"].tolist(), probs.tolist()):
            results[sym][f"prob_{target}"] = float(p)

    return results


def _decluster_mask(mask: np.ndarray, min_gap_bars: int = 12) -> np.ndarray:
    """Keep the first bar of each signal episode separated by at least `min_gap_bars`."""
    idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
        return mask
    keep = np.zeros_like(mask, dtype=bool)
    last_kept = -min_gap_bars - 1
    for i in idxs:
        if i - last_kept >= min_gap_bars:
            keep[i] = True
            last_kept = int(i)
    return keep


def compute_event_study(
    scored_frames_by_symbol: dict[str, pd.DataFrame],
    *,
    horizon_hours: int = 48,
    min_gap_bars: int = 12,
    ml_config: MLConfig | None = None,
) -> EventStudyResult:
    """Compute forward 0-48h return paths and hit rates vs unconditional baseline.

    Reports declustered signal episodes (min 12h separation per symbol) alongside
    exact sample counts, mean/median 24h & 48h returns, directional hit rates, and
    label hit rates without smoothing or fabricated projections.
    """
    cfg = ml_config or MLConfig()
    horizons = list(range(0, horizon_hours + 1))

    signal_specs = [
        ("Short Squeeze Score >= 50", lambda d: d["short_squeeze_score"] >= ELEVATED_SCORE_THRESHOLD, "up"),
        (
            "Short Squeeze State (SETUP/TRIGGER)",
            lambda d: d["short_squeeze_state"].isin(["SETUP", "TRIGGER"]),
            "up",
        ),
        ("Long Squeeze Score >= 50", lambda d: d["long_squeeze_score"] >= ELEVATED_SCORE_THRESHOLD, "down"),
        (
            "Long Squeeze State (SETUP/TRIGGER)",
            lambda d: d["long_squeeze_state"].isin(["SETUP", "TRIGGER"]),
            "down",
        ),
        ("Unconditional Baseline", lambda d: pd.Series(True, index=d.index), "both"),
    ]

    # Collect per-signal forward return matrices across symbols
    path_store: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in signal_specs}
    up_label_store: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in signal_specs}
    dn_label_store: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in signal_specs}

    for sym, df in scored_frames_by_symbol.items():
        if len(df) <= horizon_hours + 24:
            continue
        labels = build_squeeze_labels(df, config=cfg)
        close = df["close"].to_numpy(dtype=float)
        n = len(df)
        n_usable = n - horizon_hours

        # Matrix of forward returns shape (n_usable, horizon_hours + 1)
        fwd_rets = np.column_stack(
            [(close[h : h + n_usable] / close[:n_usable]) - 1.0 for h in horizons]
        )

        for name, mask_fn, _ in signal_specs:
            raw_mask = mask_fn(df).to_numpy(dtype=bool)[:n_usable]
            if name != "Unconditional Baseline":
                sel = _decluster_mask(raw_mask, min_gap_bars=min_gap_bars)
            else:
                sel = raw_mask
            if np.any(sel):
                path_store[name].append(fwd_rets[sel])
                up_label_store[name].append(labels["squeeze_up"].to_numpy(dtype=float)[:n_usable][sel])
                dn_label_store[name].append(labels["squeeze_down"].to_numpy(dtype=float)[:n_usable][sel])

    summary_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []

    for name, _, direction in signal_specs:
        if not path_store[name]:
            summary_rows.append(
                {
                    "signal": name,
                    "direction": direction,
                    "n_events": 0,
                    "mean_ret_24h_pct": np.nan,
                    "median_ret_24h_pct": np.nan,
                    "mean_ret_48h_pct": np.nan,
                    "pos_hit_rate_24h": np.nan,
                    "neg_hit_rate_24h": np.nan,
                    "squeeze_up_label_rate": np.nan,
                    "squeeze_down_label_rate": np.nan,
                }
            )
            continue

        mat = np.vstack(path_store[name])
        y_up_vec = np.concatenate(up_label_store[name])
        y_dn_vec = np.concatenate(dn_label_store[name])

        ret_24 = mat[:, min(24, horizon_hours)]
        ret_48 = mat[:, horizon_hours]

        valid_up = y_up_vec[np.isfinite(y_up_vec)]
        valid_dn = y_dn_vec[np.isfinite(y_dn_vec)]

        summary_rows.append(
            {
                "signal": name,
                "direction": direction,
                "n_events": int(len(mat)),
                "mean_ret_24h_pct": float(np.nanmean(ret_24) * 100.0),
                "median_ret_24h_pct": float(np.nanmedian(ret_24) * 100.0),
                "mean_ret_48h_pct": float(np.nanmean(ret_48) * 100.0),
                "pos_hit_rate_24h": float(np.nanmean(ret_24 > 0.0)),
                "neg_hit_rate_24h": float(np.nanmean(ret_24 < 0.0)),
                "squeeze_up_label_rate": float(np.mean(valid_up)) if len(valid_up) else np.nan,
                "squeeze_down_label_rate": float(np.mean(valid_dn)) if len(valid_dn) else np.nan,
            }
        )

        mean_path = np.nanmean(mat, axis=0) * 100.0
        sem_path = (np.nanstd(mat, axis=0) / np.sqrt(max(1, len(mat)))) * 100.0
        for h, mp, sp in zip(horizons, mean_path.tolist(), sem_path.tolist()):
            path_rows.append(
                {
                    "signal": name,
                    "hour": h,
                    "mean_ret_pct": mp,
                    "sem_pct": sp,
                    "n_events": int(len(mat)),
                }
            )

    return EventStudyResult(
        summary_df=pd.DataFrame(summary_rows),
        paths_df=pd.DataFrame(path_rows),
    )

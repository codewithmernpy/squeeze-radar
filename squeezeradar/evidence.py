"""Precomputed empirical evidence: build, save as compact JSON, and load back.

The purged walk-forward ML run and the 0-48h event study over the full bundled history take
1-2 minutes, so `python -m squeezeradar evidence --build` runs them once and writes the
results to `data/evidence/evidence_results.json`. The Evidence page and the AI analyst load
that file instantly; the page can also recompute live.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from squeezeradar import __version__
from squeezeradar.config import DEFAULT_SAMPLE_DIR, PROJECT_ROOT, MLConfig
from squeezeradar.data import load_all_sample_symbols
from squeezeradar.features import compute_features
from squeezeradar.liqmap import build_liquidation_map
from squeezeradar.model import (
    EventStudyResult,
    WalkForwardTargetResult,
    compute_event_study,
    predict_latest_probabilities,
    run_walk_forward,
)
from squeezeradar.scores import compute_scores_and_states

DEFAULT_EVIDENCE_PATH = PROJECT_ROOT / "data" / "evidence" / "evidence_results.json"
EVIDENCE_SCHEMA_VERSION = 1

_MONTHLY_COLUMNS = (
    "month",
    "n_train",
    "n_purged",
    "n_test",
    "base_rate",
    "auc",
    "precision_top_5pct",
    "lift_top_5pct",
)


@dataclass(frozen=True)
class EvidenceBundle:
    """Walk-forward ML metrics, event study, latest purged probabilities, and run metadata."""

    walk_forward: dict[str, WalkForwardTargetResult]
    event_study: EventStudyResult
    latest_probs: dict[str, dict[str, float]]
    meta: dict[str, Any]


def score_full_history(
    symbols: list[str] | tuple[str, ...] | None = None,
    sample_dir: Path | str = DEFAULT_SAMPLE_DIR,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    cache_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute causal features, estimated liqmap, scores and states over the full bundled/cached history."""
    datasets = load_all_sample_symbols(
        symbols=symbols,
        sample_dir=sample_dir,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
    )
    scored: dict[str, pd.DataFrame] = {}
    for sym, ds in datasets.items():
        if ds.aligned.empty:
            continue
        feat = compute_features(ds.aligned)
        lm = build_liquidation_map(feat, heatmap_bars=1)
        scored[sym] = compute_scores_and_states(feat, lm.summary)
    return scored


def build_evidence(
    symbols: list[str] | tuple[str, ...] | None = None,
    sample_dir: Path | str = DEFAULT_SAMPLE_DIR,
    config: MLConfig | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    cache_dir: Path | str | None = None,
) -> EvidenceBundle:
    """Run the full purged walk-forward + event study + latest purged probabilities."""
    cfg = config or MLConfig()
    t0 = time.perf_counter()
    scored = score_full_history(
        symbols,
        sample_dir=sample_dir,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
    )
    wf = run_walk_forward(scored, config=cfg)
    es = compute_event_study(scored, ml_config=cfg)
    probs = predict_latest_probabilities(scored, config=cfg)
    runtime = time.perf_counter() - t0

    starts = [df["open_time"].min() for df in scored.values() if not df.empty]
    ends = [df["open_time"].max() for df in scored.values() if not df.empty]
    latest_close = max(ends) + pd.Timedelta(hours=1) if ends else None
    meta: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "package_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "runtime_sec": round(runtime, 1),
        "symbols": sorted(scored.keys()),
        "bars_per_symbol": {s: int(len(df)) for s, df in sorted(scored.items())},
        "data_start_utc": min(starts).strftime("%Y-%m-%d %H:%M") if starts else None,
        "data_end_utc": max(ends).strftime("%Y-%m-%d %H:%M") if ends else None,
        "latest_probs_as_of_utc": latest_close.strftime("%Y-%m-%d %H:%M UTC") if latest_close else None,
        "ml_config": cfg.model_dump(),
    }
    if start_date is not None or end_date is not None:
        meta["requested_start"] = start_date
        meta["requested_end"] = end_date
    return EvidenceBundle(walk_forward=wf, event_study=es, latest_probs=probs, meta=meta)


def _num(val: Any, digits: int = 6) -> float | int | None:
    """JSON-safe number: NaN/Inf -> None, floats rounded to `digits` significant digits."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return int(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    fv = float(val)
    if not math.isfinite(fv):
        return None
    return float(f"{fv:.{digits}g}")


def _nan(val: Any) -> float:
    return float("nan") if val is None else float(val)


def evidence_to_dict(bundle: EvidenceBundle) -> dict[str, Any]:
    """Serialize an EvidenceBundle to a compact, JSON-safe dict (no per-row OOS predictions)."""
    targets: dict[str, Any] = {}
    for name, res in bundle.walk_forward.items():
        cal = res.calibration_df
        mon = res.monthly_df
        targets[name] = {
            "n_oos": int(res.n_oos),
            "n_folds": int(res.n_folds),
            "base_rate": _num(res.base_rate),
            "auc": _num(res.auc),
            "brier_score": _num(res.brier_score),
            "precision_top_5pct": _num(res.precision_top_5pct),
            "lift_top_5pct": _num(res.lift_top_5pct),
            "calibration": {
                "pred_mean": [_num(v) for v in cal.get("pred_mean", pd.Series(dtype=float)).tolist()],
                "obs_rate": [_num(v) for v in cal.get("obs_rate", pd.Series(dtype=float)).tolist()],
            },
            "monthly": [
                {c: (str(r[c]) if c == "month" else _num(r[c])) for c in _MONTHLY_COLUMNS if c in r}
                for r in mon.to_dict(orient="records")
            ],
        }
    es_summary = [
        {k: (v if isinstance(v, str) else _num(v)) for k, v in r.items()}
        for r in bundle.event_study.summary_df.to_dict(orient="records")
    ]
    paths: dict[str, Any] = {}
    if not bundle.event_study.paths_df.empty:
        for sig, grp in bundle.event_study.paths_df.groupby("signal", sort=False):
            grp = grp.sort_values("hour")
            paths[str(sig)] = {
                "n_events": int(grp["n_events"].iloc[0]),
                "mean_ret_pct": [_num(v) for v in grp["mean_ret_pct"].tolist()],
                "sem_pct": [_num(v) for v in grp["sem_pct"].tolist()],
            }
    return {
        "meta": bundle.meta,
        "walk_forward": targets,
        "event_study": {"summary": es_summary, "paths": paths},
        "latest_probs": {
            s: {k: _num(v) for k, v in d.items()} for s, d in sorted(bundle.latest_probs.items())
        },
    }


def evidence_from_dict(payload: dict[str, Any]) -> EvidenceBundle:
    """Rebuild an EvidenceBundle (results only, no OOS rows) from `evidence_to_dict` output."""
    wf: dict[str, WalkForwardTargetResult] = {}
    for name, t in payload.get("walk_forward", {}).items():
        cal = t.get("calibration", {})
        wf[name] = WalkForwardTargetResult(
            target=name,
            n_oos=int(t.get("n_oos", 0)),
            n_folds=int(t.get("n_folds", 0)),
            base_rate=_nan(t.get("base_rate")),
            auc=_nan(t.get("auc")),
            brier_score=_nan(t.get("brier_score")),
            precision_top_5pct=_nan(t.get("precision_top_5pct")),
            lift_top_5pct=_nan(t.get("lift_top_5pct")),
            calibration_df=pd.DataFrame(
                {
                    "pred_mean": [_nan(v) for v in cal.get("pred_mean", [])],
                    "obs_rate": [_nan(v) for v in cal.get("obs_rate", [])],
                }
            ),
            monthly_df=pd.DataFrame(
                [
                    {
                        c: (
                            r[c]
                            if c == "month"
                            else (
                                int(r[c])
                                if c in ("n_train", "n_purged", "n_test") and r.get(c) is not None
                                else _nan(r.get(c))
                            )
                        )
                        for c in _MONTHLY_COLUMNS
                        if c in r
                    }
                    for r in t.get("monthly", [])
                ]
            ),
            oos_predictions=pd.DataFrame(),
        )
    es = payload.get("event_study", {})
    summary_df = pd.DataFrame(
        [
            {
                k: (
                    v
                    if isinstance(v, str)
                    else (int(v or 0) if k == "n_events" else _nan(v))
                )
                for k, v in r.items()
            }
            for r in es.get("summary", [])
        ]
    )
    path_rows: list[dict[str, Any]] = []
    for sig, p in es.get("paths", {}).items():
        for hour, (mp, sp) in enumerate(zip(p.get("mean_ret_pct", []), p.get("sem_pct", []))):
            path_rows.append(
                {
                    "signal": sig,
                    "hour": hour,
                    "mean_ret_pct": _nan(mp),
                    "sem_pct": _nan(sp),
                    "n_events": int(p.get("n_events", 0)),
                }
            )
    probs = {
        s: {k: _nan(v) for k, v in d.items()} for s, d in payload.get("latest_probs", {}).items()
    }
    return EvidenceBundle(
        walk_forward=wf,
        event_study=EventStudyResult(summary_df=summary_df, paths_df=pd.DataFrame(path_rows)),
        latest_probs=probs,
        meta=dict(payload.get("meta", {})),
    )


def save_evidence(bundle: EvidenceBundle, path: Path | str = DEFAULT_EVIDENCE_PATH) -> Path:
    """Write the compact evidence JSON (creating parent directories)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(evidence_to_dict(bundle), separators=(",", ":"), allow_nan=False)
    out.write_text(text + "\n", encoding="utf-8")
    return out


def load_evidence(path: Path | str = DEFAULT_EVIDENCE_PATH) -> EvidenceBundle | None:
    """Load precomputed evidence, or None if the file is absent or has another schema version."""
    p = Path(path)
    if not p.exists():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    if payload.get("meta", {}).get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        return None
    return evidence_from_dict(payload)


def format_evidence_summary(bundle: EvidenceBundle) -> str:
    """Plain-text summary of the headline out-of-sample metrics (for the CLI)."""
    m = bundle.meta
    lines = [
        f"Evidence generated {m.get('generated_at_utc')} | symbols: {', '.join(m.get('symbols', []))} | "
        f"data {m.get('data_start_utc')} -> {m.get('data_end_utc')} UTC | runtime {m.get('runtime_sec')}s",
    ]
    for name, res in bundle.walk_forward.items():
        lines.append(
            f"{name:>12}: OOS AUC {res.auc:.3f} | base rate {res.base_rate * 100:.2f}% | "
            f"precision@top5% {res.precision_top_5pct * 100:.2f}% | lift {res.lift_top_5pct:.2f}x | "
            f"Brier {res.brier_score:.4f} | {res.n_oos:,} OOS rows, {res.n_folds} monthly folds"
        )
    es = bundle.event_study.summary_df
    if not es.empty:
        lines.append("Event study (declustered, 24h forward):")
        for r in es.to_dict(orient="records"):
            lines.append(
                f"  {r['signal']:<38} n={int(r['n_events']):>6} | mean 24h {r['mean_ret_24h_pct']:+.2f}% | "
                f"up-label {r['squeeze_up_label_rate'] * 100:.1f}% | down-label {r['squeeze_down_label_rate'] * 100:.1f}%"
            )
    return "\n".join(lines)

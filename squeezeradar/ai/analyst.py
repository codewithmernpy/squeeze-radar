"""Claude AI analyst orchestration using official anthropic 2.x beta tool_runner + offline brief."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic
import pandas as pd

from squeezeradar.ai.prompts import ANALYST_PROMPT, build_kickoff_message
from squeezeradar.ai.tools import AnalystContext, build_analyst_tools
from squeezeradar.config import DISCLAIMER_TEXT, ELEVATED_SCORE_THRESHOLD, format_price
from squeezeradar.scores import explain_score_row, state_for_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalystBrief:
    """Generated squeeze market brief with provenance, tool log, and fallback state."""

    mode: str  # "claude" | "offline"
    model: str | None
    market_summary: str
    setups: list[str]
    risks: list[str]
    watch_levels: list[str]
    narrative_markdown: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    fallback_reason: str | None = None
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )


def _format_offline_narrative(
    *,
    summary: str,
    setups: list[str],
    risks: list[str],
    watch_levels: list[str],
    fallback_reason: str | None,
) -> str:
    lines = []
    if fallback_reason:
        lines.append(f"> **Offline Template Brief**: {fallback_reason}\n")
    lines.append(f"### Market Summary\n{summary}\n")
    lines.append("### Active Setups & Scenarios")
    lines.extend(f"- {item}" for item in setups)
    lines.append("\n### Watch Levels (Estimated Clusters)")
    lines.extend(f"- {item}" for item in watch_levels)
    lines.append("\n### Key Risks & Limitations")
    lines.extend(f"- {item}" for item in risks)
    lines.append(f"\n---\n*{DISCLAIMER_TEXT}*")
    return "\n".join(lines)


def build_offline_brief(
    ctx: AnalystContext,
    *,
    focus_symbol: str | None = None,
    fallback_reason: str | None = None,
) -> AnalystBrief:
    """Generate a deterministic, evidence-grounded brief from local numbers without an API key."""
    ranked: list[tuple[str, float, pd.Series]] = []
    for sym, df in ctx.scored_frames.items():
        if df.empty:
            continue
        row = df.iloc[-1]
        max_sc = float(row.get("max_squeeze_score", 0.0) or 0.0)
        ranked.append((sym, max_sc, row))
    ranked.sort(key=lambda x: x[1], reverse=True)

    if not ranked:
        summary = "No perpetual contract data is currently loaded."
        return AnalystBrief(
            mode="offline",
            model=None,
            market_summary=summary,
            setups=[],
            risks=["Load replay or live market data before generating a brief."],
            watch_levels=[],
            narrative_markdown=summary + " Not financial advice.",
            tool_calls=[],
            fallback_reason=fallback_reason,
        )

    if focus_symbol and focus_symbol.upper() in ctx.scored_frames:
        fsym = focus_symbol.upper()
        top_sym, top_sc, top_row = next(
            (item for item in ranked if item[0] == fsym),
            ranked[0],
        )
    else:
        top_sym, top_sc, top_row = ranked[0]

    top_bd = explain_score_row(top_row, symbol=top_sym)
    elevated = [
        f"{s} ({sc:.1f}/100, {str(r.get('squeeze_state'))})"
        for s, sc, r in ranked
        if state_for_score(sc) != "Calm"
    ]
    elevated_text = (
        ", ".join(elevated)
        if elevated
        else f"none above {int(ELEVATED_SCORE_THRESHOLD)}/100 (highest: {top_sym} at {top_sc:.1f}/100)"
    )

    summary = (
        f"As of {ctx.as_of_label} ({ctx.data_source_label}), {len(ranked)} perpetual contracts were scanned. "
        f"Elevated squeeze crowding contracts: {elevated_text}. "
        f"Top focus is {top_sym} on the {top_bd.side.replace('_', ' ')} side "
        f"(score {top_bd.score:.1f}/100, state {top_bd.state})."
    )
    if ctx.walk_forward:
        up_res = ctx.walk_forward.get("squeeze_up")
        dn_res = ctx.walk_forward.get("squeeze_down")
        if up_res and dn_res and up_res.n_oos > 0:
            summary += (
                f" Purged walk-forward ML OOS AUC: squeeze_up={up_res.auc:.3f} "
                f"(lift@5%={up_res.lift_top_5pct:.2f}x), "
                f"squeeze_down={dn_res.auc:.3f} (lift@5%={dn_res.lift_top_5pct:.2f}x)."
            )

    setups: list[str] = []
    watch_levels: list[str] = []
    for sym, _, row in ranked[:3]:
        bd = explain_score_row(row, symbol=sym)
        spot = float(row.get("close", 0.0) or 0.0)
        atr = float(row.get("atr_14", 0.0) or 0.0)
        fr = row.get("funding_rate_last")
        oi24 = row.get("oi_chg_24h")
        fr_str = f"{fr*100:+.4f}%/8h" if pd.notna(fr) else "N/A"
        oi_str = f"{oi24*100:+.1f}%" if pd.notna(oi24) else "N/A"

        if bd.side == "short_squeeze":
            dist = row.get("nearest_short_liq_dist_pct")
            usd = row.get("nearest_short_liq_usd")
            trig_px = spot * (1.0 + (float(dist) / 100.0 if pd.notna(dist) else 0.02))
            inv_px = spot - max(atr, spot * 0.01)
            clust_txt = (
                f"estimated short-liq cluster +{dist:.2f}% (${(usd or 0)/1e6:,.1f}M)"
                if pd.notna(dist)
                else "no nearby short-liq cluster"
            )
            setups.append(
                f"{sym} SHORT-SQUEEZE ({bd.score:.1f}/100, {bd.state}): spot {format_price(spot)}, "
                f"funding {fr_str}, OI 24h {oi_str}. Trigger scenario above {format_price(trig_px)} "
                f"toward {clust_txt} if OI falls on aggressive taker buying; invalidation below {format_price(inv_px)} (1 ATR)."
            )
            watch_levels.append(
                f"{sym}: Upside estimated short-liq magnet ~{format_price(trig_px)}; downside invalidation {format_price(inv_px)}."
            )
        else:
            dist = row.get("nearest_long_liq_dist_pct")
            usd = row.get("nearest_long_liq_usd")
            trig_px = spot * (1.0 - (float(dist) / 100.0 if pd.notna(dist) else 0.02))
            inv_px = spot + max(atr, spot * 0.01)
            clust_txt = (
                f"estimated long-liq cluster -{dist:.2f}% (${(usd or 0)/1e6:,.1f}M)"
                if pd.notna(dist)
                else "no nearby long-liq cluster"
            )
            setups.append(
                f"{sym} LONG-SQUEEZE / CASCADE ({bd.score:.1f}/100, {bd.state}): spot {format_price(spot)}, "
                f"funding {fr_str}, OI 24h {oi_str}. Cascade scenario below {format_price(trig_px)} "
                f"into {clust_txt} if longs unwind; invalidation above {format_price(inv_px)} (1 ATR)."
            )
            watch_levels.append(
                f"{sym}: Downside estimated long-liq magnet ~{format_price(trig_px)}; upside invalidation {format_price(inv_px)}."
            )

    risks = [
        "Liquidation levels and USD cluster sizes are analytical ESTIMATES inferred from hourly delta-OI and modeled leverage buckets, not live exchange orderbooks.",
        "Crowded positioning can persist or unwind gradually via time-decay without triggering a forced liquidation cascade.",
        "In Hyperliquid or degraded fallback modes, Binance-specific account/position L/S ratios may be unavailable and omitted from scoring.",
    ]
    if ctx.event_study is not None and not ctx.event_study.summary_df.empty:
        base_row = ctx.event_study.summary_df[
            ctx.event_study.summary_df["signal"] == "Unconditional Baseline"
        ]
        if not base_row.empty:
            br = base_row.iloc[0]
            risks.append(
                f"Historical unconditional 24h baseline across bundled sample: "
                f"mean={br['mean_ret_24h_pct']:+.2f}%, up_label={br['squeeze_up_label_rate']*100:.1f}%, "
                f"down_label={br['squeeze_down_label_rate']*100:.1f}%."
            )

    narrative = _format_offline_narrative(
        summary=summary,
        setups=setups,
        risks=risks,
        watch_levels=watch_levels,
        fallback_reason=fallback_reason,
    )
    return AnalystBrief(
        mode="offline",
        model=None,
        market_summary=summary,
        setups=setups,
        risks=risks,
        watch_levels=watch_levels,
        narrative_markdown=narrative,
        tool_calls=[],
        fallback_reason=fallback_reason,
    )


def generate_analyst_brief(
    ctx: AnalystContext,
    *,
    api_key: str | None = None,
    model: str | None = None,
    focus_symbol: str | None = None,
    client_factory: Any = None,
) -> AnalystBrief:
    """Run the Claude AI Analyst via official anthropic 2.x tool_runner, with safe offline fallback."""
    effective_key = (api_key or os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not effective_key and client_factory is None:
        return build_offline_brief(
            ctx,
            focus_symbol=focus_symbol,
            fallback_reason="No Anthropic API key provided; rendered deterministic offline brief.",
        )

    tools, captured_brief = build_analyst_tools(ctx)
    chosen_model = model or os.getenv("SR_MODEL", "claude-opus-5-5")
    kickoff = build_kickoff_message(
        as_of_label=ctx.as_of_label,
        data_source_label=ctx.data_source_label,
        focus_symbol=focus_symbol,
    )

    tool_log: list[dict[str, Any]] = []
    text_chunks: list[str] = []

    try:
        client = (
            client_factory(api_key=effective_key)
            if client_factory is not None
            else anthropic.Anthropic(api_key=effective_key)
        )
        runner = client.beta.messages.tool_runner(
            model=chosen_model,
            max_tokens=16000,
            max_iterations=16,
            system=ANALYST_PROMPT,
            output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            tools=tools,
            messages=[{"role": "user", "content": kickoff}],
        )
        for message in runner:
            stop_reason = getattr(message, "stop_reason", None)
            if stop_reason == "refusal":
                return build_offline_brief(
                    ctx,
                    focus_symbol=focus_symbol,
                    fallback_reason="Claude returned stop_reason='refusal'; fell back to deterministic offline brief.",
                )
            for block in getattr(message, "content", []) or []:
                btype = getattr(block, "type", None)
                if btype == "tool_use":
                    tool_log.append(
                        {
                            "tool": getattr(block, "name", "unknown"),
                            "input": getattr(block, "input", {}),
                        }
                    )
                elif btype == "text":
                    txt = getattr(block, "text", "")
                    if txt:
                        text_chunks.append(txt)
    except (
        anthropic.APIConnectionError,
        anthropic.RateLimitError,
        anthropic.APIStatusError,
        anthropic.APIError,
    ) as exc:
        logger.warning("Anthropic API error, falling back to offline brief: %s", exc)
        return build_offline_brief(
            ctx,
            focus_symbol=focus_symbol,
            fallback_reason=f"Anthropic API error ({type(exc).__name__}): {exc}",
        )

    narrative = "\n\n".join(text_chunks).strip()
    summary = str(captured_brief.get("market_summary") or "").strip()
    setups = [str(s) for s in captured_brief.get("setups", [])]
    risks = [str(r) for r in captured_brief.get("risks", [])]
    watch_levels = [str(w) for w in captured_brief.get("watch_levels", [])]

    if not summary:
        summary = narrative or "Claude analysis completed."
    if not narrative:
        narrative = _format_offline_narrative(
            summary=summary,
            setups=setups,
            risks=risks,
            watch_levels=watch_levels,
            fallback_reason=None,
        )
    if "not financial advice" not in narrative.lower():
        narrative += f"\n\n*{DISCLAIMER_TEXT}*"

    return AnalystBrief(
        mode="claude",
        model=chosen_model,
        market_summary=summary,
        setups=setups,
        risks=risks,
        watch_levels=watch_levels,
        narrative_markdown=narrative,
        tool_calls=tool_log,
        fallback_reason=None,
    )

"""Tests for Anthropic AI Analyst tools (no API key), runner invocation, and fallbacks."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import anthropic
import httpx2 as httpx  # the anthropic 2.x SDK ships httpx2 (plain httpx is not installed with it)
import numpy as np

from squeezeradar.ai import AnalystContext, build_analyst_tools, build_offline_brief, generate_analyst_brief
from squeezeradar.config import format_price
from squeezeradar.pipeline import run_radar_scan


def _build_test_context() -> AnalystContext:
    scan_res = run_radar_scan(symbols=["BTCUSDT", "ETHUSDT"], mode="replay", limit=400)
    return AnalystContext(
        scored_frames={s: a.scored_df for s, a in scan_res.analyses.items()},
        liq_maps={s: a.liq_map for s, a in scan_res.analyses.items()},
        as_of_label=scan_res.as_of_label,
        data_source_label=scan_res.badge.label,
    )


def test_ai_tools_build_valid_schemas_and_compact_json_without_key() -> None:
    """Verify all 7 @beta_tool schemas and compact JSON outputs (<3000 chars) without API key."""
    ctx = _build_test_context()
    tools, captured = build_analyst_tools(ctx)
    names = [t.name for t in tools]
    assert names == [
        "scan_market",
        "get_symbol_snapshot",
        "get_liquidation_map",
        "get_score_breakdown",
        "get_event_study",
        "get_model_evidence",
        "submit_brief",
    ]

    for tool in tools:
        d = tool.to_dict()
        assert d["name"] == tool.name
        assert isinstance(d["description"], str) and len(d["description"]) > 5
        schema = d["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema

    # Invoke every tool directly
    tool_map = {t.name: t for t in tools}
    for call_res in (
        tool_map["scan_market"].call({}),
        tool_map["get_symbol_snapshot"].call({"symbol": "BTCUSDT"}),
        tool_map["get_liquidation_map"].call({"symbol": "BTCUSDT"}),
        tool_map["get_score_breakdown"].call({"symbol": "BTCUSDT"}),
        tool_map["get_event_study"].call({"signal": "short"}),
        tool_map["get_model_evidence"].call({}),
        tool_map["submit_brief"].call(
            {
                "market_summary": "Summary",
                "setups": ["BTCUSDT setup"],
                "risks": ["Estimate risk"],
                "watch_levels": ["$60,000"],
            }
        ),
    ):
        assert isinstance(call_res, str)
        assert len(call_res) < 3000
        parsed = json.loads(call_res)
        assert isinstance(parsed, dict)

    assert captured["market_summary"] == "Summary"
    assert captured["setups"] == ["BTCUSDT setup"]


def test_offline_analyst_brief_works_without_api_key(monkeypatch) -> None:
    """Verify deterministic offline brief renders cleanly with no ANTHROPIC_API_KEY."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = _build_test_context()
    brief = generate_analyst_brief(ctx, api_key=None, focus_symbol="BTCUSDT")

    assert brief.mode == "offline"
    assert "BTCUSDT" in brief.market_summary
    assert len(brief.setups) >= 1
    assert len(brief.risks) >= 1
    assert len(brief.watch_levels) >= 1
    assert "not financial advice" in brief.narrative_markdown.lower()


def test_claude_tool_runner_exact_kwargs_and_fallbacks(monkeypatch) -> None:
    """Verify exact beta.messages.tool_runner call args, refusal handling, and API error fallback."""
    ctx = _build_test_context()
    captured_kwargs: list[dict] = []

    class _FakeMessages:
        def __init__(self, behavior: str) -> None:
            self.behavior = behavior

        def tool_runner(self, **kwargs):
            captured_kwargs.append(kwargs)
            if self.behavior == "ok":
                return [
                    SimpleNamespace(
                        stop_reason="tool_use",
                        content=[SimpleNamespace(type="tool_use", name="scan_market", input={})],
                    ),
                    SimpleNamespace(
                        stop_reason="end_turn",
                        content=[SimpleNamespace(type="text", text="BTCUSDT brief grounded in tools. Not financial advice.")],
                    ),
                ]
            if self.behavior == "refusal":
                return [SimpleNamespace(stop_reason="refusal", content=[])]
            if self.behavior == "rate_limit":
                req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                resp = httpx.Response(429, request=req)
                raise anthropic.RateLimitError("rate limited", response=resp, body=None)
            raise RuntimeError("unexpected")

    def _make_factory(behavior: str):
        return lambda api_key: SimpleNamespace(
            beta=SimpleNamespace(messages=_FakeMessages(behavior))
        )

    # 1) Normal success path
    brief_ok = generate_analyst_brief(ctx, api_key="test-key", client_factory=_make_factory("ok"))
    assert brief_ok.mode == "claude"
    assert len(brief_ok.tool_calls) == 1
    assert brief_ok.tool_calls[0]["tool"] == "scan_market"
    kw = captured_kwargs[0]
    assert kw["output_config"] == {"effort": "high"}
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    assert "temperature" not in kw
    assert "top_p" not in kw
    assert "thinking" not in kw
    assert "tool_choice" not in kw

    # 2) stop_reason == "refusal" -> offline fallback
    brief_refusal = generate_analyst_brief(
        ctx, api_key="test-key", client_factory=_make_factory("refusal")
    )
    assert brief_refusal.mode == "offline"
    assert "refusal" in (brief_refusal.fallback_reason or "").lower()

    # 3) RateLimitError -> offline fallback
    brief_rl = generate_analyst_brief(
        ctx, api_key="test-key", client_factory=_make_factory("rate_limit")
    )
    assert brief_rl.mode == "offline"
    assert "RateLimitError" in (brief_rl.fallback_reason or "")


def test_format_price_and_no_scientific_notation_in_brief_and_tools() -> None:
    """Verify magnitude-based price formatting across offline brief, scenarios, watch levels, and AI tools."""
    assert format_price(109557.3) == "$109,557"
    assert format_price(3846.42) == "$3,846.42"
    assert format_price(0.18634) == "$0.1863"
    assert format_price(0.003430) == "$0.003430"
    assert format_price(0.00001234) == "$0.00001234"
    assert format_price(0.0) == "$0.00"
    assert format_price(float("nan")) == "N/A"
    assert format_price(None) == "N/A"

    scan_res = run_radar_scan(symbols=["BTCUSDT", "ETHUSDT", "DOGEUSDT"], mode="replay", limit=400)
    frames = {s: a.scored_df.copy() for s, a in scan_res.analyses.items()}
    # Add a synthetic sub-cent contract to verify 6-decimal formatting in brief + tools
    pepe_df = frames["DOGEUSDT"].copy()
    pepe_df["close"] = 0.003430
    pepe_df["atr_14"] = 0.000120
    frames["1000PEPEUSDT"] = pepe_df
    liq_maps = dict(scan_res.analyses["DOGEUSDT"].liq_map.__dict__)
    from dataclasses import replace

    pepe_lm = replace(scan_res.analyses["DOGEUSDT"].liq_map, as_of_price=0.003430)
    ctx = AnalystContext(
        scored_frames=frames,
        liq_maps={**{s: a.liq_map for s, a in scan_res.analyses.items()}, "1000PEPEUSDT": pepe_lm},
        as_of_label=scan_res.as_of_label,
        data_source_label=scan_res.badge.label,
    )

    brief = build_offline_brief(ctx, focus_symbol="BTCUSDT")
    sci_pat = re.compile(r"\d+(?:\.\d+)?e[+-]\d+", re.IGNORECASE)
    for text in [brief.narrative_markdown, *brief.setups, *brief.watch_levels]:
        assert not sci_pat.search(text), f"Unexpected scientific notation in: {text}"
    assert brief.narrative_markdown.lower().count("not financial advice") == 1
    assert "not financial advice. not financial advice" not in brief.narrative_markdown.lower()

    btc_spot = float(frames["BTCUSDT"].iloc[-1]["close"])
    assert format_price(btc_spot) in "\n".join(brief.setups)

    # AI tool text includes formatted prices with no scientific notation
    tools, _ = build_analyst_tools(ctx)
    tool_map = {t.name: t for t in tools}
    scan_json = json.loads(tool_map["scan_market"].call({}))
    by_sym = {row["sym"]: row for row in scan_json["symbols"]}
    assert by_sym["BTCUSDT"]["px_fmt"] == format_price(btc_spot)
    assert by_sym["1000PEPEUSDT"]["px_fmt"] == "$0.003430"

    snap_json = json.loads(tool_map["get_symbol_snapshot"].call({"symbol": "BTCUSDT"}))
    assert snap_json["close_fmt"] == format_price(btc_spot)

    lm_json = json.loads(tool_map["get_liquidation_map"].call({"symbol": "BTCUSDT"}))
    assert lm_json["spot_price_fmt"] == format_price(ctx.liq_maps["BTCUSDT"].as_of_price)

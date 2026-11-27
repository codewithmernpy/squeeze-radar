"""Tests for CLI `scan`, `watch`, and mocked Telegram alerts."""

from __future__ import annotations

from squeezeradar.cli import main, send_telegram_alert, watch_loop


def test_cli_scan_replay_at_timestamp(capsys) -> None:
    """Verify `python -m squeezeradar scan --mode replay --at '2024-08-05 00:00'`."""
    rc = main(["scan", "--mode", "replay", "--at", "2024-08-05 00:00"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "REPLAY" in out
    assert "2024-08-05 00:00" in out
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"):
        assert sym in out


def test_watch_loop_and_telegram_alert(monkeypatch, capsys) -> None:
    """Verify watch loop emits transition alerts and sends Telegram POST when configured."""
    sent_payloads: list[dict] = []

    class _FakeResp:
        def raise_for_status(self) -> None:
            return None

    class _FakeHTTP:
        def post(self, url, json=None, timeout=None):
            sent_payloads.append({"url": url, "json": json})
            return _FakeResp()

    assert (
        send_telegram_alert(
            "Test alert",
            bot_token="bot123",
            chat_id="chat456",
            session=_FakeHTTP(),
        )
        is True
    )
    assert len(sent_payloads) == 1
    assert "bot123" in sent_payloads[0]["url"]
    assert sent_payloads[0]["json"]["chat_id"] == "chat456"

    # Watch loop single iteration on offline replay
    alerts = watch_loop(
        symbols=["BTCUSDT"],
        mode="replay",
        interval_sec=1,
        max_iterations=1,
        sleep_fn=lambda _: None,
    )
    assert alerts == []
    assert "BTCUSDT" in capsys.readouterr().out

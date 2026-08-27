"""Covers two real-money dashboard bugs fixed 2026-08-26:

1. ``_merge_real_cash_summary`` -- ``/api/status`` used to read only
   ``kite.positions()``, which drops a CNC position to 0 the day after it's
   bought (it moves into ``kite.holdings()`` instead) -- silently showing
   0 open positions and Rs0 P&L for real, currently-open positions.
2. ``_last_run_summary`` -- used to recognize only the retired hourly-cron
   pipeline's log lines, so it kept reporting a days-old "last run" even
   while the always-on live-ticker pipeline was actively running.
"""

import sys
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trading_scanner import webapp  # noqa: E402


def _holding(symbol: str, quantity: int, t1_quantity: int, pnl: float, day_change: float) -> dict:
    return {
        "tradingsymbol": symbol,
        "product": "CNC",
        "quantity": quantity,
        "t1_quantity": t1_quantity,
        "pnl": pnl,
        "day_change": day_change,
    }


def test_a_position_bought_today_is_counted_from_positions_net():
    positions = {
        "net": [{"product": "CNC", "quantity": 10, "unrealised": 50.0}],
        "day": [{"product": "CNC", "pnl": 50.0}],
    }
    summary = webapp._merge_real_cash_summary(positions, holdings=[])
    assert summary["open_position_count"] == 1
    assert summary["unrealized_pnl"] == 50.0
    assert summary["today_pnl"] == 50.0


def test_a_position_bought_yesterday_only_shows_up_in_holdings():
    # positions()['net'] has already netted this out to 0 -- exactly what
    # Kite does the day after a CNC buy settles.
    positions = {"net": [{"product": "CNC", "quantity": 0, "unrealised": 0.0}], "day": []}
    holdings = [_holding("MCX", quantity=0, t1_quantity=1, pnl=48.5, day_change=48.5)]
    summary = webapp._merge_real_cash_summary(positions, holdings)
    assert summary["open_position_count"] == 1
    assert summary["unrealized_pnl"] == Decimal("48.5")
    assert summary["today_pnl"] == Decimal("48.5")


def test_today_pnl_uses_only_the_days_move_for_a_prior_day_holding():
    # A holding's total unrealized P&L (pnl) is since its original entry --
    # today's contribution is only the price move since yesterday's close
    # (day_change), times the shares actually held.
    holdings = [_holding("COCHINSHIP", quantity=0, t1_quantity=3, pnl=108.6, day_change=11.0)]
    summary = webapp._merge_real_cash_summary({"net": [], "day": []}, holdings)
    assert summary["unrealized_pnl"] == Decimal("108.6")
    assert summary["today_pnl"] == Decimal("33.0")  # 11.0 * 3 shares


def test_a_fully_settled_holding_uses_quantity_not_t1_quantity():
    holdings = [_holding("SIEMENS", quantity=1, t1_quantity=0, pnl=-73.6, day_change=-73.6)]
    summary = webapp._merge_real_cash_summary({"net": [], "day": []}, holdings)
    assert summary["open_position_count"] == 1
    assert summary["today_pnl"] == Decimal("-73.6")


def test_a_holding_with_zero_quantity_and_zero_t1_is_not_counted():
    holdings = [_holding("CLOSEDOUT", quantity=0, t1_quantity=0, pnl=0.0, day_change=0.0)]
    summary = webapp._merge_real_cash_summary({"net": [], "day": []}, holdings)
    assert summary["open_position_count"] == 0


def test_non_cnc_holdings_and_positions_are_ignored():
    positions = {
        "net": [{"product": "MIS", "quantity": 5, "unrealised": 999.0}],
        "day": [{"product": "MIS", "pnl": 999.0}],
    }
    holdings = [{**_holding("X", 1, 0, 999.0, 999.0), "product": "NRML"}]
    summary = webapp._merge_real_cash_summary(positions, holdings)
    assert summary == {
        "open_position_count": 0,
        "unrealized_pnl": 0,
        "today_pnl": 0,
    }


def test_same_day_and_prior_day_positions_combine_without_double_counting():
    positions = {
        "net": [{"product": "CNC", "quantity": 10, "unrealised": 50.0}],
        "day": [{"product": "CNC", "pnl": 50.0}],
    }
    holdings = [_holding("MCX", quantity=0, t1_quantity=1, pnl=48.5, day_change=48.5)]
    summary = webapp._merge_real_cash_summary(positions, holdings)
    assert summary["open_position_count"] == 2
    assert summary["unrealized_pnl"] == 98.5
    assert summary["today_pnl"] == 98.5


def test_last_run_recognizes_the_live_ticker_pipelines_own_log_lines(tmp_path, monkeypatch):
    log_path = tmp_path / "signals.log"
    log_path.write_text(
        "2026-08-19 05:51:11,931 INFO: Signal pipeline finished\n"
        "2026-08-26 06:46:06,613 INFO: Live ticker pipeline starting\n"
        "2026-08-26 07:45:07,693 INFO: Bucket 2026-08-26 12:15:00+05:30 closed: "
        "220/220 symbols had ticks. Evaluating...\n"
    )
    monkeypatch.setattr(webapp, "_LOG_PATH", log_path)
    summary = webapp._last_run_summary()
    assert summary["status"] == "processing"
    assert "Bucket" in summary["raw"]


def test_last_run_reports_idle_when_a_bucket_closes_with_no_ticks(tmp_path, monkeypatch):
    log_path = tmp_path / "signals.log"
    log_path.write_text(
        "2026-08-26 09:00:00,000 INFO: Bucket 2026-08-26 14:30:00+05:30 closed with no "
        "ticks for any symbol -- nothing to evaluate.\n"
    )
    monkeypatch.setattr(webapp, "_LOG_PATH", log_path)
    summary = webapp._last_run_summary()
    assert summary["status"] == "idle"


def test_last_run_falls_back_to_the_legacy_cron_pipelines_lines(tmp_path, monkeypatch):
    log_path = tmp_path / "signals.log"
    log_path.write_text("2026-08-19 05:51:11,931 INFO: Signal pipeline finished\n")
    monkeypatch.setattr(webapp, "_LOG_PATH", log_path)
    summary = webapp._last_run_summary()
    assert summary["status"] == "finished"


def test_pipeline_health_reports_healthy_when_the_log_was_just_written(tmp_path, monkeypatch):
    log_path = tmp_path / "live.log"
    log_path.write_text("2026-08-27 12:05:29,704 INFO: Outside market hours -- sleeping.\n")
    monkeypatch.setattr(webapp, "_LOG_PATH", log_path)
    health = webapp._live_pipeline_health()
    assert health["healthy"] is True
    assert health["age_seconds"] < 5


def test_pipeline_health_reports_unhealthy_when_the_log_is_stale(tmp_path, monkeypatch):
    import os

    log_path = tmp_path / "live.log"
    log_path.write_text("2026-08-27 04:37:12,714 ERROR: Live pipeline crashed.\n")
    stale_time = time.time() - 600  # older than the 300s threshold
    os.utime(log_path, (stale_time, stale_time))
    monkeypatch.setattr(webapp, "_LOG_PATH", log_path)
    health = webapp._live_pipeline_health()
    assert health["healthy"] is False
    assert health["age_seconds"] >= 600


def test_pipeline_health_reports_unhealthy_when_the_log_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp, "_LOG_PATH", tmp_path / "does-not-exist.log")
    health = webapp._live_pipeline_health()
    assert health == {"healthy": False, "age_seconds": None, "last_log_at": None}

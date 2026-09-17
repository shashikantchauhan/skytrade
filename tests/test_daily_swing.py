from datetime import UTC, date, datetime

import pytest

from trading_scanner.application.daily_swing import SwingSetup, confirm_entry, trailed_stop
from trading_scanner.daily_swing_live import expiry_exit_required


def _setup(side: int = 1) -> SwingSetup:
    return SwingSetup(
        symbol="TEST.NS",
        setup_date=date(2026, 9, 16),
        side=side,
        trigger=105 if side == 1 else 95,
        stop=104 if side == 1 else 98,
        atr=5,
        setup_close=100,
        outer_structure=130 if side == 1 else 70,
        score=1.0,
    )


def test_long_confirmation_enters_at_same_candle_close() -> None:
    candle = {
        "timestamp": datetime(2026, 9, 17, 4, 15, tzinfo=UTC),
        "open": 103,
        "high": 108,
        "low": 102,
        "close": 107,
        "volume": 1200,
    }
    market = {"open": 100, "close": 101}

    result = confirm_entry(
        _setup(), candle, market, slot=1, slot_volume_base=1000, session_open=103
    )

    assert result is not None
    assert result.entry == 107
    assert result.timestamp == candle["timestamp"]
    assert result.risk == 3
    assert result.target == 116


def test_confirmation_rejects_abnormal_opening_gap() -> None:
    candle = {
        "timestamp": datetime(2026, 9, 17, 3, 45, tzinfo=UTC),
        "open": 110,
        "high": 112,
        "low": 109,
        "close": 111,
        "volume": 1200,
    }
    assert confirm_entry(_setup(), candle, {"open": 100, "close": 100}, 0, 1000) is None


def test_later_confirmation_still_uses_first_bar_open_for_gap_gate() -> None:
    candle = {
        "timestamp": datetime(2026, 9, 17, 4, 45, tzinfo=UTC),
        "open": 104,
        "high": 108,
        "low": 103,
        "close": 107,
        "volume": 1200,
    }
    assert (
        confirm_entry(
            _setup(),
            candle,
            {"open": 100, "close": 100},
            2,
            1000,
            session_open=110,
        )
        is None
    )


@pytest.mark.parametrize(
    ("best", "expected"),
    [(106, 96), (123, 107), (130, 123.75)],
)
def test_long_trailing_stop_matches_frozen_rules(best: float, expected: float) -> None:
    assert trailed_stop(1, 107, 96, 11, 5, best) == expected


def test_short_trailing_stop_moves_in_mirrored_direction() -> None:
    assert trailed_stop(-1, 95, 104, 9, 4, 75) == 80


def test_expiry_guard_exits_with_ten_calendar_days_remaining() -> None:
    assert expiry_exit_required("2026-09-24", date(2026, 9, 14))
    assert not expiry_exit_required("2026-09-24", date(2026, 9, 13))
    assert not expiry_exit_required(None, date(2026, 9, 20))

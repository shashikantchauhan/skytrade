"""Tests for the entry-gate wrappers -- see application/entry_gates.py's
own docstring. These only need to prove the wrappers report the right
GateResult/EntryDecision for each underlying filter's outcome; the
filters' own thresholds are already covered by test_paper_trading.py,
test_entry_quality_filter.py, and test_conviction_filter.py."""

from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.application import entry_gates
from trading_scanner.application.entry_quality_filter import (
    _REGIME_NORMALIZED_FLOOR,
    _VOLATILITY_MARGIN_FLOOR,
)
from trading_scanner.domain.models import SignalSide, Trade

_ABOVE_QUALITY_FLOOR = (_VOLATILITY_MARGIN_FLOOR + 1, _REGIME_NORMALIZED_FLOOR + 1)
_BELOW_QUALITY_FLOOR = (0.0, 0.0)


class _FakeTradeRepository:
    """Minimal in-memory TradeRepository -- only ``get_trades`` is
    exercised by ``paper_trading.is_eligible``."""

    def __init__(self, trades: list[Trade]) -> None:
        self._trades = trades

    async def get_trades(self, symbol: str, interval: str) -> list[Trade]:
        return self._trades

    async def open_trade(self, interval, trade) -> None:  # pragma: no cover -- unused here
        raise NotImplementedError

    async def close_open_trade(self, *args, **kwargs) -> None:  # pragma: no cover
        raise NotImplementedError

    async def abandon_open_trade(self, *args, **kwargs) -> None:  # pragma: no cover
        raise NotImplementedError


def _closed_buy(pnl_percent: Decimal) -> Trade:
    return Trade(
        symbol="RELIANCE.NS",
        side=SignalSide.BUY,
        entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        entry_price=Decimal("100"),
        prediction_at_entry=1,
        is_early_signal_flip=False,
        exit_timestamp=datetime(2026, 1, 5, tzinfo=UTC),
        exit_price=Decimal("110") if pnl_percent > 0 else Decimal("90"),
        pnl_percent=pnl_percent,
        status="closed",
    )


async def test_track_record_gate_passes_at_or_above_the_win_rate_bar():
    # 5 closed trades (the minimum), 3 wins -- 60% >= MIN_WIN_RATE (55%).
    trades = [_closed_buy(Decimal("5"))] * 3 + [_closed_buy(Decimal("-5"))] * 2
    repo = _FakeTradeRepository(trades)

    gate = await entry_gates.evaluate_track_record_gate("RELIANCE.NS", "1h", repo)

    assert gate.name == "track_record"
    assert gate.passed is True
    assert gate.reason is None


async def test_track_record_gate_fails_with_insufficient_history():
    repo = _FakeTradeRepository([])

    gate = await entry_gates.evaluate_track_record_gate("RELIANCE.NS", "1h", repo)

    assert gate.passed is False
    assert gate.reason == "not eligible yet (win_rate<55% or insufficient trade history)"


async def test_cash_quality_gates_both_pass():
    volatility_margin, regime_normalized = _ABOVE_QUALITY_FLOOR
    decision = entry_gates.evaluate_cash_quality_gates(
        volatility_margin, regime_normalized, Decimal("110"), Decimal("100"), Decimal("109")
    )

    assert decision.allowed is True
    assert [g.passed for g in decision.gates] == [True, True]
    assert decision.blocked_reason is None


def test_cash_quality_gates_reports_quality_failure_reason():
    volatility_margin, regime_normalized = _BELOW_QUALITY_FLOOR
    # Conviction candle is strong (closes at the high) -- only the quality
    # gate should fail here.
    decision = entry_gates.evaluate_cash_quality_gates(
        volatility_margin, regime_normalized, Decimal("110"), Decimal("100"), Decimal("110")
    )

    assert decision.allowed is False
    assert decision.blocked_reason == "entry_quality_filter"


def test_cash_quality_gates_reports_conviction_failure_reason_when_quality_passes():
    volatility_margin, regime_normalized = _ABOVE_QUALITY_FLOOR
    # Weak entry candle (closes at the low) -- quality passes, conviction
    # doesn't; quality's own reason must not shadow it.
    decision = entry_gates.evaluate_cash_quality_gates(
        volatility_margin, regime_normalized, Decimal("110"), Decimal("100"), Decimal("100")
    )

    assert decision.allowed is False
    assert decision.blocked_reason == "conviction filter -- weak entry candle"


def test_cash_quality_gates_prefers_quality_reason_when_both_fail():
    volatility_margin, regime_normalized = _BELOW_QUALITY_FLOOR
    decision = entry_gates.evaluate_cash_quality_gates(
        volatility_margin, regime_normalized, Decimal("110"), Decimal("100"), Decimal("100")
    )

    assert decision.allowed is False
    assert decision.blocked_reason == "entry_quality_filter"

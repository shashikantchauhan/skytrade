from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_scanner.application.ranking_features import build_feature_table
from trading_scanner.domain.models import Candle, SignalSide, Trade


class _FakeCandleRepository:
    def __init__(self, candles_by_symbol: dict[str, list[Candle]]):
        self._candles_by_symbol = candles_by_symbol

    async def upsert_candles(self, symbol, interval, candles):
        raise NotImplementedError

    async def get_candles(self, symbol, interval, limit=None):
        return self._candles_by_symbol.get(symbol, [])


def _candles(symbol: str, start: datetime, count: int, seed: float) -> list[Candle]:
    candles = []
    price = Decimal("100")
    for i in range(count):
        # Deterministic pseudo-price walk so two symbols can be genuinely
        # correlated (same seed) or not (different seed) for the test below.
        price = price * (Decimal("1") + Decimal(str(0.001 * ((i * seed) % 7 - 3))))
        candles.append(
            Candle(
                symbol=symbol, timestamp=start + timedelta(hours=i),
                open=price, high=price, low=price, close=price, volume=1000,
            )
        )
    return candles


def _trade(symbol, entry, exit_, prediction=5, adx=0.3, regime=0.1, vol=0.05, win=True):
    return Trade(
        symbol=symbol, side=SignalSide.BUY, entry_timestamp=entry, entry_price=Decimal("100"),
        prediction_at_entry=prediction, is_early_signal_flip=False,
        exit_timestamp=exit_, exit_price=Decimal("110") if win else Decimal("90"),
        pnl_percent=Decimal("10") if win else Decimal("-10"), status="closed",
        adx_at_entry=adx, regime_normalized_at_entry=regime, volatility_margin_at_entry=vol,
        volatility_filter_passed=True, regime_filter_passed=True, adx_filter_passed=True,
    )


@pytest.mark.asyncio
async def test_build_feature_table_produces_one_row_per_closed_buy_trade():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trades = [
        _trade("RELIANCE.NS", start, start + timedelta(hours=5)),
        _trade("TCS.NS", start + timedelta(hours=1), start + timedelta(hours=6), win=False),
    ]
    candle_repo = _FakeCandleRepository({
        "RELIANCE.NS": _candles("RELIANCE.NS", start - timedelta(hours=100), 150, seed=1.0),
        "TCS.NS": _candles("TCS.NS", start - timedelta(hours=100), 150, seed=1.0),
    })
    rows = await build_feature_table(trades, candle_repo, "60minute")
    assert len(rows) == 2
    assert {row.symbol for row in rows} == {"RELIANCE.NS", "TCS.NS"}
    assert {row.label for row in rows} == {0, 1}


@pytest.mark.asyncio
async def test_sector_is_looked_up_from_the_curated_mapping():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trades = [_trade("HDFCBANK.NS", start, start + timedelta(hours=5))]
    candle_repo = _FakeCandleRepository({
        "HDFCBANK.NS": _candles("HDFCBANK.NS", start - timedelta(hours=100), 150, seed=1.0),
    })
    rows = await build_feature_table(trades, candle_repo, "60minute")
    assert rows[0].sector == "^NSEBANK"


@pytest.mark.asyncio
async def test_unmapped_symbol_gets_unknown_sector():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trades = [_trade("NOT_A_REAL_SYMBOL.NS", start, start + timedelta(hours=5))]
    symbol = "NOT_A_REAL_SYMBOL.NS"
    candle_repo = _FakeCandleRepository({
        symbol: _candles(symbol, start - timedelta(hours=100), 150, seed=1.0),
    })
    rows = await build_feature_table(trades, candle_repo, "60minute")
    assert rows[0].sector == "UNKNOWN"


@pytest.mark.asyncio
async def test_trades_missing_feature_columns_are_skipped_not_imputed():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    pre_migration_trade = Trade(
        symbol="INFY.NS", side=SignalSide.BUY, entry_timestamp=start, entry_price=Decimal("100"),
        prediction_at_entry=5, is_early_signal_flip=False,
        exit_timestamp=start + timedelta(hours=5), exit_price=Decimal("110"),
        pnl_percent=Decimal("10"), status="closed",  # adx_at_entry left as None (pre-migration row)
    )
    candle_repo = _FakeCandleRepository({
        "INFY.NS": _candles("INFY.NS", start - timedelta(hours=100), 150, seed=1.0),
    })
    rows = await build_feature_table([pre_migration_trade], candle_repo, "60minute")
    assert rows == []


@pytest.mark.asyncio
async def test_correlation_to_open_positions_is_zero_when_nothing_else_is_open():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trades = [_trade("ONLY.NS", start, start + timedelta(hours=5))]
    candle_repo = _FakeCandleRepository({
        "ONLY.NS": _candles("ONLY.NS", start - timedelta(hours=100), 150, seed=1.0),
    })
    rows = await build_feature_table(trades, candle_repo, "60minute")
    assert rows[0].correlation_to_open_positions == 0.0


@pytest.mark.asyncio
async def test_correlation_to_open_positions_is_nonzero_when_something_else_is_open():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    # SYM_B opens well before and stays open through SYM_A's entry.
    trades = [
        _trade("SYM_B.NS", start - timedelta(hours=10), start + timedelta(hours=20)),
        _trade("SYM_A.NS", start, start + timedelta(hours=5)),
    ]
    candle_repo = _FakeCandleRepository({
        # Identical seed -> identical price walk for both symbols.
        "SYM_A.NS": _candles("SYM_A.NS", start - timedelta(hours=100), 150, seed=1.0),
        "SYM_B.NS": _candles("SYM_B.NS", start - timedelta(hours=100), 150, seed=1.0),
    })
    rows = await build_feature_table(trades, candle_repo, "60minute")
    candidate_row = next(row for row in rows if row.symbol == "SYM_A.NS")
    # Same seed -> identical price walk -> should show strong positive correlation.
    assert candidate_row.correlation_to_open_positions > 0.9

from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.application.sector_confirmation import (
    annotate_sector_confirmation,
    compare_confirmed_vs_unconfirmed,
)
from trading_scanner.domain.models import SignalSide, Trade


def _trade(symbol, side, entry, win=True, pnl=Decimal("5")):
    return Trade(
        symbol=symbol, side=side, entry_timestamp=entry, entry_price=Decimal("100"),
        prediction_at_entry=5, is_early_signal_flip=False,
        exit_timestamp=entry, exit_price=Decimal("105") if win else Decimal("95"),
        pnl_percent=pnl if win else -pnl, status="closed",
    )


def test_confirmed_when_sector_index_enters_same_side_same_time():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    # PNB.NS maps to ^NSEBANK in the real SECTOR_MAP.
    stock = _trade("PNB.NS", SignalSide.BUY, t)
    index = _trade("^NSEBANK", SignalSide.BUY, t)

    confirmation = annotate_sector_confirmation([stock, index])

    assert confirmation[id(stock)] is True


def test_unconfirmed_when_sector_index_silent_at_that_time():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    stock = _trade("PNB.NS", SignalSide.BUY, t)
    # Index only entered on a different day -- no confirmation.
    index = _trade("^NSEBANK", SignalSide.BUY, datetime(2026, 1, 2, tzinfo=UTC))

    confirmation = annotate_sector_confirmation([stock, index])

    assert confirmation[id(stock)] is False


def test_unconfirmed_when_sector_index_enters_opposite_side():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    stock = _trade("PNB.NS", SignalSide.BUY, t)
    index = _trade("^NSEBANK", SignalSide.SELL, t)  # opposite side, same time

    confirmation = annotate_sector_confirmation([stock, index])

    assert confirmation[id(stock)] is False


def test_unmapped_symbol_gets_none_not_false():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    stock = _trade("NOT_A_REAL_SYMBOL.NS", SignalSide.BUY, t)

    confirmation = annotate_sector_confirmation([stock])

    assert confirmation[id(stock)] is None


def test_compare_splits_population_by_confirmation_and_computes_stats():
    t1 = datetime(2026, 1, 1, tzinfo=UTC)
    t2 = datetime(2026, 1, 2, tzinfo=UTC)
    confirmed_win = _trade("PNB.NS", SignalSide.BUY, t1, win=True, pnl=Decimal("10"))
    confirming_index = _trade("^NSEBANK", SignalSide.BUY, t1)
    unconfirmed_loss = _trade("PNB.NS", SignalSide.BUY, t2, win=False, pnl=Decimal("3"))
    # No ^NSEBANK trade at t2 -> unconfirmed_loss has no confirmation.

    results = compare_confirmed_vs_unconfirmed(
        [confirmed_win, confirming_index, unconfirmed_loss], SignalSide.BUY
    )

    assert results["confirmed"].n == 1
    assert results["confirmed"].win_rate == Decimal("100")
    assert results["unconfirmed"].n == 1
    assert results["unconfirmed"].win_rate == Decimal("0")

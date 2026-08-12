from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.application.capital_constrained_backtest import SimulationConfig, simulate
from trading_scanner.domain.models import SignalSide, Trade


def _closed_trade(symbol, day, prediction=4, pnl_percent=Decimal("10"), win=True):
    entry = datetime(2026, 1, day, tzinfo=UTC)
    exit_ = datetime(2026, 1, day + 1, tzinfo=UTC)
    entry_price = Decimal("100")
    exit_price = entry_price * (1 + pnl_percent / 100) if win else entry_price * (
        1 - pnl_percent / 100
    )
    return Trade(
        symbol=symbol, side=SignalSide.BUY, entry_timestamp=entry, entry_price=entry_price,
        prediction_at_entry=prediction, is_early_signal_flip=False,
        exit_timestamp=exit_, exit_price=exit_price,
        pnl_percent=pnl_percent if win else -pnl_percent, status="closed",
    )


def _track_record(symbol, count=5, win=True):
    """5 closed BUY trades far enough in the past to establish eligibility."""
    return [_closed_trade(symbol, day, win=win) for day in range(1, count + 1)]


def test_final_equity_counts_capital_still_locked_in_an_open_position():
    """A position still open at the end of the dataset never returns its
    capital to final_cash (it never closes) -- final_equity must still
    count that capital as real equity, not silently drop it."""
    history = _track_record("GOOD")
    still_open_entry = Trade(
        symbol="GOOD", side=SignalSide.BUY, entry_timestamp=datetime(2026, 2, 1, tzinfo=UTC),
        entry_price=Decimal("100"), prediction_at_entry=5, is_early_signal_flip=False,
        exit_timestamp=None, exit_price=None, status="open",
    )
    config = SimulationConfig(
        initial_capital=Decimal("800000"), target_slots=10, min_position_size=Decimal("75000"),
    )
    result = simulate([*history, still_open_entry], config)

    assert result.trades_taken == 1
    # position_size = max(800000/10, 75000) = max(80000, 75000) = 80000.
    assert result.final_open_capital == Decimal("80000")
    # No realized pnl yet (nothing closed) -- equity is exactly capital in,
    # neither gained nor lost, not silently short by the open position's stake.
    assert result.final_equity == config.initial_capital


def test_ineligible_symbol_with_no_track_record_is_skipped():
    trade = Trade(
        symbol="NEW", side=SignalSide.BUY, entry_timestamp=datetime(2026, 2, 1, tzinfo=UTC),
        entry_price=Decimal("100"), prediction_at_entry=5, is_early_signal_flip=False,
        exit_timestamp=None, exit_price=None, status="open",
    )
    result = simulate(
        [trade],
        SimulationConfig(
            initial_capital=Decimal("800000"), target_slots=10,
            min_position_size=Decimal("75000"),
        ),
    )
    assert result.trades_taken == 0
    assert result.trades_skipped_ineligible == 1


def test_eligible_symbol_opens_a_position():
    history = _track_record("GOOD")
    entry = Trade(
        symbol="GOOD", side=SignalSide.BUY, entry_timestamp=datetime(2026, 2, 1, tzinfo=UTC),
        entry_price=Decimal("100"), prediction_at_entry=5, is_early_signal_flip=False,
        exit_timestamp=None, exit_price=None, status="open",
    )
    result = simulate(
        [*history, entry],
        SimulationConfig(
            initial_capital=Decimal("800000"), target_slots=10,
            min_position_size=Decimal("75000"),
        ),
    )
    assert result.trades_taken == 1
    assert result.trades_skipped_no_capital == 0


def test_capital_exhaustion_skips_lower_ranked_candidates():
    history_a = _track_record("A")
    history_b = _track_record("B")
    same_time = datetime(2026, 2, 1, tzinfo=UTC)
    strong = Trade(
        symbol="A", side=SignalSide.BUY, entry_timestamp=same_time, entry_price=Decimal("100"),
        prediction_at_entry=8, is_early_signal_flip=False,
        exit_timestamp=None, exit_price=None, status="open",
    )
    weak = Trade(
        symbol="B", side=SignalSide.BUY, entry_timestamp=same_time, entry_price=Decimal("100"),
        prediction_at_entry=1, is_early_signal_flip=False,
        exit_timestamp=None, exit_price=None, status="open",
    )
    # Small account, only room for one slot at the floor.
    config = SimulationConfig(
        initial_capital=Decimal("80000"), target_slots=1, min_position_size=Decimal("75000"),
    )
    result = simulate([*history_a, *history_b, strong, weak], config)
    assert result.trades_taken == 1
    assert result.trades_skipped_no_capital == 1


def test_unranked_mode_preserves_input_order_instead_of_scoring():
    history_a = _track_record("A")
    history_b = _track_record("B")
    same_time = datetime(2026, 2, 1, tzinfo=UTC)
    weak_first = Trade(
        symbol="B", side=SignalSide.BUY, entry_timestamp=same_time, entry_price=Decimal("100"),
        prediction_at_entry=1, is_early_signal_flip=False,
        exit_timestamp=None, exit_price=None, status="open",
    )
    strong_second = Trade(
        symbol="A", side=SignalSide.BUY, entry_timestamp=same_time, entry_price=Decimal("100"),
        prediction_at_entry=8, is_early_signal_flip=False,
        exit_timestamp=None, exit_price=None, status="open",
    )
    config = SimulationConfig(
        initial_capital=Decimal("80000"), target_slots=1, min_position_size=Decimal("75000"),
        use_ranking=False,
    )
    # Input order lists B (weak) before A (strong); unranked mode should take
    # whichever came first in the input, not the stronger one.
    result = simulate([*history_a, *history_b, weak_first, strong_second], config)
    assert result.trades_taken == 1
    assert result.trades_skipped_no_capital == 1


def test_exit_frees_capital_for_a_same_cycle_entry():
    history_a = _track_record("A")
    history_c = _track_record("C")
    exit_and_entry_time = datetime(2026, 2, 5, tzinfo=UTC)
    still_open_from_earlier = Trade(
        symbol="A", side=SignalSide.BUY, entry_timestamp=datetime(2026, 2, 1, tzinfo=UTC),
        entry_price=Decimal("100"), prediction_at_entry=5, is_early_signal_flip=False,
        exit_timestamp=exit_and_entry_time, exit_price=Decimal("110"),
        pnl_percent=Decimal("10"), status="closed",
    )
    new_entry = Trade(
        symbol="C", side=SignalSide.BUY, entry_timestamp=exit_and_entry_time,
        entry_price=Decimal("100"), prediction_at_entry=5, is_early_signal_flip=False,
        exit_timestamp=None, exit_price=None, status="open",
    )
    # Tight capital: only enough for one slot's worth, freed by the exit
    # above in the same cycle as the new entry.
    config = SimulationConfig(
        initial_capital=Decimal("80000"), target_slots=1, min_position_size=Decimal("75000"),
    )
    result = simulate([*history_a, *history_c, still_open_from_earlier, new_entry], config)
    # Two entries actually happen here: A opens on Feb 1 (this same
    # simulation is what puts it "still open"), then closes and frees
    # capital on Feb 5 in the same cycle C's new entry needs it.
    assert result.trades_taken == 2
    assert result.trades_skipped_no_capital == 0
    assert result.wins == 1  # only A's trade has closed; C is still open

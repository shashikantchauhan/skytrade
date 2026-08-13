"""Replay stored BUY trades through the same capital/eligibility/ranking gates
``paper_trading.py`` and ``signal_pipeline.py`` apply live, to answer the
question the raw strategy backtest (``application/backtest.py``) cannot:
how many signals actually get capital under a given slot/floor configuration,
and does ranking (``application/ranking.py``) change the outcome versus
first-come-first-served.

``application/backtest.py``'s replay answers "what would the strategy do
with unlimited capital" -- useful for training data, useless for capacity
planning. This module answers "what would the *account* do" -- walking the
same trade history chronologically, causally recomputing each symbol's
eligibility from only its own already-closed trades as of that point in
time (no look-ahead), and applying the real position-sizing math
(``paper_trading.try_open_position``'s formula, not a copy of it -- this
imports and calls the exact same functions the live account uses, so a
future change to that sizing logic is automatically reflected here too).

Simplification: trades from the same scan cycle should compete for capital
together (that is what ranking is for), but this module only groups BUY
entries with an *identical* ``entry_timestamp`` as one cycle. In practice
symbols' hourly candles usually align to the same boundary, so this holds
for most cycles; entries that drift by even a second fall into their own
single-candidate group and get no ranking benefit. This makes the ranked
comparison a conservative (not optimistic) estimate of ranking's benefit.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_scanner.application.ranking import RankedCandidate, rank_candidates, score_candidate
from trading_scanner.application.trading_costs import round_trip_cost
from trading_scanner.domain.models import SignalSide, Trade

MIN_WIN_RATE = Decimal("55")
MIN_CLOSED_TRADES = 5


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Mirrors paper_trading.py's constants -- pass a symbol's live .env
    values here rather than hardcoding the repo defaults, so this simulates
    the account actually deployed, not a theoretical one."""

    initial_capital: Decimal
    target_slots: int
    min_position_size: Decimal
    use_ranking: bool = True
    # Mirrors ranking.MIN_SCORE -- 0 (default) means no floor, matching
    # ranking.py's own backward-compatible default. Only applied when
    # use_ranking is True (a floor without ranking has no ordering to
    # apply it against consistently).
    min_score: float = 0.0
    # Every result before 2026-08-14 was gross, not net of real trading
    # costs (see trading_costs.py -- known gap, never modeled). Defaults
    # False so existing callers/tests keep today's (unrealistic) numbers
    # unless they explicitly opt in.
    apply_realistic_costs: bool = False


@dataclass(slots=True)
class SimulationResult:
    config: SimulationConfig
    trades_taken: int = 0
    trades_skipped_ineligible: int = 0
    trades_skipped_no_capital: int = 0
    trades_skipped_below_score_floor: int = 0
    final_cash: Decimal = Decimal("0")
    # Capital still locked in positions open at the end of the dataset --
    # never returned to final_cash since they never close within this
    # history, but it is still real equity, not lost money. Needed because
    # initial_capital + total_pnl_amount alone silently drops it (it only
    # accounts for *closed* trades), which would misstate final_equity by
    # exactly this amount.
    final_open_capital: Decimal = Decimal("0")
    total_pnl_amount: Decimal = Decimal("0")
    total_costs_paid: Decimal = Decimal("0")
    wins: int = 0
    losses: int = 0

    @property
    def total_signals(self) -> int:
        return (
            self.trades_taken
            + self.trades_skipped_ineligible
            + self.trades_skipped_no_capital
            + self.trades_skipped_below_score_floor
        )

    @property
    def win_rate(self) -> Decimal | None:
        decided = self.wins + self.losses
        return None if decided == 0 else Decimal(100 * self.wins) / decided

    @property
    def final_equity(self) -> Decimal:
        return self.final_cash + self.final_open_capital


@dataclass(slots=True)
class _OpenPosition:
    symbol: str
    entry_price: Decimal
    quantity: int
    capital_allocated: Decimal


def simulate(trades: Sequence[Trade], config: SimulationConfig) -> SimulationResult:
    """Replay BUY-side trades chronologically against the given account config.

    ``trades`` should be every historical BUY-side trade for every symbol
    (``TradeRepository.get_trades(None, interval)``, filtered to
    ``side == SignalSide.BUY``) -- SELL-side never touches the paper
    account (see paper_trading.py's own docstring). Trades still open at
    the end of history are ignored for P&L purposes (their outcome is
    unknown) but do still occupy a slot/consume capital while open, exactly
    as a real open position would.
    """
    buy_trades = [trade for trade in trades if trade.side == SignalSide.BUY]
    result = SimulationResult(config=config)
    cash = config.initial_capital
    open_positions: dict[str, _OpenPosition] = {}
    # Per symbol, the closed BUY trades' pnl history in chronological order
    # of when they closed -- this is what a causal is_eligible check can see
    # "as of now" without look-ahead.
    closed_by_symbol: dict[str, list[Decimal]] = {}

    entries_by_timestamp: dict[datetime, list[Trade]] = {}
    exits_by_timestamp: dict[datetime, list[Trade]] = {}
    for trade in buy_trades:
        entries_by_timestamp.setdefault(trade.entry_timestamp, []).append(trade)
        if trade.exit_timestamp is not None:
            exits_by_timestamp.setdefault(trade.exit_timestamp, []).append(trade)

    all_timestamps = sorted(set(entries_by_timestamp) | set(exits_by_timestamp))

    def total_equity() -> Decimal:
        return cash + sum(
            (position.capital_allocated for position in open_positions.values()),
            start=Decimal("0"),
        )

    def is_eligible(symbol: str) -> bool:
        history = closed_by_symbol.get(symbol, [])
        if len(history) < MIN_CLOSED_TRADES:
            return False
        wins = sum(1 for pnl in history if pnl > 0)
        return Decimal(100 * wins) / len(history) >= MIN_WIN_RATE

    def try_open(trade: Trade) -> bool:
        nonlocal cash
        position_size = max(total_equity() / config.target_slots, config.min_position_size)
        if cash < position_size:
            return False
        quantity = int(position_size / trade.entry_price)
        if quantity < 1:
            return False
        capital_allocated = quantity * trade.entry_price
        cash -= capital_allocated
        open_positions[trade.symbol] = _OpenPosition(
            symbol=trade.symbol,
            entry_price=trade.entry_price,
            quantity=quantity,
            capital_allocated=capital_allocated,
        )
        return True

    for timestamp in all_timestamps:
        # Exits first: a position closing this same cycle frees capital that
        # can fund a new entry in the same cycle, matching try_open_position
        # reading cash_balance fresh on every call.
        for trade in exits_by_timestamp.get(timestamp, []):
            # Eligibility bookkeeping mirrors production's is_eligible, which
            # reads the *strategy's* full trade history (every closed BUY
            # signal, recorded regardless of paper-account capital) -- not
            # just the subset this constrained account actually funded. Keep
            # this unconditional on whether a position was opened below.
            if trade.status == "closed" and trade.pnl_percent is not None:
                closed_by_symbol.setdefault(trade.symbol, []).append(trade.pnl_percent)

            position = open_positions.pop(trade.symbol, None)
            if position is None or trade.exit_price is None:
                continue
            pnl_amount = position.quantity * (trade.exit_price - position.entry_price)
            if config.apply_realistic_costs:
                entry_value = position.quantity * position.entry_price
                exit_value = position.quantity * trade.exit_price
                cost = round_trip_cost(entry_value, exit_value)
                pnl_amount -= cost
                result.total_costs_paid += cost
            cash += position.capital_allocated + pnl_amount
            result.total_pnl_amount += pnl_amount
            # Win/loss (and therefore win_rate, and therefore is_eligible's
            # own downstream eligibility bar) is judged on the same
            # cost-adjusted pnl_amount -- a trade that was gross-profitable
            # but net-negative after real costs is a loss, not a win.
            if pnl_amount > 0:
                result.wins += 1
            else:
                result.losses += 1

        candidates_this_cycle = entries_by_timestamp.get(timestamp, [])
        if not candidates_this_cycle:
            continue

        eligible: list[Trade] = []
        for trade in candidates_this_cycle:
            if trade.symbol in open_positions:
                continue  # Mirrors live behavior: one open position per symbol.
            if is_eligible(trade.symbol):
                eligible.append(trade)
            else:
                result.trades_skipped_ineligible += 1

        ordered = eligible
        if config.use_ranking:
            ranked = rank_candidates([
                RankedCandidate(
                    symbol=trade.symbol,
                    entry_timestamp=trade.entry_timestamp,
                    entry_price=trade.entry_price,
                    prediction_at_entry=trade.prediction_at_entry,
                    adx=trade.adx_at_entry or 0.0,
                    regime_normalized=trade.regime_normalized_at_entry or 0.0,
                    volatility_margin=trade.volatility_margin_at_entry or 0.0,
                )
                for trade in eligible
            ])
            by_symbol = {trade.symbol: trade for trade in eligible}
            # Ranked strongest-first -- once one candidate falls below the
            # floor, every remaining one does too, so cut the ordered list
            # there rather than filtering after (keeps trades_skipped_no_capital
            # meaning what it says: real capacity misses, not floor rejects).
            below_floor_count = 0
            cutoff = len(ranked)
            for index, candidate in enumerate(ranked):
                if score_candidate(candidate) < config.min_score:
                    cutoff = index
                    below_floor_count = len(ranked) - index
                    break
            result.trades_skipped_below_score_floor += below_floor_count
            ordered = [by_symbol[candidate.symbol] for candidate in ranked[:cutoff]]

        for trade in ordered:
            if try_open(trade):
                result.trades_taken += 1
            else:
                result.trades_skipped_no_capital += 1

    result.final_cash = cash
    result.final_open_capital = sum(
        (position.capital_allocated for position in open_positions.values()), start=Decimal("0")
    )
    return result

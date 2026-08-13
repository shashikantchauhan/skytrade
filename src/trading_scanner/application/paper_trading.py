"""Paper-trading account: long-only simulated real-money positions.

NSE cash market does not allow short selling for multi-day (delivery) holds
-- only intraday MIS positions can be short, squared off same day. This
strategy's average holding period is ~3.5 days, so SELL/short signals can
never be executed as real cash-market positions; the paper account only ever
opens a position on a BUY entry. SELL signals still notify (see
``signal_pipeline.py``) but are informational only.

Two gates decide whether a BUY entry actually becomes a paper position:

1. **Eligibility**: the symbol's own closed-trade, BUY-only win rate (see
   ``application/backtest.py``/``signal_pipeline.py``'s trade bookkeeping)
   must be at least ``MIN_WIN_RATE``, and it must have at least
   ``MIN_CLOSED_TRADES`` closed BUY trades to compute a meaningful rate from.
   A symbol with no track record yet, or a poor one, is skipped -- still
   notified, just tagged as not paper-traded.
2. **Capacity**: the account only has ``INITIAL_CAPITAL`` to work with, split
   into ``TARGET_SLOTS`` dynamically-sized slots. If the cash balance can't
   cover one more slot, the entry is skipped and tagged accordingly rather
   than silently dropped.

``TARGET_SLOTS`` (32) matches real signal demand: Little's Law
(concurrent positions needed ~= entries/day x average holding period),
computed only over symbols that actually clear the eligibility bar above
(ineligible symbols never reach ``try_open_position`` at all, so they don't
count toward real capacity demand). ``INITIAL_CAPITAL`` (Rs 8,00,000) is
sized so 32 slots at the resulting ~Rs 25,000/slot fully covers that demand
with no capital-driven skips under normal conditions.

Slot size is **dynamic**, not fixed: every entry recomputes
``total_equity / TARGET_SLOTS``, where total_equity is cash plus all open
positions' allocated capital. As the account compounds profit week over
week, each slot grows proportionally -- no manual re-tuning needed. A floor
(``MIN_POSITION_SIZE``, Rs 25,000) keeps the flat per-trade DP charge
(~Rs 18, sell-side only) under ~5% of an average winning trade's profit;
below that floor, flat fees start eating a disproportionate share of returns.
"""

import os
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal

from dotenv import load_dotenv

from trading_scanner.domain.models import PaperPosition, SignalSide
from trading_scanner.domain.ports import CandleRepository, PaperAccountRepository, TradeRepository

# Loaded here (not just in config/settings.py) because the constants below
# are read from the environment at import time, which can happen before
# signals.py's main() gets around to calling load_config(). Safe to call
# more than once -- dotenv never overwrites an already-set env var.
load_dotenv()

# Overridable via .env so the dashboard's config editor can adjust sizing
# without a code change/redeploy -- defaults match the Little's Law sizing
# derived above (32 slots, Rs 8,00,000, Rs 25,000 floor).
INITIAL_CAPITAL = Decimal(os.getenv("TRADING_SCANNER_PAPER_CAPITAL", "800000"))
TARGET_SLOTS = int(os.getenv("TRADING_SCANNER_PAPER_SLOTS", "32"))
MIN_POSITION_SIZE = Decimal(os.getenv("TRADING_SCANNER_PAPER_MIN_POSITION", "25000"))
MIN_WIN_RATE = Decimal("55")
MIN_CLOSED_TRADES = 5

# The strategy has no price-based risk control of its own -- a losing
# position only closes when the model's opposite signal eventually fires,
# however far price has moved by then. application/stop_loss_replay.py
# validated this threshold against every real historical trade's actual
# candle-by-candle path (2026-08-13, post corrupted-candle cleanup):
# BUY expectancy 0.194% -> 1.099%, SELL expectancy -15.27% -> +0.99% at a
# 3% cap. This only ever fires *before* the strategy's own exit signal --
# if the signal comes first, nothing changes; see live_pipeline.py's
# tick-level stop check.
STOP_LOSS_PCT = Decimal(os.getenv("TRADING_SCANNER_STOP_LOSS_PCT", "3"))


def stop_loss_price(entry_price: Decimal) -> Decimal:
    """The price at which an open BUY position should be force-closed,
    instead of waiting for the strategy's own opposite signal. Long-only,
    so the stop is always below entry."""
    return entry_price * (1 - STOP_LOSS_PCT / 100)


async def is_eligible(symbol: str, interval: str, trade_repository: TradeRepository) -> bool:
    """Return whether a symbol's BUY-only track record clears the paper-trading bar.

    Long-only, so only BUY-side closed trades count -- a symbol whose edge is
    entirely on the SELL side is still not tradeable here.
    """
    win_rate = await _buy_only_win_rate(symbol, interval, trade_repository)
    return win_rate is not None and win_rate >= MIN_WIN_RATE


async def try_open_position(
    symbol: str,
    entry_timestamp: datetime,
    entry_price: Decimal,
    paper_account_repository: PaperAccountRepository,
    prediction_at_entry: Decimal | None = None,
) -> PaperPosition | None:
    """Open a paper position sized off current total equity if capital allows.

    Slot size is recomputed fresh on every call from total_equity /
    TARGET_SLOTS (floored at MIN_POSITION_SIZE) so the account scales
    proportionally as profit compounds in, without a hardcoded slot size
    going stale. Returns None (no position opened) if the remaining cash
    balance can't cover one more slot -- the caller is responsible for
    notifying that the signal was skipped for lack of capital, not silently
    dropping it (or, see ``try_evict_and_open`` below, deciding whether to
    make room instead).

    ``prediction_at_entry`` (the ranking candidate's score) is stored on the
    position purely for later use by ``try_evict_and_open`` -- it plays no
    role in whether *this* entry succeeds.
    """
    cash_balance = await paper_account_repository.get_cash_balance()
    open_positions = await paper_account_repository.get_open_positions()
    total_equity = cash_balance + sum(
        (position.capital_allocated for position in open_positions), start=Decimal("0")
    )
    position_size = max(total_equity / TARGET_SLOTS, MIN_POSITION_SIZE)

    if cash_balance < position_size:
        return None
    quantity = int(position_size / entry_price)
    if quantity < 1:
        return None
    position = PaperPosition(
        symbol=symbol,
        entry_timestamp=entry_timestamp,
        entry_price=entry_price,
        quantity=quantity,
        capital_allocated=quantity * entry_price,
        prediction_at_entry=prediction_at_entry,
    )
    await paper_account_repository.open_position(position)
    return position


# How much clearer a new candidate's score must be than an open position's
# own entry score before it's worth evicting that position -- a small edge
# isn't worth the churn (transaction costs, whipsawing between two
# similar-strength signals). Same units as AlphaEngine's prediction score.
EVICTION_MIN_SCORE_MARGIN = Decimal("1")

# NSE's closing-auction session (~15:40-16:00 IST) makes it impossible for
# a real trader to sell one position and buy another after regular trading
# ends (~15:30 IST) -- see the conversation that prompted this feature
# (2026-08-13): rotating capital this late leaves no time to actually
# execute it with real money. The paper account has no such constraint of
# its own (closing a simulated position is just a DB write, any time), but
# it exists to mirror what's realistically executable -- so eviction stops
# offering itself well before the real cutoff, not at it.
_IST_OFFSET = timedelta(hours=5, minutes=30)
ROTATION_CUTOFF_IST = time(14, 30)


def _before_rotation_cutoff(at: datetime) -> bool:
    """Whether ``at`` (any timezone, or naive -- assumed UTC) falls before
    ``ROTATION_CUTOFF_IST`` in India time."""
    aware = at if at.tzinfo is not None else at.replace(tzinfo=UTC)
    ist_time = (aware.astimezone(UTC) + _IST_OFFSET).time()
    return ist_time < ROTATION_CUTOFF_IST


async def try_evict_and_open(
    symbol: str,
    entry_timestamp: datetime,
    entry_price: Decimal,
    prediction_at_entry: Decimal,
    paper_account_repository: PaperAccountRepository,
    candle_repository: CandleRepository,
    interval: str,
) -> PaperPosition | None:
    """When ``try_open_position`` fails for lack of capital, decide whether
    a weaker, currently-losing open position should be evicted to make
    room for this stronger new candidate -- instead of just skipping it.

    Both conditions must hold for a position to be eligible for eviction
    (the conservative choice, confirmed 2026-08-13): its own entry score
    must be at least ``EVICTION_MIN_SCORE_MARGIN`` weaker than the new
    candidate's, AND it must currently be at an unrealized loss. A
    profitable position is never sold just to make room, however weak its
    original score was. Among eligible positions, the worst-performing one
    (by current unrealized %) is evicted -- at most one per call, matching
    ``try_open_position``'s one-slot-per-call sizing.

    Refuses to act at all past ``ROTATION_CUTOFF_IST`` (see its own
    docstring) or for positions with no stored entry score (opened before
    this feature existed -- there's nothing to compare).
    """
    if not _before_rotation_cutoff(entry_timestamp):
        return None

    open_positions = await paper_account_repository.get_open_positions()
    # (unrealized_pct, position, current_price) of the worst eligible candidate so far.
    worst: tuple[Decimal, PaperPosition, Decimal] | None = None
    for position in open_positions:
        if position.prediction_at_entry is None:
            continue
        if prediction_at_entry - position.prediction_at_entry < EVICTION_MIN_SCORE_MARGIN:
            continue
        recent = await candle_repository.get_candles(position.symbol, interval, limit=1)
        if not recent:
            continue
        current_price = recent[-1].close
        unrealized_pct = (current_price - position.entry_price) / position.entry_price * 100
        if unrealized_pct >= 0:
            continue  # never evict a position that's currently winning
        if worst is None or unrealized_pct < worst[0]:
            worst = (unrealized_pct, position, current_price)

    if worst is None:
        return None
    _, evicted, evicted_price = worst
    await paper_account_repository.close_position(evicted.symbol, entry_timestamp, evicted_price)
    return await try_open_position(
        symbol, entry_timestamp, entry_price, paper_account_repository, prediction_at_entry
    )


async def _buy_only_win_rate(
    symbol: str, interval: str, trade_repository: TradeRepository
) -> Decimal | None:
    """Compute the closed BUY-only win rate, or None if too few trades exist."""
    trades = await trade_repository.get_trades(symbol, interval)
    closed_buys = [
        trade for trade in trades if trade.side == SignalSide.BUY and trade.status == "closed"
    ]
    if len(closed_buys) < MIN_CLOSED_TRADES:
        return None
    wins = sum(
        1 for trade in closed_buys if trade.pnl_percent is not None and trade.pnl_percent > 0
    )
    return Decimal(100 * wins) / len(closed_buys)

"""Futures paper account: simulated real-money futures+hedge combo positions,
capped by real margin instead of full notional.

Sibling to ``paper_trading.py``, not a replacement -- separate capital pool
(see ``domain/models.py``'s ``FuturesPaperPosition`` docstring for why),
separate eligibility track record, separate slot math. Where the cash paper
account is long-only (NSE cash market can't short) and sizes each position
off ``total_equity / TARGET_SLOTS`` of *notional*, this account can take
both long and short (futures are the real short mechanism, see
``application/futures_shadow.py``) and sizes off ``total_equity /
TARGET_SLOTS`` of *margin* -- the actual SPAN+exposure Kite's basket-margin
API says the futures+hedge combo together will block, netted against the
account's existing positions (``KiteDerivativesChain.margin_benefit``'s
``combined_margin``), not a guessed percentage of the futures leg's full
notional.

``application/futures_shadow.py`` remains uncapped and analysis-only by its
own design (shadow-tracks *every* signal, no capital gate at all) -- this
module is a new, additional gate that decides whether a signal *also*
becomes a tracked, capital-constrained futures paper position, exactly
mirroring how ``paper_trading.py`` sits alongside the strategy's raw
``trades`` bookkeeping for the cash side.

Not yet wired into ``signal_pipeline.py``'s live scan loop -- see
NOTES.md's futures-capital-gate roadmap. Calling Kite's margin API
synchronously per signal has its own latency/rate-limit design questions
that deserve the same review Phase 2's live ranking wiring got, so this
stops at the module boundary for now.
"""

import asyncio
import os
from datetime import datetime
from decimal import Decimal

from dotenv import load_dotenv

from trading_scanner.domain.models import FuturesPaperPosition, SignalSide
from trading_scanner.domain.ports import FuturesPaperAccountRepository, TradeRepository
from trading_scanner.infrastructure.kite import KiteDerivativesChain

load_dotenv()

# Deliberately separate .env keys from the cash paper account's -- see this
# module's docstring on why the two books are not shared.
FUTURES_INITIAL_CAPITAL = Decimal(os.getenv("TRADING_SCANNER_FUTURES_CAPITAL", "400000"))
FUTURES_TARGET_SLOTS = int(os.getenv("TRADING_SCANNER_FUTURES_SLOTS", "16"))
FUTURES_MIN_SLOT_MARGIN = Decimal(os.getenv("TRADING_SCANNER_FUTURES_MIN_MARGIN", "15000"))
# Never deploy 100% of a slot's margin budget -- MTM losses on positions
# already open can raise real maintenance-margin requirements intraday
# (a margin call), so headroom is kept unallocated on purpose. 25% mirrors
# the kind of buffer Zerodha itself recommends holding above the bare SPAN+
# exposure figure for exactly this reason.
FUTURES_MARGIN_BUFFER_PCT = Decimal(os.getenv("TRADING_SCANNER_FUTURES_MARGIN_BUFFER", "0.25"))

MIN_WIN_RATE = Decimal("55")
MIN_CLOSED_TRADES = 5


async def is_eligible(
    symbol: str, side: SignalSide, interval: str, trade_repository: TradeRepository
) -> bool:
    """Return whether a symbol's own track record on this side clears the bar.

    Unlike the cash paper account's BUY-only check, futures can take either
    side, so eligibility is checked against the matching side's own closed
    trades -- a symbol whose edge is only proven long should not
    automatically also get a green light to short it, and vice versa.
    """
    trades = await trade_repository.get_trades(symbol, interval)
    closed = [trade for trade in trades if trade.side == side and trade.status == "closed"]
    if len(closed) < MIN_CLOSED_TRADES:
        return False
    wins = sum(1 for trade in closed if trade.pnl_percent is not None and trade.pnl_percent > 0)
    return Decimal(100 * wins) / len(closed) >= MIN_WIN_RATE


async def try_open_futures_position(
    symbol: str,
    side: str,
    entry_timestamp: datetime,
    futures_entry_price: Decimal,
    futures_tradingsymbol: str,
    hedge_tradingsymbol: str,
    lot_size: int,
    derivatives_chain: KiteDerivativesChain,
    futures_account_repository: FuturesPaperAccountRepository,
) -> FuturesPaperPosition | None:
    """Open a futures+hedge combo paper position if this account's own
    capital (sized off live margin, not notional) allows it.

    One open combo per symbol, mirroring the cash paper account. Margin is
    priced fresh via Kite's real basket-margin API for this exact combo
    (see ``KiteDerivativesChain.margin_benefit``) -- if that call fails
    (no live Kite session, API hiccup), this returns None rather than
    guessing a margin figure, matching ``try_open_futures_position`` in
    ``futures_shadow.py``'s own best-effort/silent-on-failure convention.
    """
    open_positions = await futures_account_repository.get_open_positions()
    if any(position.symbol == symbol for position in open_positions):
        return None

    cash_balance = await futures_account_repository.get_cash_balance()
    total_equity = cash_balance + sum(
        (position.margin_allocated for position in open_positions), start=Decimal("0")
    )
    slot_budget = max(total_equity / FUTURES_TARGET_SLOTS, FUTURES_MIN_SLOT_MARGIN)

    futures_transaction = "BUY" if side == "long" else "SELL"
    # margin_benefit calls the blocking kiteconnect client -- run off the
    # event loop, matching webapp.py's own asyncio.to_thread usage of it.
    margin_result = await asyncio.to_thread(
        derivatives_chain.margin_benefit,
        [
            (futures_tradingsymbol, futures_transaction, lot_size),
            # The hedge option is always bought (paying premium) regardless
            # of which side the future is on -- see signal_pipeline.py's
            # _open_derivatives_shadow, this mirrors the same structure.
            (hedge_tradingsymbol, "BUY", lot_size),
        ],
    )
    if margin_result is None:
        return None
    required_margin = Decimal(str(margin_result["combined_margin"])) * (
        1 + FUTURES_MARGIN_BUFFER_PCT
    )
    if required_margin > slot_budget or required_margin > cash_balance:
        return None

    position = FuturesPaperPosition(
        symbol=symbol,
        side=side,
        entry_timestamp=entry_timestamp,
        futures_entry_price=futures_entry_price,
        futures_tradingsymbol=futures_tradingsymbol,
        hedge_tradingsymbol=hedge_tradingsymbol,
        lot_size=lot_size,
        margin_allocated=required_margin,
    )
    await futures_account_repository.open_position(position)
    return position

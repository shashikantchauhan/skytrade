"""Position/derivatives-shadow close bookkeeping and exit notifications --
split out of ``signal_pipeline.py`` (Phase 8, see
``application/pipeline/__init__.py``). No behavior changed; every
function's body moved as-is.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal

from trading_scanner.application import (
    futures_shadow,
    futures_trading,
    live_execution,
    options_shadow,
)
from trading_scanner.application.pipeline.market_data import _STRATEGY_NAME
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import Signal, SignalSide
from trading_scanner.domain.ports import (
    FuturesPaperAccountRepository,
    Notifier,
    PaperAccountRepository,
    SignalRepository,
    TradeRepository,
)
from trading_scanner.infrastructure.db import (
    TursoFuturesTradeRepository,
    TursoLiveOrderRepository,
    TursoOptionsTradeRepository,
)
from trading_scanner.infrastructure.kite import KiteDerivativesChain, KiteOrderExecutor
from trading_scanner.infrastructure.telegram import LoggingNotifier

# The hedge option leg (Structure B, see _open_derivatives_shadow) targets a
# strike this far OTM instead of ATM -- confirmed against Kite's own
# margin-benefit numbers that ATM cancels out most of the primary position's
# profit (delta near -1/+1) for only a modest extra margin benefit over a
# further-OTM strike. See try_open_option_position's strike_target_price
# docstring for the full reasoning.
_HEDGE_OTM_PCT = Decimal("0.05")


async def _notify_exit(
    symbol: str,
    config: AppConfig,
    side: SignalSide,
    exit_timestamp,
    exit_price: Decimal,
    trade_repository: TradeRepository,
    signal_repository: SignalRepository,
    notifier: Notifier,
) -> None:
    """Record that an open position closed, with its realized pnl_percent.

    Reads the still-open trade's entry price before ``close_open_trade`` is
    called, so this must run first. A distinct ``strategy`` tag keeps this
    signal's fingerprint from ever colliding with an entry notification at
    the same symbol/side/timestamp (see Signal.fingerprint).

    2026-08-21: never sends via Telegram any more -- only real cash-market
    order events (see live_cash_execution.py) are notified now. Retiring
    paper trading meant the old suppress-when-a-paper-close-covers-it logic
    almost never suppressed any more (no new paper positions exist to cover
    it), so every strategy exit across the whole universe started notifying
    -- exactly the notification flood reported. Still recorded here (and
    ``notifier`` kept as a parameter, unused, for interface parity with the
    entry path) so fingerprint/dedup and dashboard signal history are
    unaffected.
    """
    trades = await trade_repository.get_trades(symbol, config.candle_interval)
    open_trade = next(
        (trade for trade in trades if trade.side == side and trade.status == "open"), None
    )
    if open_trade is None:
        return  # Nothing was actually open (e.g. abandoned by an opposite entry earlier).

    pnl_percent = (
        (exit_price - open_trade.entry_price) / open_trade.entry_price * 100
        if side == SignalSide.BUY
        else (open_trade.entry_price - exit_price) / open_trade.entry_price * 100
    )
    signal = Signal(
        symbol=symbol,
        side=side,
        strategy=f"{_STRATEGY_NAME}-exit",
        timestamp=exit_timestamp,
        price=exit_price,
        rationale=(
            f"exit; entry={open_trade.entry_price}, exit={exit_price}, pnl={pnl_percent:.2f}%"
        ),
        category="exit",
    )
    if await signal_repository.contains(signal.fingerprint):
        return
    await signal_repository.record(signal.fingerprint, signal.timestamp)


async def _close_paper_position(
    symbol: str,
    exit_timestamp,
    exit_price: Decimal,
    paper_account_repository: PaperAccountRepository,
    signal_repository: SignalRepository,
    notifier: Notifier,
    paper_account_lock: asyncio.Lock,
) -> None:
    """Close an open paper position (if any) and record the realized P&L.

    2026-08-21: never sends via Telegram any more -- follow only real
    cash-market order events now (see live_cash_execution.py). Paper
    trading itself is retired (no new positions open), so this only ever
    fires for whatever was still open from before retirement; ``notifier``
    stays a parameter, unused, for interface parity with the other close
    paths.
    """
    async with paper_account_lock:
        closed = await paper_account_repository.close_position(symbol, exit_timestamp, exit_price)
    if closed is None:
        return  # Nothing was ever opened for this symbol (not eligible / no capacity).
    pnl_percent = (exit_price - closed.entry_price) / closed.entry_price * 100
    signal = Signal(
        symbol=symbol,
        side=SignalSide.BUY,
        strategy=f"{_STRATEGY_NAME}-paper-exit",
        timestamp=exit_timestamp,
        price=exit_price,
        rationale=(
            f"paper CLOSE {closed.quantity} qty; entry={closed.entry_price}, "
            f"exit={exit_price}, pnl=₹{closed.pnl_amount:.0f} ({pnl_percent:.2f}%)"
        ),
        category="paper_exit",
    )
    if await signal_repository.contains(signal.fingerprint):
        return
    await signal_repository.record(signal.fingerprint, signal.timestamp)


async def _open_derivatives_shadow(
    symbol: str,
    side: SignalSide,
    entry_timestamp: datetime,
    market_price: Decimal,
    derivatives_chain: KiteDerivativesChain | None,
    options_trade_repository: TursoOptionsTradeRepository | None,
    futures_trade_repository: TursoFuturesTradeRepository | None,
    config: AppConfig | None = None,
    order_executor: KiteOrderExecutor | None = None,
    live_order_repository: TursoLiveOrderRepository | None = None,
    notifier: Notifier | None = None,
) -> str | None:
    """Best-effort: one leg per signal, never a real order by default -- a
    futures position (long for BUY, short for SELL, carries real open-ended
    margin risk) hedged by an option at the opposite delta (PUT hedging a
    long future, CALL hedging a short future), targeting ~2% OTM rather
    than ATM (see _HEDGE_OTM_PCT's docstring for why). No standalone
    naked-option leg -- dropped after review; if you want a pure
    directional option bet tracked again, that's a separate decision, not
    implied by this one.

    When ``config.live_trading_enabled`` is set (see ``config/
    settings.py``'s kill switch) and ``symbol`` is on the allowlist, this
    *additionally* places a real basket via ``live_execution.
    execute_basket_entry`` -- see that module's docstring for the
    option-first-then-futures sequencing and rollback behavior. The shadow
    trade above always runs regardless, so the dashboard's analysis view
    stays consistent whether or not real money followed it.

    Any failure here is a side observation, not a dependency of the main
    signal/paper-trading flow, so this never raises into the caller.
    """
    if derivatives_chain is None:
        return None
    hedge_option_type = "PE" if side == SignalSide.BUY else "CE"
    futures_side = "long" if side == SignalSide.BUY else "short"
    # PE is OTM below spot, CE is OTM above spot -- see _HEDGE_OTM_PCT.
    hedge_strike_target = (
        market_price * (1 - _HEDGE_OTM_PCT)
        if side == SignalSide.BUY
        else market_price * (1 + _HEDGE_OTM_PCT)
    )
    notes: list[str] = []
    try:
        # Futures position, hedged by an option.
        if futures_trade_repository is not None:
            note = await futures_shadow.try_open_futures_position(
                symbol, futures_side, entry_timestamp, market_price,
                derivatives_chain, futures_trade_repository, purpose="primary",
            )
            if note is not None:
                notes.append(note)
            if options_trade_repository is not None:
                hedge_note = await options_shadow.try_open_option_position(
                    symbol, hedge_option_type, "hedge", entry_timestamp, market_price,
                    derivatives_chain, options_trade_repository,
                    strike_target_price=hedge_strike_target,
                )
                if hedge_note is not None:
                    notes.append(hedge_note)
    except Exception:
        logging.getLogger(__name__).warning(
            "Derivatives shadow open failed for %s (%s) -- continuing without it.",
            symbol, side, exc_info=True,
        )
    if config is not None and order_executor is not None and live_order_repository is not None:
        try:
            await live_execution.execute_basket_entry(
                symbol, side, hedge_option_type, hedge_strike_target,
                config, derivatives_chain, order_executor, live_order_repository,
                notifier if notifier is not None else LoggingNotifier(),
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "Live order execution raised for %s (%s) -- shadow tracking above still stands.",
                symbol, side,
            )
    return "; ".join(notes) if notes else None


async def _close_derivatives_shadow(
    symbol: str,
    side: SignalSide,
    exit_timestamp: datetime,
    market_price: Decimal,
    derivatives_chain: KiteDerivativesChain | None,
    options_trade_repository: TursoOptionsTradeRepository | None,
    futures_trade_repository: TursoFuturesTradeRepository | None,
    config: AppConfig | None = None,
    order_executor: KiteOrderExecutor | None = None,
    live_order_repository: TursoLiveOrderRepository | None = None,
    notifier: Notifier | None = None,
) -> None:
    """Best-effort close of whatever ``_open_derivatives_shadow`` opened --
    see that function's docstring, including the real-order gate. Shadow
    close is logged, not notified via Telegram (analysis-only, see the
    module docstring); a real close always notifies regardless, since it's
    real money moving."""
    if derivatives_chain is None:
        return
    hedge_option_type = "PE" if side == SignalSide.BUY else "CE"
    logger = logging.getLogger(__name__)
    try:
        # Futures position, hedged by an option.
        if futures_trade_repository is not None:
            note = await futures_shadow.close_futures_position(
                symbol, exit_timestamp, market_price, futures_trade_repository, purpose="primary",
            )
            if note is not None:
                logger.info(note)
        if options_trade_repository is not None:
            note = await options_shadow.close_option_position(
                symbol, hedge_option_type, "hedge", exit_timestamp, market_price,
                derivatives_chain, options_trade_repository,
            )
            if note is not None:
                logger.info(note)
    except Exception:
        logger.warning(
            "Derivatives shadow close failed for %s (%s) -- continuing without it.",
            symbol, side, exc_info=True,
        )
    if config is not None and order_executor is not None and live_order_repository is not None:
        try:
            await live_execution.execute_basket_exit(
                symbol, config, order_executor, live_order_repository,
                notifier if notifier is not None else LoggingNotifier(),
            )
        except Exception:
            logger.exception(
                "Live order exit execution raised for %s (%s) -- shadow close above still stands.",
                symbol, side,
            )


async def _open_futures_paper(
    symbol: str,
    side: SignalSide,
    entry_timestamp: datetime,
    market_price: Decimal,
    interval: str,
    trade_repository: TradeRepository,
    derivatives_chain: KiteDerivativesChain | None,
    futures_account_repository: FuturesPaperAccountRepository | None,
    futures_paper_symbols: frozenset[str],
) -> str | None:
    """Best-effort wrapper around ``futures_trading.open_futures_paper_position``
    -- see ``_process_symbol``'s docstring for the allowlist/gating this
    checks first. Same exception-isolation shape as
    ``_open_derivatives_shadow``: a Kite margin-API hiccup here must never
    break the main signal/paper-trading flow around it.
    """
    if (
        futures_account_repository is None
        or derivatives_chain is None
        or symbol not in futures_paper_symbols
    ):
        return None
    try:
        return await futures_trading.open_futures_paper_position(
            symbol, side, entry_timestamp, market_price, interval,
            derivatives_chain, trade_repository, futures_account_repository,
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "Futures paper open failed for %s (%s) -- continuing without it.",
            symbol, side, exc_info=True,
        )
        return None


async def _close_futures_paper(
    symbol: str,
    side: SignalSide,
    exit_timestamp: datetime,
    market_price: Decimal,
    derivatives_chain: KiteDerivativesChain | None,
    futures_account_repository: FuturesPaperAccountRepository | None,
    futures_paper_symbols: frozenset[str],
    signal_repository: SignalRepository,
    notifier: Notifier,
) -> str | None:
    """Best-effort close of whatever ``_open_futures_paper`` opened -- see
    that function's docstring. Exit price is resolved internally from the
    real futures contract's live LTP, not the equity price -- see
    ``close_futures_paper_position``'s docstring.

    2026-08-21: never sends via Telegram any more -- follow only real
    cash-market order events now (see live_cash_execution.py); ``notifier``
    stays a parameter, unused, for interface parity with the other close
    paths. Own strategy tag so the fingerprint can't collide with the
    entry signal at the same symbol/side/timestamp."""
    if (
        futures_account_repository is None
        or derivatives_chain is None
        or symbol not in futures_paper_symbols
    ):
        return None
    try:
        note = await futures_trading.close_futures_paper_position(
            symbol, exit_timestamp, derivatives_chain, futures_account_repository,
        )
        if note is None:
            return None
        logging.getLogger(__name__).info(note)
        signal = Signal(
            symbol=symbol,
            side=side,
            strategy=f"{_STRATEGY_NAME}-futures-exit",
            timestamp=exit_timestamp,
            price=market_price,
            rationale=note,
            category="futures_exit",
        )
        if not await signal_repository.contains(signal.fingerprint):
            await signal_repository.record(signal.fingerprint, signal.timestamp)
        return note
    except Exception:
        logging.getLogger(__name__).warning(
            "Futures paper close failed for %s -- continuing without it.",
            symbol, exc_info=True,
        )
        return None


async def _win_rate_summary(
    symbol: str, config: AppConfig, trade_repository: TradeRepository
) -> str | None:
    """Summarize this symbol's historical closed BUY-trade win rate for a notification.

    BUY-only, matching ``paper_trading._buy_only_win_rate`` exactly -- this is
    the same number the paper account's eligibility gate actually uses, so a
    signal tagged "not eligible yet" is never paired with a rationale showing
    a healthier combined BUY+SELL number that would make the rejection look
    wrong. SELL trades are excluded even for a SELL-signal notification since
    they can never affect the (BUY-only) paper account either way.

    Returns None if there's no closed BUY-trade history yet (a brand-new
    symbol, or one whose only trades are still open) -- nothing meaningful to
    show.
    """
    trades = await trade_repository.get_trades(symbol, config.candle_interval)
    closed = [
        trade for trade in trades if trade.side == SignalSide.BUY and trade.status == "closed"
    ]
    if not closed:
        return None
    wins = sum(1 for trade in closed if trade.pnl_percent is not None and trade.pnl_percent > 0)
    losses = sum(1 for trade in closed if trade.pnl_percent is not None and trade.pnl_percent < 0)
    win_rate = 100 * wins / len(closed)
    return f"win_rate={win_rate:.1f}%({wins}W/{losses}L)"

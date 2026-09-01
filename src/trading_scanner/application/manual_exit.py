"""Manual, dashboard-triggered exit of one real cash-equity position.

2026-08-26: until now, the only way to close a real position outside the
strategy's own dynamic exit or its GTT bracket was a one-off script run by
hand (see the BDL.NS/MCX.NS bracket-repair incident). This gives the
dashboard a real "exit now" control that reuses exactly the same
reconcile-then-exit path ``signal_pipeline.py``'s own strategy-exit branch
already uses, so a manual exit behaves identically to a strategy exit --
same GTT reconciliation, same protected-limit order, same Telegram
notification, same paper-benchmark recording.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.application import (
    broker_reconciliation,
    gtt_bracket,
    live_cash_execution,
    paper_benchmark,
)
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.ports import Notifier
from trading_scanner.infrastructure.db import (
    TursoGttRepository,
    TursoLiveOrderRepository,
    TursoPaperBenchmarkRepository,
)
from trading_scanner.infrastructure.kite import KiteOrderExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ManualExitResult:
    ok: bool
    message: str


async def exit_position(
    symbol: str,
    market_price: Decimal,
    config: AppConfig,
    order_executor: KiteOrderExecutor,
    gtt_repository: TursoGttRepository | None,
    live_order_repository: TursoLiveOrderRepository,
    paper_benchmark_repository: TursoPaperBenchmarkRepository | None,
    notifier: Notifier,
) -> ManualExitResult:
    """Close whatever real position is open for ``symbol``, if any.

    ``market_price`` should be a fresh quote, not a stale one -- it prices
    the protected limit order ``execute_cash_exit`` places (see that
    function's own docstring), same as a strategy-driven exit uses the
    current bar's close.
    """
    # broker_reconciliation.get_unclosed_entry_leg, not get_open_cash_legs
    # directly -- the latter is COMPLETE-only and would report "nothing to
    # exit" for a real entry stuck at status=UNKNOWN (fill unconfirmed),
    # which is exactly the case a manual exit is most likely to be needed
    # for. See that module's own docstring.
    entry_leg = await broker_reconciliation.get_unclosed_entry_leg(symbol, live_order_repository)
    if entry_leg is None:
        return ManualExitResult(False, f"No real open position for {symbol}.")

    should_exit = True
    if gtt_repository is not None:
        should_exit = await gtt_bracket.reconcile_before_exit(
            symbol, entry_leg.tradingsymbol, config, order_executor, gtt_repository
        )
    if not should_exit:
        # 2026-08-28: the real position was already flat (GTT fired, or a
        # manual square-off outside this app) -- close it in our own
        # ledger too, not just note it, or the next dashboard load and the
        # next entry-eligibility check both still see it as open (this is
        # exactly how COCHINSHIP.NS/VMM.NS got stuck).
        exit_basket_id = await live_cash_execution.record_broker_side_exit(
            symbol, entry_leg, market_price, live_order_repository, notifier,
        )
        if paper_benchmark_repository is not None:
            exit_legs = await live_order_repository.get_legs(exit_basket_id)
            if exit_legs:
                try:
                    await paper_benchmark.record_exit(
                        symbol, entry_leg.basket_id, market_price, datetime.now(UTC),
                        exit_legs[0], paper_benchmark_repository,
                    )
                except Exception:
                    logger.exception(
                        "Paper-benchmark exit recording raised for %s -- "
                        "real exit unaffected.", symbol,
                    )
        return ManualExitResult(
            True, f"{symbol} was already flat at the broker -- reconciled in our records."
        )

    exit_basket_id = await live_cash_execution.execute_cash_exit(
        symbol, market_price, config, order_executor, live_order_repository, notifier,
    )
    if exit_basket_id is None:
        return ManualExitResult(False, f"{symbol}: nothing to exit.")

    exit_legs = await live_order_repository.get_legs(exit_basket_id)
    if not exit_legs or exit_legs[0].status != "COMPLETE":
        return ManualExitResult(
            False,
            f"{symbol}: exit order did not complete -- check the Kite account directly, "
            "this may need manual squaring off.",
        )

    exit_leg = exit_legs[0]
    if paper_benchmark_repository is not None:
        try:
            await paper_benchmark.record_exit(
                symbol, entry_leg.basket_id, market_price, datetime.now(UTC),
                exit_leg, paper_benchmark_repository,
            )
        except Exception:
            logger.exception(
                "Paper-benchmark exit recording raised for %s -- real exit unaffected.", symbol
            )

    price_note = f" @ {exit_leg.average_price}" if exit_leg.average_price else ""
    return ManualExitResult(True, f"{symbol} closed{price_note}.")

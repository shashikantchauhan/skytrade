"""Paper-simulated benchmark run 1:1 alongside every real live-cash trade
(see ``application/live_cash_execution.py``) -- measures execution quality
by comparing the decision price (paper "fill", no friction) against Kite's
real ``average_price`` for the same basket. Strictly paired: only ever
opened right after a real entry actually filled, only ever closed right
after a real exit actually filled.

DB-only -- deliberately no ``Notifier`` param at all (no Telegram
notifications, by explicit product decision, consistent with keeping
Telegram limited to real cash-order events; see ``infrastructure/
telegram.py``). Fresh table (``paper_benchmark_positions``), independent of
the retired ``paper_trading.py``/``paper_account`` system -- do not import
from or extend that module here.
"""

from datetime import datetime
from decimal import Decimal

from trading_scanner.domain.models import LiveOrderLeg, PaperBenchmarkPosition
from trading_scanner.infrastructure.db import TursoPaperBenchmarkRepository


async def record_entry(
    symbol: str,
    decision_price: Decimal,
    real_entry_leg: LiveOrderLeg,
    paper_benchmark_repository: TursoPaperBenchmarkRepository,
) -> None:
    """Opens the paper-benchmark side right after a real cash entry filled.

    ``real_entry_leg`` must be the just-opened COMPLETE leg (from
    ``live_order_repository.get_open_cash_legs(symbol)`` right after
    ``live_cash_execution.execute_cash_entry``) -- its own ``basket_id``
    becomes this row's key, so the exit side can close it by exact match
    later."""
    await paper_benchmark_repository.open_position(
        PaperBenchmarkPosition(
            symbol=symbol,
            basket_id=real_entry_leg.basket_id,
            quantity=real_entry_leg.quantity,
            entry_timestamp=real_entry_leg.placed_at,
            paper_entry_price=decision_price,
            # A COMPLETE leg should always carry a real average_price; the
            # fallback only guards against that invariant somehow not
            # holding, mirroring the same defensive pattern used when the
            # GTT bracket prices itself off this leg (see
            # signal_pipeline.py's ``leg.average_price or market_price``).
            real_entry_price=real_entry_leg.average_price or decision_price,
        )
    )


async def record_exit(
    symbol: str,
    entry_basket_id: str,
    decision_price: Decimal,
    exit_timestamp: datetime,
    real_exit_leg: LiveOrderLeg,
    paper_benchmark_repository: TursoPaperBenchmarkRepository,
) -> None:
    """Closes the paper-benchmark side by the real entry's own basket_id
    (captured by the caller before the real exit ran, while the position
    was still open) -- only call this once the real exit leg is confirmed
    COMPLETE; an incomplete/failed real exit must leave both sides open."""
    await paper_benchmark_repository.close_position(
        symbol,
        entry_basket_id,
        exit_timestamp,
        decision_price,
        real_exit_leg.average_price or decision_price,
    )

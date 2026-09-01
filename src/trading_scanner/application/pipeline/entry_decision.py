"""Persisting/finalizing one real cash-entry candidate's outcome -- split
out of ``signal_pipeline.py`` (Phase 8, see
``application/pipeline/__init__.py``). No behavior changed; every
function's body moved as-is.
"""

import logging
from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.application import broker_reconciliation, gtt_bracket, paper_benchmark
from trading_scanner.domain.models import EntryDecisionRecord, SignalSide
from trading_scanner.domain.ports import Notifier
from trading_scanner.infrastructure.db import (
    LiveCashToggleState,
    TursoEntryDecisionRepository,
    TursoGttRepository,
    TursoLiveOrderRepository,
    TursoPaperBenchmarkRepository,
)
from trading_scanner.infrastructure.kite import KiteOrderExecutor


async def _persist_entry_decision(
    entry_decision_repository: TursoEntryDecisionRepository | None,
    *,
    symbol: str,
    signal_timestamp: datetime,
    signal_price: Decimal,
    track_record_passed: bool | None,
    quality_passed: bool | None,
    conviction_passed: bool | None,
    ranking_score: Decimal | None = None,
    ranking_passed: bool | None = None,
    final_decision: str,
    blocked_reason: str | None,
) -> None:
    """Best-effort write of one ``EntryDecisionRecord`` row (Phase 3) --
    ``capital_passed``/``position_limit_passed``/``cutoff_passed`` are
    always ``None`` here (see that dataclass's own docstring for why);
    ``blocked_reason`` still carries the free-text explanation for those.
    A no-op when no repository was threaded through (every caller stays
    optional, same convention as ``gtt_repository``/
    ``paper_benchmark_repository`` elsewhere in this module). Never raises
    into the caller's own decision flow -- this is an audit trail, not
    something real trading depends on."""
    if entry_decision_repository is None:
        return
    try:
        await entry_decision_repository.record(
            EntryDecisionRecord(
                symbol=symbol,
                strategy="alpha_engine",
                signal_timestamp=signal_timestamp,
                signal_side=SignalSide.BUY,
                signal_price=signal_price,
                track_record_passed=track_record_passed,
                quality_passed=quality_passed,
                conviction_passed=conviction_passed,
                ranking_score=ranking_score,
                ranking_passed=ranking_passed,
                capital_passed=None,
                position_limit_passed=None,
                cutoff_passed=None,
                final_decision=final_decision,
                blocked_reason=blocked_reason,
                created_at=datetime.now(UTC),
            )
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "Failed to persist entry decision for %s -- decision itself unaffected.", symbol,
        )


async def _finalize_cash_entry(
    symbol: str,
    market_price: Decimal,
    cash_state: LiveCashToggleState,
    live_order_repository: TursoLiveOrderRepository,
    notifier: Notifier,
    gtt_repository: TursoGttRepository | None,
    paper_benchmark_repository: TursoPaperBenchmarkRepository | None,
    order_executor: KiteOrderExecutor,
) -> str:
    """Called right after ``live_cash_execution.execute_cash_entry`` was
    attempted for a candidate that already cleared every real gate
    (eligibility, entry_quality_filter, conviction_filter): places the GTT
    bracket and records the paper-benchmark entry if a real fill happened,
    or sends the "missed" notification if it didn't. Shared by
    ``_rank_and_open_cash_positions`` (ranked path) and
    ``_process_symbol``'s own unranked fallback, so this bookkeeping has a
    single implementation instead of being duplicated between them.

    ``get_open_cash_legs`` only ever returns a leg that is
    ``status='COMPLETE'`` -- a rejected/incomplete real order never appears
    there, so "a leg exists" reliably means "a real fill actually
    happened," no extra status checking needed (same reasoning
    ``application/paper_benchmark.py``'s own design doc already relies on).
    """
    opened = await live_order_repository.get_open_cash_legs(symbol)
    if opened:
        leg = opened[0]
        if gtt_repository is not None:
            await gtt_bracket.place_bracket(
                symbol, leg.tradingsymbol, leg.quantity, leg.average_price or market_price,
                cash_state, order_executor, gtt_repository, notifier,
            )
        if paper_benchmark_repository is not None:
            try:
                await paper_benchmark.record_entry(
                    symbol, market_price, leg, paper_benchmark_repository,
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "Paper-benchmark entry recording raised for %s -- real trade unaffected.",
                    symbol,
                )
        return f"cash: opened {leg.quantity} qty (avg ₹{leg.average_price or market_price:.2f})"
    # 2026-08-26: notify only for the specific miss that's actually costly --
    # a signal that cleared every real gate and still didn't get a real
    # order, which today only happens because execute_cash_entry's own
    # capacity check (max_positions already full) or its cutoff/execution
    # check turned it away. See _notify_missed_cash_entry's own docstring.
    await _notify_missed_cash_entry(
        symbol, market_price, cash_state, live_order_repository, notifier
    )
    return "cash: SKIPPED (no free slot, past cutoff, or execution failed -- see logs)"


async def _notify_missed_cash_entry(
    symbol: str,
    market_price: Decimal,
    cash_state: LiveCashToggleState,
    live_order_repository: TursoLiveOrderRepository,
    notifier: Notifier,
) -> None:
    """A BUY signal cleared all real gates (track record,
    entry_quality_filter, conviction_filter) -- a genuinely good signal --
    but still got no real order. Tell the user, so a missed opportunity is
    visible instead of silently dropped.

    2026-08-26: deliberately narrow -- a signal that failed track record,
    the quality filter, or the conviction filter was correctly rejected,
    not "missed", and does not notify here (see the call site). This only
    fires for the case that's actually costly: capital was tied up (or,
    more rarely, the entry cutoff/an execution hiccup) while a signal worth
    taking passed by. Distinct from the entry-signal notification silenced
    on 2026-08-21, which fired for every signal regardless of outcome.
    """
    all_open = await broker_reconciliation.get_all_unclosed_positions(live_order_repository)
    if len(all_open) >= cash_state.max_positions:
        reason = f"all {cash_state.max_positions} real slots are already full"
    else:
        reason = "past the entry cutoff or the order didn't go through -- check logs"
    await notifier.send_text(
        "\U0001f6ab <b>MISSED BUY SIGNAL</b>\n"
        f"{symbol} @ {market_price} cleared eligibility, the quality filter, and the "
        f"conviction filter, but {reason}."
    )

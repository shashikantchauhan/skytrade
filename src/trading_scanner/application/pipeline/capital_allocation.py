"""One scan cycle's ranked capital allocation across the paper/futures/cash
books -- split out of ``signal_pipeline.py`` (Phase 8, see
``application/pipeline/__init__.py``). No behavior changed; every
function's body moved as-is.
"""

import asyncio
import logging
from decimal import Decimal

from trading_scanner.application import (
    entry_gates,
    futures_trading,
    live_cash_execution,
    paper_trading,
)
from trading_scanner.application.fast_predict import FastPredictResult
from trading_scanner.application.pipeline.entry_decision import (
    _finalize_cash_entry,
    _persist_entry_decision,
)
from trading_scanner.application.pipeline.market_data import _market_price
from trading_scanner.application.ranking import (
    MIN_SCORE,
    RankedCandidate,
    rank_candidates,
    score_candidate,
    symbol_expectancy,
)
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import Candle, SignalSide
from trading_scanner.domain.order_intent import compute_intent_id
from trading_scanner.domain.ports import (
    FuturesPaperAccountRepository,
    Notifier,
    PaperAccountRepository,
    TradeRepository,
)
from trading_scanner.infrastructure.db import (
    LiveCashToggleState,
    TursoEntryDecisionRepository,
    TursoGttRepository,
    TursoLiveOrderRepository,
    TursoPaperBenchmarkRepository,
)
from trading_scanner.infrastructure.kite import KiteDerivativesChain, KiteOrderExecutor


async def _open_paper_position(
    symbol: str,
    config: AppConfig,
    entry_timestamp,
    entry_price: Decimal,
    trade_repository: TradeRepository,
    paper_account_repository: PaperAccountRepository,
    paper_account_lock: asyncio.Lock,
) -> str | None:
    """Attempt to open a real paper position for a BUY entry; return a status note.

    Returns a short human-readable reason whenever no position was opened
    (not yet eligible, or the account is out of free capital) so the caller
    can surface it in the notification instead of silently skipping.

    The eligibility check is read-only and per-symbol, so it stays outside
    the lock; only the capital check-then-act in try_open_position needs
    exclusive access to the shared account.
    """
    if not await paper_trading.is_eligible(symbol, config.candle_interval, trade_repository):
        return "paper: not eligible yet (win_rate<55% or insufficient trade history)"
    async with paper_account_lock:
        position = await paper_trading.try_open_position(
            symbol, entry_timestamp, entry_price, paper_account_repository
        )
    if position is None:
        return "paper: SKIPPED (no capital available)"
    return f"paper: opened {position.quantity} qty (₹{position.capital_allocated:.0f})"


async def _rank_and_open_paper_positions(
    candidates: list[tuple[str, RankedCandidate]],
    paper_account_repository: PaperAccountRepository,
    paper_account_lock: asyncio.Lock,
) -> dict[str, str]:
    """Open paper positions for one cycle's already-eligible BUY candidates,
    strongest-ranked first, instead of whichever symbol's task happened to
    finish first (see ``application/ranking.py``).

    Each ``try_open_position`` call is still individually gated on free
    capital exactly as ``_open_paper_position`` does today -- this only
    changes the *order* candidates compete for that capital, not the
    capital math itself. Once capital runs out, every remaining (weaker)
    candidate is skipped and tagged distinctly from a plain
    no-capital-at-all skip, so it is visible in the notification that
    ranking, not just capital, decided the outcome.

    (2026-08-13: a capital-rotation/eviction variant of this -- selling a
    weaker, losing open position to make room for a stronger new signal --
    was built, tested, and deployed, then explicitly rejected and reverted
    same day. Not wanted; do not reintroduce without being asked.)
    """
    notes: dict[str, str] = {}
    symbol_by_candidate = {candidate: symbol for symbol, candidate in candidates}
    ranked = rank_candidates([candidate for _, candidate in candidates])
    for index, candidate in enumerate(ranked):
        symbol = symbol_by_candidate[candidate]
        score = score_candidate(candidate)
        if score < MIN_SCORE:
            # Ranked strongest-first, so every remaining candidate also
            # scores below the floor -- reject them all and stop, rather
            # than spending capital on a signal too weak by policy just
            # because a slot happens to be free (see MIN_SCORE's own
            # docstring for the 2026-08-14 baseline this threshold is
            # tuned against).
            for weaker_candidate in ranked[index:]:
                weaker_symbol = symbol_by_candidate[weaker_candidate]
                weaker_score = score_candidate(weaker_candidate)
                notes[weaker_symbol] = (
                    f"paper: REJECTED (score {weaker_score:.0f} below minimum {MIN_SCORE:.0f})"
                )
            break
        async with paper_account_lock:
            position = await paper_trading.try_open_position(
                symbol, candidate.entry_timestamp, candidate.entry_price, paper_account_repository
            )
        if position is None:
            notes[symbol] = (
                "paper: SKIPPED (ranked below capacity, no capital left)"
                if index > 0
                else "paper: SKIPPED (no capital available)"
            )
        else:
            notes[symbol] = (
                f"paper: opened {position.quantity} qty (₹{position.capital_allocated:.0f})"
            )
    return notes


async def _rank_and_open_cash_positions(
    candidates: list[tuple[str, RankedCandidate]],
    config: AppConfig,
    cash_state: LiveCashToggleState,
    order_executor: KiteOrderExecutor,
    live_order_repository: TursoLiveOrderRepository,
    notifier: Notifier,
    gtt_repository: TursoGttRepository | None,
    paper_benchmark_repository: TursoPaperBenchmarkRepository | None,
    entry_decision_repository: TursoEntryDecisionRepository | None = None,
) -> dict[str, str]:
    """Open REAL cash-equity BUY positions for one cycle's already-gated
    candidates (track record, entry_quality_filter, and conviction_filter
    all already cleared -- see ``_collect_and_open_ranked_positions``),
    strongest-ranked first, instead of whichever symbol's task happened to
    finish first -- same idea as ``_rank_and_open_paper_positions``, for
    real money instead of the paper book.

    Sequential by construction (candidates are attempted one at a time, in
    ranked order) -- unlike the old per-symbol-concurrent path this
    replaces, no lock is needed here: there is no way for two of these
    calls to race on ``execute_cash_entry``'s own check-then-act capacity
    check, because there's only ever one in flight. ``execute_cash_entry``
    still does its own real capacity/cutoff/already-open check per call --
    this function doesn't pre-compute whether each candidate will fit, it
    just tries them in ranked order and lets that real check (surfaced via
    ``_finalize_cash_entry``'s own broker_reconciliation-backed lookup)
    decide.
    """
    notes: dict[str, str] = {}
    symbol_by_candidate = {candidate: symbol for symbol, candidate in candidates}
    ranked = rank_candidates([candidate for _, candidate in candidates])
    for index, candidate in enumerate(ranked):
        symbol = symbol_by_candidate[candidate]
        score = score_candidate(candidate)
        if score < MIN_SCORE:
            # Ranked strongest-first, so every remaining candidate also
            # scores below the floor -- reject them all and stop, rather
            # than spending real capital on a signal too weak by policy
            # just because a slot happens to be free (see MIN_SCORE's own
            # docstring -- 0 by default, a no-op, pending real evidence
            # this heuristic score discriminates quality on its own).
            for weaker_candidate in ranked[index:]:
                weaker_symbol = symbol_by_candidate[weaker_candidate]
                weaker_score = score_candidate(weaker_candidate)
                notes[weaker_symbol] = (
                    f"cash: REJECTED (score {weaker_score:.0f} below minimum {MIN_SCORE:.0f})"
                )
                await _persist_entry_decision(
                    entry_decision_repository,
                    symbol=weaker_symbol,
                    signal_timestamp=weaker_candidate.entry_timestamp,
                    signal_price=weaker_candidate.entry_price,
                    track_record_passed=True, quality_passed=True, conviction_passed=True,
                    ranking_score=Decimal(str(weaker_score)), ranking_passed=False,
                    final_decision="rejected", blocked_reason=notes[weaker_symbol],
                )
            break
        try:
            await live_cash_execution.execute_cash_entry(
                symbol, candidate.entry_price, config, cash_state, order_executor,
                live_order_repository, notifier,
                signal_timestamp=candidate.entry_timestamp,
            )
            notes[symbol] = await _finalize_cash_entry(
                symbol, candidate.entry_price, cash_state, live_order_repository, notifier,
                gtt_repository, paper_benchmark_repository, order_executor,
            )
            opened = notes[symbol].startswith("cash: opened")
            await _persist_entry_decision(
                entry_decision_repository,
                symbol=symbol,
                signal_timestamp=candidate.entry_timestamp, signal_price=candidate.entry_price,
                track_record_passed=True, quality_passed=True, conviction_passed=True,
                ranking_score=Decimal(str(score)), ranking_passed=True,
                final_decision="opened" if opened else "skipped",
                blocked_reason=None if opened else notes[symbol],
                intent_id=compute_intent_id(symbol, "BUY", candidate.entry_timestamp, "cash"),
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "Live cash order entry raised for %s -- rest of signal handling still stands.",
                symbol,
            )
            notes[symbol] = "cash: ERROR placing order (see logs)"
            await _persist_entry_decision(
                entry_decision_repository,
                symbol=symbol,
                signal_timestamp=candidate.entry_timestamp, signal_price=candidate.entry_price,
                track_record_passed=True, quality_passed=True, conviction_passed=True,
                ranking_score=Decimal(str(score)), ranking_passed=True,
                final_decision="error", blocked_reason=notes[symbol],
                intent_id=compute_intent_id(symbol, "BUY", candidate.entry_timestamp, "cash"),
            )
    return notes


async def _rank_and_open_futures_positions(
    candidates: list[tuple[str, RankedCandidate]],
    interval: str,
    trade_repository: TradeRepository,
    derivatives_chain: KiteDerivativesChain,
    futures_account_repository: FuturesPaperAccountRepository,
) -> dict[str, str]:
    """Same idea as ``_rank_and_open_paper_positions``, for the futures
    paper account instead of cash -- strongest-ranked candidate wins this
    cycle's margin budget first, instead of whichever symbol's task
    happened to finish first.

    One real difference from the cash version: candidates here can be
    EITHER direction (BUY or SELL both compete for the same futures margin
    pool, unlike cash which is BUY-only) -- ``score_candidate`` already
    scores conviction in each candidate's own direction correctly (see
    ``RankedCandidate.direction``), so ranking them together is safe.

    ``open_futures_paper_position`` does its own eligibility recheck and
    real Kite margin call per candidate -- this function doesn't
    pre-compute whether each one will fit, it just tries them in ranked
    order and lets that real check reject once the account's margin budget
    (or a symbol's own eligibility) says no, exactly like the cash version
    relies on ``try_open_position``'s own capital check.
    """
    notes: dict[str, str] = {}
    symbol_by_candidate = {candidate: symbol for symbol, candidate in candidates}
    ranked = rank_candidates([candidate for _, candidate in candidates])
    for candidate in ranked:
        symbol = symbol_by_candidate[candidate]
        note = await futures_trading.open_futures_paper_position(
            symbol, candidate.direction, candidate.entry_timestamp, candidate.entry_price,
            interval, derivatives_chain, trade_repository, futures_account_repository,
        )
        notes[symbol] = (
            note
            if note is not None
            else "futures-paper: SKIPPED (ranked below capacity, margin, or eligibility)"
        )
    return notes


async def _collect_and_open_ranked_positions(
    evaluated_by_symbol: dict[str, tuple[FastPredictResult, Candle] | None],
    config: AppConfig,
    trade_repository: TradeRepository,
    paper_account_repository: PaperAccountRepository,
    paper_account_lock: asyncio.Lock,
    derivatives_chain: KiteDerivativesChain | None,
    futures_account_repository: FuturesPaperAccountRepository | None,
    futures_paper_symbols: frozenset[str],
    notifier: Notifier,
    order_executor: KiteOrderExecutor | None = None,
    live_order_repository: TursoLiveOrderRepository | None = None,
    gtt_repository: TursoGttRepository | None = None,
    paper_benchmark_repository: TursoPaperBenchmarkRepository | None = None,
    cash_state: LiveCashToggleState | None = None,
    entry_decision_repository: TursoEntryDecisionRepository | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """One scan cycle's ranked capital allocation for ALL THREE books --
    paper (BUY-only, ``_rank_and_open_paper_positions``), futures (BUY+SELL,
    Nifty50-allowlisted, ``_rank_and_open_futures_positions``), and REAL
    cash (BUY-only, ``_rank_and_open_cash_positions``) -- from the same
    batch of already-evaluated symbols.

    2026-08-14: this used to be duplicated inline inside
    ``run_signal_pipeline`` (the cron path) with no futures-side
    equivalent at all -- meaning ``live_pipeline.py`` (the actual
    production driver) never ranked either book, cash or futures, despite
    the ranking module existing and being "deployed." Factored out here so
    both the cron path and the live WebSocket path call the exact same
    ranking logic and can't silently drift apart again. Real cash orders
    joined this same ranked flow later -- previously ``execute_cash_entry``
    was first-come-first-served, no ranking step existed before it.

    Cash competes on a strictly narrower candidate set than paper: paper is
    gated only on ``paper_trading.is_eligible`` (track record); real cash
    ALSO requires ``entry_quality_filter`` and ``conviction_filter`` to
    pass (see both modules' own docstrings for the evidence) before a
    candidate is even allowed to compete for ranked capital -- a real-money
    order is held to a higher bar than the paper benchmark. Cash ranking
    only runs at all when ``order_executor``/``live_order_repository``/
    ``cash_state`` are all given.

    ``cash_state`` -- see ``live_cash_execution.py``'s module docstring:
    the single ``LiveCashToggleState`` the caller built for this cycle
    (from the DB toggle in ``live_pipeline.py``, from static config once
    per run in ``run_signal_pipeline``), passed explicitly rather than
    read off a possibly-stale clone of ``config``.

    Returns ``(paper_notes, futures_notes, cash_notes)``, all
    ``symbol -> note``, for the caller to pass into ``_process_symbol`` as
    ``precomputed_paper_note``/``precomputed_futures_note``/
    ``precomputed_cash_note`` so its second pass doesn't redecide what this
    already decided.
    """
    paper_notes: dict[str, str] = {}
    ranked_candidates: list[tuple[str, RankedCandidate]] = []
    futures_notes: dict[str, str] = {}
    futures_candidates: list[tuple[str, RankedCandidate]] = []
    cash_notes: dict[str, str] = {}
    cash_candidates: list[tuple[str, RankedCandidate]] = []
    futures_enabled = derivatives_chain is not None and futures_account_repository is not None
    cash_enabled = (
        order_executor is not None and live_order_repository is not None and cash_state is not None
    )

    for symbol, evaluated in evaluated_by_symbol.items():
        if evaluated is None:
            continue
        result, newest_candle = evaluated
        if result.signal not in ("BUY", "SELL"):
            continue
        entry_price = _market_price(newest_candle)

        if result.signal == "BUY":
            track_record_gate = await entry_gates.evaluate_track_record_gate(
                symbol, config.candle_interval, trade_repository
            )
            if track_record_gate.passed:
                expectancy = await symbol_expectancy(
                    symbol, SignalSide.BUY, config.candle_interval, trade_repository
                )
                candidate = RankedCandidate(
                    symbol=symbol,
                    entry_timestamp=newest_candle.timestamp,
                    entry_price=entry_price,
                    prediction_at_entry=result.prediction,
                    adx=result.adx,
                    regime_normalized=result.regime_normalized,
                    volatility_margin=result.volatility_margin,
                    expectancy=expectancy,
                )
                ranked_candidates.append((symbol, candidate))

                if cash_enabled:
                    quality_decision = entry_gates.evaluate_cash_quality_gates(
                        result.volatility_margin, result.regime_normalized,
                        newest_candle.high, newest_candle.low, newest_candle.close,
                    )
                    if quality_decision.allowed:
                        cash_candidates.append((symbol, candidate))
                    else:
                        cash_notes[symbol] = f"cash: REJECTED ({quality_decision.blocked_reason})"
                        await _persist_entry_decision(
                            entry_decision_repository,
                            symbol=symbol,
                            signal_timestamp=newest_candle.timestamp, signal_price=entry_price,
                            track_record_passed=True,
                            quality_passed=quality_decision.gates[0].passed,
                            conviction_passed=quality_decision.gates[1].passed,
                            final_decision="rejected",
                            blocked_reason=quality_decision.blocked_reason,
                        )
            else:
                paper_notes[symbol] = f"paper: {track_record_gate.reason}"
                if cash_enabled:
                    cash_notes[symbol] = f"cash: {track_record_gate.reason}"
                    await _persist_entry_decision(
                        entry_decision_repository,
                        symbol=symbol,
                        signal_timestamp=newest_candle.timestamp, signal_price=entry_price,
                        track_record_passed=False, quality_passed=None, conviction_passed=None,
                        final_decision="rejected", blocked_reason=track_record_gate.reason,
                    )

        if futures_enabled and symbol in futures_paper_symbols:
            side = SignalSide.BUY if result.signal == "BUY" else SignalSide.SELL
            if await futures_trading.is_eligible(
                symbol, side, config.candle_interval, trade_repository
            ):
                # 2026-08-17: was missing here (only ever passed on the cash
                # side above) -- meant every futures candidate scored at the
                # flat median-50 fallback in score_candidate, ignoring
                # _EXPECTANCY_WEIGHT (1.5, the highest weight in the
                # formula and the only factor with real walk-forward
                # support -- see ranking.py). Scarce futures slots were
                # competing on volatility/regime/ADX/prediction alone.
                expectancy = await symbol_expectancy(
                    symbol, side, config.candle_interval, trade_repository
                )
                futures_candidates.append((
                    symbol,
                    RankedCandidate(
                        symbol=symbol,
                        entry_timestamp=newest_candle.timestamp,
                        entry_price=entry_price,
                        prediction_at_entry=result.prediction,
                        adx=result.adx,
                        regime_normalized=result.regime_normalized,
                        volatility_margin=result.volatility_margin,
                        direction=side,
                        expectancy=expectancy,
                    ),
                ))
            else:
                futures_notes[symbol] = (
                    "futures-paper: not eligible yet (this side's win_rate<55% or "
                    "insufficient trade history)"
                )

    paper_notes.update(
        await _rank_and_open_paper_positions(
            ranked_candidates, paper_account_repository, paper_account_lock
        )
    )
    if futures_enabled:
        futures_notes.update(
            await _rank_and_open_futures_positions(
                futures_candidates, config.candle_interval, trade_repository,
                derivatives_chain, futures_account_repository,
            )
        )
    if cash_enabled:
        cash_notes.update(
            await _rank_and_open_cash_positions(
                cash_candidates, config, cash_state, order_executor, live_order_repository,
                notifier, gtt_repository, paper_benchmark_repository, entry_decision_repository,
            )
        )
    return paper_notes, futures_notes, cash_notes

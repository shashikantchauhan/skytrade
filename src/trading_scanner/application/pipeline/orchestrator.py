"""``run_signal_pipeline`` (the cron/CLI entry point) and ``_process_symbol``
(per-symbol trade/notification bookkeeping, shared by the cron and live-
ticker paths) -- split out of ``signal_pipeline.py`` (Phase 8, see
``application/pipeline/__init__.py``). No behavior changed; every
function's body moved as-is. Ties every other ``application/pipeline/*``
submodule together into one scan cycle.
"""

import asyncio
import logging
from datetime import UTC, datetime

from trading_scanner.alpha_engine import AlphaEngine
from trading_scanner.application import (
    broker_reconciliation,
    entry_gates,
    gtt_bracket,
    live_cash_execution,
    paper_benchmark,
)
from trading_scanner.application.fast_predict import FastPredictResult
from trading_scanner.application.pipeline.capital_allocation import (
    _collect_and_open_ranked_positions,
    _open_paper_position,
)
from trading_scanner.application.pipeline.entry_decision import _finalize_cash_entry
from trading_scanner.application.pipeline.evaluation import (
    _evaluate_symbol,
    _notify_stale_catch_up_signals,
)
from trading_scanner.application.pipeline.lifecycle import (
    _close_derivatives_shadow,
    _close_futures_paper,
    _close_paper_position,
    _notify_exit,
    _open_derivatives_shadow,
    _open_futures_paper,
    _win_rate_summary,
)
from trading_scanner.application.pipeline.market_data import (
    _STRATEGY_NAME,
    MarketDataProvider,
    NoValidKiteSession,
    _is_kite_token_error,
    _market_price,
    _notify_kite_expired_periodically,
    _select_provider,
)
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import Candle, GateStatusSnapshot, Signal, SignalSide, Trade
from trading_scanner.domain.ports import (
    CandleRepository,
    EngineStateRepository,
    FuturesPaperAccountRepository,
    Notifier,
    PaperAccountRepository,
    SignalRepository,
    TradeRepository,
)
from trading_scanner.infrastructure.db import (
    LiveCashToggleState,
    TursoEntryDecisionRepository,
    TursoFuturesTradeRepository,
    TursoGateStatusRepository,
    TursoGttRepository,
    TursoKiteSessionRepository,
    TursoLiveOrderRepository,
    TursoOptionsTradeRepository,
    TursoPaperBenchmarkRepository,
)
from trading_scanner.infrastructure.kite import KiteDerivativesChain, KiteOrderExecutor

# AlphaEngine's constructor defaults mirror the Pine script's own stock
# defaults. This deployment's actual TradingView chart has two inputs
# overridden from those defaults -- confirmed by comparing the chart's Inputs
# panel against AlphaEngine's default constructor -- so signals here must be
# generated with the same overrides or they will not match the chart:
#   - includeFullHistory is checked (True): the ANN neighbor search scans the
#     entire stored history rather than only the most recent max_bars_back
#     window, which directly changes `prediction` values.
#   - useDynamicExits is checked (True): affects end_long/end_short only, not
#     BUY/SELL entries, but is included so exit behavior matches too.
_ENGINE_SETTINGS = {"include_full_history": True, "use_dynamic_exits": True}

# Per-symbol processing is dominated by network I/O (Yahoo Finance downloads),
# so symbols are processed concurrently rather than one at a time -- cuts a
# 220-symbol run from ~12 minutes to roughly one, without hammering Yahoo with
# 220 simultaneous requests (which risks IP-level throttling, especially from
# GitHub's shared runner IPs). This bound is deliberately conservative rather
# than tuned for maximum throughput.
_MAX_CONCURRENT_SYMBOLS = 12


async def run_signal_pipeline(
    config: AppConfig,
    symbols: list[str],
    candle_repository: CandleRepository,
    signal_repository: SignalRepository,
    engine_state_repository: EngineStateRepository,
    trade_repository: TradeRepository,
    paper_account_repository: PaperAccountRepository,
    notifier: Notifier,
    kite_session_repository: TursoKiteSessionRepository | None = None,
    options_trade_repository: TursoOptionsTradeRepository | None = None,
    futures_trade_repository: TursoFuturesTradeRepository | None = None,
    live_order_repository: TursoLiveOrderRepository | None = None,
    market_data_provider: MarketDataProvider | None = None,
    futures_account_repository: FuturesPaperAccountRepository | None = None,
    futures_paper_symbols: frozenset[str] = frozenset(),
    gtt_repository: TursoGttRepository | None = None,
    paper_benchmark_repository: TursoPaperBenchmarkRepository | None = None,
    entry_decision_repository: TursoEntryDecisionRepository | None = None,
    gate_status_repository: TursoGateStatusRepository | None = None,
) -> None:
    """Ingest recent candles and notify on new BUY/SELL signals for each symbol.

    ``market_data_provider``, when given, bypasses ``_select_provider``
    entirely and is used as-is -- test-only injection point (there is no
    production code path that ever passes this). Exists so tests can
    supply a fake/stubbed provider directly instead of needing a real Kite
    session; production always goes through the real Kite-only selection
    below.

    If ``config.index_symbol`` is set, it is evaluated once per run (same
    machinery, no trades/notifications of its own) and its current state is
    attached to every stock signal's rationale -- purely informational, so
    you can judge whether a stock signal lines up with the broader market or
    looks like noise against it. It is never used to suppress a signal.

    Data source: Kite Connect's Historical Data API only (the same feed NSE
    brokers use -- see ``infrastructure/kite.py``). No Yahoo fallback (see
    ``_select_provider``'s docstring for why that was removed 2026-08-13).
    Kite sessions expire daily and are refreshed via the dashboard's
    ``/kite/login`` flow, not anything in this pipeline -- a stale/missing
    session here just means this run is skipped entirely until the user
    logs in again, rather than silently degrading to a worse data source.

    When Kite is active and the shadow-trade repositories are provided,
    every BUY/SELL entry-exit also shadow-tracks (analysis only, never a
    real order, entirely separate from the paper account's capital):
    a directional option (CALL for BUY, PUT for SELL -- see
    ``application/options_shadow.py``), and a futures position (long for
    BUY, short for SELL) with its own protective hedge option (PUT hedging
    a long future, CALL hedging a short future -- see
    ``application/futures_shadow.py``).
    """
    logger = logging.getLogger(__name__)
    if market_data_provider is not None:
        provider, provider_name, kite = market_data_provider, "injected (test)", None
    else:
        try:
            provider, provider_name, kite = await _select_provider(
                config, kite_session_repository, notifier
            )
        except NoValidKiteSession:
            logger.warning(
                "No valid Kite session -- skipping this run entirely (no Yahoo fallback)."
            )
            return
    logger.info("Using %s as the market data source for this run.", provider_name)
    engine = AlphaEngine(**_ENGINE_SETTINGS)
    derivatives_chain = KiteDerivativesChain(kite) if kite is not None else None
    # Real order placement, if the kill switch is on -- see config/
    # settings.py and application/live_execution.py. None whenever kite is
    # None (no valid session, e.g. Yahoo fallback) since there's nothing to
    # place orders through; _open_derivatives_shadow/_close_derivatives_
    # shadow already no-op on a None order_executor regardless of the flag.
    order_executor = KiteOrderExecutor(kite) if kite is not None else None

    # This path (the hourly cron/CLI runner) has no per-cycle dashboard-
    # toggle refresh -- see AppConfig.live_cash_trading_enabled's own
    # docstring -- so cash_state is just a static repackaging of the same
    # 4 fields config already carries, built once here rather than re-read
    # off config at each call site. See live_cash_execution.py's module
    # docstring for why this is threaded as its own explicit parameter.
    cash_state = LiveCashToggleState(
        enabled=config.live_cash_trading_enabled,
        symbols=config.live_cash_trading_symbols,
        notional=config.live_cash_trading_notional,
        max_positions=config.live_cash_trading_max_positions,
    )

    index_result = None
    if config.index_symbol:
        try:
            index_evaluated = await _evaluate_symbol(
                config.index_symbol, config, provider, engine, candle_repository,
                engine_state_repository,
            )
            # Only the current (last) bar -- the index is never itself
            # traded, just used as market-regime context, so any stale
            # catch-up entries from a gap don't need their own notification.
            index_result = index_evaluated[-1][0] if index_evaluated else None
        except Exception:
            logger.exception("Unexpected exception while evaluating index %s", config.index_symbol)

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SYMBOLS)
    # Paper-account reads/writes are check-then-act (read cash_balance, decide,
    # then write) -- concurrent symbols must not interleave through that
    # section, or two symbols could both see the same "enough capital" balance
    # and both commit, overspending the account. Everything else per symbol
    # (Yahoo download, AlphaEngine evaluation, per-symbol trade/candle rows)
    # is fully independent across symbols and safe to run in parallel.
    paper_account_lock = asyncio.Lock()

    # Real cash entries have the exact same check-then-act shape (read the
    # current open-position count against live_cash_trading_max_positions,
    # decide, then place a real order and record the leg) -- 2026-08-25:
    # without this lock, several symbols signaling BUY in the same cycle
    # could each see the same pre-entry open count and all pass the cap
    # check concurrently, together opening more real positions than
    # max_positions ever intended to allow. Scoped to entries only --
    # exits are deliberately never serialized/blocked (de-risking an
    # already-open real position must never wait on this).
    live_cash_lock = asyncio.Lock()

    # _select_provider only validates Kite once, at the start of the run --
    # the token can still go invalid partway through (observed live: a run
    # validated fine, then every symbol still queued after that point failed
    # with TokenException, with no alert sent at all, since _select_provider
    # never saw it). This guard fires the same day-deduped notification the
    # moment any symbol's download hits that specific error mid-run, so a
    # mid-run expiry is never silent again. mid_run_notify_lock exists only
    # to stop N concurrently-failing symbols from sending N Telegram
    # messages -- the per-day dedup in the notification itself still applies
    # across separate runs.
    mid_run_notified = False
    mid_run_notify_lock = asyncio.Lock()

    async def _notify_mid_run_kite_expiry_once(error: BaseException) -> None:
        nonlocal mid_run_notified
        if kite_session_repository is not None and not mid_run_notified and _is_kite_token_error(
            error
        ):
            async with mid_run_notify_lock:
                if not mid_run_notified:
                    mid_run_notified = True
                    await _notify_kite_expired_periodically(kite_session_repository, notifier)

    # Ranking (see application/ranking.py) needs one scan cycle's full set of
    # BUY candidates before any of them can open a paper position -- so
    # evaluation and paper-position-opening can no longer happen inside a
    # single per-symbol pass the way _process_symbol traditionally did it.
    # This splits the run into: (1) evaluate every symbol concurrently
    # (download/compute only, no trade or paper-account writes -- safe to
    # run in any order), (2) rank this cycle's eligible BUY candidates and
    # open paper positions for them strongest-first, (3) run the rest of
    # each symbol's bookkeeping/notifications concurrently again, now with
    # the paper-position outcome already decided so it isn't redecided.
    evaluated_by_symbol: dict[str, list[tuple[FastPredictResult, Candle]]] = {}

    async def _evaluate_with_limit(symbol: str) -> None:
        async with semaphore:
            try:
                evaluated_by_symbol[symbol] = await _evaluate_symbol(
                    symbol, config, provider, engine, candle_repository, engine_state_repository
                )
            except Exception as error:
                logger.exception("Unexpected exception while evaluating %s", symbol)
                evaluated_by_symbol[symbol] = []
                await _notify_mid_run_kite_expiry_once(error)

    await asyncio.gather(*(_evaluate_with_limit(symbol) for symbol in symbols))

    # 2026-09-02: tell the user about anything caught up from an outage gap
    # (more than one candle closed since the last run) before ranking --
    # see evaluation.py's _evaluate_from_stored_candles docstring. Every
    # entry but the last in each symbol's list is stale catch-up.
    for symbol, evaluations in evaluated_by_symbol.items():
        if len(evaluations) > 1:
            await _notify_stale_catch_up_signals(symbol, evaluations[:-1], notifier)

    paper_notes, futures_notes, cash_notes = await _collect_and_open_ranked_positions(
        evaluated_by_symbol, config, trade_repository, paper_account_repository,
        paper_account_lock, derivatives_chain, futures_account_repository, futures_paper_symbols,
        notifier, order_executor, live_order_repository, gtt_repository, paper_benchmark_repository,
        cash_state, entry_decision_repository,
    )

    async def _process_with_limit(symbol: str) -> None:
        evaluations = evaluated_by_symbol.get(symbol)
        if not evaluations:
            return  # Nothing new for this symbol this cycle -- same as before.
        # Only the current (last) entry is processed for trades/orders/
        # notifications here -- any stale catch-up entries were already
        # handled (notify-only) above, before ranking.
        evaluated = evaluations[-1]
        async with semaphore:
            try:
                await _process_symbol(
                    symbol,
                    config,
                    provider,
                    engine,
                    candle_repository,
                    signal_repository,
                    engine_state_repository,
                    trade_repository,
                    paper_account_repository,
                    notifier,
                    index_result,
                    paper_account_lock,
                    derivatives_chain,
                    options_trade_repository,
                    futures_trade_repository,
                    evaluated,
                    order_executor,
                    live_order_repository,
                    precomputed_paper_note=paper_notes.get(symbol),
                    futures_account_repository=futures_account_repository,
                    futures_paper_symbols=futures_paper_symbols,
                    precomputed_futures_note=futures_notes.get(symbol),
                    gtt_repository=gtt_repository,
                    paper_benchmark_repository=paper_benchmark_repository,
                    live_cash_lock=live_cash_lock,
                    precomputed_cash_note=cash_notes.get(symbol),
                    cash_state=cash_state,
                    gate_status_repository=gate_status_repository,
                )
            except Exception as error:
                logger.exception("Unexpected exception while processing %s", symbol)
                await _notify_mid_run_kite_expiry_once(error)

    await asyncio.gather(*(_process_with_limit(symbol) for symbol in symbols))


async def _record_gate_status(
    symbol: str,
    config: AppConfig,
    result: FastPredictResult,
    newest_candle: Candle,
    trade_repository: TradeRepository,
    gate_status_repository: TursoGateStatusRepository,
) -> None:
    """Snapshot this symbol's gate state for the dashboard's Gates tab --
    2026-09-02, see domain/models.py's GateStatusSnapshot and docs/
    decisions/008-gate-status-snapshot.md.

    Computed for *every* symbol evaluated this cycle regardless of
    ``result.signal`` (BUY, SELL, or NEUTRAL) -- unlike the real cash-entry
    gates below (only reached for an active BUY/SELL that already won
    ranking), this exists so a NEUTRAL symbol that's close to qualifying is
    visible too. Reuses the exact same ``entry_gates`` calls a real cash
    order would use -- no separate/duplicated gate logic -- so what the
    dashboard shows always matches what a real decision would actually see.

    Best-effort: never allowed to raise into the caller (a dashboard
    convenience must not be able to break real trade processing)."""
    try:
        track_record_gate = await entry_gates.evaluate_track_record_gate(
            symbol, config.candle_interval, trade_repository
        )
        quality_decision = entry_gates.evaluate_cash_quality_gates(
            result.volatility_margin, result.regime_normalized,
            newest_candle.high, newest_candle.low, newest_candle.close,
        )
        quality_passed = quality_decision.gates[0].passed
        conviction_passed = quality_decision.gates[1].passed
        now = datetime.now(UTC)
        await gate_status_repository.set_snapshot(
            GateStatusSnapshot(
                symbol=symbol,
                interval=config.candle_interval,
                signal=result.signal,
                adx=result.adx,
                regime_normalized=result.regime_normalized,
                volatility_margin=result.volatility_margin,
                track_record_passed=track_record_gate.passed,
                quality_passed=quality_passed,
                conviction_passed=conviction_passed,
                evaluated_at=newest_candle.timestamp,
                updated_at=now,
            )
        )
    except Exception:
        logging.getLogger(__name__).exception("Failed to record gate status for %s", symbol)


async def _process_symbol(
    symbol: str,
    config: AppConfig,
    provider: MarketDataProvider,
    engine: AlphaEngine,
    candle_repository: CandleRepository,
    signal_repository: SignalRepository,
    engine_state_repository: EngineStateRepository,
    trade_repository: TradeRepository,
    paper_account_repository: PaperAccountRepository,
    notifier: Notifier,
    index_result: FastPredictResult | None,
    paper_account_lock: asyncio.Lock,
    derivatives_chain: KiteDerivativesChain | None = None,
    options_trade_repository: TursoOptionsTradeRepository | None = None,
    futures_trade_repository: TursoFuturesTradeRepository | None = None,
    precomputed_evaluation: tuple[FastPredictResult, Candle] | None = None,
    order_executor: KiteOrderExecutor | None = None,
    live_order_repository: TursoLiveOrderRepository | None = None,
    precomputed_paper_note: str | None = None,
    futures_account_repository: FuturesPaperAccountRepository | None = None,
    futures_paper_symbols: frozenset[str] = frozenset(),
    precomputed_futures_note: str | None = None,
    gtt_repository: TursoGttRepository | None = None,
    paper_benchmark_repository: TursoPaperBenchmarkRepository | None = None,
    live_cash_lock: asyncio.Lock | None = None,
    precomputed_cash_note: str | None = None,
    cash_state: LiveCashToggleState | None = None,
    gate_status_repository: TursoGateStatusRepository | None = None,
) -> None:
    """Evaluate the newest bar, record/close trades, and notify for one symbol.

    Trade bookkeeping mirrors Pine's own ``ml.backtest`` block order and
    scoring exactly (see ``application/backtest.py``'s module docstring): a
    new entry first abandons -- without scoring -- whatever opposite-side
    position was still open, then that side's own exit (if it fires the same
    bar) is applied. Entry/exit price uses Pine's ``(high+low+open+open)/4``
    scoring convention, not the close, so live trades stay consistent with
    the historical backtest.

    BUY entries additionally attempt to open a real paper-trading position
    (see ``application/paper_trading.py``) -- gated on the symbol's own
    BUY-only win-rate track record and on the account having free capital.
    SELL signals never touch the paper account: NSE cash market doesn't
    allow short selling for multi-day holds, so they stay informational only.

    When ``derivatives_chain`` is available (Kite active), every entry/exit
    also shadow-tracks (analysis only, never a real order) a directional
    option and a futures+hedge position -- see ``_open_derivatives_shadow``.

    When ``futures_account_repository`` is given AND ``symbol`` is in
    ``futures_paper_symbols`` (Nifty50 by default -- see ``AppConfig.
    futures_paper_symbols_file``), entries/exits *additionally* attempt a
    real, capital-gated futures paper position on BOTH sides (unlike the
    cash paper account, futures can short) -- see
    ``application/futures_trading.py``'s own 55%-win-rate eligibility bar
    and live-margin sizing. Entirely separate book/capital from the cash
    paper account and from the always-on, uncapped derivatives shadow
    tracking above.

    ``precomputed_evaluation``, when given, skips calling ``_evaluate_symbol``
    (the download-based path) entirely and uses this instead -- the live-
    ticker path (``live_pipeline.py``) already has a freshly-closed candle
    from aggregated ticks and has already called
    ``_evaluate_from_stored_candles`` itself; this lets it reuse every bit of
    the trade/paper/derivatives/notification logic below unchanged, rather
    than duplicating it.

    ``precomputed_paper_note``, when given, is used as-is for the BUY
    paper-position outcome instead of calling ``_open_paper_position`` --
    the caller (``_collect_and_open_ranked_positions``) already decided
    (and opened, if applicable) this cycle's paper positions in ranked
    order before calling this function, so it must not redecide/reopen
    here. ``precomputed_futures_note`` is the same idea for the futures
    paper account -- ``is None`` (not given) is what tells this function
    whether to fall back to deciding it itself (``_open_futures_paper``,
    unranked -- only used when a caller doesn't pre-rank a whole cycle at
    once) or trust what's already been decided.

    ``precomputed_cash_note``, same idea again, for the real cash order --
    when given, the entire eligibility/quality-filter/conviction-filter/
    execute_cash_entry/GTT/paper-benchmark sequence below is skipped
    entirely (``_collect_and_open_ranked_positions`` already ran it, in
    ranked order, before calling this function) and the note is used as-is.
    ``None`` falls back to deciding and executing it right here, unranked --
    same "only used when a caller doesn't pre-rank a whole cycle at once"
    situation as the other two, dead in production today since both real
    call sites always pre-rank first.
    """
    if precomputed_evaluation is not None:
        evaluated = precomputed_evaluation
    else:
        # Dead in production (see this docstring above) -- only the
        # current bar is used here even if a gap left stale catch-up
        # entries too, since this fallback path has no ranking/notifier
        # wiring around it to handle them the way the real call sites do.
        evaluations = await _evaluate_symbol(
            symbol, config, provider, engine, candle_repository, engine_state_repository
        )
        evaluated = evaluations[-1] if evaluations else None
    if evaluated is None:
        return
    result, newest_candle = evaluated
    market_price = _market_price(newest_candle)
    paper_note = None
    derivatives_note = None
    futures_note = None
    cash_note = None

    if gate_status_repository is not None:
        await _record_gate_status(
            symbol, config, result, newest_candle, trade_repository, gate_status_repository
        )

    if order_executor is not None and gtt_repository is not None and cash_state is not None:
        try:
            await gtt_bracket.check_and_extend(
                symbol, market_price, cash_state, order_executor, gtt_repository, notifier,
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "GTT extension check raised for %s -- original bracket (if any) left in place.",
                symbol,
            )

    if result.signal == "BUY":
        await trade_repository.abandon_open_trade(symbol, config.candle_interval, SignalSide.SELL)
        await trade_repository.open_trade(
            config.candle_interval,
            Trade(
                symbol=symbol,
                side=SignalSide.BUY,
                entry_timestamp=newest_candle.timestamp,
                entry_price=market_price,
                prediction_at_entry=result.prediction,
                is_early_signal_flip=result.is_early_signal_flip,
                adx_at_entry=result.adx,
                regime_normalized_at_entry=result.regime_normalized,
                volatility_margin_at_entry=result.volatility_margin,
            ),
        )
        paper_note = (
            precomputed_paper_note
            if precomputed_paper_note is not None
            else await _open_paper_position(
                symbol, config, newest_candle.timestamp, market_price,
                trade_repository, paper_account_repository, paper_account_lock,
            )
        )
        derivatives_note = await _open_derivatives_shadow(
            symbol, SignalSide.BUY, newest_candle.timestamp, market_price,
            derivatives_chain, options_trade_repository, futures_trade_repository,
            config, order_executor, live_order_repository, notifier,
        )
        if precomputed_cash_note is not None:
            # _collect_and_open_ranked_positions already ran the whole
            # eligibility/quality-filter/conviction-filter/execute_cash_
            # entry/GTT/paper-benchmark sequence below, in ranked order,
            # for this cycle -- trust its outcome instead of redeciding.
            cash_note = precomputed_cash_note
        elif (
            order_executor is not None
            and live_order_repository is not None
            and cash_state is not None
        ):
            try:
                # 2026-08-25: real cash entries now clear the same 55%-win-
                # rate/>=5-closed-trades bar the old paper simulator used
                # (paper_trading.is_eligible) -- previously execute_cash_entry
                # fired on a bare AlphaEngine BUY signal regardless of this
                # symbol's own track record, unlike every other capital-gated
                # book (paper account, futures paper account) which already
                # required it. Does not affect exits -- squaring off an
                # already-open real position is never gated on eligibility.
                # Also gated on entry_quality_filter (walk-forward-tested
                # indicator floor) and conviction_filter (entry-candle
                # close-location-value >= 0.7, see that module's own
                # docstring for the evidence behind both).
                track_record_gate = await entry_gates.evaluate_track_record_gate(
                    symbol, config.candle_interval, trade_repository
                )
                quality_decision = entry_gates.evaluate_cash_quality_gates(
                    result.volatility_margin, result.regime_normalized,
                    newest_candle.high, newest_candle.low, newest_candle.close,
                )
                cash_eligible = track_record_gate.passed and quality_decision.allowed
                if cash_eligible:
                    # 2026-08-25: serialize real entries -- execute_cash_entry's
                    # own max_positions check reads the current open count and
                    # decides in two separate steps; without a lock, several
                    # symbols signaling BUY in the same cycle could all read
                    # the same pre-entry count and all pass concurrently,
                    # together opening more real positions than intended
                    # (observed live: 8 opened against a configured cap of 8,
                    # only safe by coincidence). Falls back to a fresh,
                    # call-scoped lock if none was threaded through -- never
                    # crashes, just doesn't serialize across other calls.
                    # (Only needed on this unranked fallback path -- the
                    # ranked path, _rank_and_open_cash_positions, attempts
                    # entries sequentially by construction, so no race is
                    # possible there.)
                    async with (live_cash_lock or asyncio.Lock()):
                        await live_cash_execution.execute_cash_entry(
                            symbol, market_price, config, cash_state, order_executor,
                            live_order_repository, notifier,
                            signal_timestamp=newest_candle.timestamp,
                        )
                    cash_note = await _finalize_cash_entry(
                        symbol, market_price, cash_state, live_order_repository, notifier,
                        gtt_repository, paper_benchmark_repository, order_executor,
                    )
            except Exception:
                logging.getLogger(__name__).exception(
                    "Live cash order entry raised for %s -- rest of signal handling still stands.",
                    symbol,
                )
        futures_note = (
            precomputed_futures_note
            if precomputed_futures_note is not None
            else await _open_futures_paper(
                symbol, SignalSide.BUY, newest_candle.timestamp, market_price,
                config.candle_interval, trade_repository, derivatives_chain,
                futures_account_repository, futures_paper_symbols,
            )
        )
    if result.end_long:
        # Notification failures (e.g. a Telegram network timeout) must never
        # block actually recording the close -- this was a real bug: a
        # ConnectTimeout here used to propagate up through _process_symbol's
        # caller and skip close_open_trade entirely, leaving the position
        # stuck open in the database even though the strategy had already
        # decided to exit. _notify_exit still has to run *before* the close
        # below (it reads the still-open entry price), just no longer
        # allowed to prevent the close from happening.
        try:
            await _notify_exit(
                symbol, config, SignalSide.BUY, newest_candle.timestamp, market_price,
                trade_repository, signal_repository, notifier,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "Exit notification failed for %s (BUY) -- closing the trade anyway.",
                symbol, exc_info=True,
            )
        await trade_repository.close_open_trade(
            symbol, config.candle_interval, SignalSide.BUY, newest_candle.timestamp, market_price
        )
        await _close_paper_position(
            symbol, newest_candle.timestamp, market_price,
            paper_account_repository, signal_repository, notifier, paper_account_lock,
        )
        await _close_derivatives_shadow(
            symbol, SignalSide.BUY, newest_candle.timestamp, market_price,
            derivatives_chain, options_trade_repository, futures_trade_repository,
            config, order_executor, live_order_repository, notifier,
        )
        if order_executor is not None and live_order_repository is not None:
            try:
                # broker_reconciliation.get_unclosed_entry_leg, not get_
                # open_cash_legs directly -- the latter is COMPLETE-only
                # and would leave entry_leg None (skipping the
                # reconcile_before_exit ground-truth check entirely) for a
                # real entry stuck at status=UNKNOWN. See that module's
                # own docstring for the incident this fixes.
                entry_leg = await broker_reconciliation.get_unclosed_entry_leg(
                    symbol, live_order_repository
                )
                should_exit = True
                if gtt_repository is not None and entry_leg is not None:
                    should_exit = await gtt_bracket.reconcile_before_exit(
                        symbol, entry_leg.tradingsymbol, config, order_executor, gtt_repository,
                    )
                closed_basket_id = None
                if should_exit:
                    closed_basket_id = await live_cash_execution.execute_cash_exit(
                        symbol, market_price, config, order_executor, live_order_repository,
                        notifier,
                    )
                elif entry_leg is not None:
                    # 2026-08-28: reconcile_before_exit determined the real
                    # position is already flat -- close it in our own
                    # ledger too. Previously this branch did nothing, which
                    # is exactly how COCHINSHIP.NS got stuck "open" for
                    # days with no way to exit or re-enter it; see
                    # live_cash_execution.record_broker_side_exit.
                    closed_basket_id = await live_cash_execution.record_broker_side_exit(
                        symbol, entry_leg, market_price, live_order_repository, notifier,
                    )
                if (
                    paper_benchmark_repository is not None
                    and closed_basket_id is not None
                    and entry_leg is not None
                ):
                    closed_legs = await live_order_repository.get_legs(closed_basket_id)
                    if closed_legs and closed_legs[0].status == "COMPLETE":
                        try:
                            await paper_benchmark.record_exit(
                                symbol, entry_leg.basket_id, market_price,
                                newest_candle.timestamp, closed_legs[0],
                                paper_benchmark_repository,
                            )
                        except Exception:
                            logging.getLogger(__name__).exception(
                                "Paper-benchmark exit recording raised for %s -- "
                                "real trade unaffected.", symbol,
                            )
            except Exception:
                logging.getLogger(__name__).exception(
                    "Live cash order exit raised for %s -- close above still stands.", symbol,
                )
        await _close_futures_paper(
            symbol, SignalSide.BUY, newest_candle.timestamp, market_price, derivatives_chain,
            futures_account_repository, futures_paper_symbols, signal_repository, notifier,
        )
    if result.signal == "SELL":
        await trade_repository.abandon_open_trade(symbol, config.candle_interval, SignalSide.BUY)
        await trade_repository.open_trade(
            config.candle_interval,
            Trade(
                symbol=symbol,
                side=SignalSide.SELL,
                entry_timestamp=newest_candle.timestamp,
                entry_price=market_price,
                prediction_at_entry=result.prediction,
                is_early_signal_flip=result.is_early_signal_flip,
                adx_at_entry=result.adx,
                regime_normalized_at_entry=result.regime_normalized,
                volatility_margin_at_entry=result.volatility_margin,
            ),
        )
        derivatives_note = await _open_derivatives_shadow(
            symbol, SignalSide.SELL, newest_candle.timestamp, market_price,
            derivatives_chain, options_trade_repository, futures_trade_repository,
            config, order_executor, live_order_repository, notifier,
        )
        futures_note = (
            precomputed_futures_note
            if precomputed_futures_note is not None
            else await _open_futures_paper(
                symbol, SignalSide.SELL, newest_candle.timestamp, market_price,
                config.candle_interval, trade_repository, derivatives_chain,
                futures_account_repository, futures_paper_symbols,
            )
        )
    if result.end_short:
        # See the end_long branch above for why this is wrapped -- a
        # notification failure must never block recording the actual close.
        try:
            await _notify_exit(
                symbol, config, SignalSide.SELL, newest_candle.timestamp, market_price,
                trade_repository, signal_repository, notifier,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "Exit notification failed for %s (SELL) -- closing the trade anyway.",
                symbol, exc_info=True,
            )
        await trade_repository.close_open_trade(
            symbol, config.candle_interval, SignalSide.SELL, newest_candle.timestamp, market_price
        )
        await _close_derivatives_shadow(
            symbol, SignalSide.SELL, newest_candle.timestamp, market_price,
            derivatives_chain, options_trade_repository, futures_trade_repository,
            config, order_executor, live_order_repository, notifier,
        )
        await _close_futures_paper(
            symbol, SignalSide.SELL, newest_candle.timestamp, market_price, derivatives_chain,
            futures_account_repository, futures_paper_symbols, signal_repository, notifier,
        )

    if result.signal not in ("BUY", "SELL"):
        return

    side = SignalSide.BUY if result.signal == "BUY" else SignalSide.SELL
    rationale = f"prediction={result.prediction}"
    win_rate_summary = await _win_rate_summary(symbol, config, trade_repository)
    if win_rate_summary is not None:
        rationale += f"; {win_rate_summary}"
    if side == SignalSide.SELL:
        rationale += "; informational only -- not tradeable in NSE cash market"
    elif paper_note is not None:
        rationale += f"; {paper_note}"
    if cash_note is not None:
        rationale += f"; {cash_note}"
    # Unlike paper_note (cash, BUY-only), futures_note applies to both
    # sides -- futures are the real short mechanism, see futures_trading.py.
    if futures_note is not None:
        rationale += f"; {futures_note}"
    if derivatives_note is not None:
        rationale += f"; {derivatives_note}"
    if index_result is not None:
        rationale += (
            f"; index({config.index_symbol})={index_result.signal},"
            f"pred={index_result.prediction},early_flip={index_result.is_early_signal_flip}"
        )
    signal = Signal(
        symbol=symbol,
        side=side,
        strategy=_STRATEGY_NAME,
        timestamp=newest_candle.timestamp,
        price=newest_candle.close,
        rationale=rationale,
        category="entry",
    )
    if await signal_repository.contains(signal.fingerprint):
        return
    # 2026-08-21: entry-signal Telegram notifications turned off entirely --
    # follow only real cash-market order events (see live_cash_execution.py)
    # from here on; everything else (this informational strategy signal,
    # paper/futures-paper/derivatives-shadow tracking) is analysis-only and
    # was flooding Telegram once paper trading's own suppression logic
    # stopped covering it. Still recorded for the dashboard's signal history.
    await signal_repository.record(signal.fingerprint, signal.timestamp)

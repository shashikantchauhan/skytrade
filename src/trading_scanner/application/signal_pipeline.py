"""Hourly pipeline: accumulate candles per symbol and notify on AlphaEngine signals."""

import asyncio
import logging
from datetime import UTC, date, datetime
from decimal import Decimal

import pandas as pd
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException as KiteTokenException

from trading_scanner.alpha_engine import AlphaEngine
from trading_scanner.application import (
    entry_gates,
    futures_shadow,
    futures_trading,
    gtt_bracket,
    live_cash_execution,
    live_execution,
    options_shadow,
    paper_benchmark,
    paper_trading,
)
from trading_scanner.application.fast_predict import (
    ExitState,
    FastPredictResult,
    QueueState,
    bootstrap_queue_state,
    evaluate_latest_bar,
)
from trading_scanner.application.ranking import (
    MIN_SCORE,
    RankedCandidate,
    rank_candidates,
    score_candidate,
    symbol_expectancy,
)
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import Candle, Signal, SignalSide, Trade
from trading_scanner.domain.ports import (
    CandleRepository,
    EngineState,
    EngineStateRepository,
    FuturesPaperAccountRepository,
    Notifier,
    PaperAccountRepository,
    SignalRepository,
    TradeRepository,
)
from trading_scanner.infrastructure.db import (
    LiveCashToggleState,
    TursoFuturesTradeRepository,
    TursoGttRepository,
    TursoKiteSessionRepository,
    TursoLiveOrderRepository,
    TursoOptionsTradeRepository,
    TursoPaperBenchmarkRepository,
)
from trading_scanner.infrastructure.kite import (
    KiteDerivativesChain,
    KiteInstrumentMap,
    KiteOrderExecutor,
    KiteProvider,
)
from trading_scanner.infrastructure.telegram import LoggingNotifier
from trading_scanner.infrastructure.yahoo import YahooProvider

# Both providers implement the same duck-typed get_recent_history(symbol,
# interval, days) -> pd.DataFrame interface; nothing below needs to know
# which one it got.
MarketDataProvider = YahooProvider | KiteProvider

# AlphaEngine's regime filter needs ~200 bars of warm-up before predictions are
# meaningful; below this the pipeline skips a symbol instead of notifying on
# noise. This mirrors application.validation.validate_candles's own minimum.
_MINIMUM_CANDLES = 200

# Small recent window fetched every run once a symbol already has history --
# just enough to cover any gap since the last hourly run.
_RECENT_WINDOW_DAYS = 5

# Yahoo Finance caps 1h/60m intraday history at 730 days regardless of what is
# requested; 729 is the largest window that reliably stays under that cap. A
# brand-new symbol is backfilled with this once, so AlphaEngine analyzes
# roughly the same depth of history TradingView's chart typically has, rather
# than waiting months for the small recent window to accumulate that far.
_BACKFILL_WINDOW_DAYS = 729

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

# evaluate_latest_bar's neighbor search always looks at indices [0, max_bars_back-1]
# of whatever candle history is passed in -- i.e. the *oldest* stored candles.
# Fetching anything less than the full accumulated history (e.g. a "most recent
# N" cap) would drop those oldest rows and shift what index 0 means, silently
# breaking the TradingView match this pipeline exists to preserve. See
# fast_predict.py's module docstring for the full explanation.
_FULL_HISTORY = None

_STRATEGY_NAME = "lorentzian"

# The hedge option leg (Structure B, see _open_derivatives_shadow) targets a
# strike this far OTM instead of ATM -- confirmed against Kite's own
# margin-benefit numbers that ATM cancels out most of the primary position's
# profit (delta near -1/+1) for only a modest extra margin benefit over a
# further-OTM strike. See try_open_option_position's strike_target_price
# docstring for the full reasoning.
_HEDGE_OTM_PCT = Decimal("0.05")

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
            index_result = index_evaluated[0] if index_evaluated is not None else None
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
                    await _notify_kite_expired_once_per_day(kite_session_repository, notifier)

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
    evaluated_by_symbol: dict[str, tuple[FastPredictResult, Candle] | None] = {}

    async def _evaluate_with_limit(symbol: str) -> None:
        async with semaphore:
            try:
                evaluated_by_symbol[symbol] = await _evaluate_symbol(
                    symbol, config, provider, engine, candle_repository, engine_state_repository
                )
            except Exception as error:
                logger.exception("Unexpected exception while evaluating %s", symbol)
                evaluated_by_symbol[symbol] = None
                await _notify_mid_run_kite_expiry_once(error)

    await asyncio.gather(*(_evaluate_with_limit(symbol) for symbol in symbols))

    paper_notes, futures_notes, cash_notes = await _collect_and_open_ranked_positions(
        evaluated_by_symbol, config, trade_repository, paper_account_repository,
        paper_account_lock, derivatives_chain, futures_account_repository, futures_paper_symbols,
        notifier, order_executor, live_order_repository, gtt_repository, paper_benchmark_repository,
        cash_state,
    )

    async def _process_with_limit(symbol: str) -> None:
        evaluated = evaluated_by_symbol.get(symbol)
        if evaluated is None:
            return  # Nothing new for this symbol this cycle -- same as before.
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
                )
            except Exception as error:
                logger.exception("Unexpected exception while processing %s", symbol)
                await _notify_mid_run_kite_expiry_once(error)

    await asyncio.gather(*(_process_with_limit(symbol) for symbol in symbols))


def _is_kite_token_error(error: BaseException) -> bool:
    """Walks the exception chain -- ``KiteProvider.get_recent_history``
    wraps the underlying ``TokenException`` in a ``RuntimeError`` (see
    ``infrastructure/kite.py``), so a plain ``isinstance`` check on the
    caught exception alone would miss it."""
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, KiteTokenException):
            return True
        current = current.__cause__ or current.__context__
    return False


class NoValidKiteSession(RuntimeError):
    """Raised by ``_select_provider`` when no usable Kite session exists.

    There is deliberately no Yahoo fallback here anymore (see this
    function's own docstring) -- the caller's job is to notify and skip
    this cycle cleanly, not to substitute a different data source.
    """


async def _select_provider(
    config: AppConfig,
    kite_session_repository: TursoKiteSessionRepository | None,
    notifier: Notifier | None = None,
) -> tuple[MarketDataProvider, str, KiteConnect | None]:
    """Require a valid Kite session; never silently substitute Yahoo.

    Used to fall back to Yahoo Finance when no Kite session was available.
    Removed 2026-08-13: Yahoo's ``yf.download`` has a documented failure
    mode under concurrent request load where a response silently belongs
    to the *wrong* ticker entirely (not just a malformed/partial one --
    see ``infrastructure/yahoo.py``'s ``_clean_data`` docstring, which
    already guarded the partial case). Root-caused to exactly this: 176
    symbols' candles were found byte-for-byte identical across unrelated
    stocks over a ~2-week window, matching runs where Kite's daily token
    hadn't been refreshed yet and this fallback silently kicked in. Kite
    Connect's Historical Data API is a paid, official data source with no
    such issue -- there is no good reason to ever substitute a free,
    unofficial scrape for real trading data again. A stale/missing Kite
    session now means this cycle is skipped entirely (see
    ``NoValidKiteSession``) rather than degrading to a worse, riskier data
    source.

    "Valid" is checked by actually calling Kite's instruments endpoint and
    confirming the hardcoded index mapping still resolves (see
    ``KiteInstrumentMap.validate_index_mapping``) rather than trusting a
    stored token blindly -- catches both an expired/revoked token and a
    stale index mapping in one check.

    Also sends one "please log in again" notification per calendar day
    (deduped via ``kite_session_repository``'s ``expiry_notified_date``,
    see ``TursoKiteSessionRepository``) -- Kite tokens expire daily with
    no documented exact time, so this piggybacks on the pipeline's own
    hourly cron rather than needing separate infrastructure to detect
    expiry.
    """
    logger = logging.getLogger(__name__)
    if config.kite_api_key and kite_session_repository is not None:
        token_row = await kite_session_repository.get_token()
        if token_row is not None:
            access_token, obtained_at = token_row
            try:
                kite = KiteConnect(api_key=config.kite_api_key)
                kite.set_access_token(access_token)
                instrument_map = KiteInstrumentMap(kite)
                await asyncio.to_thread(instrument_map.validate_index_mapping)
                return KiteProvider(kite, instrument_map), "kite", kite
            except Exception:
                logger.warning(
                    "Kite session unusable (obtained_at=%s); skipping this run.",
                    obtained_at,
                    exc_info=True,
                )
        await _notify_kite_expired_once_per_day(kite_session_repository, notifier)
    elif kite_session_repository is not None:
        await _notify_kite_expired_once_per_day(kite_session_repository, notifier)
    raise NoValidKiteSession("No valid Kite session -- skipping this run rather than using Yahoo.")


async def _notify_kite_expired_once_per_day(
    kite_session_repository: TursoKiteSessionRepository, notifier: Notifier | None
) -> None:
    if notifier is None:
        return
    today = date.today().isoformat()
    try:
        last_notified = await kite_session_repository.get_expiry_notified_date()
        if last_notified == today:
            return
        await notifier.send_text(
            "⚠️ <b>SYSTEM ALERT</b>\n"
            "Kite session expired/missing -- this run is being skipped (no Yahoo fallback).\n"
            "Log in again: https://skytrade.oneatem.com/kite/login"
        )
        await kite_session_repository.set_expiry_notified_date(today)
    except Exception:
        logging.getLogger(__name__).exception("Failed to send Kite-expiry notification")


async def _evaluate_symbol(
    symbol: str,
    config: AppConfig,
    provider: MarketDataProvider,
    engine: AlphaEngine,
    candle_repository: CandleRepository,
    engine_state_repository: EngineStateRepository,
) -> tuple[FastPredictResult, Candle] | None:
    """Download, store, and evaluate the newest bar for one symbol -- the
    polling/historical-API path (cron, manual dashboard trigger, backfill).

    Returns None if the symbol is still warming up (<200 candles) or if no
    new candle has arrived since the last run (nothing new to evaluate).

    Zerodha's own guidance is that the Historical Data API is for backfill/
    backtesting, not live signals (it can lag the current session's candles
    by hours -- confirmed live and via Kite's dev forum). The live-ticker
    path (``infrastructure/kite_ticker.py`` -> ``live_pipeline.py``) is the
    real-time replacement for market hours; this function's download step
    stays around for backfill, catch-up after downtime, and the dashboard's
    manual "run pipeline now" button.
    """
    logger = logging.getLogger(__name__)
    existing = await candle_repository.get_candles(
        symbol, config.candle_interval, limit=_MINIMUM_CANDLES
    )
    needs_backfill = len(existing) < _MINIMUM_CANDLES
    window_days = _BACKFILL_WINDOW_DAYS if needs_backfill else _RECENT_WINDOW_DAYS
    if needs_backfill:
        logger.info("Backfilling %s: downloading %d days of history.", symbol, window_days)

    # get_recent_history is a blocking call (yfinance is synchronous); running
    # it in a thread lets other symbols' downloads proceed concurrently on
    # the event loop instead of serializing behind this one.
    downloaded = await asyncio.to_thread(
        provider.get_recent_history, symbol, config.candle_interval, window_days
    )
    await candle_repository.upsert_candles(
        symbol, config.candle_interval, _dataframe_to_candles(symbol, downloaded)
    )
    return await _evaluate_from_stored_candles(
        symbol, config, engine, candle_repository, engine_state_repository
    )


async def _evaluate_from_stored_candles(
    symbol: str,
    config: AppConfig,
    engine: AlphaEngine,
    candle_repository: CandleRepository,
    engine_state_repository: EngineStateRepository,
) -> tuple[FastPredictResult, Candle] | None:
    """Evaluate the newest already-*stored* bar for one symbol -- shared by
    both the download-based path above (which upserts before calling this)
    and the live-ticker path (``live_pipeline.py``, which upserts one
    freshly-closed candle from aggregated ticks before calling this).

    Returns None if the symbol is still warming up (<200 candles) or if no
    new candle has arrived since the last run (nothing new to evaluate).
    """
    logger = logging.getLogger(__name__)
    accumulated = await candle_repository.get_candles(
        symbol, config.candle_interval, limit=_FULL_HISTORY
    )
    if len(accumulated) < _MINIMUM_CANDLES:
        logger.info(
            "Warming up %s: %d/%d candles stored.", symbol, len(accumulated), _MINIMUM_CANDLES
        )
        return None

    engine_state = await engine_state_repository.get_state(symbol, config.candle_interval)
    newest_timestamp = accumulated[-1].timestamp.isoformat()
    if newest_timestamp == engine_state.last_bar_timestamp:
        # The newest stored candle is the same bar already advanced into the
        # queue last run (no new candle has closed since) -- advancing it
        # again would double-count its contribution to the persisted
        # neighbor queue, corrupting future predictions. Nothing changed;
        # nothing to do.
        logger.info("No new candle for %s since the last run; skipping.", symbol)
        return None

    history = _candles_to_dataframe(accumulated)
    if engine_state.queue_json is None:
        logger.info("Bootstrapping ANN neighbor queue for %s (one-time, slower).", symbol)
        bootstrap = bootstrap_queue_state(engine, history.iloc[:-1])
        signal_previous = bootstrap.signal_previous
        queue_state = bootstrap.queue_state
        exit_state = bootstrap.exit_state
    else:
        signal_previous = engine_state.signal
        queue_state = QueueState.from_json(engine_state.queue_json)
        exit_state = ExitState.from_json(engine_state.exit_state_json)

    result = evaluate_latest_bar(engine, history, signal_previous, queue_state, exit_state)
    await engine_state_repository.set_state(
        symbol,
        config.candle_interval,
        EngineState(
            signal=result.signal_previous,
            queue_json=result.queue_state.to_json(),
            exit_state_json=result.exit_state.to_json(),
            last_bar_timestamp=newest_timestamp,
        ),
    )
    return result, accumulated[-1]


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
        evaluated = await _evaluate_symbol(
            symbol, config, provider, engine, candle_repository, engine_state_repository
        )
    if evaluated is None:
        return
    result, newest_candle = evaluated
    market_price = _market_price(newest_candle)
    paper_note = None
    derivatives_note = None
    futures_note = None
    cash_note = None

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
                open_legs = await live_order_repository.get_open_cash_legs(symbol)
                entry_leg = open_legs[0] if open_legs else None
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
    all_open = await live_order_repository.get_all_open_cash_legs()
    if len(all_open) >= cash_state.max_positions:
        reason = f"all {cash_state.max_positions} real slots are already full"
    else:
        reason = "past the entry cutoff or the order didn't go through -- check logs"
    await notifier.send_text(
        "\U0001f6ab <b>MISSED BUY SIGNAL</b>\n"
        f"{symbol} @ {market_price} cleared eligibility, the quality filter, and the "
        f"conviction filter, but {reason}."
    )


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
    ``_finalize_cash_entry``'s empty-``get_open_cash_legs`` case) decide.
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
            break
        try:
            await live_cash_execution.execute_cash_entry(
                symbol, candidate.entry_price, config, cash_state, order_executor,
                live_order_repository, notifier,
            )
            notes[symbol] = await _finalize_cash_entry(
                symbol, candidate.entry_price, cash_state, live_order_repository, notifier,
                gtt_repository, paper_benchmark_repository, order_executor,
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "Live cash order entry raised for %s -- rest of signal handling still stands.",
                symbol,
            )
            notes[symbol] = "cash: ERROR placing order (see logs)"
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
            else:
                paper_notes[symbol] = f"paper: {track_record_gate.reason}"
                if cash_enabled:
                    cash_notes[symbol] = f"cash: {track_record_gate.reason}"

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
                notifier, gtt_repository, paper_benchmark_repository,
            )
        )
    return paper_notes, futures_notes, cash_notes


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


def _market_price(candle: Candle) -> Decimal:
    """Pine's ``ml.backtest`` scoring price: (high + low + open + open) / 4.

    Not the close -- matches ``application/backtest.py``'s historical
    replay so live and backtested trades use the same price convention.
    """
    return (candle.high + candle.low + candle.open + candle.open) / 4


def _dataframe_to_candles(symbol: str, data: pd.DataFrame) -> list[Candle]:
    """Convert a downloaded OHLCV DataFrame into domain Candle objects.

    Normalized to UTC here -- Yahoo returns NSE timestamps tz-aware in IST,
    and storing that offset as-is produces an ISO string ("+05:30") that
    sorts incorrectly against UTC-stored rows ("+00:00") under a plain text
    ORDER BY, scrambling chronological order at every day boundary. Always
    normalizing to UTC before it ever reaches the repository keeps every
    row's timestamp column in the same offset, so text sort order matches
    real chronological order.
    """
    return [
        Candle(
            symbol=symbol,
            timestamp=timestamp.to_pydatetime().astimezone(UTC),
            open=Decimal(str(row["Open"])),
            high=Decimal(str(row["High"])),
            low=Decimal(str(row["Low"])),
            close=Decimal(str(row["Close"])),
            volume=int(row["Volume"]),
        )
        for timestamp, row in data.iterrows()
    ]


def _candles_to_dataframe(candles) -> pd.DataFrame:
    """Convert chronological Candle objects into the OHLCV DataFrame AlphaEngine expects.

    Normalizes every timestamp to UTC before building the index -- candles
    stored across different runs can carry equivalent-offset but distinct
    tzinfo objects (e.g. a fixed +05:30 offset vs. a zoneinfo-based one),
    which pandas refuses to unify into one DatetimeIndex without this
    (raises "Tz-aware datetime.datetime cannot be converted to datetime64
    unless utc=True"). AlphaEngine only depends on chronological order and
    OHLCV values, never the displayed hour, so this is safe -- the original
    Candle objects (with their real tzinfo) are still used everywhere else.
    """
    return pd.DataFrame(
        {
            "Open": [float(candle.open) for candle in candles],
            "High": [float(candle.high) for candle in candles],
            "Low": [float(candle.low) for candle in candles],
            "Close": [float(candle.close) for candle in candles],
            "Volume": [candle.volume for candle in candles],
        },
        index=pd.DatetimeIndex([candle.timestamp.astimezone(UTC) for candle in candles]),
    )

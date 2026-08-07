"""Evaluate only the newest bar's BUY/SELL signal, instead of AlphaEngine's
full-history batch ``analyze()``.

Why this exists: with ``include_full_history=True`` (this deployment's real
TradingView chart setting -- see ``signal_pipeline._ENGINE_SETTINGS``),
``AlphaEngine._predict``'s ANN neighbor search runs once for *every* one of
the ~2,000 most-recent bars on every call, because that method always
computes predictions for the whole history. The hourly pipeline only ever
acts on the newest bar's result, so re-deriving ~2,000 bars' worth of
predictions every run (measured at ~43s/symbol on real data) to read the
last one is pure waste.

The one subtlety that makes this non-trivial: in ``_predict``, the
``distances``/``predictions`` neighbor queue is declared *before* the
``for bar in range(length)`` loop -- it is a single persistent queue that
accumulates and evicts across *every* bar, not a fresh search per bar. So a
given bar's prediction can (occasionally) still include a handful of
neighbors admitted while processing earlier bars. Reproducing this exactly
means the queue's state must itself be carried forward between hourly runs
(``QueueState`` below), the same way ``signal_previous`` already is. The
very first time a symbol is evaluated there is no prior queue to carry, so
one one-time (expensive) bootstrap replay builds it from
``max_bars_back_index`` up through the bar *before* the newest one; every
run after that only pays for the newest bar's own O(max_bars_back) scan.

``start_long``/``start_short`` (the only fields the pipeline notifies on)
depend only on the *current* bar's ``signal``, trend filters, and kernel
state -- never on ``bars_held`` or the dynamic-exit ``bars_since_*``
tracking (that machinery only feeds ``end_long``/``end_short``). And
``signal[bar]`` only needs ``signal[bar-1]`` as a carry-forward value, not a
full history of past signal states.

This is not an approximation: the formulas are copied term-for-term from
``AlphaEngine._predict``/``_entries_and_exits`` for entry (BUY/SELL) signals
specifically. ``tests/test_fast_predict.py`` verifies this against the full
batch ``analyze()`` directly -- a single call, and a simulated sequence of
consecutive hourly runs -- and is what caught the queue-persistence subtlety
above during development.

Critical invariant: the neighbor search always looks at indices
``[0, max_bars_back-1]`` of *whatever array is passed in* -- i.e. the
*oldest* stored candles, not a sliding recent window. This only stays
correct if the stored candle history's starting point never shifts forward
(no truncating from the front). Callers must fetch the full accumulated
history (``limit=None``), never a "most recent N" window.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_scanner.alpha_engine import (
    AlphaEngine,
    _bars_since,
    _ema,
    _finite_compare,
    _lorentzian_distance,
    _sma,
)


@dataclass(frozen=True, slots=True)
class QueueState:
    """The ANN neighbor queue's state -- small (bounded near neighbors_count),
    persisted between hourly runs so each run only replays the newest bar's
    own scan instead of the entire history."""

    distances: tuple[float, ...] = ()
    predictions: tuple[int, ...] = ()

    def to_json(self) -> str:
        """Serialize for persistence (e.g. EngineState.queue_json)."""
        return json.dumps(
            {"distances": list(self.distances), "predictions": list(self.predictions)}
        )

    @staticmethod
    def from_json(queue_json: str | None) -> "QueueState":
        """Deserialize a persisted state, or return an empty queue if None
        (a symbol never evaluated before -- caller should bootstrap first)."""
        if queue_json is None:
            return QueueState()
        data = json.loads(queue_json)
        return QueueState(
            distances=tuple(data["distances"]), predictions=tuple(data["predictions"])
        )


@dataclass(frozen=True, slots=True)
class ExitState:
    """State needed for ``end_long``/``end_short`` (dynamic exits) and
    ``is_early_signal_flip``, carried between hourly runs.

    Only ``start_long``/``start_short``-derived counts need persisting here:
    ``since_red_exit``/``since_green_exit`` (based on the kernel's own
    ``alert_bullish``/``alert_bearish``) are cheap to recompute fresh every
    call via ``_bars_since`` on the already-vectorized kernel arrays, so they
    are *not* persisted -- only ``bars_since_red_entry``/
    ``bars_since_green_entry`` (based on ``start_short``/``start_long``,
    which are not cheaply recomputable without the full ANN sweep this
    module exists to avoid) are.
    """

    bars_since_red_entry: int | None = None
    bars_since_green_entry: int | None = None
    changed_window: tuple[bool, bool, bool] = (False, False, False)

    def to_json(self) -> str:
        return json.dumps(
            {
                "bars_since_red_entry": self.bars_since_red_entry,
                "bars_since_green_entry": self.bars_since_green_entry,
                "changed_window": list(self.changed_window),
            }
        )

    @staticmethod
    def from_json(exit_state_json: str | None) -> "ExitState":
        if exit_state_json is None:
            return ExitState()
        data = json.loads(exit_state_json)
        return ExitState(
            bars_since_red_entry=data["bars_since_red_entry"],
            bars_since_green_entry=data["bars_since_green_entry"],
            changed_window=tuple(data["changed_window"]),
        )


@dataclass(frozen=True, slots=True)
class FastPredictResult:
    """The newest bar's entry/exit signals, and the state to persist for next run."""

    signal: str  # "BUY", "SELL", or "NEUTRAL" (entry)
    prediction: int
    end_long: bool
    end_short: bool
    is_early_signal_flip: bool
    signal_previous: int  # persist this as next run's `signal_previous` input
    queue_state: QueueState  # persist this as next run's `queue_state` input
    exit_state: ExitState  # persist this as next run's `exit_state` input


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """The state to seed a symbol's first ``evaluate_latest_bar`` call."""

    signal_previous: int
    queue_state: QueueState
    exit_state: ExitState


def bootstrap_queue_state(
    engine: AlphaEngine, history_before_newest_bar: pd.DataFrame
) -> BootstrapResult:
    """One-time expensive replay building both the neighbor queue *and* the
    raw signal state up through the bar *before* the newest one.

    Both must be bootstrapped, not just the queue: ``signal[bar]`` carries
    forward its previous value whenever ``filter_all[bar]`` is False or
    ``prediction[bar] == 0`` (see ``AlphaEngine._predict``), so the correct
    starting `signal_previous` can depend on bars arbitrarily far in the
    past -- it is not safe to assume 0 for a symbol with existing history.
    Call this only when a symbol has no persisted state yet; every run after
    that only needs ``evaluate_latest_bar``, which is cheap.
    """
    if history_before_newest_bar.empty:
        return BootstrapResult(signal_previous=0, queue_state=QueueState(), exit_state=ExitState())
    data = history_before_newest_bar.sort_index()
    open_ = data["Open"].to_numpy(dtype=float)
    high = data["High"].to_numpy(dtype=float)
    low = data["Low"].to_numpy(dtype=float)
    close = data["Close"].to_numpy(dtype=float)
    source = close.copy()
    length = len(source)

    features = engine._calculate_features(close, high, low)
    filter_all = engine._calculate_filters(open_, source, high, low, close)
    kernel = engine._kernel_regression(source)
    labels = _labels(source)
    max_bars_back_index = _max_bars_back_index(engine, length)
    since_red_exit_all = _bars_since(kernel["alert_bullish"])
    since_green_exit_all = _bars_since(kernel["alert_bearish"])

    ema = _ema(close, engine.ema_period)
    sma = _sma(close, engine.sma_period)
    ema_up_all = _finite_compare(close, ema, np.greater)
    ema_down_all = _finite_compare(close, ema, np.less)
    sma_up_all = _finite_compare(close, sma, np.greater)
    sma_down_all = _finite_compare(close, sma, np.less)

    queue_state = QueueState()
    exit_state = ExitState()
    signal_previous = 0
    for bar in range(max_bars_back_index, length):
        prediction, queue_state = _advance_queue(
            engine, features, labels, bar, max_bars_back_index, queue_state
        )
        signal_new = (
            1
            if prediction > 0 and filter_all[bar]
            else -1 if prediction < 0 and filter_all[bar] else signal_previous
        )
        changed = signal_new != signal_previous
        start_long, start_short = _entry_signals(
            engine,
            signal_new,
            changed,
            kernel,
            bar,
            ema_up_all[bar] if engine.use_ema_filter else True,
            sma_up_all[bar] if engine.use_sma_filter else True,
            ema_down_all[bar] if engine.use_ema_filter else True,
            sma_down_all[bar] if engine.use_sma_filter else True,
        )
        _, _, _, exit_state = _advance_exit_state(
            exit_state,
            start_long,
            start_short,
            changed,
            kernel,
            since_red_exit_all,
            since_green_exit_all,
            bar,
        )
        signal_previous = signal_new
    return BootstrapResult(
        signal_previous=signal_previous, queue_state=queue_state, exit_state=exit_state
    )


def evaluate_latest_bar(
    engine: AlphaEngine,
    history: pd.DataFrame,
    signal_previous: int,
    queue_state: QueueState,
    exit_state: ExitState,
) -> FastPredictResult:
    """Compute the newest bar's entry/exit signals by advancing the
    persisted neighbor queue and exit-tracking state with only the newest
    bar's own contribution.

    ``history`` must be chronological OHLCV data (same contract as
    ``AlphaEngine.analyze``). ``signal_previous``, ``queue_state``, and
    ``exit_state`` are all persisted from the previous call (use
    ``bootstrap_queue_state`` once per symbol before the first call).
    """
    data = history.sort_index()
    open_ = data["Open"].to_numpy(dtype=float)
    high = data["High"].to_numpy(dtype=float)
    low = data["Low"].to_numpy(dtype=float)
    close = data["Close"].to_numpy(dtype=float)
    source = close.copy()
    length = len(source)
    bar = length - 1

    features = engine._calculate_features(close, high, low)
    filter_all = engine._calculate_filters(open_, source, high, low, close)
    kernel = engine._kernel_regression(source)
    labels = _labels(source)
    since_red_exit_all = _bars_since(kernel["alert_bullish"])
    since_green_exit_all = _bars_since(kernel["alert_bearish"])

    max_bars_back_index = _max_bars_back_index(engine, length)
    prediction = 0
    new_queue_state = queue_state
    if bar >= max_bars_back_index:
        prediction, new_queue_state = _advance_queue(
            engine, features, labels, bar, max_bars_back_index, queue_state
        )

    signal_new = (
        1
        if prediction > 0 and filter_all[bar]
        else -1 if prediction < 0 and filter_all[bar] else signal_previous
    )
    changed = signal_new != signal_previous

    ema = _ema(close, engine.ema_period)
    sma = _sma(close, engine.sma_period)
    ema_up = _finite_compare(close, ema, np.greater)[bar] if engine.use_ema_filter else True
    ema_down = _finite_compare(close, ema, np.less)[bar] if engine.use_ema_filter else True
    sma_up = _finite_compare(close, sma, np.greater)[bar] if engine.use_sma_filter else True
    sma_down = _finite_compare(close, sma, np.less)[bar] if engine.use_sma_filter else True

    start_long, start_short = _entry_signals(
        engine, signal_new, changed, kernel, bar, ema_up, sma_up, ema_down, sma_down
    )
    end_long, end_short, is_early_signal_flip, new_exit_state = _advance_exit_state(
        exit_state,
        start_long,
        start_short,
        changed,
        kernel,
        since_red_exit_all,
        since_green_exit_all,
        bar,
    )

    signal = "BUY" if start_long else "SELL" if start_short else "NEUTRAL"
    return FastPredictResult(
        signal=signal,
        prediction=prediction,
        end_long=end_long,
        end_short=end_short,
        is_early_signal_flip=is_early_signal_flip,
        signal_previous=signal_new,
        queue_state=new_queue_state,
        exit_state=new_exit_state,
    )


def _entry_signals(
    engine: AlphaEngine,
    signal_new: int,
    changed: bool,
    kernel: dict[str, np.ndarray],
    bar: int,
    ema_up: bool,
    sma_up: bool,
    ema_down: bool,
    sma_down: bool,
) -> tuple[bool, bool]:
    """``start_long``/``start_short`` for one bar -- term-for-term identical
    to the corresponding slice of ``AlphaEngine._entries_and_exits``."""
    is_bullish = bool(kernel["bullish"][bar]) if engine.use_kernel_filter else True
    is_bearish = bool(kernel["bearish"][bar]) if engine.use_kernel_filter else True
    buy = signal_new == 1 and ema_up and sma_up
    sell = signal_new == -1 and ema_down and sma_down
    new_buy = buy and changed
    new_sell = sell and changed
    start_long = new_buy and is_bullish and ema_up and sma_up
    start_short = new_sell and is_bearish and ema_down and sma_down
    return start_long, start_short


def _advance_exit_state(
    state: ExitState,
    start_long: bool,
    start_short: bool,
    changed: bool,
    kernel: dict[str, np.ndarray],
    since_red_exit_all: np.ndarray,
    since_green_exit_all: np.ndarray,
    bar: int,
) -> tuple[bool, bool, bool, ExitState]:
    """``end_long``/``end_short``/``is_early_signal_flip`` for one bar --
    term-for-term identical to ``AlphaEngine._entries_and_exits``'s dynamic
    exit branch (this deployment's production config always takes that
    branch -- see ``signal_pipeline._ENGINE_SETTINGS``), continuing (not
    resetting) the persisted entry-side "bars since" counters and the
    3-bar `changed` window.
    """
    since_red_entry_prev = state.bars_since_red_entry
    since_green_entry_prev = state.bars_since_green_entry
    # `_shift_bool(valid_short_exit, 1)`/`_shift_bool(valid_long_exit, 1)` in
    # the original: use the *previous* bar's since-exit/since-entry counts,
    # i.e. the persisted (not-yet-updated-for-this-bar) state.
    since_red_exit_prev = since_red_exit_all[bar - 1] if bar > 0 else np.nan
    since_green_exit_prev = since_green_exit_all[bar - 1] if bar > 0 else np.nan
    valid_short_exit_prev = (
        since_red_entry_prev is not None
        and np.isfinite(since_red_exit_prev)
        and since_red_exit_prev > since_red_entry_prev
    )
    valid_long_exit_prev = (
        since_green_entry_prev is not None
        and np.isfinite(since_green_exit_prev)
        and since_green_exit_prev > since_green_entry_prev
    )
    end_long = bool(kernel["bearish_change"][bar]) and valid_long_exit_prev
    end_short = bool(kernel["bullish_change"][bar]) and valid_short_exit_prev

    is_early_signal_flip = changed and any(state.changed_window)

    new_since_red_entry = (
        0
        if start_short
        else (since_red_entry_prev + 1 if since_red_entry_prev is not None else None)
    )
    new_since_green_entry = (
        0
        if start_long
        else (since_green_entry_prev + 1 if since_green_entry_prev is not None else None)
    )
    new_changed_window = (state.changed_window[1], state.changed_window[2], changed)
    new_state = ExitState(
        bars_since_red_entry=new_since_red_entry,
        bars_since_green_entry=new_since_green_entry,
        changed_window=new_changed_window,
    )
    return end_long, end_short, is_early_signal_flip, new_state


def _max_bars_back_index(engine: AlphaEngine, length: int) -> int:
    return length - 1 - engine.max_bars_back if length - 1 >= engine.max_bars_back else 0


def _labels(source: np.ndarray) -> np.ndarray:
    length = len(source)
    labels = np.zeros(length, dtype=int)
    for i in range(4, length):
        labels[i] = -1 if source[i - 4] < source[i] else 1 if source[i - 4] > source[i] else 0
    return labels


def _advance_queue(
    engine: AlphaEngine,
    features: list[np.ndarray],
    labels: np.ndarray,
    bar: int,
    max_bars_back_index: int,
    state: QueueState,
) -> tuple[int, QueueState]:
    """Process one bar's own candidate scan, continuing (not resetting) the
    persisted neighbor queue -- term-for-term identical to one iteration of
    ``AlphaEngine._predict``'s outer loop body.
    """
    # `last_distance` resets every bar in the original loop (it is declared
    # *inside* the outer "if bar >= max_bars_back_index" block) -- only
    # `distances`/`predictions` are declared outside the loop and persist.
    last_distance = -1.0
    distances = list(state.distances)
    predictions = list(state.predictions)
    size = min(engine.max_bars_back - 1, bar)
    start_index = 0 if engine.include_full_history else max_bars_back_index
    for i in range(start_index, size + 1):
        distance = _lorentzian_distance(features, bar, i, engine.feature_count)
        if np.isfinite(distance) and distance >= last_distance and i % 4 != 0:
            last_distance = distance
            distances.append(distance)
            predictions.append(int(labels[i]))
            if len(predictions) > engine.neighbors_count:
                last_distance = distances[round(engine.neighbors_count * 3 / 4)]
                distances.pop(0)
                predictions.pop(0)
    return sum(predictions), QueueState(distances=tuple(distances), predictions=tuple(predictions))

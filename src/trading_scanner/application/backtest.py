"""One-time historical backtest: replay full stored history into real trades.

Unlike ``application/fast_predict.py`` (which only evaluates the newest bar
for speed), this walks a symbol's *entire* stored candle history through
AlphaEngine's real vectorized signal/exit logic once, and returns every
historical BUY/SELL entry and end_long/end_short exit in chronological
order -- so the ``trades`` table can hold the strategy's full backtest, not
just live entries seen after the trade-tracking feature shipped.

``alpha_engine.py`` is not modified. This reuses its already-vectorized
private methods, plus the same ``_signal_state``/``_changed`` helpers
``validation/runner.py`` already uses to expose the raw signal-state array
Pine hides (``events`` from ``_predict`` never returns it directly).

Trade scoring faithfully replicates Pine's own ``ml.backtest`` helper
(``strategy/MLExtensions_v2.pine``), which TradingView's on-chart Winrate/WL
Ratio table is built from -- not an independent approximation. Two details
matter for matching TradingView's numbers exactly:

1. **Price convention**: Pine scores using ``(high + low + open + open) / 4``
   (its default, ``useWorstCase=false``), not the close.
2. **Single active position per symbol**: Pine tracks entries with one
   overwritable variable per side, not independent open positions. If a new
   opposite-side entry fires before the current position's own exit
   condition triggers, that position is silently abandoned -- never scored
   as a win or a loss. Replicating this (rather than letting a position
   dangle open until its own exit eventually fires, possibly much later at a
   worse price) is what actually explains a large win-rate mismatch, not the
   price convention above.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC

import numpy as np
import pandas as pd

from trading_scanner.alpha_engine import (
    AlphaEngine,
    _atr,
    _ema,
    _filter_adx,
    _filter_volatility,
    _regime_filter,
)
from trading_scanner.domain.models import Candle
from trading_scanner.validation.runner import _changed, _signal_state


@dataclass(frozen=True, slots=True)
class HistoricalEvent:
    """One bar's entry/exit signal state, in chronological order."""

    timestamp: object
    market_price: float
    prediction: int
    start_long: bool
    start_short: bool
    end_long: bool
    end_short: bool
    is_early_signal_flip: bool
    # Feature/filter snapshot for this bar -- for training a future
    # ranking/meta-labeling model (see NOTES.md Phase 2). These are already
    # computed by AlphaEngine's vectorized filter/feature pass in
    # ``compute_historical_events``; this just keeps them instead of
    # discarding them per bar.
    adx: float
    regime_normalized: float
    volatility_margin: float
    volatility_filter_passed: bool
    regime_filter_passed: bool
    adx_filter_passed: bool


@dataclass(frozen=True, slots=True)
class PineTrade:
    """One trade as Pine's ``ml.backtest`` would score it."""

    side: str  # "buy" | "sell"
    entry_timestamp: object
    entry_price: float
    prediction_at_entry: int
    is_early_signal_flip: bool
    exit_timestamp: object | None
    exit_price: float | None
    status: str  # "open" | "closed"
    adx_at_entry: float
    regime_normalized_at_entry: float
    volatility_margin_at_entry: float
    volatility_filter_passed: bool
    regime_filter_passed: bool
    adx_filter_passed: bool


def compute_historical_events(
    engine: AlphaEngine, candles: Sequence[Candle]
) -> list[HistoricalEvent]:
    """Run one full-history AlphaEngine pass and return every bar's signal/exit state.

    A single vectorized pass over the whole history -- the same cost as one
    ``AlphaEngine.analyze()`` call, not a per-bar re-derivation.
    """
    data = _candles_to_dataframe(candles)
    open_ = data["Open"].to_numpy(dtype=float)
    high = data["High"].to_numpy(dtype=float)
    low = data["Low"].to_numpy(dtype=float)
    close = data["Close"].to_numpy(dtype=float)
    source = close.copy()

    features = engine._calculate_features(close, high, low)
    filter_all = engine._calculate_filters(open_, source, high, low, close)
    kernel = engine._kernel_regression(source)
    events = engine._predict(features, source, filter_all, kernel, close)

    signal_state = _signal_state(events["prediction"], filter_all)
    changed = _changed(signal_state)

    # Pine's own `ml.backtest` scoring price -- (high + low + open + open) / 4,
    # not the close (its default, useWorstCase=false).
    market_price = (high + low + open_ + open_) / 4.0

    # Individual filter states and the continuous values behind two of them
    # -- `_calculate_filters` above already ANDs these into one bool per bar
    # for live trading, which is all the live path needs. For training data
    # we want each filter's own pass/fail plus, for regime and volatility,
    # the underlying margin (not just the threshold crossing). Calling the
    # same filter functions again here is cheap (one more vectorized pass)
    # and does not change anything about the live signal.
    ohlc4 = (open_ + high + low + close) / 4.0
    volatility_passed = _filter_volatility(high, low, close, 1, 10, engine.use_volatility_filter)
    regime_passed = _regime_filter(
        ohlc4, high, low, engine.regime_threshold, engine.use_regime_filter
    )
    adx_passed = _filter_adx(source, high, low, 14, engine.adx_threshold, engine.use_adx_filter)
    adx_value = features[3]  # `_n_adx`, already computed for the model's own features
    regime_normalized = _regime_normalized(ohlc4, high, low)
    volatility_margin = _volatility_margin(high, low, close)

    timestamps = data.index
    return [
        HistoricalEvent(
            timestamp=timestamps[bar],
            market_price=float(market_price[bar]),
            prediction=int(events["prediction"][bar]),
            start_long=bool(events["start_long"][bar]),
            start_short=bool(events["start_short"][bar]),
            end_long=bool(events["end_long"][bar]),
            end_short=bool(events["end_short"][bar]),
            is_early_signal_flip=_is_early_signal_flip(changed, bar),
            adx=float(adx_value[bar]),
            regime_normalized=float(regime_normalized[bar]),
            volatility_margin=float(volatility_margin[bar]),
            volatility_filter_passed=bool(volatility_passed[bar]),
            regime_filter_passed=bool(regime_passed[bar]),
            adx_filter_passed=bool(adx_passed[bar]),
        )
        for bar in range(len(data))
    ]


def _regime_normalized(source: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """The continuous KLMF-slope value ``_regime_filter`` thresholds, kept for training data.

    Same formula as ``alpha_engine._regime_filter``, just returning the
    normalized value instead of collapsing it into a pass/fail bool --
    ``alpha_engine.py`` itself is untouched.
    """
    value1 = np.full(len(source), np.nan)
    value2 = np.full(len(source), np.nan)
    klmf = np.full(len(source), np.nan)
    for index in range(len(source)):
        old_value1 = value1[index - 1] if index else 0.0
        old_value2 = value2[index - 1] if index else 0.0
        old_klmf = klmf[index - 1] if index else 0.0
        delta = source[index] - source[index - 1] if index else np.nan
        value1[index] = 0.2 * delta + 0.8 * (0.0 if np.isnan(old_value1) else old_value1)
        value2[index] = 0.1 * (high[index] - low[index]) + 0.8 * (
            0.0 if np.isnan(old_value2) else old_value2
        )
        omega = abs(value1[index] / value2[index])
        alpha = (-(omega**2) + np.sqrt(omega**4 + 16.0 * omega**2)) / 8.0
        klmf[index] = alpha * source[index] + (1.0 - alpha) * (
            0.0 if np.isnan(old_klmf) else old_klmf
        )
    slope = np.abs(klmf - np.concatenate(([np.nan], klmf[:-1])))
    average = _ema(slope, 200)
    return (slope - average) / average


def _volatility_margin(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """How far the short-lookback ATR clears the long-lookback ATR.

    ``_filter_volatility`` only reports whether ``atr(1) > atr(10)``; this
    keeps the gap itself as a continuous training feature.
    """
    return _atr(high, low, close, 1) - _atr(high, low, close, 10)


def replay_pine_backtest(events: Sequence[HistoricalEvent]) -> list[PineTrade]:
    """Score historical events exactly as Pine's ``ml.backtest`` would.

    Mirrors the Pine function's four ordered checks per bar (startLong,
    endLong, startShort, endShort), each acting on a single current-position
    variable -- a new opposite-side entry silently abandons whatever
    position was still open, exactly like Pine's ``start_long_trade := 0.``/
    ``start_short_trade := 0.`` resets. An abandoned position is never
    scored and never stored; only a position that reaches its own matching
    exit (or is still open at the end of history) becomes a ``PineTrade``.
    """
    trades: list[PineTrade] = []
    current: dict[str, object] | None = None

    for event in events:
        if event.start_long:
            current = _open_position("buy", event)
        if event.end_long and current is not None and current["side"] == "buy":
            trades.append(_close_trade(current, event))
            current = None
        if event.start_short:
            current = _open_position("sell", event)
        if event.end_short and current is not None and current["side"] == "sell":
            trades.append(_close_trade(current, event))
            current = None

    if current is not None:
        trades.append(
            PineTrade(
                side=current["side"],
                entry_timestamp=current["entry_timestamp"],
                entry_price=current["entry_price"],
                prediction_at_entry=current["prediction_at_entry"],
                is_early_signal_flip=current["is_early_signal_flip"],
                exit_timestamp=None,
                exit_price=None,
                status="open",
                adx_at_entry=current["adx_at_entry"],
                regime_normalized_at_entry=current["regime_normalized_at_entry"],
                volatility_margin_at_entry=current["volatility_margin_at_entry"],
                volatility_filter_passed=current["volatility_filter_passed"],
                regime_filter_passed=current["regime_filter_passed"],
                adx_filter_passed=current["adx_filter_passed"],
            )
        )
    return trades


def _open_position(side: str, event: HistoricalEvent) -> dict[str, object]:
    """Snapshot an entry event's price/prediction plus its feature state."""
    return {
        "side": side,
        "entry_timestamp": event.timestamp,
        "entry_price": event.market_price,
        "prediction_at_entry": event.prediction,
        "is_early_signal_flip": event.is_early_signal_flip,
        "adx_at_entry": event.adx,
        "regime_normalized_at_entry": event.regime_normalized,
        "volatility_margin_at_entry": event.volatility_margin,
        "volatility_filter_passed": event.volatility_filter_passed,
        "regime_filter_passed": event.regime_filter_passed,
        "adx_filter_passed": event.adx_filter_passed,
    }


def _close_trade(current: dict[str, object], exit_event: HistoricalEvent) -> PineTrade:
    """Build a closed PineTrade from the active position and its exit bar."""
    return PineTrade(
        side=current["side"],
        entry_timestamp=current["entry_timestamp"],
        entry_price=current["entry_price"],
        prediction_at_entry=current["prediction_at_entry"],
        is_early_signal_flip=current["is_early_signal_flip"],
        exit_timestamp=exit_event.timestamp,
        exit_price=exit_event.market_price,
        status="closed",
        adx_at_entry=current["adx_at_entry"],
        regime_normalized_at_entry=current["regime_normalized_at_entry"],
        volatility_margin_at_entry=current["volatility_margin_at_entry"],
        volatility_filter_passed=current["volatility_filter_passed"],
        regime_filter_passed=current["regime_filter_passed"],
        adx_filter_passed=current["adx_filter_passed"],
    )


def _is_early_signal_flip(changed, bar: int) -> bool:
    """Mirror Pine's ``isEarlySignalFlip``: a flip within 3 bars of the last one."""
    if bar < 3:
        return False
    return bool(changed[bar] and (changed[bar - 1] or changed[bar - 2] or changed[bar - 3]))


def _candles_to_dataframe(candles: Sequence[Candle]) -> pd.DataFrame:
    """Convert chronological Candle objects into the OHLCV DataFrame AlphaEngine expects.

    Normalizes every timestamp to UTC before building the index -- see
    ``application/signal_pipeline.py``'s identical helper for why this is
    required (candles can carry equivalent-offset but distinct tzinfo
    objects, which pandas refuses to unify without this).
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

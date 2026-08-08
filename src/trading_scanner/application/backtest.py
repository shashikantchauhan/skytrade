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

import pandas as pd

from trading_scanner.alpha_engine import AlphaEngine
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
        )
        for bar in range(len(data))
    ]


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
            current = {
                "side": "buy",
                "entry_timestamp": event.timestamp,
                "entry_price": event.market_price,
                "prediction_at_entry": event.prediction,
                "is_early_signal_flip": event.is_early_signal_flip,
            }
        if event.end_long and current is not None and current["side"] == "buy":
            trades.append(_close_trade(current, event))
            current = None
        if event.start_short:
            current = {
                "side": "sell",
                "entry_timestamp": event.timestamp,
                "entry_price": event.market_price,
                "prediction_at_entry": event.prediction,
                "is_early_signal_flip": event.is_early_signal_flip,
            }
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
            )
        )
    return trades


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

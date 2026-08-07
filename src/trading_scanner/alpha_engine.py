"""Faithful Python translation of ``strategy/lorentzian.pine``.

The implementation follows the Pine script's default input values and evaluates
each bar in chronological order.  This is deliberate: the source uses Pine
``var`` arrays and recursive series values, so a purely vectorized replacement
would change the classifier's output.

Pine-only behavior not represented by ``SignalResult``: chart plots, labels,
bar colors, alert declarations, tables, and the ``ml.backtest`` statistics.
Those operations do not participate in BUY/SELL calculation.  Accordingly,
``show_exits`` and ``color_compression`` are retained as source inputs but do
not alter the returned signal.  ``showTradeStats`` and ``useWorstCase`` are
not constructor inputs because they affect only that omitted display statistic.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True, slots=True)
class SignalResult:
    """The latest-bar result produced by the Pine indicator."""

    signal: Literal["BUY", "SELL", "NEUTRAL"]
    prediction: int
    timestamp: object
    start_long_trade: bool
    start_short_trade: bool
    end_long_trade: bool
    end_short_trade: bool


class AlphaEngine:
    """Execute the Lorentzian Classification Pine script against OHLCV data.

    Constructor defaults are the script's ``input.*`` defaults.  They are
    exposed only to mirror Pine inputs; no formula or threshold is changed.
    """

    def __init__(
        self,
        neighbors_count: int = 8,
        max_bars_back: int = 2000,
        feature_count: int = 5,
        color_compression: int = 1,
        show_exits: bool = False,
        use_dynamic_exits: bool = False,
        include_full_history: bool = False,
        use_volatility_filter: bool = True,
        use_regime_filter: bool = True,
        use_adx_filter: bool = False,
        regime_threshold: float = -0.1,
        adx_threshold: int = 20,
        use_ema_filter: bool = False,
        ema_period: int = 200,
        use_sma_filter: bool = False,
        sma_period: int = 200,
        use_kernel_filter: bool = True,
        use_kernel_smoothing: bool = False,
        h: int = 8,
        r: float = 8.0,
        x: int = 25,
        lag: int = 2,
    ) -> None:
        self.neighbors_count = neighbors_count
        self.max_bars_back = max_bars_back
        self.feature_count = feature_count
        self.color_compression = color_compression
        self.show_exits = show_exits
        self.use_dynamic_exits = use_dynamic_exits
        self.include_full_history = include_full_history
        self.use_volatility_filter = use_volatility_filter
        self.use_regime_filter = use_regime_filter
        self.use_adx_filter = use_adx_filter
        self.regime_threshold = regime_threshold
        self.adx_threshold = adx_threshold
        self.use_ema_filter = use_ema_filter
        self.ema_period = ema_period
        self.use_sma_filter = use_sma_filter
        self.sma_period = sma_period
        self.use_kernel_filter = use_kernel_filter
        self.use_kernel_smoothing = use_kernel_smoothing
        self.h = h
        self.r = r
        self.x = x
        self.lag = lag

    def analyze(self, dataframe: pd.DataFrame) -> SignalResult:
        """Return the most recent BUY, SELL, or NEUTRAL Pine signal.

        ``dataframe`` must be chronological OHLCV data.  Its index is returned
        unchanged as the result timestamp.
        """
        _validate_input(dataframe)
        data = dataframe.sort_index()
        open_ = data["Open"].to_numpy(dtype=float)
        high = data["High"].to_numpy(dtype=float)
        low = data["Low"].to_numpy(dtype=float)
        close = data["Close"].to_numpy(dtype=float)
        source = close.copy()  # Pine input.source default: close.
        features = self._calculate_features(close, high, low)
        filters = self._calculate_filters(open_, source, high, low, close)
        kernel = self._kernel_regression(source)
        events = self._predict(features, source, filters, kernel, close)
        last = len(data) - 1
        signal = (
            "BUY"
            if events["start_long"][last]
            else "SELL" if events["start_short"][last] else "NEUTRAL"
        )
        return SignalResult(
            signal=signal,
            prediction=int(events["prediction"][last]),
            timestamp=data.index[last],
            start_long_trade=bool(events["start_long"][last]),
            start_short_trade=bool(events["start_short"][last]),
            end_long_trade=bool(events["end_long"][last]),
            end_short_trade=bool(events["end_short"][last]),
        )

    def _calculate_features(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray
    ) -> list[np.ndarray]:
        """Translate ``series_from`` and the five default FeatureSeries inputs."""
        hlc3 = (high + low + close) / 3.0
        return [
            _n_rsi(close, 14, 1),
            _n_wt(hlc3, 10, 11),
            _n_cci(close, 20, 1),
            _n_adx(high, low, close, 20),
            _n_rsi(close, 9, 1),
        ]

    def _calculate_filters(
        self,
        open_: np.ndarray,
        source: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """Translate the MLExtensions volatility, regime, and ADX filters."""
        ohlc4 = (open_ + high + low + close) / 4.0
        volatility = _filter_volatility(high, low, close, 1, 10, self.use_volatility_filter)
        regime = _regime_filter(ohlc4, high, low, self.regime_threshold, self.use_regime_filter)
        adx = _filter_adx(source, high, low, 14, self.adx_threshold, self.use_adx_filter)
        return volatility & regime & adx

    def _kernel_regression(self, source: np.ndarray) -> dict[str, np.ndarray]:
        """Translate KernelFunctions rationalQuadratic and gaussian calls."""
        yhat1 = _rational_quadratic(source, self.h, self.r, self.x)
        yhat2 = _gaussian(source, self.h - self.lag, self.x)
        bearish_rate = _compare_shift(yhat1, 1, yhat1, "gt")
        bullish_rate = _compare_shift(yhat1, 1, yhat1, "lt")
        was_bearish_rate = _compare_shift(yhat1, 2, yhat1, "gt", right_shift=1)
        was_bullish_rate = _compare_shift(yhat1, 2, yhat1, "lt", right_shift=1)
        bearish_change = bearish_rate & was_bullish_rate
        bullish_change = bullish_rate & was_bearish_rate
        bullish_cross = _crossover(yhat2, yhat1)
        bearish_cross = _crossunder(yhat2, yhat1)
        bullish_smooth = _finite_compare(yhat2, yhat1, np.greater_equal)
        bearish_smooth = _finite_compare(yhat2, yhat1, np.less_equal)
        return {
            "bullish": bullish_smooth if self.use_kernel_smoothing else bullish_rate,
            "bearish": bearish_smooth if self.use_kernel_smoothing else bearish_rate,
            "alert_bullish": bullish_cross if self.use_kernel_smoothing else bullish_change,
            "alert_bearish": bearish_cross if self.use_kernel_smoothing else bearish_change,
            "bearish_change": bearish_change,
            "bullish_change": bullish_change,
        }

    def _predict(
        self,
        features: list[np.ndarray],
        source: np.ndarray,
        filters: np.ndarray,
        kernel: dict[str, np.ndarray],
        close: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Translate the Pine ANN loop, signal state, entries, and exits."""
        length = len(source)
        max_bars_back_index = (
            length - 1 - self.max_bars_back if length - 1 >= self.max_bars_back else 0
        )
        labels = np.zeros(length, dtype=int)
        for bar in range(4, length):
            labels[bar] = (
                -1
                if source[bar - 4] < source[bar]
                else 1 if source[bar - 4] > source[bar] else 0
            )

        prediction = np.zeros(length, dtype=int)
        signal = np.zeros(length, dtype=int)
        predictions: list[int] = []
        distances: list[float] = []
        for bar in range(length):
            if bar >= max_bars_back_index:
                last_distance = -1.0
                size = min(self.max_bars_back - 1, bar)
                start_index = 0 if self.include_full_history else max_bars_back_index
                for i in range(start_index, size + 1):
                    distance = _lorentzian_distance(features, bar, i, self.feature_count)
                    # This intentionally preserves Pine's ``i % 4 != 0`` condition.
                    if np.isfinite(distance) and distance >= last_distance and i % 4 != 0:
                        last_distance = distance
                        distances.append(distance)
                        predictions.append(int(labels[i]))
                        if len(predictions) > self.neighbors_count:
                            last_distance = distances[round(self.neighbors_count * 3 / 4)]
                            distances.pop(0)
                            predictions.pop(0)
                prediction[bar] = sum(predictions)
            previous = signal[bar - 1] if bar else 0
            signal[bar] = (
                1
                if prediction[bar] > 0 and filters[bar]
                else -1 if prediction[bar] < 0 and filters[bar] else previous
            )

        ema = _ema(close, self.ema_period)
        sma = _sma(close, self.sma_period)
        ema_up = (
            _finite_compare(close, ema, np.greater)
            if self.use_ema_filter
            else np.ones(length, dtype=bool)
        )
        ema_down = (
            _finite_compare(close, ema, np.less)
            if self.use_ema_filter
            else np.ones(length, dtype=bool)
        )
        sma_up = (
            _finite_compare(close, sma, np.greater)
            if self.use_sma_filter
            else np.ones(length, dtype=bool)
        )
        sma_down = (
            _finite_compare(close, sma, np.less)
            if self.use_sma_filter
            else np.ones(length, dtype=bool)
        )
        return self._entries_and_exits(
            signal, prediction, ema_up, ema_down, sma_up, sma_down, kernel
        )

    def _entries_and_exits(
        self,
        signal: np.ndarray,
        prediction: np.ndarray,
        ema_up: np.ndarray,
        ema_down: np.ndarray,
        sma_up: np.ndarray,
        sma_down: np.ndarray,
        kernel: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Translate Pine's fractal, kernel, fixed, and dynamic exit booleans."""
        length = len(signal)
        changed = np.zeros(length, dtype=bool)
        changed[1:] = signal[1:] != signal[:-1]
        buy = (signal == 1) & ema_up & sma_up
        sell = (signal == -1) & ema_down & sma_down
        new_buy = buy & changed
        new_sell = sell & changed
        is_bullish = kernel["bullish"] if self.use_kernel_filter else np.ones(length, dtype=bool)
        is_bearish = kernel["bearish"] if self.use_kernel_filter else np.ones(length, dtype=bool)
        start_long = new_buy & is_bullish & ema_up & sma_up
        start_short = new_sell & is_bearish & ema_down & sma_down

        bars_held = np.zeros(length, dtype=int)
        for bar in range(length):
            bars_held[bar] = 0 if changed[bar] else (bars_held[bar - 1] + 1 if bar else 1)
        held_four = bars_held == 4
        held_less = (bars_held > 0) & (bars_held < 4)
        last_buy = _shift_bool((signal == 1) & ema_up & sma_up, 4)
        last_sell = _shift_bool((signal == -1) & ema_down & sma_down, 4)
        strict_long = (
            ((held_four & last_buy) | (held_less & new_sell & last_buy))
            & _shift_bool(start_long, 4)
        )
        strict_short = (
            ((held_four & last_sell) | (held_less & new_buy & last_sell))
            & _shift_bool(start_short, 4)
        )

        since_red_entry = _bars_since(start_short)
        since_red_exit = _bars_since(kernel["alert_bullish"])
        since_green_entry = _bars_since(start_long)
        since_green_exit = _bars_since(kernel["alert_bearish"])
        valid_short_exit = _greater_or_false(since_red_exit, since_red_entry)
        valid_long_exit = _greater_or_false(since_green_exit, since_green_entry)
        dynamic_long = kernel["bearish_change"] & _shift_bool(valid_long_exit, 1)
        dynamic_short = kernel["bullish_change"] & _shift_bool(valid_short_exit, 1)
        dynamic_valid = (
            not self.use_ema_filter and not self.use_sma_filter and not self.use_kernel_smoothing
        )
        end_long = dynamic_long if self.use_dynamic_exits and dynamic_valid else strict_long
        end_short = dynamic_short if self.use_dynamic_exits and dynamic_valid else strict_short
        return {
            "prediction": prediction,
            "start_long": start_long,
            "start_short": start_short,
            "end_long": end_long,
            "end_short": end_short,
        }


def _validate_input(data: pd.DataFrame) -> None:
    """Validate the DataFrame columns needed by the Pine script."""
    missing = [column for column in _REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"AlphaEngine requires OHLCV columns; missing: {', '.join(missing)}.")
    if data.empty:
        raise ValueError("AlphaEngine requires at least one candle.")


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    alpha = 2.0 / (period + 1.0)
    for index, value in enumerate(values):
        if np.isnan(value):
            continue
        previous = result[index - 1] if index else np.nan
        result[index] = value if np.isnan(previous) else alpha * value + (1.0 - alpha) * previous
    return result


def _rma(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    for index in range(period - 1, len(values)):
        window = values[index - period + 1 : index + 1]
        if np.isnan(window).any():
            continue
        previous = result[index - 1] if index else np.nan
        result[index] = (
            window.mean()
            if np.isnan(previous)
            else (previous * (period - 1) + values[index]) / period
        )
    return result


def _sma(values: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(values).rolling(period, min_periods=period).mean().to_numpy()


def _rsi(values: np.ndarray, period: int) -> np.ndarray:
    change = np.full(len(values), np.nan)
    change[1:] = values[1:] - values[:-1]
    gains = np.where(change > 0.0, change, 0.0)
    losses = np.where(change < 0.0, -change, 0.0)
    gains[0] = losses[0] = np.nan
    average_gain, average_loss = _rma(gains, period), _rma(losses, period)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(average_gain) & np.isfinite(average_loss)
    result[valid & (average_loss == 0.0)] = 100.0
    ratio = np.divide(
        average_gain, average_loss, out=np.zeros(len(values)), where=average_loss != 0.0
    )
    has_loss = valid & (average_loss != 0.0)
    result[has_loss] = 100.0 - 100.0 / (1.0 + ratio[has_loss])
    return result


def _cci(values: np.ndarray, period: int) -> np.ndarray:
    series = pd.Series(values)
    mean = series.rolling(period, min_periods=period).mean()
    deviation = series.rolling(period, min_periods=period).apply(
        lambda window: np.mean(np.abs(window - window.mean())), raw=True
    )
    return ((series - mean) / (0.015 * deviation)).to_numpy()


def _normalize(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    result = np.full(len(values), np.nan)
    historic_min, historic_max = 1e11, -1e11
    for index, value in enumerate(values):
        if np.isfinite(value):
            historic_min = min(value, historic_min)
            historic_max = max(value, historic_max)
            result[index] = minimum + (maximum - minimum) * (value - historic_min) / max(
                historic_max - historic_min, 1e-9
            )
    return result


def _n_rsi(values: np.ndarray, n1: int, n2: int) -> np.ndarray:
    return _ema(_rsi(values, n1), n2) / 100.0


def _n_cci(values: np.ndarray, n1: int, n2: int) -> np.ndarray:
    return _normalize(_ema(_cci(values, n1), n2), 0.0, 1.0)


def _n_wt(values: np.ndarray, n1: int, n2: int) -> np.ndarray:
    ema1 = _ema(values, n1)
    ema2 = _ema(np.abs(values - ema1), n1)
    ci = np.divide(values - ema1, 0.015 * ema2, out=np.full(len(values), np.nan), where=ema2 != 0.0)
    wt1 = _ema(ci, n2)
    return _normalize(wt1 - _sma(wt1, 4), 0.0, 1.0)


def _directional_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    length = len(close)
    tr_smooth = np.zeros(length)
    plus_smooth = np.zeros(length)
    minus_smooth = np.zeros(length)
    dx = np.full(length, np.nan)
    for index in range(length):
        previous_close = close[index - 1] if index else 0.0
        previous_high = high[index - 1] if index else 0.0
        previous_low = low[index - 1] if index else 0.0
        tr = max(
            high[index] - low[index],
            abs(high[index] - previous_close),
            abs(low[index] - previous_close),
        )
        plus = (
            max(high[index] - previous_high, 0.0)
            if high[index] - previous_high > previous_low - low[index]
            else 0.0
        )
        minus = (
            max(previous_low - low[index], 0.0)
            if previous_low - low[index] > high[index] - previous_high
            else 0.0
        )
        before_tr = tr_smooth[index - 1] if index else 0.0
        before_plus = plus_smooth[index - 1] if index else 0.0
        before_minus = minus_smooth[index - 1] if index else 0.0
        tr_smooth[index] = before_tr - before_tr / period + tr
        plus_smooth[index] = before_plus - before_plus / period + plus
        minus_smooth[index] = before_minus - before_minus / period + minus
        positive = plus_smooth[index] / tr_smooth[index] * 100.0
        negative = minus_smooth[index] / tr_smooth[index] * 100.0
        if positive + negative:
            dx[index] = abs(positive - negative) / (positive + negative) * 100.0
    return _rma(dx, period)


def _n_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    return _directional_adx(high, low, close, period) / 100.0


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    tr = np.empty(len(close))
    tr[0] = high[0] - low[0]
    for index in range(1, len(close)):
        tr[index] = max(
            high[index] - low[index],
            abs(high[index] - close[index - 1]),
            abs(low[index] - close[index - 1]),
        )
    return _rma(tr, period)


def _filter_volatility(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    minimum: int,
    maximum: int,
    enabled: bool,
) -> np.ndarray:
    if not enabled:
        return np.ones(len(close), dtype=bool)
    return _finite_compare(
        _atr(high, low, close, minimum), _atr(high, low, close, maximum), np.greater
    )


def _regime_filter(
    source: np.ndarray, high: np.ndarray, low: np.ndarray, threshold: float, enabled: bool
) -> np.ndarray:
    if not enabled:
        return np.ones(len(source), dtype=bool)
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
        alpha = (-omega**2 + np.sqrt(omega**4 + 16.0 * omega**2)) / 8.0
        klmf[index] = alpha * source[index] + (1.0 - alpha) * (
            0.0 if np.isnan(old_klmf) else old_klmf
        )
    slope = np.abs(klmf - _shift(klmf, 1))
    average = _ema(slope, 200)
    normalized = (slope - average) / average
    return np.isfinite(normalized) & (normalized >= threshold)


def _filter_adx(
    source: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    length: int,
    threshold: int,
    enabled: bool,
) -> np.ndarray:
    if not enabled:
        return np.ones(len(source), dtype=bool)
    return _finite_compare(
        _directional_adx(high, low, source, length),
        np.full(len(source), threshold),
        np.greater,
    )


def _rational_quadratic(
    source: np.ndarray, lookback: int, relative_weight: float, start_at_bar: int
) -> np.ndarray:
    return _kernel(
        source,
        start_at_bar,
        lambda i: (1.0 + i**2 / (lookback**2 * 2.0 * relative_weight)) ** (-relative_weight),
    )


def _gaussian(source: np.ndarray, lookback: int, start_at_bar: int) -> np.ndarray:
    return _kernel(source, start_at_bar, lambda i: np.exp(-(i**2) / (2.0 * lookback**2)))


def _kernel(source: np.ndarray, start_at_bar: int, weight_function) -> np.ndarray:
    """Mirror ``for i = 0 to array.size(array.from(src)) + startAtBar``."""
    result = np.full(len(source), np.nan)
    offsets = range(start_at_bar + 2)  # array.from(series) has Pine size one.
    for bar in range(len(source)):
        weighted, total = 0.0, 0.0
        valid = True
        for offset in offsets:
            if bar - offset < 0 or not np.isfinite(source[bar - offset]):
                valid = False
                break
            weight = weight_function(offset)
            weighted += source[bar - offset] * weight
            total += weight
        if valid:
            result[bar] = weighted / total
    return result


def _lorentzian_distance(
    features: list[np.ndarray], bar: int, index: int, feature_count: int
) -> float:
    total = 0.0
    for feature in features[:feature_count]:
        current, historical = feature[bar], feature[index]
        if not np.isfinite(current) or not np.isfinite(historical):
            return np.nan
        total += np.log(1.0 + abs(current - historical))
    return total


def _shift(values: np.ndarray, periods: int) -> np.ndarray:
    if periods == 0:
        return values.copy()
    shifted = np.full(len(values), np.nan)
    if periods < len(values):
        shifted[periods:] = values[:-periods]
    return shifted


def _shift_bool(values: np.ndarray, periods: int) -> np.ndarray:
    shifted = np.zeros(len(values), dtype=bool)
    if periods < len(values):
        shifted[periods:] = values[:-periods]
    return shifted


def _finite_compare(left: np.ndarray, right: np.ndarray, comparison) -> np.ndarray:
    return np.isfinite(left) & np.isfinite(right) & comparison(left, right)


def _compare_shift(
    left: np.ndarray,
    left_shift: int,
    right: np.ndarray,
    operator: str,
    right_shift: int = 0,
) -> np.ndarray:
    comparison = np.greater if operator == "gt" else np.less
    return _finite_compare(_shift(left, left_shift), _shift(right, right_shift), comparison)


def _crossover(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _finite_compare(left, right, np.greater) & _finite_compare(
        _shift(left, 1), _shift(right, 1), np.less_equal
    )


def _crossunder(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _finite_compare(left, right, np.less) & _finite_compare(
        _shift(left, 1), _shift(right, 1), np.greater_equal
    )


def _bars_since(condition: np.ndarray) -> np.ndarray:
    result = np.full(len(condition), np.nan)
    last_true: int | None = None
    for index, value in enumerate(condition):
        if value:
            last_true = index
        if last_true is not None:
            result[index] = index - last_true
    return result


def _greater_or_false(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.isfinite(left) & np.isfinite(right) & (left > right)

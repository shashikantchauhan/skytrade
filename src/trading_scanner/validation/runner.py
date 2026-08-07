"""Chronological CSV validation for the Alpha Engine."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from trading_scanner.alpha_engine import AlphaEngine
from trading_scanner.alpha_engine import _ema, _finite_compare, _rational_quadratic, _sma
from trading_scanner.infrastructure.yahoo import YahooFinanceProvider


class ValidationRunner:
    """Download candles, evaluate every bar, and export comparison data."""

    def __init__(self, output_directory: Path = Path("validation")) -> None:
        self.output_directory = output_directory

    def run(
        self, symbol: str, interval: str = "1h", days: int = 10, verbose: bool = False
    ) -> Path:
        """Export chronological AlphaEngine results for a Yahoo symbol."""
        logger = logging.getLogger(__name__)
        logger.info("Downloading data...")
        data = YahooFinanceProvider().get_recent_history(symbol, interval, days)
        logger.info("Running AlphaEngine...")
        history, debug_history = self._run_every_bar(data)
        logger.info("Processed %d candles.", len(history))
        output_path = self._output_path(symbol, interval)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(output_path, index=False)
        logger.info("CSV written to %s", output_path)
        debug_output_path = self._debug_output_path(symbol, interval)
        debug_history.to_csv(debug_output_path, index=False)
        logger.info("Debug CSV written to %s", debug_output_path)
        if verbose:
            self._print_verbose(debug_history)
        return output_path

    def _run_every_bar(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run a fresh full-history AlphaEngine evaluation for each candle.

        Each prefix is evaluated independently so AlphaEngine recreates Pine's
        recursive state exactly as it existed at that bar.  This intentionally
        avoids changing the strategy or adding an alternate calculation path.
        """
        engine = AlphaEngine()
        records = [
            self._evaluate_prefix(engine, data.iloc[: index + 1])
            for index in range(len(data))
        ]
        history = pd.DataFrame.from_records(
            [record["row"] for record in records], columns=_CSV_COLUMNS
        )
        debug_history = pd.DataFrame.from_records(
            [record["debug_row"] for record in records], columns=_DEBUG_CSV_COLUMNS
        )
        return history, debug_history

    def _evaluate_prefix(
        self, engine: AlphaEngine, data: pd.DataFrame
    ) -> dict[str, dict[str, object]]:
        """Create one validation row and one debug row from a data prefix."""
        result = engine.analyze(data)
        row = data.iloc[-1]
        open_ = data["Open"].to_numpy(dtype=float)
        high = data["High"].to_numpy(dtype=float)
        low = data["Low"].to_numpy(dtype=float)
        close = data["Close"].to_numpy(dtype=float)
        source = close.copy()

        features = engine._calculate_features(close, high, low)
        filter_all = engine._calculate_filters(open_, source, high, low, close)
        kernel = engine._kernel_regression(source)
        kernel_estimate = _rational_quadratic(source, engine.h, engine.r, engine.x)
        events = engine._predict(features, source, filter_all, kernel, close)
        ema_uptrend, ema_downtrend, sma_uptrend, sma_downtrend = _trend_states(engine, close)

        prediction = events["prediction"]
        signal_state = _signal_state(prediction, filter_all)
        changed = _changed(signal_state)
        bars_held = _bars_held(changed)
        new_buy = (signal_state == 1) & ema_uptrend & sma_uptrend & changed
        new_sell = (signal_state == -1) & ema_downtrend & sma_downtrend & changed

        index = len(data) - 1
        is_bullish = kernel["bullish"][index] if engine.use_kernel_filter else True
        is_bearish = kernel["bearish"][index] if engine.use_kernel_filter else True
        common = {
            "timestamp": data.index[index],
            "open": row["Open"],
            "high": row["High"],
            "low": row["Low"],
            "close": row["Close"],
            "prediction": result.prediction,
        }
        validation_row = {
            **common,
            "prediction_direction": _prediction_direction(result.prediction),
            "is_bullish": bool(is_bullish),
            "is_bearish": bool(is_bearish),
            "is_ema_uptrend": bool(ema_uptrend[index]),
            "is_ema_downtrend": bool(ema_downtrend[index]),
            "is_sma_uptrend": bool(sma_uptrend[index]),
            "is_sma_downtrend": bool(sma_downtrend[index]),
            "start_long_trade": result.start_long_trade,
            "start_short_trade": result.start_short_trade,
            "end_long_trade": result.end_long_trade,
            "end_short_trade": result.end_short_trade,
            "signal": result.signal,
        }
        debug_row = {
            **common,
            "is_bullish": bool(is_bullish),
            "is_bearish": bool(is_bearish),
            "is_ema_uptrend": bool(ema_uptrend[index]),
            "is_ema_downtrend": bool(ema_downtrend[index]),
            "is_sma_uptrend": bool(sma_uptrend[index]),
            "is_sma_downtrend": bool(sma_downtrend[index]),
            "filter_all": bool(filter_all[index]),
            "bars_held": int(bars_held[index]),
            "is_new_buy_signal": bool(new_buy[index]),
            "is_new_sell_signal": bool(new_sell[index]),
            "start_long_trade": result.start_long_trade,
            "start_short_trade": result.start_short_trade,
            "end_long_trade": result.end_long_trade,
            "end_short_trade": result.end_short_trade,
            "kernel_estimate": kernel_estimate[index],
            "kernel_bullish": bool(kernel["bullish"][index]),
            "kernel_bearish": bool(kernel["bearish"][index]),
            "signal": result.signal,
        }
        return {"row": validation_row, "debug_row": debug_row}

    def _print_verbose(self, debug_history: pd.DataFrame) -> None:
        """Print timestamp, prediction, and signal for every BUY or SELL bar."""
        signals = debug_history[debug_history["signal"].isin(["BUY", "SELL"])]
        for _, row in signals.iterrows():
            print(row["timestamp"])
            print(f"Prediction: {row['prediction']}")
            print(row["signal"])
            print("--------------------------------")

    def _output_path(self, symbol: str, interval: str) -> Path:
        """Build the specified CSV name without Yahoo exchange suffixes."""
        symbol_stem = symbol.split(".", maxsplit=1)[0]
        safe_symbol = "".join(
            character if character.isalnum() or character == "-" else "_"
            for character in symbol_stem
        )
        safe_interval = "".join(
            character if character.isalnum() else "_" for character in interval
        )
        return self.output_directory / f"validation_{safe_symbol}_{safe_interval}.csv"

    def _debug_output_path(self, symbol: str, interval: str) -> Path:
        """Build the debug CSV name without Yahoo exchange suffixes."""
        symbol_stem = symbol.split(".", maxsplit=1)[0]
        safe_symbol = "".join(
            character if character.isalnum() or character == "-" else "_"
            for character in symbol_stem
        )
        safe_interval = "".join(
            character if character.isalnum() else "_" for character in interval
        )
        return self.output_directory / f"validation_debug_{safe_symbol}_{safe_interval}.csv"


def _signal_state(prediction: np.ndarray, filter_all: np.ndarray) -> np.ndarray:
    """Mirror the Pine ANN signal carry-forward: ``prediction``/``filter_all`` only."""
    length = len(prediction)
    signal = np.zeros(length, dtype=int)
    for bar in range(length):
        previous = signal[bar - 1] if bar else 0
        signal[bar] = (
            1
            if prediction[bar] > 0 and filter_all[bar]
            else -1 if prediction[bar] < 0 and filter_all[bar] else previous
        )
    return signal


def _changed(signal: np.ndarray) -> np.ndarray:
    """Mirror Pine's ``ta.change(signal) != 0`` state-change detector."""
    changed = np.zeros(len(signal), dtype=bool)
    changed[1:] = signal[1:] != signal[:-1]
    return changed


def _bars_held(changed: np.ndarray) -> np.ndarray:
    """Mirror Pine's bars-since-signal-change counter."""
    bars_held = np.zeros(len(changed), dtype=int)
    for bar in range(len(changed)):
        bars_held[bar] = 0 if changed[bar] else (bars_held[bar - 1] + 1 if bar else 1)
    return bars_held


def _trend_states(
    engine: AlphaEngine, close: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the exact EMA/SMA state calculations used inside AlphaEngine."""
    size = len(close)
    ema = _ema(close, engine.ema_period)
    sma = _sma(close, engine.sma_period)
    ema_uptrend = (
        _finite_compare(close, ema, np.greater)
        if engine.use_ema_filter
        else np.ones(size, dtype=bool)
    )
    ema_downtrend = (
        _finite_compare(close, ema, np.less)
        if engine.use_ema_filter
        else np.ones(size, dtype=bool)
    )
    sma_uptrend = (
        _finite_compare(close, sma, np.greater)
        if engine.use_sma_filter
        else np.ones(size, dtype=bool)
    )
    sma_downtrend = (
        _finite_compare(close, sma, np.less)
        if engine.use_sma_filter
        else np.ones(size, dtype=bool)
    )
    return ema_uptrend, ema_downtrend, sma_uptrend, sma_downtrend


def _prediction_direction(prediction: int) -> str:
    """Return the signed direction of the Pine neighbor-sum prediction."""
    return "BULLISH" if prediction > 0 else "BEARISH" if prediction < 0 else "NEUTRAL"


_CSV_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "prediction",
    "prediction_direction",
    "is_bullish",
    "is_bearish",
    "is_ema_uptrend",
    "is_ema_downtrend",
    "is_sma_uptrend",
    "is_sma_downtrend",
    "start_long_trade",
    "start_short_trade",
    "end_long_trade",
    "end_short_trade",
    "signal",
]

_DEBUG_CSV_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "prediction",
    "is_bullish",
    "is_bearish",
    "is_ema_uptrend",
    "is_ema_downtrend",
    "is_sma_uptrend",
    "is_sma_downtrend",
    "filter_all",
    "bars_held",
    "is_new_buy_signal",
    "is_new_sell_signal",
    "start_long_trade",
    "start_short_trade",
    "end_long_trade",
    "end_short_trade",
    "kernel_estimate",
    "kernel_bullish",
    "kernel_bearish",
    "signal",
]

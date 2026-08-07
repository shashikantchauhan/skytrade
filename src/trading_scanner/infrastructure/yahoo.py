"""Yahoo Finance market-data provider."""

import pandas as pd
import yfinance as yf


class YahooProvider:
    """Download historical OHLCV candles from Yahoo Finance."""

    def get_history(self, symbol: str, interval: str, history: int) -> pd.DataFrame:
        """Return cleaned, chronological candles for one symbol.

        The requested history is the maximum number of candles returned.  Yahoo
        accepts time periods rather than candle counts, so it is given a period
        large enough to obtain the requested number before the result is capped.

        Raises:
            RuntimeError: If Yahoo cannot provide usable market data.
        """
        if history <= 0:
            raise ValueError("History must be greater than zero.")

        try:
            data = yf.download(
                tickers=symbol,
                period=f"{history}d",
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
        except Exception as error:
            raise RuntimeError(f"Failed to download history for {symbol}: {error}") from error

        data = _clean_data(data, history)
        if data.empty:
            raise RuntimeError(f"No usable history returned for {symbol}.")
        return data

    def get_recent_history(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Return all cleaned candles Yahoo provides for the last calendar days.

        Unlike :meth:`get_history`, this method deliberately does not cap the
        result to a candle count.  It is used by the validation exporter, where
        every available bar in a calendar-day window must be compared with
        TradingView.
        """
        if days <= 0:
            raise ValueError("Days must be greater than zero.")
        try:
            data = yf.download(
                tickers=symbol,
                period=f"{days}d",
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
        except Exception as error:
            raise RuntimeError(f"Failed to download history for {symbol}: {error}") from error

        data = _clean_data(data, limit=None)
        if data.empty:
            raise RuntimeError(f"No usable history returned for {symbol}.")
        return data


def _clean_data(data: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Normalize Yahoo output and remove incomplete candles."""
    cleaned = data.copy()
    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = cleaned.columns.get_level_values(0)
    cleaned = cleaned.sort_index().dropna()
    return cleaned.tail(limit) if limit is not None else cleaned


# Name used by the validation workflow.  It intentionally refers to the same
# existing Yahoo implementation rather than introducing a second provider.
YahooFinanceProvider = YahooProvider

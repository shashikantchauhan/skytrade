"""Frozen daily liquidity-sweep strategy with 30-minute confirmation.

This module contains calculations only.  It is independent of AlphaEngine and
does not perform I/O or place orders.  ``daily_swing_live.py`` owns the live
WebSocket, persistence, and broker lifecycle around these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SwingSetup:
    symbol: str
    setup_date: date
    side: int  # 1 long, -1 short
    trigger: float
    stop: float
    atr: float
    setup_close: float
    outer_structure: float | None
    score: float


@dataclass(frozen=True, slots=True)
class EntryCandidate:
    setup: SwingSetup
    timestamp: datetime
    entry: float
    target: float
    risk: float
    slot: int


def _valid_bars(bars: pd.DataFrame) -> pd.DataFrame:
    valid = (
        (bars.open > 0)
        & (bars.high >= bars[["open", "close"]].max(axis=1))
        & (bars.low <= bars[["open", "close"]].min(axis=1))
        & (bars.high >= bars.low)
        & (bars.volume >= 0)
    )
    return bars.loc[valid].sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def build_live_context(
    bars: pd.DataFrame,
    signal_symbols: frozenset[str],
) -> tuple[dict[str, SwingSetup], dict[tuple[str, int], float], date]:
    """Build today's setups and same-slot volume baselines from completed history.

    ``bars`` must contain 30-minute candles through the previous completed
    session.  The returned date is that most recent session date. A setup is
    retained only for ``signal_symbols``; all equities still participate in
    breadth, matching the frozen Nifty 500 backtest.
    """
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    frame["date"] = frame.timestamp.dt.date
    frame = _valid_bars(frame)
    if frame.empty:
        raise ValueError("No valid completed 30-minute candles.")

    frame["slot"] = frame.groupby(["symbol", "date"]).cumcount()
    slot_base = (
        frame.groupby(["symbol", "slot"], sort=False)
        .volume.apply(lambda values: values.tail(20).median() if len(values) >= 12 else np.nan)
        .to_dict()
    )

    daily = (
        frame.groupby(["symbol", "date"], sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            bars=("close", "size"),
        )
        .reset_index()
    )
    daily = daily[daily.bars >= 10].copy()
    last_session = daily.date.max()
    group = daily.groupby("symbol", group_keys=False)
    daily["prev"] = group.close.shift()
    daily["ret"] = daily.close / daily.prev - 1
    daily["tr"] = np.maximum.reduce(
        [
            (daily.high - daily.low).to_numpy(),
            (daily.high - daily.prev).abs().to_numpy(),
            (daily.low - daily.prev).abs().to_numpy(),
        ]
    )
    daily["atr"] = group.tr.transform(
        lambda values: values.shift().rolling(20, min_periods=15).median()
    )
    daily["vol_ratio"] = daily.volume / group.volume.transform(
        lambda values: values.shift().rolling(20, min_periods=15).median()
    )
    daily["dollar_liq"] = (
        (daily.close * daily.volume)
        .groupby(daily.symbol)
        .transform(lambda values: values.shift().rolling(20, min_periods=15).median())
    )
    daily["res20"] = group.high.transform(
        lambda values: values.shift().rolling(20, min_periods=10).max()
    )
    daily["sup20"] = group.low.transform(
        lambda values: values.shift().rolling(20, min_periods=10).min()
    )
    daily["res60"] = group.high.transform(
        lambda values: values.shift().rolling(60, min_periods=30).max()
    )
    daily["sup60"] = group.low.transform(
        lambda values: values.shift().rolling(60, min_periods=30).min()
    )
    daily["mom20"] = daily.close / group.close.shift(20) - 1

    benchmark = daily[daily.symbol == "^NSEI"][["date", "mom20"]].rename(
        columns={"mom20": "market_mom20"}
    )
    equity = daily[daily.symbol != "^NSEI"].merge(benchmark, on="date", how="left")
    equity["breadth_up"] = equity.groupby("date").ret.transform(lambda values: (values > 0).mean())
    current = equity[(equity.date == last_session) & equity.symbol.isin(signal_symbols)].copy()

    spread = (current.high - current.low).replace(0, np.nan)
    lower_wick = (current[["open", "close"]].min(axis=1) - current.low) / spread
    upper_wick = (current.high - current[["open", "close"]].max(axis=1)) / spread
    liquid = current.dollar_liq >= 20_000_000
    long_mask = (
        liquid
        & (current.low < current.sup20 - 0.05 * current.atr)
        & (current.close > current.sup20)
        & (lower_wick >= 0.35)
        & (current.vol_ratio >= 1.0)
        & (current.market_mom20.abs() <= 0.04)
        & current.breadth_up.between(0.35, 0.65)
    )
    short_mask = (
        liquid
        & (current.high > current.res20 + 0.05 * current.atr)
        & (current.close < current.res20)
        & (upper_wick >= 0.45)
        & (current.vol_ratio >= 1.5)
        & (current.market_mom20.abs() <= 0.04)
        & current.breadth_up.between(0.40, 0.60)
    )

    setups: dict[str, SwingSetup] = {}
    for row in current[long_mask].itertuples():
        penetration = max(0.0, (row.sup20 - row.low) / row.atr)
        setups[row.symbol] = SwingSetup(
            symbol=row.symbol,
            setup_date=row.date,
            side=1,
            trigger=float(row.high),
            stop=float(row.low - 0.10 * row.atr),
            atr=float(row.atr),
            setup_close=float(row.close),
            outer_structure=float(row.res60) if pd.notna(row.res60) else None,
            score=float(lower_wick.loc[row.Index] * row.vol_ratio * (1 + penetration)),
        )
    for row in current[short_mask].itertuples():
        penetration = max(0.0, (row.high - row.res20) / row.atr)
        setups[row.symbol] = SwingSetup(
            symbol=row.symbol,
            setup_date=row.date,
            side=-1,
            trigger=float(row.low),
            stop=float(row.high + 0.10 * row.atr),
            atr=float(row.atr),
            setup_close=float(row.close),
            outer_structure=float(row.sup60) if pd.notna(row.sup60) else None,
            score=float(upper_wick.loc[row.Index] * row.vol_ratio * (1 + penetration)),
        )
    return setups, slot_base, last_session


def confirm_entry(
    setup: SwingSetup,
    candle: dict,
    market_candle: dict,
    slot: int,
    slot_volume_base: float | None,
    session_open: float | None = None,
) -> EntryCandidate | None:
    """Apply the frozen next-session confirmation and risk/room gates."""
    if slot >= 4 or not slot_volume_base or np.isnan(slot_volume_base):
        return None
    if slot > 0 and session_open is None:
        return None
    opening_price = float(session_open if session_open is not None else candle["open"])
    if abs(opening_price / setup.setup_close - 1) > (0.75 * setup.atr / setup.setup_close):
        return None
    open_ = float(candle["open"])
    high = float(candle["high"])
    low = float(candle["low"])
    close = float(candle["close"])
    spread = high - low
    location = (close - low) / spread if spread > 0 else 0.5
    bar_return = close / open_ - 1
    market_return = float(market_candle["close"]) / float(market_candle["open"]) - 1
    relative = bar_return - market_return
    volume_ok = float(candle["volume"]) / slot_volume_base >= 1.0
    if setup.side == 1:
        confirmed = (
            close > setup.trigger
            and close > open_
            and location >= 0.65
            and relative >= 0
            and volume_ok
        )
    else:
        confirmed = (
            close < setup.trigger
            and close < open_
            and location <= 0.35
            and relative <= 0
            and volume_ok
        )
    if not confirmed:
        return None

    entry = close  # the completed confirmation candle's own close
    # The frozen research gates included 0.05% adverse fill slippage. Keep
    # that conservative eligibility test while the live decision price and
    # persisted entry remain this candle's actual close.
    gate_entry = close * (1 + setup.side * 0.0005)
    gate_risk = setup.side * (gate_entry - setup.stop)
    if gate_risk <= 0 or gate_risk > 2.2 * setup.atr or gate_risk / gate_entry * 100 > 3.5:
        return None
    if setup.outer_structure is not None:
        room = setup.side * (setup.outer_structure - gate_entry)
        if 0 < room < 2.0 * gate_risk:
            return None
    risk = setup.side * (entry - setup.stop)
    timestamp = pd.Timestamp(candle["timestamp"]).to_pydatetime()
    return EntryCandidate(
        setup=setup,
        timestamp=timestamp,
        entry=entry,
        target=entry + setup.side * 3.0 * risk,
        risk=risk,
        slot=slot,
    )


def trailed_stop(
    side: int,
    entry: float,
    initial_stop: float,
    risk: float,
    atr: float,
    best_close: float,
) -> float:
    """Frozen close-based profit protection used by the historical test."""
    active = initial_stop
    progress = side * (best_close - entry)
    if progress >= 2 * risk:
        candidate = best_close - side * 1.25 * atr
        one_r = entry + side * risk
        active = max(active, one_r, candidate) if side == 1 else min(active, one_r, candidate)
    elif progress >= risk:
        active = max(active, entry) if side == 1 else min(active, entry)
    return active

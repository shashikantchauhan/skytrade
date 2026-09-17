from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    friction_pct: float = 0.20
    confirmation_bars: int = 4
    max_holding_sessions: int = 10
    max_risk_atr: float = 2.2
    max_risk_pct: float = 3.5
    max_gap_atr: float = 0.75


def load_bars(db: str, symbols_file: str | None) -> pd.DataFrame:
    con = sqlite3.connect(db)
    if symbols_file:
        allowed = {x.strip() for x in Path(symbols_file).read_text().splitlines() if x.strip()}
        allowed.add("^NSEI")
    else:
        allowed = {row[0] for row in con.execute("SELECT DISTINCT symbol FROM candles")}
    chunks = []
    names = sorted(allowed)
    for start in range(0, len(names), 200):
        batch = names[start : start + 200]
        marks = ",".join("?" * len(batch))
        chunks.append(
            pd.read_sql_query(
                f"""select symbol,timestamp,open,high,low,close,volume
                from candles where interval='30m' and symbol in ({marks})
                order by symbol,timestamp""",
                con,
                params=batch,
            )
        )
    con.close()
    bars = pd.concat(chunks, ignore_index=True)
    bars.timestamp = pd.to_datetime(bars.timestamp, utc=True)
    bars["date"] = bars.timestamp.dt.date
    valid = (
        (bars.open > 0)
        & (bars.high >= bars[["open", "close"]].max(axis=1))
        & (bars.low <= bars[["open", "close"]].min(axis=1))
        & (bars.high >= bars.low)
        & (bars.volume >= 0)
    )
    return bars[valid].sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _cusum(z: pd.Series, drift: float = 0.20) -> pd.Series:
    pos = neg = 0.0
    out = np.zeros(len(z), dtype=float)
    for i, value in enumerate(z.fillna(0.0).to_numpy()):
        pos = max(0.0, pos + value - drift)
        neg = min(0.0, neg + value + drift)
        if pos and neg:
            if pos >= -neg:
                neg = 0.0
            else:
                pos = 0.0
        out[i] = pos + neg
    return pd.Series(out, index=z.index)


def daily_features(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = (
        bars.groupby(["symbol", "date"], sort=True)
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
    daily["atr"] = group.tr.transform(lambda x: x.shift().rolling(20, min_periods=15).median())
    daily["ret_vol"] = group.ret.transform(lambda x: x.shift().rolling(20, min_periods=15).std())
    daily["vol_ratio"] = daily.volume / group.volume.transform(
        lambda x: x.shift().rolling(20, min_periods=15).median()
    )
    daily["dollar_liq"] = (
        (daily.close * daily.volume)
        .groupby(daily.symbol)
        .transform(lambda x: x.shift().rolling(20, min_periods=15).median())
    )
    for n in (5, 20, 60):
        daily[f"res{n}"] = group.high.transform(
            lambda x, n=n: x.shift().rolling(n, min_periods=max(4, n // 2)).max()
        )
        daily[f"sup{n}"] = group.low.transform(
            lambda x, n=n: x.shift().rolling(n, min_periods=max(4, n // 2)).min()
        )
    daily["tr5"] = group.tr.transform(lambda x: x.shift().rolling(5, min_periods=4).median())
    daily["loc"] = (daily.close - daily.low) / (daily.high - daily.low).replace(0, np.nan)
    daily["mom20"] = daily.close / group.close.shift(20) - 1
    daily["eff20"] = (daily.close - group.close.shift(20)).abs() / group.close.transform(
        lambda x: x.diff().abs().rolling(20, min_periods=15).sum()
    )
    z = daily.ret / daily.ret_vol.replace(0, np.nan)
    daily["cusum"] = (
        z.groupby(daily.symbol, group_keys=False).apply(_cusum).reset_index(level=0, drop=True)
    )

    benchmark = daily[daily.symbol == "^NSEI"][["date", "ret", "mom20"]].rename(
        columns={"ret": "market_ret", "mom20": "market_mom20"}
    )
    equity = daily[daily.symbol != "^NSEI"].merge(benchmark, on="date", how="left")
    equity["excess20"] = equity.mom20 - equity.market_mom20
    equity["rs_rank"] = equity.groupby("date").excess20.rank(pct=True)
    equity["breadth_up"] = equity.groupby("date").ret.transform(lambda x: (x > 0).mean())
    return equity.sort_values(["symbol", "date"]), benchmark


def generate_setups(d: pd.DataFrame) -> pd.DataFrame:
    spread = (d.high - d.low).replace(0, np.nan)
    upper_wick = (d.high - d[["open", "close"]].max(axis=1)) / spread
    lower_wick = (d[["open", "close"]].min(axis=1) - d.low) / spread
    liquid = d.dollar_liq >= 20_000_000
    setups: list[pd.DataFrame] = []

    def add(
        mask: pd.Series,
        name: str,
        side: int,
        trigger: pd.Series,
        stop_ref: pd.Series,
        level: pd.Series,
    ) -> None:
        eligible = mask & liquid
        cols = ["symbol", "date", "open", "high", "low", "close", "atr", "res60", "sup60"]
        frame = d.loc[eligible, cols].copy()
        frame["setup"] = name
        frame["side"] = side
        frame["trigger"] = trigger[eligible]
        frame["stop_ref"] = stop_ref[eligible]
        frame["level"] = level[eligible]
        setups.append(frame)

    # A high-volume liquidity sweep penetrates a 20-day extreme and closes
    # well back inside it. The next session must break the sweep candle in
    # the reversal direction before an entry is allowed.
    add(
        (d.low < d.sup20 - 0.05 * d.atr)
        & (d.close > d.sup20)
        & (lower_wick >= 0.35)
        & (d.vol_ratio >= 1.00)
        & (d.market_mom20.abs() <= 0.04)
        & d.breadth_up.between(0.35, 0.65),
        "liquidity_sweep_long",
        1,
        d.high,
        d.low - 0.10 * d.atr,
        d.sup20,
    )
    add(
        (d.high > d.res20 + 0.05 * d.atr)
        & (d.close < d.res20)
        & (upper_wick >= 0.45)
        & (d.vol_ratio >= 1.50)
        & (d.market_mom20.abs() <= 0.04)
        & d.breadth_up.between(0.40, 0.60),
        "liquidity_sweep_short",
        -1,
        d.low,
        d.high + 0.10 * d.atr,
        d.res20,
    )
    return pd.concat(setups, ignore_index=True).sort_values(["date", "symbol", "setup"])


def prepare_intraday(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["slot"] = bars.groupby(["symbol", "date"]).cumcount()
    bars["slot_vol_base"] = bars.groupby(["symbol", "slot"], group_keys=False).volume.transform(
        lambda x: x.shift().rolling(20, min_periods=12).median()
    )
    bench = bars[bars.symbol == "^NSEI"][["timestamp", "open", "close"]].copy()
    bench["market_bar_ret"] = bench.close / bench.open - 1
    bars = bars.merge(bench[["timestamp", "market_bar_ret"]], on="timestamp", how="left")
    bars["bar_ret"] = bars.close / bars.open - 1
    return bars[bars.symbol != "^NSEI"].sort_values(["symbol", "timestamp"])


def backtest(
    bars: pd.DataFrame,
    daily: pd.DataFrame,
    setups: pd.DataFrame,
    cfg: Config,
    execution_bars: pd.DataFrame | None = None,
) -> pd.DataFrame:
    confirmation_by = {
        symbol: frame.reset_index(drop=True) for symbol, frame in bars.groupby("symbol", sort=False)
    }
    execution_source = bars if execution_bars is None else execution_bars
    execution_by = {
        symbol: frame.reset_index(drop=True)
        for symbol, frame in execution_source.groupby("symbol", sort=False)
    }
    dates = {s: list(x.date) for s, x in daily.groupby("symbol", sort=False)}
    out = []
    for e in setups.itertuples():
        ds = dates[e.symbol]
        i = ds.index(e.date)
        if i + 1 >= len(ds):
            continue
        next_date = ds[i + 1]
        confirmation = confirmation_by[e.symbol]
        morning = confirmation[confirmation.date == next_date].head(cfg.confirmation_bars)
        if (
            morning.empty
            or abs(float(morning.iloc[0].open) / e.close - 1) > cfg.max_gap_atr * e.atr / e.close
        ):
            continue
        entrybar = None
        for br in morning.itertuples():
            loc = (br.close - br.low) / (br.high - br.low) if br.high > br.low else 0.5
            vol_ok = (
                pd.notna(br.slot_vol_base)
                and br.slot_vol_base > 0
                and br.volume / br.slot_vol_base >= 1.0
            )
            rel = br.bar_ret - br.market_bar_ret if pd.notna(br.market_bar_ret) else 0.0
            if (
                e.side == 1
                and br.close > e.trigger
                and br.close > br.open
                and loc >= 0.65
                and rel >= 0
                and vol_ok
            ):
                entrybar = br
                break
            if (
                e.side == -1
                and br.close < e.trigger
                and br.close < br.open
                and loc <= 0.35
                and rel <= 0
                and vol_ok
            ):
                entrybar = br
                break
        if entrybar is None:
            continue
        entry = float(entrybar.close) * (1 + e.side * 0.0005)
        stop = float(e.stop_ref)
        risk = e.side * (entry - stop)
        if risk <= 0 or risk > cfg.max_risk_atr * e.atr or risk / entry * 100 > cfg.max_risk_pct:
            continue
        outer = e.res60 if e.side == 1 else e.sup60
        room = e.side * (outer - entry) if pd.notna(outer) else np.inf
        if room > 0 and room < 2.0 * risk:
            continue
        target = entry + e.side * 3.0 * risk
        last_date = ds[min(i + cfg.max_holding_sessions, len(ds) - 1)]
        execution = execution_by[e.symbol]
        future = execution[
            (execution.timestamp > entrybar.timestamp) & (execution.date <= last_date)
        ]
        if future.empty:
            continue
        active_stop = stop
        exitp = None
        why = "time"
        exit_ts = None
        high_close = entry
        for br in future.itertuples():
            if e.side == 1:
                if br.low <= active_stop:
                    exitp, why, exit_ts = active_stop, "stop", br.timestamp
                    break
                if br.high >= target:
                    exitp, why, exit_ts = target, "target", br.timestamp
                    break
                high_close = max(high_close, br.close)
                if high_close >= entry + 2 * risk:
                    active_stop = max(active_stop, entry + risk, high_close - 1.25 * e.atr)
                elif high_close >= entry + risk:
                    active_stop = max(active_stop, entry)
            else:
                if br.high >= active_stop:
                    exitp, why, exit_ts = active_stop, "stop", br.timestamp
                    break
                if br.low <= target:
                    exitp, why, exit_ts = target, "target", br.timestamp
                    break
                high_close = min(high_close, br.close)
                if high_close <= entry - 2 * risk:
                    active_stop = min(active_stop, entry - risk, high_close + 1.25 * e.atr)
                elif high_close <= entry - risk:
                    active_stop = min(active_stop, entry)
        if exitp is None:
            exitp = float(future.iloc[-1].close)
            exit_ts = future.iloc[-1].timestamp
        net = e.side * (exitp - entry) / entry * 100 - cfg.friction_pct
        out.append(
            (
                e.symbol,
                e.setup,
                e.side,
                e.date,
                entrybar.timestamp,
                entry,
                stop,
                target,
                exit_ts,
                exitp,
                net,
                e.side * (exitp - entry) / risk,
                why,
            )
        )
    return pd.DataFrame(
        out,
        columns=[
            "symbol",
            "setup",
            "side",
            "setup_date",
            "entry_timestamp",
            "entry",
            "stop",
            "target",
            "exit_timestamp",
            "exit",
            "net_pct",
            "r",
            "exit_reason",
        ],
    )


def report(t: pd.DataFrame) -> None:
    print("trades", len(t))
    for name, x in [("all", t), *list(t.groupby("setup"))]:
        if x.empty:
            continue
        loss = -x.loc[x.net_pct < 0, "net_pct"].sum()
        pf = x.loc[x.net_pct > 0, "net_pct"].sum() / loss if loss else np.inf
        print(
            name,
            "n",
            len(x),
            "win",
            round((x.net_pct > 0).mean() * 100, 1),
            "avg%",
            round(x.net_pct.mean(), 3),
            "PF",
            round(pf, 2),
            "avgR",
            round(x.r.mean(), 3),
        )
    t["year"] = pd.to_datetime(t.setup_date).dt.year
    for year, x in t.groupby("year"):
        loss = -x.loc[x.net_pct < 0, "net_pct"].sum()
        pf = x.loc[x.net_pct > 0, "net_pct"].sum() / loss if loss else np.inf
        print(year, "n", len(x), "avg%", round(x.net_pct.mean(), 3), "PF", round(pf, 2))
    t["month"] = pd.to_datetime(t.setup_date).dt.to_period("M").astype(str)
    m = t.groupby("month").agg(trades=("net_pct", "size"), total=("net_pct", "sum"))
    print(
        "months",
        len(m),
        "median trades",
        m.trades.median(),
        "positive months",
        round((m.total > 0).mean() * 100, 1),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/skytrade-swing-30m.db")
    parser.add_argument("--symbols", help="optional newline-delimited fixed universe")
    parser.add_argument("--output", default="/tmp/swing_30m_v2.csv")
    args = parser.parse_args()
    raw = load_bars(args.db, args.symbols)
    print("bars", len(raw), "symbols", raw.symbol.nunique())
    daily, _ = daily_features(raw)
    setups = generate_setups(daily)
    print("setups", len(setups))
    intraday = prepare_intraday(raw)
    trades = backtest(intraday, daily, setups, Config())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.output, index=False)
    report(trades)

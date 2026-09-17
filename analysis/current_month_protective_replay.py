"""Read-only exact-price replay for current-month protective futures combos.

Kite's live instrument dump only resolves unexpired contracts, so this tool is
limited to the current contract month. It never places an order or writes to the
application database.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import timedelta
from pathlib import Path

import pandas as pd
from kiteconnect import KiteConnect

from trading_scanner.config.settings import load_config


def candle_close(kite: KiteConnect, token: int, when: pd.Timestamp) -> float | None:
    candles = kite.historical_data(
        token,
        when.to_pydatetime() - timedelta(days=1),
        when.to_pydatetime() + timedelta(days=1),
        "30minute",
    )
    target = when.timestamp()
    exact = [row for row in candles if abs(row["date"].timestamp() - target) < 1]
    return float(exact[0]["close"]) if exact else None


def isolated_margin(
    kite: KiteConnect,
    future_symbol: str,
    future_transaction: str,
    option_symbol: str,
    lot_size: int,
) -> float:
    def leg(symbol: str, transaction: str) -> dict[str, str | int]:
        return {
            "exchange": "NFO",
            "tradingsymbol": symbol,
            "transaction_type": transaction,
            "variety": "regular",
            "product": "NRML",
            "order_type": "MARKET",
            "quantity": lot_size,
        }

    result = kite.basket_order_margins(
        [leg(future_symbol, future_transaction), leg(option_symbol, "BUY")],
        consider_positions=False,
    )
    return float(result["final"]["total"])


def replay(trades: pd.DataFrame, kite: KiteConnect, month: str) -> pd.DataFrame:
    instruments = kite.instruments("NFO")
    rows = []
    for trade in trades.itertuples():
        if trade.entry_timestamp.strftime("%Y-%m") != month:
            continue
        name = trade.symbol.removesuffix(".NS")
        futures = [
            item
            for item in instruments
            if item["name"] == name
            and item["instrument_type"] == "FUT"
            and item["expiry"] >= trade.exit_timestamp.date()
        ]
        if not futures:
            rows.append({"symbol": trade.symbol, "status": "no_current_future"})
            continue
        future = min(futures, key=lambda item: item["expiry"])
        option_type = "PE" if trade.side == 1 else "CE"
        options = [
            item
            for item in instruments
            if item["name"] == name
            and item["instrument_type"] == option_type
            and item["expiry"] == future["expiry"]
        ]
        if not options:
            rows.append({"symbol": trade.symbol, "status": "no_current_option"})
            continue
        option = min(options, key=lambda item: abs(float(item["strike"]) - trade.stop))
        future_entry = candle_close(kite, int(future["instrument_token"]), trade.entry_timestamp)
        future_exit = candle_close(kite, int(future["instrument_token"]), trade.exit_timestamp)
        option_entry = candle_close(kite, int(option["instrument_token"]), trade.entry_timestamp)
        option_exit = candle_close(kite, int(option["instrument_token"]), trade.exit_timestamp)
        if None in (future_entry, future_exit, option_entry, option_exit):
            rows.append(
                {
                    "symbol": trade.symbol,
                    "status": "missing_exact_30m_candle",
                    "future": future["tradingsymbol"],
                    "option": option["tradingsymbol"],
                }
            )
            continue
        lot_size = int(future["lot_size"])
        direction = 1 if trade.side == 1 else -1
        future_pnl = direction * (future_exit - future_entry) * lot_size
        option_pnl = (option_exit - option_entry) * lot_size
        margin = isolated_margin(
            kite,
            future["tradingsymbol"],
            "BUY" if trade.side == 1 else "SELL",
            option["tradingsymbol"],
            lot_size,
        )
        rows.append(
            {
                "symbol": trade.symbol,
                "side": "long" if trade.side == 1 else "short",
                "status": "priced",
                "entry_timestamp": trade.entry_timestamp,
                "exit_timestamp": trade.exit_timestamp,
                "exit_reason": trade.exit_reason,
                "future": future["tradingsymbol"],
                "option": option["tradingsymbol"],
                "option_strike": float(option["strike"]),
                "structural_stop": trade.stop,
                "lot_size": lot_size,
                "future_entry": future_entry,
                "future_exit": future_exit,
                "option_entry": option_entry,
                "option_exit": option_exit,
                "future_pnl": future_pnl,
                "option_pnl": option_pnl,
                "combined_pnl": future_pnl + option_pnl,
                "current_isolated_margin": margin,
                "return_on_margin_pct": (future_pnl + option_pnl) / margin * 100,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config()
    if not config.kite_api_key:
        raise RuntimeError("TRADING_SCANNER_KITE_API_KEY is required")
    with sqlite3.connect(f"file:{args.database}?mode=ro", uri=True) as connection:
        token = connection.execute(
            "SELECT access_token FROM kite_session ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if token is None:
        raise RuntimeError("No Kite session")
    kite = KiteConnect(api_key=config.kite_api_key)
    kite.set_access_token(token[0])

    trades = pd.read_csv(
        args.trades,
        parse_dates=["entry_timestamp", "exit_timestamp"],
    )
    result = replay(trades, kite, args.month)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

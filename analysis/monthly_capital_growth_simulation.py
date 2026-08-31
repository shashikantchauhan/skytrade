"""Month-by-month capital growth simulation: same real trade history, same
winning 3-gate stack (entry_quality_filter + ranking + conviction
entry-candle CLV>=0.7 -- see ``capital_simulation_50k_8slots.py`` and
``analysis/conviction_threshold_filter.py``), but instead of a fixed
Rs50,000/slot forever, models the user's actual plan:

  - Slot COUNT stays fixed at 8 (not "add more slots").
  - Slot SIZE grows every calendar month: this month's trading profit
    stays in the account (already true of every other simulation here --
    P&L compounds), PLUS a fresh Rs40,000/month external contribution is
    added on top, and slot size is recomputed as
    ``running_capital / 8`` at the start of each new month. So position
    size drifts up over time (Rs50,000 -> Rs60,000 -> ... as capital
    grows), never the number of concurrent positions.

This is a fork of ``capital_simulation_50k_8slots.py``'s ``simulate()``,
not a call into it -- the fixed-position-size assumption is baked
throughout that function's quantity/capital bookkeeping, so a dynamic
per-month position size needs its own loop. Every gate (eligibility,
quality filter, ranking, conviction) is reused unchanged from that module
and ``application/ranking.py`` -- only the position-sizing/capital-growth
mechanics are new.

Run:
    PYTHONPATH=src python analysis/monthly_capital_growth_simulation.py <db-path> \
      [--contribution 40000] [--start-capital 400000]
"""

import argparse
import asyncio
import statistics
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from capital_simulation_50k_8slots import load_entry_clv  # noqa: E402

from trading_scanner.application.entry_quality_filter import passes_indicator_filter  # noqa: E402
from trading_scanner.application.ranking import (  # noqa: E402
    MIN_EXPECTANCY_TRADES,
    RankedCandidate,
    rank_candidates,
)
from trading_scanner.application.trading_costs import round_trip_cost  # noqa: E402
from trading_scanner.domain.models import SignalSide, Trade  # noqa: E402
from trading_scanner.infrastructure.db import (  # noqa: E402
    TursoTradeRepository,
    create_turso_client,
)

MAX_POSITIONS = 8
MIN_WIN_RATE = Decimal("55")
MIN_CLOSED_TRADES = 5
CONVICTION_THRESHOLD = 0.7


@dataclass(slots=True)
class _OpenPosition:
    symbol: str
    entry_price: Decimal
    quantity: int
    capital_allocated: Decimal


@dataclass(slots=True)
class MonthSnapshot:
    month: str
    capital_start: Decimal
    position_size: Decimal
    contribution_added: Decimal
    trades_taken: int
    wins: int
    losses: int
    pnl: Decimal
    capital_end: Decimal


def simulate_growth(
    buy_trades: list[Trade],
    conviction_clv: dict[tuple[str, str], float],
    *,
    start_capital: Decimal,
    monthly_contribution: Decimal,
    max_positions: int = MAX_POSITIONS,
) -> list[MonthSnapshot]:
    open_positions: dict[str, _OpenPosition] = {}
    closed_by_symbol: dict[str, list[Decimal]] = {}
    capital = start_capital
    position_size = capital / max_positions
    current_month: tuple[int, int] | None = None
    contribution_this_month = Decimal("0")
    month_trades = month_wins = month_losses = 0
    month_pnl = Decimal("0")
    month_start_capital = capital
    snapshots: list[MonthSnapshot] = []

    entries_by_ts: dict = {}
    exits_by_ts: dict = {}
    for trade in buy_trades:
        entries_by_ts.setdefault(trade.entry_timestamp, []).append(trade)
        if trade.exit_timestamp is not None:
            exits_by_ts.setdefault(trade.exit_timestamp, []).append(trade)
    all_ts = sorted(set(entries_by_ts) | set(exits_by_ts))

    def is_eligible(symbol: str) -> bool:
        history = closed_by_symbol.get(symbol, [])
        if len(history) < MIN_CLOSED_TRADES:
            return False
        wins = sum(1 for pnl in history if pnl > 0)
        return Decimal(100 * wins) / len(history) >= MIN_WIN_RATE

    def expectancy_value(symbol: str) -> float | None:
        history = closed_by_symbol.get(symbol, [])
        if len(history) < MIN_EXPECTANCY_TRADES:
            return None
        wins = [float(p) for p in history if p > 0]
        losses = [float(p) for p in history if p < 0]
        win_rate = len(wins) / len(history)
        avg_win = statistics.mean(wins) if wins else 0.0
        avg_loss = statistics.mean(losses) if losses else 0.0
        return win_rate * avg_win + (1 - win_rate) * avg_loss

    def flush_month() -> None:
        if current_month is None:
            return
        snapshots.append(
            MonthSnapshot(
                month=f"{current_month[0]:04d}-{current_month[1]:02d}",
                capital_start=month_start_capital,
                position_size=position_size,
                contribution_added=contribution_this_month,
                trades_taken=month_trades,
                wins=month_wins,
                losses=month_losses,
                pnl=month_pnl,
                capital_end=capital,
            )
        )

    for ts in all_ts:
        month_key = (ts.year, ts.month)
        if month_key != current_month:
            flush_month()
            if current_month is not None:
                # New month rolled over: top up with the fresh monthly
                # contribution, then re-size each of the 8 slots off the
                # new (bigger) capital base -- "increase slot size, not
                # slot count".
                capital += monthly_contribution
                contribution_this_month = monthly_contribution
            else:
                contribution_this_month = Decimal("0")
            position_size = capital / max_positions
            current_month = month_key
            month_start_capital = capital
            month_trades = month_wins = month_losses = 0
            month_pnl = Decimal("0")

        for trade in exits_by_ts.get(ts, []):
            if trade.status == "closed" and trade.pnl_percent is not None:
                closed_by_symbol.setdefault(trade.symbol, []).append(trade.pnl_percent)
            position = open_positions.pop(trade.symbol, None)
            if position is None or trade.exit_price is None:
                continue
            pnl_amount = position.quantity * (trade.exit_price - position.entry_price)
            entry_value = position.quantity * position.entry_price
            exit_value = position.quantity * trade.exit_price
            cost = round_trip_cost(entry_value, exit_value)
            pnl_amount -= cost
            capital += pnl_amount
            month_pnl += pnl_amount
            if pnl_amount > 0:
                month_wins += 1
            else:
                month_losses += 1

        survivors: list[Trade] = []
        for trade in entries_by_ts.get(ts, []):
            if trade.symbol in open_positions:
                continue
            if not is_eligible(trade.symbol):
                continue
            if not passes_indicator_filter(
                trade.volatility_margin_at_entry or 0.0, trade.regime_normalized_at_entry or 0.0
            ):
                continue
            clv = conviction_clv.get((trade.symbol, trade.entry_timestamp.isoformat()))
            if clv is None or clv < CONVICTION_THRESHOLD:
                continue
            survivors.append(trade)

        ordered = survivors
        if len(survivors) > 1:
            candidates = [
                RankedCandidate(
                    symbol=trade.symbol,
                    entry_timestamp=trade.entry_timestamp,
                    entry_price=trade.entry_price,
                    prediction_at_entry=trade.prediction_at_entry,
                    adx=trade.adx_at_entry or 0.0,
                    regime_normalized=trade.regime_normalized_at_entry or 0.0,
                    volatility_margin=trade.volatility_margin_at_entry or 0.0,
                    expectancy=expectancy_value(trade.symbol),
                )
                for trade in survivors
            ]
            by_symbol = {trade.symbol: trade for trade in survivors}
            ordered = [by_symbol[c.symbol] for c in rank_candidates(candidates)]

        for trade in ordered:
            if len(open_positions) >= max_positions:
                continue
            quantity = max(1, int(position_size / trade.entry_price))
            capital_allocated = quantity * trade.entry_price
            open_positions[trade.symbol] = _OpenPosition(
                trade.symbol, trade.entry_price, quantity, capital_allocated
            )
            month_trades += 1

    flush_month()
    return snapshots


def report(
    snapshots: list[MonthSnapshot], start_capital: Decimal, monthly_contribution: Decimal
) -> None:
    print(
        f"{'Month':<9}{'Start cap':>13}{'Slot size':>12}{'+Contrib':>11}"
        f"{'Trades':>8}{'W/L':>8}{'Month P&L':>13}{'End cap':>13}"
    )
    for s in snapshots:
        wl = f"{s.wins}/{s.losses}"
        print(
            f"{s.month:<9}{s.capital_start:>13,.0f}{s.position_size:>12,.0f}"
            f"{s.contribution_added:>11,.0f}{s.trades_taken:>8}{wl:>8}"
            f"{s.pnl:>13,.0f}{s.capital_end:>13,.0f}"
        )

    total_months = len(snapshots)
    total_contributions = sum(s.contribution_added for s in snapshots)
    total_trading_pnl = sum(s.pnl for s in snapshots)
    final_capital = snapshots[-1].capital_end if snapshots else start_capital
    total_wins = sum(s.wins for s in snapshots)
    total_losses = sum(s.losses for s in snapshots)
    total_trades = sum(s.trades_taken for s in snapshots)
    money_in = start_capital + total_contributions

    print("\n=== Summary ===")
    print(f"Months simulated:              {total_months}")
    print(f"Starting capital:              Rs{start_capital:,.0f}")
    print(
        f"Total fresh contributions:     Rs{total_contributions:,.0f}  "
        f"(Rs{monthly_contribution:,.0f}/month)"
    )
    print(f"Total money put in:            Rs{money_in:,.0f}")
    print(f"Total trading P&L (compounded):Rs{total_trading_pnl:,.0f}")
    print(f"Final capital:                 Rs{final_capital:,.0f}")
    win_rate = 100 * total_wins / (total_wins + total_losses)
    print(f"Trades taken:                  {total_trades}  (win rate {win_rate:.1f}%)")
    print(
        f"Slot size grew:                Rs{start_capital / MAX_POSITIONS:,.0f} -> "
        f"Rs{snapshots[-1].position_size:,.0f}"
    )
    print(
        f"Return on money actually put in: "
        f"{100 * (final_capital / money_in - 1):+.1f}%  "
        f"(pure trading skill, strips out that contributions themselves aren't profit)"
    )
    print(
        f"Naive 'final vs starting capital' %: "
        f"{100 * (final_capital / start_capital - 1):+.1f}%  "
        f"(misleading here -- inflated by new money in, not just trading)"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path")
    parser.add_argument("--contribution", type=Decimal, default=Decimal("40000"))
    parser.add_argument("--start-capital", type=Decimal, default=Decimal("400000"))
    args = parser.parse_args()

    client = create_turso_client(f"file:{args.db_path}", None)
    try:
        trade_repository = TursoTradeRepository(client)
        await trade_repository.ensure_schema()
        all_trades = await trade_repository.get_trades(None, "1h")
        buys = [
            t
            for t in all_trades
            if t.side == SignalSide.BUY
            and t.adx_at_entry is not None
            and not t.symbol.startswith("^")
        ]
    finally:
        await client.close()

    clv = load_entry_clv(args.db_path, buys)
    snapshots = simulate_growth(
        buys, clv, start_capital=args.start_capital, monthly_contribution=args.contribution
    )
    report(snapshots, args.start_capital, args.contribution)


if __name__ == "__main__":
    asyncio.run(main())

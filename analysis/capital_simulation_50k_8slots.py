"""One-off: replay the full real BUY trade history through the *actual*
production capital rules -- fixed Rs50,000/position, hard cap of 8
concurrent positions (a plain open-position COUNT cap, exactly what
``live_cash_execution.execute_cash_entry`` checks -- ``len(all_open) >=
max_positions`` -- not the dynamic equity/target_slots formula
``application/capital_constrained_backtest.py`` was built for, which models
the *retired* paper account's sizing, not live cash trading's) -- with and
without the new ``entry_quality_filter`` gate, so the two can be compared
side by side at the real deployed size.

No ranking (production doesn't rank candidates today -- see
``application/ranking.py``'s own docstring: "not wired into
signal_pipeline.py yet"), so ties for a free slot are broken by whichever
signal appears first in the sorted event stream -- the same
first-come-first-served reality live trading actually has.

Real per-trade costs (``application/trading_costs.py``) are applied on
every exit, so the reported P&L is net, not the raw strategy return.

Run:
    PYTHONPATH=src .venv/bin/python analysis/capital_simulation_50k_8slots.py <db-path>
"""

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trading_scanner.application.entry_quality_filter import passes_indicator_filter  # noqa: E402
from trading_scanner.application.trading_costs import round_trip_cost  # noqa: E402
from trading_scanner.domain.models import SignalSide, Trade  # noqa: E402
from trading_scanner.infrastructure.db import (  # noqa: E402
    TursoTradeRepository,
    create_turso_client,
)

POSITION_SIZE = Decimal("50000")
MAX_POSITIONS = 8
INITIAL_CAPITAL = Decimal("400000")
MIN_WIN_RATE = Decimal("55")
MIN_CLOSED_TRADES = 5


@dataclass(slots=True)
class _OpenPosition:
    symbol: str
    entry_price: Decimal
    quantity: int
    capital_allocated: Decimal


@dataclass(slots=True)
class Result:
    label: str
    trades_taken: int = 0
    skipped_ineligible: int = 0
    skipped_quality_filter: int = 0
    skipped_no_slot: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_amount: Decimal = Decimal("0")
    total_costs_paid: Decimal = Decimal("0")
    peak_concurrent: int = 0
    peak_capital_deployed: Decimal = Decimal("0")
    max_drawdown_amount: Decimal = Decimal("0")
    equity_curve: list[tuple[datetime, Decimal]] = field(default_factory=list)

    @property
    def win_rate(self) -> Decimal | None:
        decided = self.wins + self.losses
        return None if decided == 0 else Decimal(100 * self.wins) / decided

    @property
    def final_equity(self) -> Decimal:
        return INITIAL_CAPITAL + self.total_pnl_amount - self.total_costs_paid


def simulate(buy_trades: list[Trade], label: str, apply_quality_filter: bool) -> Result:
    result = Result(label=label)
    open_positions: dict[str, _OpenPosition] = {}
    closed_by_symbol: dict[str, list[Decimal]] = {}
    running_equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL

    entries_by_ts: dict[datetime, list[Trade]] = {}
    exits_by_ts: dict[datetime, list[Trade]] = {}
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

    for ts in all_ts:
        equity_after_exits = running_equity
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
            result.total_costs_paid += cost
            result.total_pnl_amount += pnl_amount
            equity_after_exits += pnl_amount
            if pnl_amount > 0:
                result.wins += 1
            else:
                result.losses += 1

        running_equity = equity_after_exits
        peak_equity = max(peak_equity, running_equity)
        drawdown = peak_equity - running_equity
        result.max_drawdown_amount = max(result.max_drawdown_amount, drawdown)

        for trade in entries_by_ts.get(ts, []):
            if trade.symbol in open_positions:
                continue  # one open position per symbol, matches live behavior
            if not is_eligible(trade.symbol):
                result.skipped_ineligible += 1
                continue
            if apply_quality_filter and not passes_indicator_filter(
                trade.volatility_margin_at_entry or 0.0, trade.regime_normalized_at_entry or 0.0
            ):
                result.skipped_quality_filter += 1
                continue
            if len(open_positions) >= MAX_POSITIONS:
                result.skipped_no_slot += 1
                continue
            quantity = max(1, int(POSITION_SIZE / trade.entry_price))
            capital_allocated = quantity * trade.entry_price
            open_positions[trade.symbol] = _OpenPosition(
                trade.symbol, trade.entry_price, quantity, capital_allocated
            )
            result.trades_taken += 1
            result.peak_concurrent = max(result.peak_concurrent, len(open_positions))
            deployed = sum((p.capital_allocated for p in open_positions.values()), Decimal("0"))
            result.peak_capital_deployed = max(result.peak_capital_deployed, deployed)

        result.equity_curve.append((ts, running_equity))

    return result


def report(result: Result) -> None:
    print(f"\n=== {result.label} ===")
    print(f"Trades taken:                 {result.trades_taken}")
    print(f"Skipped (ineligible symbol):  {result.skipped_ineligible}")
    if result.skipped_quality_filter or "filter" in result.label.lower():
        print(f"Skipped (quality filter):     {result.skipped_quality_filter}")
    print(f"Skipped (no free slot, 8 max):{result.skipped_no_slot:>6}")
    print(f"Wins / Losses:                {result.wins} / {result.losses}")
    wr = result.win_rate
    print(f"Win rate:                     {wr:.1f}%" if wr is not None else "Win rate: n/a")
    print(f"Peak concurrent positions:    {result.peak_concurrent} (cap {MAX_POSITIONS})")
    print(f"Peak capital deployed:        Rs{result.peak_capital_deployed:,.0f}")
    print(f"Total net P&L (after costs):  Rs{result.total_pnl_amount:,.0f}")
    print(f"Total real costs paid:        Rs{result.total_costs_paid:,.0f}")
    print(f"Final equity (from Rs{INITIAL_CAPITAL:,.0f}):  Rs{result.final_equity:,.0f}  "
          f"({100*(result.final_equity/INITIAL_CAPITAL-1):+.1f}%)")
    print(f"Max drawdown (rupees):        Rs{result.max_drawdown_amount:,.0f}")


async def main() -> None:
    db_path = sys.argv[1]
    client = create_turso_client(f"file:{db_path}", None)
    repo = TursoTradeRepository(client)
    all_trades = await repo.get_trades(None, "1h")
    await client.close()

    buys = [
        t for t in all_trades
        if t.side == SignalSide.BUY
        and t.adx_at_entry is not None
        # Sector indices (^NSEBANK etc.) are tracked for signal context only
        # -- they are not real orderable NSE cash instruments, so
        # execute_cash_entry could never actually fund one in production.
        # Excluding them here is required for this to model real capital,
        # not optional the way it was for the earlier win-rate-only study.
        and not t.symbol.startswith("^")
    ]
    print(f"Total BUY signals in history (any status, real stocks only): {len(buys)}")
    print(f"Fixed size Rs{POSITION_SIZE:,.0f}/position, hard cap {MAX_POSITIONS} concurrent, "
          f"Rs{INITIAL_CAPITAL:,.0f} deployed capital.")

    without_filter = simulate(buys, "WITHOUT entry_quality_filter (today's live behavior)", False)
    with_filter = simulate(buys, "WITH entry_quality_filter (both >= median)", True)
    report(without_filter)
    report(with_filter)

    a, b = without_filter, with_filter
    print("\n=== Side by side ===")
    print(f"{'':35}{'Without filter':>16}{'With filter':>16}")
    print(f"{'Trades taken':35}{a.trades_taken:>16}{b.trades_taken:>16}")
    print(f"{'Win rate':35}{f'{a.win_rate:.1f}%':>16}{f'{b.win_rate:.1f}%':>16}")
    print(f"{'Net P&L (Rs)':35}{a.total_pnl_amount:>16,.0f}{b.total_pnl_amount:>16,.0f}")
    print(f"{'Final equity (Rs)':35}{a.final_equity:>16,.0f}{b.final_equity:>16,.0f}")
    print(f"{'Max drawdown (Rs)':35}{a.max_drawdown_amount:>16,.0f}{b.max_drawdown_amount:>16,.0f}")
    print(f"{'Peak concurrent positions':35}{a.peak_concurrent:>16}{b.peak_concurrent:>16}")


if __name__ == "__main__":
    asyncio.run(main())

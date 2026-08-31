"""Replay the full real BUY trade history through the *actual* production
capital rules -- fixed Rs50,000/position, hard cap of 8 concurrent
positions (a plain open-position COUNT cap, exactly what
``live_cash_execution.execute_cash_entry`` checks -- ``len(all_open) >=
max_positions`` -- not the dynamic equity/target_slots formula
``application/capital_constrained_backtest.py`` was built for, which models
the *retired* paper account's sizing, not live cash trading's) -- across
several gate combinations, so each can be compared side by side at the
real deployed size, over the *whole* trade history (no train/test split --
nothing here is fit, every gate's threshold is imported from the module
that actually deploys it, or is a fixed, non-tuned rule).

Gate combinations tested:
  1. WITHOUT entry_quality_filter -- pre-2026-08-25 live behavior, kept
     only as historical reference.
  2. WITH entry_quality_filter -- today's actual live behavior. This is
     the real baseline the other rows are measured against.
  3. + positive symbol expectancy (HARD gate) -- an ADDITIONAL gate, not
     currently deployed for real cash orders. Causally requires this
     symbol's own prior closed-BUY expectancy (win_rate * avg_win +
     loss_rate * avg_loss, ranking.symbol_expectancy's own formula, >=
     MIN_EXPECTANCY_TRADES trades) to be positive -- reject outright if
     not, same all-or-nothing shape as entry_quality_filter.
  4. + ranking (SOFT preference, not a hard gate) -- ``application.
     ranking.score_candidate``, in which per-symbol expectancy is already
     the single highest-weighted factor (1.5, the only factor with real
     walk-forward support per that module's own validation) alongside
     volatility_margin/regime_normalized/adx/prediction_at_entry. Every
     candidate that clears entry_quality_filter still competes for a slot
     -- nothing is rejected outright -- but when a scan cycle produces
     more eligible candidates than free slots, the strongest-scored ones
     win the capital instead of whichever happened to be processed first.
     This is a fundamentally different mechanism from row 3's hard
     exclusion: it only ever changes *who wins when capital is already
     scarce*, never turns away a candidate capital would otherwise sit
     idle for. Not currently wired into the real cash order path either
     -- ``execute_cash_entry`` today is first-come-first-served, no
     ranking step exists before it.

Rows 1-3 keep first-come-first-served ordering (whichever signal appears
first in the sorted event stream) for a free slot, matching what
production actually does today; only row 4 reorders candidates within a
cycle by score before opening positions.

Real per-trade costs (``application/trading_costs.py``) are applied on
every exit, so the reported P&L is net, not the raw strategy return.

Run:
    PYTHONPATH=src .venv/bin/python analysis/capital_simulation_50k_8slots.py <db-path>
"""

import asyncio
import sqlite3
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
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
    initial_capital: Decimal = INITIAL_CAPITAL
    max_positions: int = MAX_POSITIONS
    trades_taken: int = 0
    skipped_ineligible: int = 0
    skipped_quality_filter: int = 0
    skipped_expectancy_filter: int = 0
    skipped_conviction_filter: int = 0
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
        return self.initial_capital + self.total_pnl_amount - self.total_costs_paid


def load_entry_clv(db_path: str, buy_trades: list[Trade]) -> dict[tuple[str, str], float]:
    """Close-location-value ((close-low)/(high-low), 0.5 for a zero-range
    bar) of each BUY's own entry candle -- see
    ``analysis/conviction_threshold_filter.py`` for the full writeup of why
    this is used as a standalone selectivity filter (fewer, stronger
    entries) rather than folded into the Lorentzian model. Read directly
    off the sqlite file (same one the trade repository already opened) --
    a plain join, not worth going through the async Turso client for."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    clv_by_key: dict[tuple[str, str], float] = {}
    for symbol, interval in {(t.symbol, "1h") for t in buy_trades}:
        cur.execute(
            "SELECT timestamp, high, low, close FROM candles WHERE symbol = ? AND interval = ?",
            (symbol, interval),
        )
        for ts, high, low, close in cur.fetchall():
            rng = high - low
            clv_by_key[(symbol, ts)] = 0.5 if rng == 0 else (close - low) / rng
    conn.close()
    return clv_by_key


def simulate(
    buy_trades: list[Trade],
    label: str,
    *,
    apply_quality_filter: bool,
    apply_expectancy_filter: bool = False,
    apply_ranking: bool = False,
    apply_conviction_filter: bool = False,
    conviction_threshold: float = 0.7,
    conviction_clv: dict[tuple[str, str], float] | None = None,
    max_positions: int = MAX_POSITIONS,
    initial_capital: Decimal = INITIAL_CAPITAL,
) -> Result:
    result = Result(label=label, initial_capital=initial_capital, max_positions=max_positions)
    open_positions: dict[str, _OpenPosition] = {}
    closed_by_symbol: dict[str, list[Decimal]] = {}
    running_equity = initial_capital
    peak_equity = initial_capital

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

    def expectancy_value(symbol: str) -> float | None:
        # Same formula as application.ranking.symbol_expectancy, computed
        # causally against this in-memory history instead of a fresh DB
        # query (that function only ever reads *current* state, so it
        # can't be called against a moment in the past). None (not 0) when
        # there isn't enough history yet -- ranking.score_candidate treats
        # None specially (scores at the neutral median, not as "bad").
        history = closed_by_symbol.get(symbol, [])
        if len(history) < MIN_EXPECTANCY_TRADES:
            return None
        wins = [float(p) for p in history if p > 0]
        losses = [float(p) for p in history if p < 0]
        win_rate = len(wins) / len(history)
        avg_win = statistics.mean(wins) if wins else 0.0
        avg_loss = statistics.mean(losses) if losses else 0.0
        return win_rate * avg_win + (1 - win_rate) * avg_loss

    def positive_expectancy(symbol: str) -> bool:
        value = expectancy_value(symbol)
        return value is not None and value > 0

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

        survivors: list[Trade] = []
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
            if apply_expectancy_filter and not positive_expectancy(trade.symbol):
                result.skipped_expectancy_filter += 1
                continue
            if apply_conviction_filter:
                clv = (conviction_clv or {}).get((trade.symbol, trade.entry_timestamp.isoformat()))
                if clv is None or clv < conviction_threshold:
                    result.skipped_conviction_filter += 1
                    continue
            survivors.append(trade)

        ordered = survivors
        if apply_ranking and len(survivors) > 1:
            # Every survivor still competes for a slot -- nothing rejected
            # here, only reordered strongest-first so a tight cycle spends
            # its free slots on the best candidates, not whichever this
            # dict happened to list first.
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
    print(f"Skipped (quality filter):     {result.skipped_quality_filter}")
    print(f"Skipped (expectancy filter):  {result.skipped_expectancy_filter}")
    print(f"Skipped (conviction filter):  {result.skipped_conviction_filter}")
    print(f"Skipped (no free slot, {result.max_positions} max):{result.skipped_no_slot:>6}")
    print(f"Wins / Losses:                {result.wins} / {result.losses}")
    wr = result.win_rate
    print(f"Win rate:                     {wr:.1f}%" if wr is not None else "Win rate: n/a")
    print(f"Peak concurrent positions:    {result.peak_concurrent} (cap {result.max_positions})")
    print(f"Peak capital deployed:        Rs{result.peak_capital_deployed:,.0f}")
    print(f"Total net P&L (after costs):  Rs{result.total_pnl_amount:,.0f}")
    print(f"Total real costs paid:        Rs{result.total_costs_paid:,.0f}")
    print(
        f"Final equity (from Rs{result.initial_capital:,.0f}):  Rs{result.final_equity:,.0f}  "
        f"({100 * (result.final_equity / result.initial_capital - 1):+.1f}%)"
    )
    print(f"Max drawdown (rupees):        Rs{result.max_drawdown_amount:,.0f}")


def side_by_side(results: list[Result]) -> None:
    print("\n=== Side by side ===")
    label_width = 30
    col_width = 15
    header = f"{'':{label_width}}" + "".join(f"{i + 1:>{col_width}}" for i in range(len(results)))
    print(header)
    for i, r in enumerate(results):
        print(f"  [{i + 1}] {r.label}")
    rows = [
        ("Max positions (slots)", lambda r: f"{r.max_positions}"),
        ("Capital required (Rs)", lambda r: f"{r.initial_capital:,.0f}"),
        ("Trades taken", lambda r: f"{r.trades_taken}"),
        ("Win rate", lambda r: f"{r.win_rate:.1f}%" if r.win_rate is not None else "n/a"),
        ("Net P&L (Rs)", lambda r: f"{r.total_pnl_amount:,.0f}"),
        ("Final equity (Rs)", lambda r: f"{r.final_equity:,.0f}"),
        (
            "ROI %",
            lambda r: f"{100 * (r.final_equity / r.initial_capital - 1):+.1f}%",
        ),
        ("Max drawdown (Rs)", lambda r: f"{r.max_drawdown_amount:,.0f}"),
        ("Peak concurrent", lambda r: f"{r.peak_concurrent}"),
        ("Skipped (no slot)", lambda r: f"{r.skipped_no_slot}"),
    ]
    for name, fn in rows:
        print(f"{name:{label_width}}" + "".join(f"{fn(r):>{col_width}}" for r in results))


async def main() -> None:
    db_path = sys.argv[1]
    client = create_turso_client(f"file:{db_path}", None)
    try:
        trade_repository = TursoTradeRepository(client)
        await trade_repository.ensure_schema()
        all_trades = await trade_repository.get_trades(None, "1h")

        buys = [
            t
            for t in all_trades
            if t.side == SignalSide.BUY
            and t.adx_at_entry is not None
            # Sector indices (^NSEBANK etc.) are tracked for signal context
            # only -- not real orderable NSE cash instruments, so
            # execute_cash_entry could never actually fund one in
            # production. Excluding them is required to model real
            # capital, not optional.
            and not t.symbol.startswith("^")
        ]
        print(f"Total BUY signals in history (any status, real stocks only): {len(buys)}")
        print(
            f"Fixed size Rs{POSITION_SIZE:,.0f}/position, hard cap {MAX_POSITIONS} concurrent, "
            f"Rs{INITIAL_CAPITAL:,.0f} deployed capital."
        )
    finally:
        await client.close()

    clv = load_entry_clv(db_path, buys)
    print(f"Entry-candle CLV computed for {len(clv)} (symbol, timestamp) candles.\n")

    results = [
        simulate(
            buys, "WITHOUT entry_quality_filter (pre-2026-08-25 reference)",
            apply_quality_filter=False,
        ),
        simulate(
            buys, "WITH entry_quality_filter (today's actual live behavior)",
            apply_quality_filter=True,
        ),
        simulate(
            buys, "+ positive symbol expectancy (NOT currently a real gate)",
            apply_quality_filter=True, apply_expectancy_filter=True,
        ),
        simulate(
            buys,
            "+ ranking, expectancy-weighted (SOFT -- reorders, never rejects)",
            apply_quality_filter=True, apply_ranking=True,
        ),
        simulate(
            buys,
            "+ conviction filter (entry-candle CLV >= 0.7, NOT currently live)",
            apply_quality_filter=True, apply_conviction_filter=True,
            conviction_threshold=0.7, conviction_clv=clv,
        ),
        simulate(
            buys,
            "+ ranking + conviction (quality + ranking + CLV>=0.7, all three)",
            apply_quality_filter=True, apply_ranking=True,
            apply_conviction_filter=True, conviction_threshold=0.7, conviction_clv=clv,
        ),
    ]
    for r in results:
        report(r)
    side_by_side(results)

    slot_counts = (4, 6, 8, 10, 12, 16, 20, 24)
    slot_results = []
    for n in slot_counts:
        slot_results.append(
            simulate(
                buys, f"{n} slots, first-come-first-served",
                apply_quality_filter=True, max_positions=n, initial_capital=POSITION_SIZE * n,
            )
        )
        slot_results.append(
            simulate(
                buys, f"{n} slots, + ranking",
                apply_quality_filter=True, apply_ranking=True,
                max_positions=n, initial_capital=POSITION_SIZE * n,
            )
        )
        slot_results.append(
            simulate(
                buys, f"{n} slots, + ranking + conviction (CLV>=0.7)",
                apply_quality_filter=True, apply_ranking=True,
                apply_conviction_filter=True, conviction_threshold=0.7, conviction_clv=clv,
                max_positions=n, initial_capital=POSITION_SIZE * n,
            )
        )
    print("\n" + "=" * 70)
    print(
        "SLOT-COUNT SENSITIVITY -- today's live gates (entry_quality_filter), "
        "fixed Rs50,000/position, capital scaled 1:1 with slot count (each "
        "extra slot needs another real Rs50,000 in the account). Each slot "
        "count run three times: first-come-first-served (today's real "
        "behavior), + ranking (expectancy-weighted, strongest candidate wins "
        "a tight cycle instead of whichever processed first), and + ranking "
        "+ conviction (also requires entry-candle CLV >= 0.7 -- fewer but "
        "stronger candidates in the first place)."
    )
    print("=" * 70)
    for r in slot_results:
        report(r)
    side_by_side(slot_results)


if __name__ == "__main__":
    asyncio.run(main())

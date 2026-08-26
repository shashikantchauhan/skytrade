"""Walk-forward screen of candidate entry filters against real BUY trade
history, to find which ones actually cut losing trades out-of-sample rather
than just fitting noise in the same data they're measured on.

Deliberately reuses only what's already accumulated in the ``trades`` table
(live pipeline signals since 2025-06-11, ~6,300 closed BUY trades as of
2026-08-25) -- no AlphaEngine recomputation, so this runs in seconds, not
hours. Recomputing with different AlphaEngine filter toggles (ADX/EMA/SMA/
kernel smoothing) is a separate, much slower follow-up (see
``filter_recompute_sweep.py``) worth doing only if these overlay filters
don't already explain the "too many trades" complaint.

Every candidate is evaluated walk-forward: thresholds/decile cuts are
computed from a TRAIN split only, then applied unchanged to a later TEST
split -- exactly ``train_ranking_model.py``'s own anti-overfitting posture
(that module's caveat: a combined heuristic score only hit AUC 0.527
walk-forward, barely above random). A filter that only looks good in-sample
is worse than no filter at all.

Run:
    TRADING_SCANNER_TURSO_URL="file:<path-to-db>" \
      PYTHONPATH=src .venv/bin/python analysis/bad_trade_filters.py
"""

import asyncio
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sector_mapping import SECTOR_MAP  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trading_scanner.config.settings import load_config  # noqa: E402
from trading_scanner.domain.models import SignalSide  # noqa: E402
from trading_scanner.infrastructure.db import (  # noqa: E402
    TursoTradeRepository,
    create_turso_client,
)

# Split point: everything before this date is TRAIN (used only to compute
# thresholds/decile cuts); everything at or after is TEST (the only place
# performance is actually measured). Chosen to leave a meaningful,
# reasonably recent out-of-sample tail (~3.5 months) while keeping enough
# train data for stable deciles.
TEST_SPLIT = datetime.fromisoformat("2026-05-10T00:00:00+00:00")


@dataclass
class Stats:
    n: int
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    expectancy: float | None
    total_pnl: float | None


def summarize(pnls: list[float]) -> Stats:
    if not pnls:
        return Stats(0, None, None, None, None, None)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return Stats(
        n=len(pnls),
        win_rate=100 * len(wins) / len(pnls),
        avg_win=statistics.mean(wins) if wins else None,
        avg_loss=statistics.mean(losses) if losses else None,
        expectancy=statistics.mean(pnls),
        total_pnl=sum(pnls),
    )


def fmt(stats: Stats, label: str, baseline_n: int) -> str:
    if stats.n == 0:
        return f"{label:<42} n=0 (nothing passed this filter)"
    kept_pct = 100 * stats.n / baseline_n if baseline_n else 0
    return (
        f"{label:<42} n={stats.n:>5} ({kept_pct:5.1f}% kept)  "
        f"win_rate={stats.win_rate:5.1f}%  avg_win={stats.avg_win:6.2f}%  "
        f"avg_loss={stats.avg_loss:6.2f}%  expectancy={stats.expectancy:+.3f}%  "
        f"sum_pnl={stats.total_pnl:+8.1f}%"
    )


def decile_cuts(values: list[float]) -> list[float]:
    values = sorted(values)
    n = len(values)
    return [values[int(n * p / 10)] for p in range(1, 10)]


def percentile_floor(values: list[float], pct: float) -> float:
    values = sorted(values)
    idx = min(int(len(values) * pct / 100), len(values) - 1)
    return values[idx]


async def main() -> None:
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config()
    url = f"file:{db_path}" if db_path else config.turso_database_url
    client = create_turso_client(url, None if db_path else config.turso_auth_token)
    trade_repository = TursoTradeRepository(client)
    all_trades = await trade_repository.get_trades(None, config.candle_interval)
    await client.close()

    buys = [
        t
        for t in all_trades
        if t.side == SignalSide.BUY
        and t.status == "closed"
        and t.pnl_percent is not None
        and t.adx_at_entry is not None  # pre-migration rows lack feature columns
    ]
    train = [t for t in buys if t.entry_timestamp < TEST_SPLIT]
    test = [t for t in buys if t.entry_timestamp >= TEST_SPLIT]
    print(f"Total closed BUY trades with features: {len(buys)}")
    print(f"Train: {len(train)} (before {TEST_SPLIT.date()})  Test: {len(test)} (on/after)")
    print()

    baseline = summarize([float(t.pnl_percent) for t in test])
    print("=== TEST-split baseline (no filter) ===")
    print(fmt(baseline, "baseline", baseline.n))
    print()

    print("=== Candidate filters (thresholds fit on TRAIN, applied to TEST) ===")
    results = []

    # --- volatility_margin floor (percentile floors computed on TRAIN only) ---
    for pct in (30, 50, 70):
        floor = percentile_floor([t.volatility_margin_at_entry for t in train], pct)
        kept = [t for t in test if t.volatility_margin_at_entry >= floor]
        results.append((f"volatility_margin >= p{pct} ({floor:.2f})", kept))

    # --- regime_normalized floor ---
    for pct in (30, 50, 70):
        floor = percentile_floor([t.regime_normalized_at_entry for t in train], pct)
        kept = [t for t in test if t.regime_normalized_at_entry >= floor]
        results.append((f"regime_normalized >= p{pct} ({floor:.2f})", kept))

    # --- combined volatility_margin + regime_normalized, both >= p50 ---
    vm_p50 = percentile_floor([t.volatility_margin_at_entry for t in train], 50)
    rn_p50 = percentile_floor([t.regime_normalized_at_entry for t in train], 50)
    kept = [
        t for t in test
        if t.volatility_margin_at_entry >= vm_p50 and t.regime_normalized_at_entry >= rn_p50
    ]
    results.append(("volatility_margin & regime_normalized both >= p50", kept))

    # --- is_early_signal_flip exclusion (whipsaw entries) ---
    kept = [t for t in test if not t.is_early_signal_flip]
    results.append(("exclude is_early_signal_flip", kept))
    kept_only_flip = [t for t in test if t.is_early_signal_flip]
    results.append(("ONLY is_early_signal_flip (control/comparison)", kept_only_flip))

    # --- ADX floor (already computed at entry, filter not enabled live) ---
    for pct in (30, 50, 70):
        floor = percentile_floor([t.adx_at_entry for t in train], pct)
        kept = [t for t in test if t.adx_at_entry >= floor]
        results.append((f"adx_at_entry >= p{pct} ({floor:.2f})", kept))

    # --- per symbol+side expectancy floor, computed from TRAIN only ---
    train_by_symbol: dict[str, list[float]] = {}
    for t in train:
        train_by_symbol.setdefault(t.symbol, []).append(float(t.pnl_percent))
    symbol_expectancy = {
        s: statistics.mean(p) for s, p in train_by_symbol.items() if len(p) >= 10
    }
    if symbol_expectancy:
        exp_values = list(symbol_expectancy.values())
        exp_median = statistics.median(exp_values)
        kept = [
            t for t in test
            if symbol_expectancy.get(t.symbol) is not None
            and symbol_expectancy[t.symbol] >= exp_median
        ]
        results.append((f"symbol expectancy (train) >= median ({exp_median:+.2f}%)", kept))
        exp_p70 = percentile_floor(exp_values, 70)
        kept = [
            t for t in test
            if symbol_expectancy.get(t.symbol) is not None
            and symbol_expectancy[t.symbol] >= exp_p70
        ]
        results.append((f"symbol expectancy (train) >= p70 ({exp_p70:+.2f}%)", kept))

    # --- sector confirmation: index fired BUY at the same entry_timestamp ---
    index_entries = {
        (t.symbol, t.side, t.entry_timestamp) for t in all_trades if t.status != "abandoned"
    }

    def confirmed(t):
        sector = SECTOR_MAP.get(t.symbol)
        if sector is None:
            return None
        return (sector, t.side, t.entry_timestamp) in index_entries
    kept = [t for t in test if confirmed(t) is True]
    results.append(("sector index confirms same-bar BUY", kept))
    kept_unconf = [t for t in test if confirmed(t) is False]
    results.append(("sector index present but did NOT confirm (control)", kept_unconf))

    # --- stricter eligibility bar sensitivity (win-rate/min-trades gate) ---
    # Recomputed per-trade using only trades *before* this trade's own entry,
    # matching how paper_trading.is_eligible works live (no look-ahead).
    def eligible_at(
        symbol: str, side: SignalSide, at: datetime, min_trades: int, min_wr: float
    ) -> bool:
        prior = [
            float(t.pnl_percent) for t in buys
            if t.symbol == symbol and t.entry_timestamp < at
        ]
        if len(prior) < min_trades:
            return False
        wr = 100 * len([p for p in prior if p > 0]) / len(prior)
        return wr >= min_wr

    for min_trades, min_wr in ((5, 55.0), (5, 60.0), (10, 55.0), (10, 60.0), (15, 60.0)):
        kept = [
            t for t in test
            if eligible_at(t.symbol, t.side, t.entry_timestamp, min_trades, min_wr)
        ]
        results.append((f"eligibility >= {min_wr:.0f}% winrate, >= {min_trades} trades", kept))

    for label, kept_trades in results:
        stats = summarize([float(t.pnl_percent) for t in kept_trades])
        print(fmt(stats, label, baseline.n))

    print()
    print("Reminder: 'kept %' below ~30-40% means this filter would cut most")
    print("candidates -- fine if win_rate/expectancy improved enough to be worth")
    print("the lost trade frequency, not fine if it barely moved either number.")


if __name__ == "__main__":
    asyncio.run(main())

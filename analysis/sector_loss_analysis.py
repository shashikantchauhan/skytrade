"""One-off analysis: cross-reference each losing BUY trade against the
movement of its stock's sector index over the same entry->exit window.

Classifies every loss into:
  - "against_tide"   -- sector index rose while the stock still lost
                         (idiosyncratic/stock-specific failure)
  - "with_sector"     -- sector index also fell (market/sector headwind,
                         not really a stock-picking failure)
  - "choppy"          -- sector index barely moved either way (<0.2%)
                         (signal noise in a directionless market)
  - "no_sector_data"  -- stock has no sector mapping, or the index has no
                         candle covering the trade window

Run after the full backtest (local or hosted) has populated ``trades``.
Reads candles + trades from whichever DB TRADING_SCANNER_TURSO_URL points at.

    TRADING_SCANNER_TURSO_URL="file:local.db" \
      PYTHONPATH=src .venv/bin/python analysis/sector_loss_analysis.py
"""

import asyncio
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sector_mapping import sector_for  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trading_scanner.config.settings import load_config  # noqa: E402
from trading_scanner.infrastructure.db import create_turso_client  # noqa: E402

CHOPPY_THRESHOLD_PCT = 0.2


def parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


async def _index_return(
    client, index_symbol: str, interval: str, start: datetime, end: datetime
) -> float | None:
    """Nearest-candle close-to-close return of an index over [start, end]."""
    r = await client.execute(
        "SELECT timestamp, close FROM candles WHERE symbol = ? AND interval = ? "
        "AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
        [index_symbol, interval, end.isoformat()],
    )
    if not r.rows:
        return None
    end_close = r.rows[0][1]

    r = await client.execute(
        "SELECT timestamp, close FROM candles WHERE symbol = ? AND interval = ? "
        "AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
        [index_symbol, interval, start.isoformat()],
    )
    if not r.rows:
        return None
    start_close = r.rows[0][1]

    if start_close == 0:
        return None
    return (end_close - start_close) / start_close * 100


async def main() -> None:
    config = load_config()
    client = create_turso_client(config.turso_database_url, config.turso_auth_token)

    r = await client.execute(
        "SELECT symbol, entry_timestamp, exit_timestamp, pnl_percent FROM trades "
        "WHERE side = 'buy' AND status = 'closed' AND pnl_percent <= 0"
    )
    losses = r.rows
    print(f"Analyzing {len(losses)} losing BUY trades...")

    buckets: dict[str, list] = defaultdict(list)
    per_sector_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    index_cache: dict[tuple, float | None] = {}

    for i, (symbol, entry_ts, exit_ts, pnl) in enumerate(losses):
        sector = sector_for(symbol)
        if sector is None or exit_ts is None:
            buckets["no_sector_data"].append((symbol, entry_ts, exit_ts, pnl))
            continue

        start = parse(entry_ts)
        end = parse(exit_ts)
        cache_key = (sector, entry_ts, exit_ts)
        if cache_key not in index_cache:
            index_cache[cache_key] = await _index_return(
                client, sector, config.candle_interval, start, end
            )
        idx_return = index_cache[cache_key]

        if idx_return is None:
            buckets["no_sector_data"].append((symbol, entry_ts, exit_ts, pnl))
            continue

        if abs(idx_return) < CHOPPY_THRESHOLD_PCT:
            category = "choppy"
        elif idx_return > 0:
            category = "against_tide"
        else:
            category = "with_sector"

        buckets[category].append((symbol, entry_ts, exit_ts, pnl, idx_return, sector))
        per_sector_counts[sector][category] += 1

        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(losses)} processed")

    print("\n=== Loss classification (all sectors) ===")
    total = len(losses)
    for category in ("against_tide", "with_sector", "choppy", "no_sector_data"):
        count = len(buckets[category])
        pct = 100 * count / total if total else 0
        print(f"{category}: {count} ({pct:.1f}%)")

    print("\n=== Per-sector breakdown ===")
    for sector, counts in sorted(per_sector_counts.items()):
        sector_total = sum(counts.values())
        against = counts.get("against_tide", 0)
        with_sec = counts.get("with_sector", 0)
        choppy = counts.get("choppy", 0)
        print(
            f"{sector}: {sector_total} losses -- "
            f"against_tide={against} ({100*against/sector_total:.0f}%), "
            f"with_sector={with_sec} ({100*with_sec/sector_total:.0f}%), "
            f"choppy={choppy} ({100*choppy/sector_total:.0f}%)"
        )

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())

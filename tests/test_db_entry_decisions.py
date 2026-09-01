from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_scanner.domain.models import EntryDecisionRecord, SignalSide
from trading_scanner.infrastructure.db import TursoEntryDecisionRepository, create_turso_client


def _local_url(tmp_path: Path) -> str:
    return f"file:{tmp_path / 'test.db'}"


def _decision(**overrides) -> EntryDecisionRecord:
    defaults = dict(
        symbol="RELIANCE.NS",
        strategy="alpha_engine",
        signal_timestamp=datetime(2026, 9, 1, 10, 15, tzinfo=UTC),
        signal_side=SignalSide.BUY,
        signal_price=Decimal("1234.50"),
        track_record_passed=True,
        quality_passed=True,
        conviction_passed=False,
        ranking_score=None,
        ranking_passed=None,
        capital_passed=None,
        position_limit_passed=None,
        cutoff_passed=None,
        final_decision="rejected",
        blocked_reason="conviction filter -- weak entry candle",
        created_at=datetime(2026, 9, 1, 10, 16, tzinfo=UTC),
    )
    defaults.update(overrides)
    return EntryDecisionRecord(**defaults)


async def test_record_and_get_recent_round_trip(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoEntryDecisionRepository(client)
        await repository.ensure_schema()

        await repository.record(_decision())
        rows = await repository.get_recent("RELIANCE.NS")

        assert len(rows) == 1
        row = rows[0]
        assert row.symbol == "RELIANCE.NS"
        assert row.track_record_passed is True
        assert row.quality_passed is True
        assert row.conviction_passed is False
        assert row.final_decision == "rejected"
        assert row.blocked_reason == "conviction filter -- weak entry candle"
        assert row.ranking_score is None
        assert row.capital_passed is None
        assert row.intent_id is None
    finally:
        await client.close()


async def test_intent_id_round_trips_when_a_real_order_was_attempted(tmp_path: Path) -> None:
    # 2026-09-01: links this audit-trail row to the exact live_order_legs/
    # broker-order trail it led to -- see EntryDecisionRecord's own
    # docstring.
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoEntryDecisionRepository(client)
        await repository.ensure_schema()

        await repository.record(_decision(final_decision="opened", intent_id="abc123"))
        rows = await repository.get_recent("RELIANCE.NS")

        assert rows[0].intent_id == "abc123"
    finally:
        await client.close()


async def test_get_recent_orders_newest_first_and_respects_limit(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoEntryDecisionRepository(client)
        await repository.ensure_schema()

        for hour in range(3):
            await repository.record(
                _decision(created_at=datetime(2026, 9, 1, 10, hour, tzinfo=UTC))
            )

        rows = await repository.get_recent("RELIANCE.NS", limit=2)

        assert len(rows) == 2
        assert rows[0].created_at > rows[1].created_at
    finally:
        await client.close()


async def test_get_recent_scopes_to_the_requested_symbol(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoEntryDecisionRepository(client)
        await repository.ensure_schema()

        await repository.record(_decision(symbol="RELIANCE.NS"))
        await repository.record(_decision(symbol="TCS.NS"))

        rows = await repository.get_recent("TCS.NS")

        assert len(rows) == 1
        assert rows[0].symbol == "TCS.NS"
    finally:
        await client.close()


async def test_opened_decision_round_trips_ranking_and_null_gate_fields(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoEntryDecisionRepository(client)
        await repository.ensure_schema()

        await repository.record(
            _decision(
                conviction_passed=True,
                ranking_score=Decimal("72.5"),
                ranking_passed=True,
                final_decision="opened",
                blocked_reason=None,
            )
        )
        rows = await repository.get_recent("RELIANCE.NS")

        assert rows[0].ranking_score == Decimal("72.5")
        assert rows[0].ranking_passed is True
        assert rows[0].final_decision == "opened"
        assert rows[0].blocked_reason is None
    finally:
        await client.close()


# --- Phase 12 (DB invariants) -- CHECK constraints -------------------------
# entry_decisions is new to this branch, never yet deployed -- unlike
# live_order_legs, these constraints WILL take effect for real once this
# table is first created in production.


async def test_an_unknown_final_decision_is_rejected(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoEntryDecisionRepository(client)
        await repository.ensure_schema()

        with pytest.raises(Exception, match="CHECK constraint failed"):
            await repository.record(_decision(final_decision="not-a-real-outcome"))
    finally:
        await client.close()


async def test_an_unknown_signal_side_string_is_rejected(tmp_path: Path) -> None:
    # record() always writes a real SignalSide's .value ("buy"/"sell") --
    # this proves the constraint is real by writing raw SQL directly,
    # bypassing the repository's own type safety.
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoEntryDecisionRepository(client)
        await repository.ensure_schema()

        with pytest.raises(Exception, match="CHECK constraint failed"):
            await client.execute(
                "INSERT INTO entry_decisions "
                "(symbol, strategy, signal_timestamp, signal_side, signal_price, "
                " final_decision, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ["RELIANCE.NS", "alpha_engine", "2026-09-01T10:15:00", "buy_and_hold",
                 1234.5, "opened", "2026-09-01T10:16:00"],
            )
    finally:
        await client.close()

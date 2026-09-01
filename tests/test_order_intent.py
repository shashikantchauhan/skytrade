"""Tests for the deterministic order-intent id -- see domain/order_intent.py."""

from datetime import UTC, datetime

from trading_scanner.domain.order_intent import compute_intent_id, new_intent

_TS = datetime(2026, 9, 1, 10, 15, tzinfo=UTC)


def test_same_inputs_always_produce_the_same_intent_id():
    first = compute_intent_id("RELIANCE.NS", "BUY", _TS, "cash")
    second = compute_intent_id("RELIANCE.NS", "BUY", _TS, "cash")
    assert first == second


def test_a_different_symbol_produces_a_different_intent_id():
    assert compute_intent_id("RELIANCE.NS", "BUY", _TS, "cash") != compute_intent_id(
        "TCS.NS", "BUY", _TS, "cash"
    )


def test_a_different_signal_timestamp_produces_a_different_intent_id():
    other_ts = datetime(2026, 9, 1, 11, 15, tzinfo=UTC)
    assert compute_intent_id("RELIANCE.NS", "BUY", _TS, "cash") != compute_intent_id(
        "RELIANCE.NS", "BUY", other_ts, "cash"
    )


def test_a_different_purpose_produces_a_different_intent_id():
    assert compute_intent_id("RELIANCE.NS", "BUY", _TS, "cash") != compute_intent_id(
        "RELIANCE.NS", "BUY", _TS, "primary"
    )


def test_new_intent_bundles_the_computed_id_with_its_inputs():
    intent = new_intent("RELIANCE.NS", "BUY", _TS, "cash")
    assert intent.intent_id == compute_intent_id("RELIANCE.NS", "BUY", _TS, "cash")
    assert intent.symbol == "RELIANCE.NS"
    assert intent.side == "BUY"
    assert intent.signal_timestamp == _TS
    assert intent.purpose == "cash"

"""Tests for the plain domain/gates.py types (GateResult, EntryDecision)."""

from trading_scanner.domain.gates import EntryDecision, GateResult


def test_blocked_reason_is_none_when_every_gate_passed():
    decision = EntryDecision(gates=(GateResult("a", True), GateResult("b", True)))
    assert decision.blocked_reason is None


def test_blocked_reason_reports_the_first_failing_gate_in_order():
    decision = EntryDecision(
        gates=(
            GateResult("a", True),
            GateResult("b", False, reason="b failed"),
            GateResult("c", False, reason="c failed"),
        ),
    )
    assert decision.blocked_reason == "b failed"


def test_blocked_reason_falls_back_to_gate_name_with_no_explicit_reason():
    decision = EntryDecision(gates=(GateResult("capacity", False),))
    assert decision.blocked_reason == "capacity"


def test_allowed_is_derived_true_only_when_every_gate_passed():
    assert EntryDecision(gates=(GateResult("a", True), GateResult("b", True))).allowed is True
    assert EntryDecision(gates=(GateResult("a", True), GateResult("b", False))).allowed is False
    assert EntryDecision(gates=()).allowed is True  # vacuously true, no gates ran

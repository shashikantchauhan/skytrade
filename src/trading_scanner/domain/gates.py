"""Composable entry-gate types -- Phase 1-2 of `projectedPlann.md`'s
architecture refactor (see docs/architecture/000-audit.md).

Pure data types only, no logic: a ``GateResult`` reports one gate's
pass/fail/reason, an ``EntryDecision`` bundles them into a single
allow/block outcome. Concrete gates that wrap this codebase's existing,
already-validated filters (track record, entry-quality, conviction, ...)
live in ``application/entry_gates.py`` -- kept out of this module so
``domain/`` stays free of any dependency on ``application/``-level
repositories or I/O, matching how every other type in ``domain/models.py``
is a plain, dependency-free dataclass.
"""

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class GateResult:
    """One gate's verdict on one candidate. ``reason`` is None when
    ``passed`` is True -- there's nothing to explain. ``metadata`` is for
    anything worth persisting alongside the verdict (e.g. the actual
    win-rate a track-record gate computed) without widening this type's
    fixed fields for every gate that might want to report something extra.
    """

    name: str
    passed: bool
    reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EntryDecision:
    """The combined outcome of every gate run for one candidate.
    ``allowed`` is the caller's own AND of the gates it ran -- this type
    doesn't compute it, since some callers only run a subset of gates (see
    ``application/entry_gates.py``'s module docstring for which gates are
    -- and deliberately are not yet -- wrapped this way) and "allowed"
    means different things depending on which subset ran.

    ``gates`` order matters: ``blocked_reason`` reports the *first*
    failing gate in this tuple, matching every existing call site's own
    "first thing that rejected it" reason string (e.g. entry_quality_filter
    before conviction_filter -- see ``evaluate_cash_quality_gates``).
    """

    allowed: bool
    gates: tuple[GateResult, ...]
    score: Decimal | None = None

    @property
    def blocked_reason(self) -> str | None:
        for gate in self.gates:
            if not gate.passed:
                return gate.reason or gate.name
        return None

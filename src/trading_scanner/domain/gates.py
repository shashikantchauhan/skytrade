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

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class GateResult:
    """One gate's verdict on one candidate. ``reason`` is None when
    ``passed`` is True -- there's nothing to explain.

    2026-09-01: this used to also carry a ``metadata: dict[str, object]``
    field ("for anything worth persisting alongside the verdict") -- a
    mutable dict on an otherwise-frozen dataclass, and one no real gate
    ever actually populated. Removed rather than fixed in place (e.g.
    ``types.MappingProxyType``): faking immutability around a field
    nothing uses yet is needless complexity for its own sake. Trivial to
    re-add, properly immutable, the day a gate actually needs to report
    something beyond pass/fail/reason.
    """

    name: str
    passed: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class EntryDecision:
    """The combined outcome of every gate run for one candidate.
    ``allowed`` is derived (``all(gate.passed for gate in gates)``), not a
    field a caller sets independently -- 2026-09-01: it used to be a
    caller-supplied field, which permitted an internally inconsistent
    ``EntryDecision(allowed=True, gates=(GateResult(..., passed=False),))``
    with nothing to catch it. There is exactly one caller
    (``application/entry_gates.py``'s ``evaluate_cash_quality_gates``) and
    it already passed exactly this AND, so deriving it changes no
    behavior.

    This type deliberately only ever wraps a *subset* of the real gates a
    cash entry runs -- ranking, capital, position-limit, and entry-cutoff
    are NOT gates here (see ``application/entry_gates.py``'s own module
    docstring for why: they're execution-time, re-checked live inside
    ``execute_cash_entry`` for TOCTOU safety, not pre-computed). ``allowed``
    therefore answers "did every gate *this decision actually ran* pass,"
    not "will a real order definitely be placed" -- the caller's own
    subsequent execution-time checks still have the final word.

    ``gates`` order matters: ``blocked_reason`` reports the *first*
    failing gate in this tuple, matching every existing call site's own
    "first thing that rejected it" reason string (e.g. entry_quality_filter
    before conviction_filter -- see ``evaluate_cash_quality_gates``).
    """

    gates: tuple[GateResult, ...]
    score: Decimal | None = None

    @property
    def allowed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def blocked_reason(self) -> str | None:
        for gate in self.gates:
            if not gate.passed:
                return gate.reason or gate.name
        return None

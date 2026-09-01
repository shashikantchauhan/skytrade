"""A stable identity for "the same logical trade" -- Phase 4 of
`projectedPlann.md` (see docs/architecture/000-audit.md).

``basket_id`` (see ``domain/models.py``'s ``LiveOrderLeg``) already groups
the legs of *one function call's* retry attempts, but it's built from
``datetime.now(UTC)`` at call time -- a fresh value every single call, even
across two independent ``execute_cash_entry`` calls for what a human would
recognize as the exact same signal (e.g. one attempt before a process
restart, a second attempt after). There is today no way to look at
``live_order_legs`` and tell "these rows, written by two different process
runs, were both attempts at the same logical trade."

``intent_id`` fixes that: computed deterministically from what actually
identifies a logical trade (symbol, side, the signal's own timestamp,
purpose) -- not a random UUID, and not tied to when an attempt happened.
The same signal always produces the same intent_id, no matter how many
times, or across how many process restarts, it's attempted.

Important scope note: this closes the "duplicate attempt from the same
symbol/signal" gap that's already caught by ``get_unclosed_cash_legs``
today (see ``live_cash_execution.py``), and gives every attempt at one
logical trade a stable, queryable, cross-restart key for the audit trail.
It does NOT, by itself, close the harder gap -- a real order accepted by
Kite but lost before any local DB write ever happened (process crashed
between ``place_order`` returning and ``_record`` running). No local key
can detect that; only checking the broker's own ground truth
(``KiteOrderExecutor.holding_quantity``) before a fresh entry can, and
that's deliberately Stage 6's job (broker reconciliation), not this one --
see ``application/broker_reconciliation.py`` once that stage lands.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class OrderIntent:
    intent_id: str
    symbol: str
    side: str  # "BUY" | "SELL"
    signal_timestamp: datetime
    purpose: str  # matches LiveOrderLeg.purpose: "cash" | "primary" | "hedge"


def compute_intent_id(symbol: str, side: str, signal_timestamp: datetime, purpose: str) -> str:
    """Deterministic id for one logical trade -- same inputs always
    produce the same id, across processes/restarts.

    Deliberately excludes quantity (a retry that recomputes quantity off a
    slightly different reference price is still the same logical trade)
    and excludes any "now" timestamp (that's exactly what would make two
    attempts at the same signal diverge). ``signal_timestamp`` is the
    candle/signal's own timestamp -- stable across retries and restarts,
    unlike wall-clock time when an attempt happens.
    """
    raw = f"{purpose}:{symbol}:{side}:{signal_timestamp.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def new_intent(symbol: str, side: str, signal_timestamp: datetime, purpose: str) -> OrderIntent:
    return OrderIntent(
        intent_id=compute_intent_id(symbol, side, signal_timestamp, purpose),
        symbol=symbol,
        side=side,
        signal_timestamp=signal_timestamp,
        purpose=purpose,
    )

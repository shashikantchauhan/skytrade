"""Order/basket + position lifecycle types -- Phases 5-6 of
`projectedPlann.md` (see docs/architecture/000-audit.md).

Pure, read-side types layered on top of the existing ``LiveOrderLeg``
ledger -- nothing here changes how a leg gets written (``live_cash_
execution.py``/``gtt_bracket.py`` are untouched), only how a set of
already-recorded legs is *interpreted*. Deliberately not a rewrite of the
storage layer: the DB column stays a plain ``TEXT`` status, and
``LegStatus`` is a compatibility adapter over it (its members' values are
exactly Kite's own status vocabulary this codebase already stores, so
``LegStatus(row.status)`` parses any existing row unchanged -- no
migration, no re-tagging historical data).

``derive_position_lifecycle`` is the type that matters most: it's what
finally answers "does a real position exist for this tradingsymbol right
now" as a single, explicit value instead of the implicit answer buried in
which of several near-identical repository methods a caller happened to
pick (``get_open_cash_legs`` vs ``get_unclosed_cash_legs`` vs
``get_all_open_cash_legs``) -- see docs/architecture/000-audit.md's gap #1
for exactly the bug that inconsistency caused. Stage 6's broker
reconciliation service is what actually wires this in to fix it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from trading_scanner.domain.models import LiveOrderLeg


class LegStatus(StrEnum):
    """Kite's own terminal/non-terminal order-status vocabulary, as
    already stored in ``live_order_legs.status`` -- member values equal
    the exact strings this codebase has always written, so parsing an
    existing row is just ``LegStatus(row.status)``, never a translation
    table."""

    COMPLETE = "COMPLETE"
    OPEN = "OPEN"
    UNKNOWN = "UNKNOWN"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal_failure(self) -> bool:
        """Never resulted in, and never will result in, a real position --
        safe to retry past (see live_cash_execution.py's retry loop)."""
        return self in (LegStatus.REJECTED, LegStatus.CANCELLED)

    @property
    def may_represent_a_real_position(self) -> bool:
        """A real position might exist because of this leg -- COMPLETE
        certainly does, OPEN/UNKNOWN might (fill unconfirmed either way).
        The broader set ``get_unclosed_cash_legs`` already uses for entry-
        blocking; ``derive_position_lifecycle`` below uses the same set
        for the *exit* side too, closing the gap ``get_open_cash_legs``
        (COMPLETE-only) left open."""
        return not self.is_terminal_failure


class PositionLifecycle(StrEnum):
    """The one place "does a real position exist for this tradingsymbol"
    gets decided, instead of implicitly depending on which repository
    method a caller reaches for."""

    NONE = "none"  # never entered, or every entry attempt failed outright
    OPENING = "opening"  # an entry is in flight, no COMPLETE leg yet
    ACTIVE = "active"  # a real position is open, no exit in flight
    EXIT_PENDING = "exit_pending"  # active position, an exit is in flight
    CLOSED = "closed"  # fully closed by COMPLETE exit quantity
    RECONCILIATION_REQUIRED = "reconciliation_required"  # an UNKNOWN leg -- broker truth needed


@dataclass(frozen=True, slots=True)
class OrderBasket:
    """One ``basket_id``'s legs (typically one entry attempt's retries, or
    one exit attempt) -- not a position; see ``derive_position_lifecycle``
    for the full tradingsymbol-level picture across every basket."""

    basket_id: str
    legs: tuple[LiveOrderLeg, ...]

    @property
    def outcome(self) -> LegStatus:
        """This basket's own terminal outcome -- the last leg's status if
        one exists that isn't a terminal failure (a retry loop's final,
        real attempt), otherwise the last leg tried (every attempt
        failed)."""
        for leg in reversed(self.legs):
            status = LegStatus(leg.status)
            if not status.is_terminal_failure:
                return status
        return LegStatus(self.legs[-1].status)


def group_into_baskets(legs: Sequence[LiveOrderLeg]) -> list[OrderBasket]:
    """Groups an unordered leg sequence into one ``OrderBasket`` per
    ``basket_id``, each basket's legs kept in their original relative
    order (already ``placed_at`` order at every existing call site)."""
    order: list[str] = []
    by_basket: dict[str, list[LiveOrderLeg]] = {}
    for leg in legs:
        if leg.basket_id not in by_basket:
            order.append(leg.basket_id)
            by_basket[leg.basket_id] = []
        by_basket[leg.basket_id].append(leg)
    return [OrderBasket(basket_id, tuple(by_basket[basket_id])) for basket_id in order]


def derive_position_lifecycle(legs: Sequence[LiveOrderLeg]) -> PositionLifecycle:
    """The lifecycle state of one tradingsymbol's real cash position, from
    every ``purpose='cash'`` leg recorded for it (any basket, any status,
    ``placed_at``-ordered) -- the single function every exit-eligibility
    check (strategy exit, manual exit, the dashboard's positions view)
    should end up calling through, replacing today's inconsistent mix of
    ``get_open_cash_legs``/``get_unclosed_cash_legs``/
    ``get_all_open_cash_legs`` call sites picking different status sets
    for what should be the same question.

    The "opening" transaction_type is whichever the first leg used --
    always BUY for this app's real cash trades (see ``_net_unclosed_legs``
    in infrastructure/db/live_orders.py, the netting logic this mirrors).
    """
    if not legs:
        return PositionLifecycle.NONE

    ordered = sorted(legs, key=lambda leg: leg.placed_at)
    opening_type = ordered[0].transaction_type

    # Any UNKNOWN leg anywhere in the still-relevant history means local
    # records alone can't answer "is this open" -- broker ground truth is
    # needed (Stage 6). Checked first: this must win over every other
    # state below, since none of them are trustworthy while one exists.
    if any(LegStatus(leg.status) is LegStatus.UNKNOWN for leg in ordered):
        return PositionLifecycle.RECONCILIATION_REQUIRED

    openers = [
        leg for leg in ordered
        if leg.transaction_type == opening_type
        and LegStatus(leg.status).may_represent_a_real_position
    ]
    if not openers:
        return PositionLifecycle.NONE

    in_flight_openers = [leg for leg in openers if LegStatus(leg.status) is LegStatus.OPEN]
    complete_openers = [leg for leg in openers if LegStatus(leg.status) is LegStatus.COMPLETE]

    closed_quantity = sum(
        leg.quantity for leg in ordered
        if leg.transaction_type != opening_type and LegStatus(leg.status) is LegStatus.COMPLETE
    )
    in_flight_closers = any(
        leg.transaction_type != opening_type and LegStatus(leg.status) is LegStatus.OPEN
        for leg in ordered
    )

    complete_quantity = sum(leg.quantity for leg in complete_openers)
    remaining = complete_quantity - closed_quantity

    if remaining > 0:
        return PositionLifecycle.EXIT_PENDING if in_flight_closers else PositionLifecycle.ACTIVE
    if in_flight_openers:
        # No net-open COMPLETE quantity left uncovered, but an entry is
        # still mid-flight (e.g. mid-retry-loop, OPEN awaiting fill).
        return PositionLifecycle.OPENING
    return PositionLifecycle.CLOSED

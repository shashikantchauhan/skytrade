# Architecture decision records

Short, dated records of real production incidents and the decisions that
fixed them -- Phase 18 of `projectedPlann.md`. Each one points back to the
actual code's own comment/docstring for full detail rather than
duplicating it; these are the short, scannable version.

- [001-live-cash-runtime-state.md](001-live-cash-runtime-state.md)
- [002-broker-reconciliation.md](002-broker-reconciliation.md)
- [003-order-idempotency.md](003-order-idempotency.md)
- [004-market-order-protection.md](004-market-order-protection.md)
- [005-entry-retry-policy.md](005-entry-retry-policy.md)
- [006-broker-crash-window.md](006-broker-crash-window.md)

Note on scope: the in-code comments these summarize were deliberately
**not** trimmed down to bare pointers, despite `projectedPlann.md`'s
Phase 18 suggesting that. This codebase's own engineering culture leans
heavily on keeping incident reasoning right next to the code it explains
-- proven valuable throughout the review that started this refactor. That
reasoning staying in both places (the short version here, the full
version still inline) is strictly better than removing something already
proven useful just to match a generic instruction that assumed comments
were bloating the code. See `docs/architecture/000-audit.md`'s stage
notes for the explicit reasoning.

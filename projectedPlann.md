# SkyTrade — Full Architecture & Reliability Refactor

You are working on the **SkyTrade** Python trading system.

Your task is to perform a **full-project software architecture, reliability, maintainability, and observability refactor** based on the current repository state.

## CRITICAL RULE #1 — DO NOT CHANGE TRADING STRATEGY BEHAVIOUR

This is the most important constraint.

Do **not** change:

* AlphaEngine logic
* Pine/TradingView strategy logic
* signal generation semantics
* existing ranking calculations
* conviction filter thresholds
* entry-quality filter behaviour
* track-record eligibility rules
* position sizing behaviour
* stop-loss behaviour
* trailing-stop behaviour
* GTT target/stop behaviour
* cash-entry cutoff behaviour
* existing live-trading safety gates
* futures-paper strategy behaviour
* options-shadow strategy behaviour

Do not "improve" a trading rule unless explicitly requested later.

The goal of this project is to improve the **software architecture around the strategy**, not to optimize the strategy itself.

The current behaviour should remain functionally equivalent.

Run the complete test suite after every significant phase and investigate any behavioural change.

---

# PHASE 0 — FULL REPOSITORY AUDIT

Before modifying code:

1. Inspect the entire repository.
2. Understand:

   * application modules
   * domain models
   * infrastructure
   * DB repositories
   * live pipeline
   * signal pipeline
   * cash execution
   * futures paper
   * options shadow
   * paper trading
   * GTT management
   * dashboard
   * authentication
   * notifications
   * deployment
   * CI/CD
   * tests
   * research/analysis tools
3. Read recent git history to understand why existing protections were introduced.
4. Identify:

   * duplicated business logic
   * God modules
   * state inconsistencies
   * implicit state transitions
   * broker/internal-state assumptions
   * fragile retry behaviour
   * race conditions
   * missing idempotency
   * missing DB invariants
   * insufficient live-pipeline testing
   * deployment weaknesses
   * observability gaps.

Do not start refactoring immediately.

First produce an internal architecture map.

---

# PHASE 1 — ESTABLISH CLEAN DOMAIN BOUNDARIES

The current system has concepts such as:

* Signal
* Trade
* Candidate
* PaperPosition
* FuturesPaperPosition
* OptionsShadowTrade
* FuturesShadowTrade
* LiveOrderLeg
* GttBracket
* PaperBenchmarkPosition

Keep existing concepts where they are valid, but introduce cleaner lifecycle boundaries.

The desired conceptual flow is:

```text
Signal
   ↓
Candidate
   ↓
EntryDecision
   ↓
OrderIntent
   ↓
Order/Basket
   ↓
OrderExecution
   ↓
Position
   ↓
PositionLifecycle
```

Do not force inheritance if composition is cleaner.

Prefer explicit domain objects over dictionaries and status strings where practical.

---

# PHASE 2 — CREATE AN ENTRY-GATE FRAMEWORK

The system now has multiple entry gates.

Examples include:

* track-record eligibility
* entry-quality filter
* conviction filter
* ranking
* symbol allowlist
* capital availability
* max-position limit
* entry cutoff
* live kill switch
* existing-position protection

Do NOT change their logic.

Instead, introduce a composable gate abstraction.

For example:

```python
class EntryGate(Protocol):
    def evaluate(self, candidate: Candidate) -> GateResult:
        ...
```

And:

```python
@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
```

Then create:

```python
@dataclass(frozen=True)
class EntryDecision:
    allowed: bool
    score: Decimal | None
    gates: tuple[GateResult, ...]
```

The exact implementation can differ if a better design exists.

The important requirements are:

* gates are independently testable
* gates are composable
* each gate reports pass/fail
* each gate provides a useful reason
* final decision is deterministic
* existing gate behaviour remains unchanged.

The pipeline should eventually be conceptually:

```text
Candidate
   ↓
EntryGateEngine
   ├── TrackRecordGate
   ├── EntryQualityGate
   ├── ConvictionGate
   ├── RankingGate
   ├── CapitalGate
   ├── PositionLimitGate
   └── TimeCutoffGate
   ↓
EntryDecision
```

---

# PHASE 3 — PERSIST ENTRY DECISIONS

Create an auditable decision record.

The system should be able to answer:

> "Why was this signal not traded?"

without inspecting logs manually.

Introduce an `entry_decisions` table or equivalent.

At minimum capture:

```text
id
symbol
strategy
signal_timestamp
signal_side
signal_price

ranking_score

track_record_passed
quality_passed
conviction_passed
ranking_passed
capital_passed
position_limit_passed
cutoff_passed

final_decision
blocked_reason

created_at
```

Prefer a structured representation if appropriate.

The exact schema may differ, but the following must be possible:

```text
Signal
→ every gate result
→ final decision
→ reason
```

Do not persist sensitive credentials.

---

# PHASE 4 — INTRODUCE ORDER INTENT + IDEMPOTENCY

Introduce an explicit `OrderIntent`.

Conceptually:

```python
@dataclass(frozen=True)
class OrderIntent:
    intent_id: str
    symbol: str
    side: str
    quantity: int
    signal_timestamp: datetime
    strategy: str
    purpose: str
```

Every attempt to execute the same logical trade must reference the same `intent_id`.

This is particularly important for:

* retries
* UNKNOWN order states
* process restarts
* network failures
* broker API timeouts.

The goal is to prevent:

```text
UNKNOWN
   ↓
retry
   ↓
duplicate real order
```

Do not assume that retrying is safe merely because the previous API call returned an error.

---

# PHASE 5 — FORMALIZE ORDER/BASKET STATE

`LiveOrderLeg` is already useful.

Keep it, but introduce a first-class aggregate where appropriate:

```text
OrderBasket
 ├── basket_id
 ├── symbol
 ├── strategy
 ├── purpose
 ├── status
 └── legs
```

Support explicit lifecycle states such as:

```text
CREATED
SUBMITTED
OPEN
PARTIALLY_FILLED
COMPLETE
REJECTED
CANCELLED
UNKNOWN
RECONCILIATION_REQUIRED
```

Use enums rather than arbitrary strings where practical.

Do not blindly rename every existing DB status if compatibility would be risky.

Create compatibility adapters/mappings if necessary.

---

# PHASE 6 — FORMALIZE POSITION LIFECYCLE

Introduce a clean position lifecycle.

Conceptually:

```text
OPENING
ACTIVE
EXIT_PENDING
CLOSED
RECONCILIATION_REQUIRED
```

Do not allow business logic to assume:

```text
COMPLETE order == position definitely exists
```

or:

```text
GTT active == position definitely exists
```

Broker state must remain authoritative for actual real-money holdings.

---

# PHASE 7 — CREATE BROKER RECONCILIATION SERVICE

Create a dedicated:

```text
BrokerReconciliationService
```

Its responsibility is to compare:

```text
Broker state
     vs
Internal state
```

for:

* holdings
* positions
* orders
* real cash positions
* GTT/protective orders where applicable.

It should produce explicit discrepancies.

Example:

```text
PERSISTENT.NS

Internal quantity: 9
Broker quantity:   9
Result: MATCH
```

or:

```text
UNIONBANK.NS

Internal quantity: 26
Broker quantity:   0
Result: BROKER_FLAT
Action: reconcile internal position
```

Important:

* broker holdings/positions are the source of truth for actual ownership
* GTT status must not be treated as proof that a position exists
* reconciliation must be idempotent
* reconciliation must not accidentally place duplicate orders
* reconciliation must not sell a position when broker quantity is zero.

Centralize this logic rather than duplicating broker checks throughout execution functions.

---

# PHASE 8 — REFACTOR `signal_pipeline.py`

`signal_pipeline.py` has accumulated too many responsibilities.

Turn it into an orchestrator.

The desired architecture is approximately:

```text
PipelineOrchestrator
    ↓
SignalEvaluationService
    ↓
CandidateSelectionService
    ↓
EntryGateEngine
    ↓
CapitalAllocationService
    ↓
ExecutionService
    ↓
ReconciliationService
    ↓
Notification/EventPublisher
```

Possible modules:

```text
application/
    pipeline/
        orchestrator.py
        evaluation.py
        candidate_selection.py
        entry_decision.py
        capital_allocation.py
        lifecycle.py
        events.py
```

Do not split files merely for the sake of having more files.

Each module must have a clear responsibility.

The final orchestrator should read roughly like:

```python
async def process_cycle(...):
    signals = evaluate(...)
    candidates = select_candidates(signals)
    decisions = evaluate_entries(candidates)
    approved = select_approved(decisions)
    intents = create_order_intents(approved)
    results = execute(intents)
    await reconcile(results)
    await publish_events(...)
```

Existing behaviour must remain unchanged.

---

# PHASE 9 — REFACTOR LIVE TICKER PIPELINE

The live ticker pipeline currently handles:

* KiteTicker
* tick queue
* candle aggregation
* hourly boundary detection
* live signals
* stop-loss
* trailing stop
* DB state
* token refresh
* watchdog
* paper positions
* live cash
* futures paper.

Keep the functionality but separate infrastructure concerns from application logic.

Target:

```text
KiteTickerAdapter
        ↓
TickStream
        ↓
CandleAggregator
        ↓
CandleCloseEvent
        ↓
PipelineOrchestrator
```

The ticker adapter should know about Kite.

The application pipeline should not need to know about KiteTicker callback mechanics.

Do not change candle semantics.

---

# PHASE 10 — ADD LIVE PIPELINE TEST HARNESS

This is mandatory.

Create proper tests for:

### Connection

* successful connection
* disconnect
* reconnect
* silent ticker hang
* token expiration
* missing token
* fresh token
* exhausted reconnect attempts

### Tick processing

* valid tick
* unknown instrument
* missing price
* multiple ticks
* duplicate ticks
* out-of-order ticks

### Candle aggregation

* first tick
* same candle
* boundary tick
* hourly close
* missing ticks
* restart around candle boundary

### Execution

* valid signal
* rejected order
* cancelled order
* UNKNOWN order
* retry
* duplicate execution attempt
* process failure after broker submission
* broker fill before internal DB write

### Reconciliation

* internal == broker
* internal open / broker flat
* internal flat / broker open
* quantity mismatch
* stale GTT
* duplicate internal records

The tests must use fake broker/ticker implementations.

Never use real money or live broker credentials.

---

# PHASE 11 — APP CONFIGURATION REFACTOR

`AppConfig` has grown too large.

Group configuration conceptually into:

```python
MarketConfig
StrategyConfig
DatabaseConfig
NotificationConfig
PaperTradingConfig
LiveCashConfig
FuturesConfig
```

Then:

```python
AppConfig(
    market=...,
    strategy=...,
    database=...,
    ...
)
```

However:

**Do not perform a risky mass rewrite just to achieve this shape.**

If compatibility is easier, introduce grouped config objects incrementally.

The important requirement is:

```text
static configuration
        ≠
runtime trading state
        ≠
broker state
        ≠
strategy state
```

Do not reintroduce `dataclasses.replace(AppConfig, ...)` for runtime trading state.

Keep runtime cash controls explicitly represented by `LiveCashToggleState` or an equivalent dedicated runtime object.

---

# PHASE 12 — DATABASE INVARIANTS

Add DB-level constraints wherever safe.

Examples:

```text
UNIQUE(intent_id)
UNIQUE(order_id)
```

and appropriate checks for:

```text
quantity > 0
valid status
valid transaction type
```

Use DB constraints as the final safety layer.

Do not add constraints that break legitimate historical data without providing a migration strategy.

Review existing production data assumptions before changing schema constraints.

---

# PHASE 13 — EVENT / AUDIT SYSTEM

Introduce domain events where useful:

```text
SignalGenerated
CandidateCreated
CandidateRejected
EntryApproved
OrderIntentCreated
OrderSubmitted
OrderFilled
OrderRejected
OrderCancelled
OrderUnknown
PositionOpened
PositionClosed
BrokerDiscrepancyDetected
ReconciliationCompleted
```

These events should make it possible to reconstruct what happened.

Do not over-engineer this into Kafka or microservices.

A lightweight in-process event model plus persisted audit records is enough.

---

# PHASE 14 — OBSERVABILITY

Improve logging and health visibility.

Introduce structured event logging where practical.

Every important execution event should contain enough context to identify:

```text
symbol
strategy
intent_id
basket_id
order_id
signal_timestamp
decision
gate result
quantity
price
status
```

Do not log:

* API secrets
* access tokens
* passwords
* sensitive authentication data.

---

# PHASE 15 — SYSTEM HEALTH

Expand the existing pipeline health concept.

Create a health model covering:

```text
Process
Ticker
Market Data
Candle Freshness
Database
Broker Authentication
Execution
Reconciliation
```

Conceptually:

```text
SYSTEM HEALTH

Pipeline          HEALTHY
Kite              CONNECTED
Market Feed       FRESH
Database          HEALTHY
Candles           CURRENT
Broker            AUTHENTICATED
Execution         ENABLED
Reconciliation    CLEAN
```

Expose this to the dashboard through a clean API.

Do not make health checks themselves capable of placing orders.

---

# PHASE 16 — DASHBOARD REFACTOR

`webapp.py` currently contains too many concerns.

Gradually split into:

```text
web/
    routes/
        auth.py
        dashboard.py
        trading.py
        kite.py
        backtest.py

    services/
        dashboard_service.py
        portfolio_service.py
        health_service.py
```

Move large inline HTML/CSS/JS out of Python where practical:

```text
templates/
    login.html
    dashboard.html
```

Do not introduce React/Vue/etc. unless there is a strong reason.

Vanilla JS + FastAPI/Jinja is sufficient.

Preserve the existing dashboard functionality.

---

# PHASE 17 — RESEARCH / ANALYSIS CLEANUP

Separate production code from research tooling.

Where appropriate:

```text
research/
    experiments/
    reports/
    backtests/
```

Production application code remains under:

```text
src/
```

Do not delete useful research scripts merely because they are not production code.

Preserve reproducibility.

---

# PHASE 18 — ARCHITECTURE DECISION RECORDS

Move historical incident explanations out of huge production-code comments.

Keep concise comments in code.

Create documentation such as:

```text
docs/
    architecture/
    decisions/
    incidents/
```

Examples:

```text
001-live-cash-runtime-state.md
002-broker-reconciliation.md
003-order-idempotency.md
004-market-order-protection.md
005-entry-retry-policy.md
```

Document:

* problem
* options considered
* chosen approach
* reason
* consequences.

Do not remove useful safety comments until their reasoning has been preserved elsewhere.

---

# PHASE 19 — CI/CD HARDENING

Review the deployment workflow.

The deployment currently pulls code and restarts services.

Ensure production dependencies are synchronized when `pyproject.toml` changes.

The deployment process should conceptually be:

```text
pull
↓
install/sync production dependencies
↓
validate configuration
↓
restart service
↓
verify service is active
↓
verify health endpoint
```

Deployment must fail if the new service does not become healthy.

Do not deploy if tests fail.

Do not expose secrets in CI logs.

---

# PHASE 20 — DO NOT TURN THIS INTO MICROSERVICES

Keep SkyTrade as a **modular monolith**.

Do NOT create unnecessary services such as:

```text
signal-service
ranking-service
execution-service
database-service
notification-service
```

The system is better served by strong internal module boundaries.

---

# PHASE 21 — BACKWARD COMPATIBILITY

Be extremely careful with:

* existing DB tables
* historical trades
* existing live-order records
* dashboard APIs
* environment variables
* systemd service names
* CLI commands
* Telegram notifications
* existing production deployment paths.

Do not perform destructive migrations.

If schema changes are required:

1. create migration
2. preserve existing data
3. support old rows
4. migrate safely
5. add tests.

---

# PHASE 22 — TESTING REQUIREMENTS

After each major phase:

```bash
pytest
ruff check .
```

Before completion:

* full test suite passes
* ruff passes
* no new warnings
* all migrations tested
* live execution remains disabled in tests
* no real broker calls occur during tests.

Add tests for every new abstraction.

Especially test failure cases, not only happy paths.

---

# PHASE 23 — FINAL REVIEW

After implementation, perform a second independent review.

Look specifically for:

### Race conditions

```text
check → act
```

patterns.

### Broker ambiguity

```text
UNKNOWN
OPEN
COMPLETE
```

handling.

### Duplicate orders

Check:

```text
intent_id
basket_id
order_id
symbol
```

relationships.

### Process crashes

Ask:

> What happens if the process dies immediately after every external broker operation?

### DB failures

Ask:

> What happens if the broker succeeds but the DB write fails?

### Broker failures

Ask:

> What happens if the DB says an order exists but Kite says it doesn't?

### Restart behaviour

Ask:

> What happens if SkyTrade restarts in the middle of a trade lifecycle?

### Time boundaries

Check:

* market open
* market close
* entry cutoff
* hourly candle boundary
* reconnect
* token refresh.

---

# IMPORTANT IMPLEMENTATION STYLE

Do not blindly rewrite large portions of the project.

Prefer:

```text
understand
→ isolate
→ introduce abstraction
→ migrate callers
→ test
→ remove old path
```

Keep commits logically separated.

Recommended commit sequence:

```text
1. architecture/audit groundwork
2. entry decision + gate abstraction
3. decision persistence
4. order intent/idempotency
5. order/basket state
6. broker reconciliation
7. pipeline decomposition
8. live ticker test harness
9. config decomposition
10. observability/health
11. dashboard decomposition
12. CI/CD hardening
13. documentation cleanup
```

Do not combine unrelated changes into one giant commit.

---

# DEFINITION OF DONE

The refactor is complete only when:

1. Trading strategy behaviour is unchanged.
2. Existing safety gates remain intact.
3. Entry decisions are explainable.
4. Every real execution has an identifiable intent.
5. Retries are idempotent.
6. Order lifecycle is explicit.
7. Position lifecycle is explicit.
8. Broker reconciliation is centralized.
9. Live pipeline has meaningful automated tests.
10. `signal_pipeline.py` is substantially smaller and easier to understand.
11. `webapp.py` is substantially smaller and cleaner.
12. Configuration separates static config from runtime state.
13. Important DB invariants are enforced.
14. Health status is observable.
15. Deployment verifies service health.
16. Research code is clearly separated from production code.
17. Historical architectural reasoning is documented.
18. Full pytest suite passes.
19. Ruff passes.
20. No real trading behaviour has been silently altered.

---

# FINAL REPORT

At the end, provide a concise engineering report containing:

## 1. What changed

List the major architectural changes.

## 2. What did NOT change

Explicitly confirm which trading/strategy behaviours were preserved.

## 3. New architecture

Show the final architecture as an ASCII diagram.

## 4. Reliability improvements

Explain how the new architecture handles:

* duplicate orders
* UNKNOWN orders
* broker/API failures
* process crashes
* DB failures
* stale broker state
* reconciliation
* retries.

## 5. Testing

Report:

```text
pytest: X passed
ruff: clean
new tests: X
```

## 6. Remaining risks

Be honest.

Do not claim the system is production-safe merely because tests pass.

Identify anything that still needs live validation.

## 7. Recommended next phase

Do NOT add new trading gates automatically.

After this refactor, the next project should be **measurement and validation of the existing strategy**, not another pile of conditions.

# SkyTrade current work and handoff

Last updated: 2026-09-18 (Asia/Kolkata)

This is the starting point for a new chat. Read this file before changing the strategy, dashboard, deployment, or Chronos research.

## Production intent

The active strategy is **Daily Swing**, an independent daily liquidity-sweep setup with 30-minute entry confirmation. It can trade long futures plus a protective put or short futures plus a protective call. The retired Alpha cash pipeline must remain stopped and disabled.

Before the changes in this working tree:

- `p-trade-daily-swing` was the active strategy service.
- `p-trade-dashboard` was active.
- `p-trade-live` (Alpha) was stopped and disabled.
- The dashboard still displayed Alpha cash positions, Alpha trade history, Alpha gates, and Alpha log freshness. That UI did not describe the active Daily Swing service.

## Phase 1 — Daily Swing visibility and notifications

Status: **implemented, tested and deployed on 2026-09-18.**

Changes in the working tree:

- The dashboard has four focused tabs: Overview, Trade history, Backtest, and System.
- Overview shows the active Daily Swing position and recent confirmed entry attempts.
- Trade history shows the full `daily_swing_positions` ledger, including open, closed and rejected records.
- Displayed return is explicitly the underlying spot move. It is not presented as the futures-plus-option basket's broker P&L.
- System health checks `p-trade-daily-swing` and `/var/log/p-trade/daily-swing.log`.
- During market hours, a log older than five minutes is unhealthy. Outside market hours, a quiet but active service is healthy.
- Admins can restart `p-trade-daily-swing` from the System tab.
- The visible Alpha Gates tab, Alpha cash configuration and old Alpha run button were removed from the dashboard.
- The Backtest tab loads the frozen 180-trade Nifty 500 research ledger from its CSV. It is visibly separate from the live ledger and labels the arithmetic sum of trade returns as non-portfolio evidence.
- Daily Swing Telegram messages now cover:
  - service start;
  - first live ticks of the trading session, including tracked symbol/setup counts;
  - missing Kite login or expired Kite session;
  - no ticks for at least 30 minutes;
  - recovery after a tick stall;
  - runner crash before its automatic restart;
  - market-session completion.
- Tick alerts are transition-based, so a persistent outage sends one alert rather than one message per minute.
- A 30-minute tick stall also ends the current WebSocket session so the runner reconnects automatically; recovery is announced after ticks resume. The dashboard's five-minute process-log health check remains separate because it detects a fully hung service, not a Telegram feed notification condition.
- HTTP transport INFO logs are suppressed for Telegram because its request URL contains the bot token; existing VPS log occurrences were sanitized during deployment verification.
- The GitHub deployment workflow now restarts and verifies `p-trade-dashboard` and `p-trade-daily-swing`. It no longer restarts retired `p-trade-live`.
- The Daily Swing systemd unit no longer declares the retired Alpha service as an `After=` dependency.

Validation completed locally:

- Full test suite: `451 passed`.
- Ruff: passed for all modified Python files.
- Dashboard JavaScript: `node --check` passed.
- No project dependency was added for this phase.

Files changed for Phase 1:

- `.github/workflows/ci.yml`
- `deploy/p-trade-daily-swing.service`
- `src/trading_scanner/daily_swing_live.py`
- `src/trading_scanner/infrastructure/db/daily_swing.py`
- `src/trading_scanner/templates/dashboard.html`
- `src/trading_scanner/webapp.py`
- `tests/test_daily_swing_notifications.py`
- `tests/test_daily_swing_repository.py`
- `tests/test_webapp_status.py`

## Phase 1B — Backtest visibility and TradingView port

Status: **implemented, tested and deployed on 2026-09-18.**

- The frozen CSV is tracked so the production dashboard can serve the same 180 research trades used by the audit.
- The Backtest tab shows 180 trades, 40.56% wins, +0.582% average net trade return, 1.47 profit factor, and +104.72 percentage points as the arithmetic sum of independent trade returns.
- `tradingview/daily_swing_strategy.pine` is a Pine Script v6 strategy for regular 30-minute NSE equity charts. It uses confirmed prior-day data, confirmation-candle-close entries, structural stops, 3R targets, the profit trail, ten-session time exit, and 0.20% round-trip commission.
- The Pine port omits Nifty 500 cross-sectional breadth and uses a same-slot mean instead of the Python median. TradingView therefore supports visual and symbol-level testing, but its trade list is not expected to match the Python universe backtest.
- `tradingview/README.md` contains usage and interpretation instructions.

Validation and deployment:

- Full suite: `452 passed`; Ruff, dashboard JavaScript syntax and `git diff --check` passed.
- Commit `61a40a0` passed both GitHub Actions test and deploy jobs.
- VPS reached `61a40a0`; dashboard and Daily Swing services were active and enabled, while retired `p-trade-live` remained inactive and disabled.
- The server loaded all 180 CSV rows and reproduced the audit summary. Its deployed tick-stall threshold is 1,800 seconds.

Deployment verification completed:

1. GitHub Actions test and deploy jobs passed.
2. VPS code reached commit `568346c`.
3. `p-trade-dashboard` and `p-trade-daily-swing` were active and enabled.
4. `p-trade-live` remained inactive and disabled.
5. The Daily Swing API reported `enabled=true`, zero active positions, zero history rows and zero attempts. This means no live Daily Swing entry has occurred yet; historical backtest trades were deliberately not inserted into the live ledger.
6. During market hours the service resolved 499 instruments, connected the ticker and reported fresh ticks.
7. Telegram returned successful delivery responses for the service-start and market-feed messages.
8. The health endpoint reported healthy with a fresh Daily Swing log.
9. Do not manufacture a tick outage in live trading merely to test the alert. The unit test covers the transition; observe the real alert only if an outage occurs.

## Phase 2 — Chronos-2 zero-shot research

Status: **paused before model download/evaluation so Phase 1 can be completed first.**

Decision:

- Use `amazon/chronos-2`, a pretrained 120M-parameter time-series foundation model under Apache-2.0.
- Do not use Ollama to predict raw candles.
- Start zero-shot: no fine-tuning and no live integration.
- The first labeled evaluation set is the frozen 1,011-trade hourly file at `analysis/reports/2026-09-16-hourly-setup/frozen-40h-trades.csv` because it has more examples than the 180 Daily Swing trades.
- Every forecast must receive candles only through the recorded entry timestamp. Later candles are scoring data only.
- Evaluate 2025 and 2026 chronologically. Do not select a threshold on 2026.
- The model must remain research-only until it improves net expectancy and profit factor on later periods and then survives new shadow data.

Local environment state:

- CPU PyTorch was installed temporarily in `.venv` for the experiment.
- `chronos-forecasting==2.3.2`, `torch==2.14.0+cpu` and their inference dependencies are installed in the local `.venv` only. The Chronos-2 model weights have not been downloaded.
- No Torch or Chronos dependency has been added to `pyproject.toml`.
- No Chronos research file has been added to the repository.

Next Chronos steps:

1. Verify/install `chronos-forecasting` in the local virtual environment without modifying project dependencies.
2. Load `amazon/chronos-2` and benchmark 10–20 trades for CPU latency and memory.
3. Implement a research script that batches causal contexts and caches forecasts.
4. Forecast future normalized price/return paths and derive target-before-stop probability using each trade's recorded stop and target.
5. Report baseline versus fixed Chronos policies for each chronological year, including trade count, win rate, average net return, profit factor and maximum drawdown.
6. If the zero-shot result fails, remove temporary packages/artifacts and record the rejection. Do not deploy it.
7. If it passes, add reproducible code and tests, then run shadow-only. The model must not place or block live orders at this stage.

## Earlier ML evidence

Do not repeat these experiments without a materially different hypothesis:

- CatBoost on 180 Daily Swing trades had combined out-of-sample AUC around 0.565 and was unstable by year.
- A custom PatchTST-style transformer with masked pretraining failed. On the larger hourly dataset, 2025 target AUC was about 0.501 and 2026 target AUC about 0.460; selected 2026 trades averaged about -0.242%.
- That failed neural experiment and its dependency changes were removed. It was never deployed.

## Safety and interpretation

- Backtest files and the live database are different evidence. Never show backtest trades as live trade history.
- `daily_swing_positions` stores spot reference prices used by strategy logic. Exact rupee basket P&L requires futures and hedge fills from the broker order ledger; label spot returns accurately until that reconciliation is implemented.
- Daily Swing currently has one global live position slot.
- Keep Chronos, other ML models and prompt-based models away from real order execution until chronological and forward-shadow validation pass.
- Never restart or enable `p-trade-live` as part of deployment.

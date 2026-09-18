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

Status: **implemented locally, tested, not yet deployed at the time this file was written.**

Changes in the working tree:

- The dashboard now has only three useful tabs: Overview, Trade history, and System.
- Overview shows the active Daily Swing position and recent confirmed entry attempts.
- Trade history shows the full `daily_swing_positions` ledger, including open, closed and rejected records.
- Displayed return is explicitly the underlying spot move. It is not presented as the futures-plus-option basket's broker P&L.
- System health checks `p-trade-daily-swing` and `/var/log/p-trade/daily-swing.log`.
- During market hours, a log older than five minutes is unhealthy. Outside market hours, a quiet but active service is healthy.
- Admins can restart `p-trade-daily-swing` from the System tab.
- The visible Alpha Gates tab, Alpha cash configuration and old Alpha run button were removed from the dashboard.
- Daily Swing Telegram messages now cover:
  - service start;
  - first live ticks of the trading session, including tracked symbol/setup counts;
  - missing Kite login or expired Kite session;
  - no ticks for at least three minutes;
  - recovery after a tick stall;
  - runner crash before its automatic restart;
  - market-session completion.
- Tick alerts are transition-based, so a persistent outage sends one alert rather than one message per minute.
- A three-minute tick stall also ends the current WebSocket session so the runner reconnects automatically; recovery is announced after ticks resume.
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

Deployment verification required after pushing:

1. Confirm GitHub Actions tests and deploy succeed.
2. Confirm `systemctl is-active p-trade-dashboard p-trade-daily-swing` returns active for both.
3. Confirm `p-trade-live` remains inactive and disabled.
4. Open the SkyTrade dashboard and verify Overview, Trade history and System load.
5. Confirm the Telegram service-start message arrives after the deploy restart.
6. During the next market session, confirm the first-ticks “Market feed live” message arrives.
7. Do not manufacture a tick outage in live trading merely to test the alert. The unit test covers the transition; observe the real alert only if an outage occurs.

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

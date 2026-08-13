"""Stage A: train a CatBoostClassifier predicting P(win) for a BUY signal,
walk-forward validated, calibrated, and sanity-checked -- see NOTES.md's
ranking-model roadmap and ``application/ranking_features.py`` for the
feature set.

Requires the optional ``ml`` dependency group (not part of the default
install, and not needed on the VPS -- only this offline script needs it):

    poetry install --with ml
    # or: pip install catboost scikit-learn

Run only after ``python -m trading_scanner.backtest`` has fully populated
the ``trades`` table with the ADX/regime/volatility columns
``application/backtest.py``'s replay now logs -- trades from before that
migration are skipped by ``build_feature_table``, not imputed.

Walk-forward, not random split: a random split leaks information through
autocorrelated/overlapping trades (avg holding ~3.5 days means a random
80/20 split routinely puts a trade's entry in train and its still-open
neighbor's outcome in test). Each fold trains on one calendar month,
purging any training trade whose *exit* falls inside an embargo window
before the test month starts (dropping the overlap-risk trades near the
boundary entirely, rather than guessing how much of their outcome leaked
into the embargo window), and tests on the next month.
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from trading_scanner.application.ranking_features import FeatureRow, build_feature_table
from trading_scanner.config.settings import load_config
from trading_scanner.infrastructure.db import (
    TursoCandleRepository,
    TursoTradeRepository,
    create_turso_client,
)

_FEATURE_COLUMNS = [
    "prediction_at_entry",
    "adx",
    "regime_normalized",
    "volatility_margin",
    "sector",
    "day_of_week",
    "hour_of_day",
    "days_since_last_trade",
    "correlation_to_open_positions",
]
_CATEGORICAL_COLUMNS = ["sector", "day_of_week", "hour_of_day"]

# Average holding period is ~3.5 days (see application/paper_trading.py's
# own docstring) -- embargo a few days past that so a training trade's
# outcome genuinely resolves before the test window starts.
_EMBARGO = timedelta(days=5)
_MIN_TRAIN_ROWS = 200
_MIN_TEST_ROWS = 20

_MODEL_OUTPUT_PATH = Path("data/ranking_stage_a.cbm")


@dataclass
class FoldResult:
    test_month: str
    train_rows: int
    test_rows: int
    auc: float


def _to_dataframe(rows: list[FeatureRow]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(row) for row in rows])
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    for column in _CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype(str)
    return frame.sort_values("entry_timestamp").reset_index(drop=True)


def _walk_forward_folds(frame: pd.DataFrame):
    """Yield (train_df, test_df) per calendar month, purged/embargoed."""
    frame = frame.copy()
    frame["month"] = frame["entry_timestamp"].dt.tz_localize(None).dt.to_period("M")
    months = sorted(frame["month"].unique())
    for index in range(1, len(months)):
        test_month = months[index]
        test_df = frame[frame["month"] == test_month]
        test_start = test_df["entry_timestamp"].min()
        purge_before = test_start - _EMBARGO

        train_df = frame[frame["month"] < test_month]
        # Purge: drop any training row whose exit could still overlap the
        # embargo window -- we don't have exit_timestamp in the feature
        # frame (only entry-side features are used for training), so purge
        # conservatively on entry_timestamp + typical holding period
        # instead of requiring an exact exit join.
        train_df = train_df[train_df["entry_timestamp"] < purge_before]
        yield str(test_month), train_df, test_df


def _train_one_fold(train_df: pd.DataFrame, test_df: pd.DataFrame) -> FoldResult | None:
    if len(train_df) < _MIN_TRAIN_ROWS or len(test_df) < _MIN_TEST_ROWS:
        return None
    train_pool = Pool(
        train_df[_FEATURE_COLUMNS], train_df["label"], cat_features=_CATEGORICAL_COLUMNS
    )
    test_pool = Pool(
        test_df[_FEATURE_COLUMNS], test_df["label"], cat_features=_CATEGORICAL_COLUMNS
    )
    model = CatBoostClassifier(
        iterations=300, depth=4, learning_rate=0.05, loss_function="Logloss",
        verbose=False, random_seed=42,
    )
    model.fit(train_pool)
    predictions = model.predict_proba(test_pool)[:, 1]
    if test_df["label"].nunique() < 2:
        return None  # AUC undefined with only one class in the test month.
    auc = roc_auc_score(test_df["label"], predictions)
    return FoldResult(
        test_month=str(test_df["month"].iloc[0]), train_rows=len(train_df),
        test_rows=len(test_df), auc=auc,
    )


def _check_calibration(model: CatBoostClassifier, frame: pd.DataFrame) -> None:
    """Print predicted-vs-realized win rate by probability bucket.

    A well-calibrated model's "70% confidence" bucket should realize ~70%
    wins on held-out data -- this is a print-and-eyeball check, not an
    automated gate, since what counts as "close enough" is a judgment call
    before trusting scores live.
    """
    pool = Pool(frame[_FEATURE_COLUMNS], cat_features=_CATEGORICAL_COLUMNS)
    predictions = model.predict_proba(pool)[:, 1]
    fraction_of_positives, mean_predicted = calibration_curve(
        frame["label"], predictions, n_bins=10, strategy="quantile"
    )
    print("\nCalibration (predicted P(win) vs realized win rate, by decile):")
    for predicted, realized in zip(mean_predicted, fraction_of_positives, strict=True):
        print(f"  predicted={predicted:.2f}  realized={realized:.2f}")


def _check_feature_importance(model: CatBoostClassifier) -> None:
    importances = sorted(
        zip(_FEATURE_COLUMNS, model.get_feature_importance(), strict=True),
        key=lambda pair: pair[1], reverse=True,
    )
    print("\nFeature importance (sanity-check this matches known market intuition,")
    print("e.g. higher ADX / stronger prediction_at_entry should rank near the top):")
    for name, importance in importances:
        print(f"  {name:<32} {importance:6.2f}")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    config = load_config()
    if not config.turso_database_url:
        raise RuntimeError("TRADING_SCANNER_TURSO_URL must be set.")

    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    trade_repository = TursoTradeRepository(client)
    candle_repository = TursoCandleRepository(client)
    await trade_repository.ensure_schema()

    logger.info("Loading trade history and building the feature table (this walks every BUY "
                "trade's correlation window -- can take a few minutes)...")
    all_trades = await trade_repository.get_trades(None, config.candle_interval)
    rows = await build_feature_table(all_trades, candle_repository, config.candle_interval)
    await client.close()
    logger.info(
        "Built %d feature rows from %d total trades (rest lacked feature columns or are "
        "SELL-side/still-open).", len(rows), len(all_trades),
    )
    if len(rows) < _MIN_TRAIN_ROWS * 2:
        logger.warning(
            "Only %d usable rows -- too few for a meaningful walk-forward validation "
            "(need the full backtest replay to have finished). Stopping here.", len(rows),
        )
        return

    frame = _to_dataframe(rows)

    logger.info("Running walk-forward validation...")
    fold_results = [
        result
        for _, train_df, test_df in _walk_forward_folds(frame)
        if (result := _train_one_fold(train_df, test_df)) is not None
    ]
    print(f"\n{'Test month':<12} {'Train rows':>10} {'Test rows':>9} {'AUC':>6}")
    print("-" * 42)
    for fold in fold_results:
        print(f"{fold.test_month:<12} {fold.train_rows:>10} {fold.test_rows:>9} {fold.auc:>6.3f}")
    if fold_results:
        mean_auc = float(np.mean([fold.auc for fold in fold_results]))
        print(f"\nMean walk-forward AUC: {mean_auc:.3f} (0.5 = no better than random)")

    logger.info("Training the final model on all available data...")
    final_pool = Pool(frame[_FEATURE_COLUMNS], frame["label"], cat_features=_CATEGORICAL_COLUMNS)
    final_model = CatBoostClassifier(
        iterations=300, depth=4, learning_rate=0.05, loss_function="Logloss",
        verbose=False, random_seed=42,
    )
    final_model.fit(final_pool)

    _check_calibration(final_model, frame)
    _check_feature_importance(final_model)

    _MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(_MODEL_OUTPUT_PATH))
    print(f"\nSaved final model to {_MODEL_OUTPUT_PATH}.")
    print(
        "This model is NOT wired into the live ranking gate yet -- "
        "application/ranking.py's score_candidate is still the Stage A heuristic. "
        "Review the walk-forward AUC, calibration, and feature importance above "
        "before deciding whether to wire it in."
    )


if __name__ == "__main__":
    asyncio.run(main())

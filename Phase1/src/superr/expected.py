"""Expected-outcome models, chronological validation, and OOF predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor


FEATURES = [
    "competition_pool", "season_year", "innings", "innings_phase", "innings_type",
    "score_before", "wickets_lost_before", "wickets_remaining", "legal_balls_before",
    "balls_remaining", "current_run_rate", "target", "runs_needed", "required_run_rate",
    "wickets_lost_bucket", "balls_remaining_bucket", "is_knockout",
    "batting_type", "bowling_type", "bowling_pace_spin", "venue_country",
]
CAT_FEATURES = [
    "competition_pool", "innings_phase", "innings_type", "wickets_lost_bucket",
    "balls_remaining_bucket", "batting_type", "bowling_type", "bowling_pace_spin", "venue_country",
]
TARGETS = {"bat_runs": "bat_runs", "bowler_conceded": "bowler_conceded", "bowler_wicket": "bowler_wicket"}
BASELINE_KEYS = ["competition_pool", "innings", "innings_phase", "wickets_lost_bucket", "balls_remaining_bucket"]
POST_MATCH_OR_OUTCOME_FIELDS = {"is_dls", "is_dls_text", "result", "team_runs", "bat_runs", "bowler_conceded", "bowler_wicket", "is_wicket", "is_wide", "is_legal", "delivery_id", "striker_id", "bowler_id"}


def assert_predictor_allowlist() -> None:
    """Fail early if any outcome, identity, or retrospective field enters predictors."""
    forbidden = POST_MATCH_OR_OUTCOME_FIELDS.intersection(FEATURES)
    if forbidden:
        raise ValueError(f"Outcome/post-match fields cannot be predictors: {sorted(forbidden)}")


def _target_population(data: pd.DataFrame, target: str) -> pd.Series:
    eligible = data[target].notna()
    if target == "bat_runs":
        eligible &= ~data["is_wide"].fillna(False)
    return eligible


def add_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create safe pre-ball categoricals and ensure model feature schema exists."""
    result = data.copy()
    result["season_year"] = pd.to_numeric(result["season_year"], errors="coerce").fillna(-1).astype(int)
    if "batting_type" not in result:
        result["batting_type"] = "unknown"
    if "bowling_type" not in result:
        result["bowling_type"] = "unknown"
    if "bowling_pace_spin" not in result:
        result["bowling_pace_spin"] = result.get("pace_or_spin", "unknown")
    if "venue_country" not in result:
        result["venue_country"] = "unknown"
    for col in CAT_FEATURES:
        result[col] = result[col].astype("string").fillna("unknown").astype(str)
    for col in FEATURES:
        if col not in result:
            result[col] = np.nan
    for col in FEATURES:
        if col not in CAT_FEATURES:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def _grouped_baseline(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """Smoothed hierarchy of contextual means, with pool-wide fallback."""
    global_mean = float(train[target].mean())
    pred = pd.Series(global_mean, index=test.index, dtype=float)
    for keys in [BASELINE_KEYS, BASELINE_KEYS[:3], BASELINE_KEYS[:2], BASELINE_KEYS[:1]]:
        means = train.groupby(keys, dropna=False)[target].mean()
        counts = train.groupby(keys, dropna=False)[target].size()
        test_keys = pd.MultiIndex.from_frame(test[keys]) if len(keys) > 1 else test[keys[0]]
        mapped_mean = pd.Series(test_keys.map(means), index=test.index, dtype=float)
        mapped_n = pd.Series(test_keys.map(counts), index=test.index, dtype=float)
        # Empirical smoothing prevents tiny cells from dominating.
        smooth = mapped_n / (mapped_n + 30.0)
        pred = mapped_mean.where(mapped_mean.notna(), pred) * smooth.fillna(0) + pred * (1 - smooth.fillna(0))
    return pred.to_numpy(dtype=float)


def _metrics(actual: np.ndarray, predicted: np.ndarray, target: str) -> dict[str, float]:
    error = predicted - actual
    if target == "bowler_wicket":
        probability = np.clip(predicted, 1e-6, 1 - 1e-6)
        return {"brier": float(np.mean(error ** 2)), "log_loss": float(-np.mean(actual * np.log(probability) + (1 - actual) * np.log(1 - probability)))}
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def _fit_catboost(train: pd.DataFrame, target: str, config: dict[str, Any], *, seed_offset: int = 0):
    common = dict(
        iterations=int(config.get("catboost_iterations", 120)),
        depth=int(config.get("catboost_depth", 6)),
        learning_rate=float(config.get("catboost_learning_rate", 0.05)),
        random_seed=int(config.get("random_seed", 2027)) + seed_offset,
        thread_count=int(config.get("thread_count", 4)),
        verbose=False,
        allow_writing_files=False,
    )
    model = CatBoostClassifier(loss_function="Logloss", **common) if target == "bowler_wicket" else CatBoostRegressor(loss_function="RMSE", **common)
    model.fit(train[FEATURES], train[target], cat_features=CAT_FEATURES)
    return model


def chronological_validation(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each pool on seasons strictly later than its training seasons."""
    assert_predictor_allowlist()
    reports: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for pool in config.get("train_pools", ["IPL", "T20I"]):
        pool_data = data.loc[data["competition_pool"].eq(pool) & data["pressure_available"] & ~data["is_super_over"]].copy()
        years = sorted(int(y) for y in pool_data["season_year"].dropna().unique())
        candidate_years = years[1:]
        for year in candidate_years:
            train = pool_data.loc[pool_data["season_year"].lt(year)]
            test = pool_data.loc[pool_data["season_year"].eq(year)]
            if train.empty or test.empty:
                continue
            cap = int(config.get("max_examples_per_fold", 0))
            if cap and len(train) > cap:
                train = train.sample(cap, random_state=int(config.get("random_seed", 2027)))
            if cap and len(test) > cap:
                test = test.sample(cap, random_state=int(config.get("random_seed", 2027)))
            for target in TARGETS:
                valid_train = train.loc[_target_population(train, target)]
                valid_test = test.loc[_target_population(test, target)]
                if valid_train.empty or valid_test.empty:
                    continue
                print(f"Validation fit: {pool} {year} {target} ({len(valid_train):,} train / {len(valid_test):,} test)", flush=True)
                base_pred = _grouped_baseline(valid_train, valid_test, target)
                model = _fit_catboost(valid_train, target, config, seed_offset=year)
                cat_pred = model.predict_proba(valid_test[FEATURES])[:, 1] if target == "bowler_wicket" else model.predict(valid_test[FEATURES])
                actual = valid_test[target].to_numpy(dtype=float)
                for name, pred in (("grouped_baseline", base_pred), ("catboost", np.asarray(cat_pred, dtype=float))):
                    metrics = _metrics(actual, pred, target)
                    subset = valid_test["high_pressure"].to_numpy(dtype=bool)
                    pressure_metrics = _metrics(actual[subset], np.asarray(pred)[subset], target) if subset.any() else {}
                    reports.append({"pool": pool, "test_season": year, "target": target, "model": name,
                                    "n": len(actual), "high_pressure_n": int(subset.sum()), **metrics,
                                    **{f"high_pressure_{key}": value for key, value in pressure_metrics.items()}})
                    if name == "catboost":
                        fold = valid_test[["delivery_id", "competition_pool", "season_year", "pressure_weight", "high_pressure"]].copy()
                        fold["target_name"] = target
                        fold["actual"] = actual
                        fold["predicted"] = np.asarray(pred, dtype=float)
                        fold["model"] = name
                        predictions.append(fold)
    return pd.DataFrame(reports), pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def leave_one_season_out_predictions(data: pd.DataFrame, config: dict[str, Any], validation: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate cross-fitted expectations; each row's season is wholly held out."""
    pred_frames: list[pd.DataFrame] = []
    model_choices: list[dict[str, Any]] = []
    if validation is None:
        validation, _ = chronological_validation(data, config)
    for pool in config.get("train_pools", ["IPL", "T20I"]):
        pool_data = data.loc[data["competition_pool"].eq(pool) & data["pressure_available"] & ~data["is_super_over"]].copy()
        years = sorted(int(y) for y in pool_data["season_year"].dropna().unique())
        for year in years:
            # Rolling-origin prediction: never train on a season after the held-out one.
            train = pool_data.loc[pool_data["season_year"].lt(year)]
            test = pool_data.loc[pool_data["season_year"].eq(year)]
            if train.empty or test.empty:
                continue
            cap = int(config.get("max_examples_per_fold", 0))
            if cap and len(train) > cap:
                train = train.sample(cap, random_state=int(config.get("random_seed", 2027)) + year)
            for target in TARGETS:
                valid_train = train.loc[_target_population(train, target)]
                valid_test = test.loc[_target_population(test, target)]
                if valid_train.empty or valid_test.empty:
                    continue
                print(f"Cross-fit: {pool} held-out {year} {target} ({len(valid_train):,} train / {len(valid_test):,} test)", flush=True)
                baseline = _grouped_baseline(valid_train, valid_test, target)
                # Select by pooled, chronological, target-specific validation. Ties go to simpler baseline.
                rows = validation.loc[
                    validation["pool"].eq(pool)
                    & validation["target"].eq(target)
                    & validation["test_season"].lt(year)
                ].copy()
                selection_count = int(config.get("selection_validation_last_seasons", 3))
                prior_seasons = sorted(int(y) for y in rows["test_season"].unique())
                if selection_count and len(prior_seasons) > selection_count:
                    rows = rows.loc[rows["test_season"].ge(prior_seasons[-selection_count])]
                metric = "brier" if target == "bowler_wicket" else "mae"
                means = rows.groupby("model")[metric].mean() if metric in rows else pd.Series(dtype=float)
                choose_cat = bool(means.get("catboost", np.inf) < means.get("grouped_baseline", np.inf))
                if choose_cat:
                    model = _fit_catboost(valid_train, target, config, seed_offset=year)
                    cat = model.predict_proba(valid_test[FEATURES])[:, 1] if target == "bowler_wicket" else model.predict(valid_test[FEATURES])
                else:
                    cat = np.full(len(valid_test), np.nan, dtype=float)
                selected = np.asarray(cat if choose_cat else baseline, dtype=float)
                # Both expectations are retained for audit and later sensitivity work.
                frame = valid_test[["delivery_id", "competition_pool", "season_year"]].copy()
                frame[f"expected_{target}_baseline"] = baseline
                frame[f"expected_{target}_catboost"] = np.asarray(cat, dtype=float)
                frame[f"expected_{target}"] = selected
                pred_frames.append(frame)
                model_choices.append({"pool": pool, "held_out_season": year, "target": target,
                                      "max_training_season": int(valid_train["season_year"].max()),
                                      "max_selection_season": int(rows["test_season"].max()) if not rows.empty else np.nan,
                                      "selected_model": "catboost" if choose_cat else "grouped_baseline",
                                      "selection_metric": metric,
                                      "validation_catboost": float(means.get("catboost", np.nan)),
                                      "validation_baseline": float(means.get("grouped_baseline", np.nan))})
    if not pred_frames:
        raise ValueError("No out-of-season predictions generated; check season labels and eligibility.")
    predictions = pd.concat(pred_frames, ignore_index=True).groupby(
        ["delivery_id", "competition_pool", "season_year"], as_index=False, sort=False
    ).first()
    return predictions, pd.DataFrame(model_choices)

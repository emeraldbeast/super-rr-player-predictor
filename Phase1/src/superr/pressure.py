"""Past-match reference rates and pre-outcome pressure weights."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prior_rate_lookup(
    data: pd.DataFrame,
    context_keys: list[str],
    *,
    exclude_dls_matches: bool,
) -> pd.DataFrame:
    """For each match/context, estimate runs per over from earlier matches only."""
    eligible_mask = ~data["is_super_over"]
    if exclude_dls_matches:
        eligible_mask &= ~data["is_dls"]
    eligible = data.loc[eligible_mask].copy()
    match_context = eligible.groupby(["competition_pool", "match_id", *context_keys], as_index=False, dropna=False).agg(
        match_date=("match_date", "first"),
        runs=("team_runs", "sum"),
        legal_balls=("is_legal", "sum"),
    )
    match_context = match_context.sort_values(["match_date", "match_id"], kind="stable")
    by_context = ["competition_pool", *context_keys]
    grouped = match_context.groupby(by_context, sort=False, dropna=False)
    match_context["prior_runs"] = grouped["runs"].cumsum() - match_context["runs"]
    match_context["prior_legal_balls"] = grouped["legal_balls"].cumsum() - match_context["legal_balls"]
    match_context["par_runs_per_over"] = 6 * match_context["prior_runs"] / match_context["prior_legal_balls"].replace(0, np.nan)
    return match_context[["competition_pool", "match_id", *context_keys, "par_runs_per_over"]]


def add_pressure(
    state: pd.DataFrame,
    *,
    cold_start_rate: float = 8.0,
    high_pressure_quantile: float = 0.75,
    chase_weights: dict[str, float] | None = None,
    first_innings_weights: dict[str, float] | None = None,
    exclude_dls_matches: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate separate first-innings and chase pressure measures.

    Reference scoring rates for a match/context exclude that entire match.
    The pressure weight depends only on pre-delivery state; actual delivery
    outcomes are not used to classify that delivery as pressure.
    """
    chase_weights = chase_weights or {"rate_strain": 0.60, "wicket_strain": 0.25, "time_strain": 0.15}
    first_innings_weights = first_innings_weights or {"score_deficit": 0.55, "wickets_lost": 0.30, "phase": 0.15}
    data = state.copy()
    exact_keys = ["innings", "innings_phase", "wickets_lost_bucket"]
    exact = _prior_rate_lookup(data, exact_keys, exclude_dls_matches=exclude_dls_matches).rename(
        columns={"par_runs_per_over": "par_rate_exact"}
    )
    broad = _prior_rate_lookup(data, ["innings", "innings_phase"], exclude_dls_matches=exclude_dls_matches).rename(
        columns={"par_runs_per_over": "par_rate_phase"}
    )
    data = data.merge(exact, on=["competition_pool", "match_id", *exact_keys], how="left", validate="many_to_one")
    data = data.merge(broad, on=["competition_pool", "match_id", "innings", "innings_phase"], how="left", validate="many_to_one")
    data["par_runs_per_over"] = data["par_rate_exact"].fillna(data["par_rate_phase"]).fillna(cold_start_rate)
    data["par_reference_source"] = np.select(
        [data["par_rate_exact"].notna(), data["par_rate_phase"].notna()],
        ["prior_match_phase_wickets", "prior_match_phase"], default="configured_cold_start",
    )

    dls_allowed = ~data["is_dls"] if exclude_dls_matches else pd.Series(True, index=data.index)
    is_chase = data["innings"].eq(2) & data["chase_target_available"] & ~data["is_super_over"] & dls_allowed
    par = data["par_runs_per_over"].clip(lower=1.0)
    rate_ratio = data["required_run_rate"] / par
    rate_strain = ((rate_ratio - 1.0) / 1.5).clip(0, 1)
    wicket_strain = (data["wickets_lost_before"] / 8.0).clip(0, 1)
    time_strain = (1 - data["balls_remaining"] / data["scheduled_legal_balls"].replace(0, np.nan)).clip(0, 1)
    chase_score = (
        chase_weights["rate_strain"] * rate_strain
        + chase_weights["wicket_strain"] * wicket_strain
        + chase_weights["time_strain"] * time_strain
    ).clip(0, 1)
    data["pressure_chase"] = chase_score.where(is_chase)

    is_first = data["innings"].eq(1) & ~data["is_super_over"] & dls_allowed
    expected_score = data["par_runs_per_over"] * data["legal_balls_before"] / 6.0
    score_deficit = ((expected_score - data["score_before"]).clip(lower=0) / expected_score.clip(lower=1)).clip(0, 1)
    resource_strain = (data["wickets_lost_before"] / 8.0).clip(0, 1)
    phase_strain = (data["legal_balls_before"] / data["scheduled_legal_balls"].replace(0, np.nan)).clip(0, 1)
    first_score = (
        first_innings_weights["score_deficit"] * score_deficit
        + first_innings_weights["wickets_lost"] * resource_strain
        + first_innings_weights["phase"] * phase_strain
    ).clip(0, 1)
    data["pressure_first_innings"] = first_score.where(is_first)

    data["pressure_weight"] = pd.to_numeric(
        data["pressure_chase"].combine_first(data["pressure_first_innings"]), errors="coerce"
    )
    data["pressure_available"] = data["pressure_weight"].notna()
    data["high_pressure_threshold"] = np.nan
    threshold_by_pool = data.loc[data["pressure_available"]].groupby("competition_pool")["pressure_weight"].quantile(high_pressure_quantile)
    data["high_pressure_threshold"] = data["competition_pool"].map(threshold_by_pool)
    data["high_pressure"] = data["pressure_available"] & data["pressure_weight"].ge(data["high_pressure_threshold"])
    data["pressure_context"] = np.select(
        [data["pressure_chase"].notna(), data["pressure_first_innings"].notna()],
        ["chase", "first_innings_adversity"], default="unavailable",
    )

    eligible = data.loc[data["pressure_available"] & ~data["is_super_over"] & ~data["is_dls"]].copy()
    diagnostic = eligible.groupby(
        ["competition_pool", "pressure_context", "innings_phase", "high_pressure"], as_index=False
    ).agg(
        deliveries=("delivery_id", "size"),
        team_runs=("team_runs", "sum"),
        legal_balls=("is_legal", "sum"),
        wickets=("is_wicket", "sum"),
        mean_pressure=("pressure_weight", "mean"),
    )
    diagnostic["runs_per_legal_ball"] = diagnostic["team_runs"] / diagnostic["legal_balls"].replace(0, np.nan)
    diagnostic["wickets_per_100_deliveries"] = 100 * diagnostic["wickets"] / diagnostic["deliveries"].replace(0, np.nan)
    reference_coverage = data.loc[data["pressure_available"]].groupby(["competition_pool", "par_reference_source"], as_index=False).agg(
        deliveries=("delivery_id", "size"),
        matches=("match_id", "nunique"),
    )

    # Standardize the ordinary/high-pressure comparison to the same pool-specific
    # pressure-context × innings-phase mixture. Use only common-support cells.
    standardized_rows: list[dict[str, float | str | int]] = []
    for pool, group in eligible.groupby("competition_pool", sort=False):
        cells = group.groupby(["pressure_context", "innings_phase", "high_pressure"], as_index=False, observed=True).agg(
            deliveries=("delivery_id", "size"), team_runs=("team_runs", "sum"),
            legal_balls=("is_legal", "sum"), wickets=("is_wicket", "sum"),
        )
        contexts = cells.groupby(["pressure_context", "innings_phase"], as_index=False, observed=True).agg(
            total_deliveries=("deliveries", "sum"),
        )
        contexts["weight"] = contexts["total_deliveries"] / contexts["total_deliveries"].sum()
        cells["runs_per_legal_ball"] = cells["team_runs"] / cells["legal_balls"].replace(0, np.nan)
        cells["wickets_per_100_deliveries"] = 100 * cells["wickets"] / cells["deliveries"].replace(0, np.nan)
        cells = cells.merge(contexts[["pressure_context", "innings_phase", "weight"]], on=["pressure_context", "innings_phase"], validate="many_to_one")
        for high, cohort in cells.groupby("high_pressure", sort=False):
            cohort = cohort.loc[cohort["runs_per_legal_ball"].notna()]
            normalized_weights = cohort["weight"] / cohort["weight"].sum()
            standardized_rows.append({
                "competition_pool": pool,
                "high_pressure": bool(high),
                "common_context_phase_cells": int(len(cohort)),
                "runs_per_legal_ball_standardized": float((cohort["runs_per_legal_ball"] * normalized_weights).sum()),
                "wickets_per_100_deliveries_standardized": float((cohort["wickets_per_100_deliveries"] * normalized_weights).sum()),
            })
    standardized = pd.DataFrame(standardized_rows)
    return data, diagnostic, reference_coverage, standardized

"""Exposure-weighted player scoring and match-cluster empirical Bayes shrinkage."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


def _player_match_rates(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    uplift_minimum_balls: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = data.merge(
        predictions,
        on=["delivery_id", "competition_pool", "season_year"],
        how="inner",
        validate="one_to_one",
    )
    model = model.loc[model["pressure_available"] & ~model["is_super_over"] & ~model["is_dls"]].copy()
    model["bat_residual"] = model["bat_runs"] - model["expected_bat_runs"]
    model["bowl_runs_saved"] = model["expected_bowler_conceded"] - model["bowler_conceded"]
    model["wicket_residual"] = model["bowler_wicket"].astype(float) - model["expected_bowler_wicket"]
    model["bat_event"] = model["striker_id"].notna() & ~model["is_wide"].fillna(False)
    model["bowl_event"] = model["bowler_id"].notna()
    model["bat_exposure"] = model["bat_event"].astype(float)
    model["bowl_exposure"] = model["is_legal"].astype(float) * model["bowl_event"].astype(float)

    # Assign a role using first non-wide striker appearance in each innings.
    appearances = model.loc[model["bat_event"], ["match_id", "innings", "delivery_id", "striker_id"]].drop_duplicates(
        ["match_id", "innings", "striker_id"]
    ).sort_values(["match_id", "innings", "delivery_id"])
    appearances["batting_order"] = appearances.groupby(["match_id", "innings"]).cumcount() + 1
    appearances["batting_role"] = pd.cut(
        appearances["batting_order"], [-1, 3, 6, 20], labels=["top_order", "middle_order", "lower_order"]
    ).astype("string")
    model = model.merge(
        appearances[["match_id", "innings", "striker_id", "batting_role"]],
        on=["match_id", "innings", "striker_id"], how="left", validate="many_to_one",
    )
    model["batting_role"] = model["batting_role"].fillna("unknown")

    bat = model.loc[model["bat_event"] & model["expected_bat_runs"].notna()].copy()
    bat["weighted_exposure"] = bat["pressure_weight"] * bat["bat_exposure"]
    bat["weighted_residual"] = bat["pressure_weight"] * bat["bat_residual"]
    bat_match = bat.groupby(
        ["competition_pool", "season_year", "match_id", "striker_id", "striker_name"],
        as_index=False, dropna=False,
    ).agg(
        pressure_balls=("weighted_exposure", "sum"),
        weighted_residual=("weighted_residual", "sum"),
    )
    bat_match = bat_match.rename(columns={"striker_name": "player_name"})
    bat_match["display_name"] = bat_match["player_name"].astype("string")
    bat_roles = bat.groupby(["competition_pool", "striker_id"])["batting_role"].agg(
        lambda values: values.mode().iloc[0] if not values.mode().empty else "unknown"
    ).rename("role").reset_index().rename(columns={"striker_id": "player_id"})
    bat_match = bat_match.merge(
        bat_roles, left_on=["competition_pool", "striker_id"], right_on=["competition_pool", "player_id"],
        how="left", validate="many_to_one",
    ).drop(columns="player_id")
    bat_match["rate"] = 100 * bat_match["weighted_residual"] / bat_match["pressure_balls"].replace(0, np.nan)
    bat_match = bat_match.loc[bat_match["pressure_balls"].gt(0)].reset_index(drop=True)

    bowl = model.loc[model["bowl_event"] & model["expected_bowler_conceded"].notna()].copy()
    bowl["weighted_exposure"] = bowl["pressure_weight"] * bowl["bowl_exposure"]
    bowl["weighted_saved"] = bowl["pressure_weight"] * bowl["bowl_runs_saved"]
    bowl["weighted_wicket_residual"] = bowl["pressure_weight"] * bowl["wicket_residual"]
    bowl["bowling_pace_spin"] = bowl["pace_or_spin"]
    bowl_match = bowl.groupby(
        ["competition_pool", "season_year", "match_id", "bowler_id", "bowler_name"],
        as_index=False, dropna=False,
    ).agg(
        pressure_balls=("weighted_exposure", "sum"),
        weighted_residual=("weighted_saved", "sum"),
        weighted_wicket_residual=("weighted_wicket_residual", "sum"),
    )
    bowl_match = bowl_match.rename(columns={"bowler_name": "player_name"})
    bowl_match["display_name"] = bowl_match["player_name"].astype("string")
    bowl_roles = bowl.groupby(["competition_pool", "bowler_id"])["bowling_type"].agg(
        lambda values: values.mode().iloc[0] if not values.mode().empty else "unknown"
    ).rename("role").reset_index().rename(columns={"bowler_id": "player_id"})
    bowl_match = bowl_match.merge(
        bowl_roles, left_on=["competition_pool", "bowler_id"], right_on=["competition_pool", "player_id"],
        how="left", validate="many_to_one",
    ).drop(columns="player_id")
    bowl_match["rate"] = 100 * bowl_match["weighted_residual"] / bowl_match["pressure_balls"].replace(0, np.nan)
    bowl_match["wicket_uplift_per_100"] = 100 * bowl_match["weighted_wicket_residual"] / bowl_match["pressure_balls"].replace(0, np.nan)
    bowl_match = bowl_match.loc[bowl_match["pressure_balls"].gt(0)].reset_index(drop=True)

    # A distinct high-pressure-minus-ordinary adjusted residual comparison.
    uplift_rows: list[pd.DataFrame] = []
    for label, frame, player_col, event_col, residual_col in (
        ("batter", model.loc[model["bat_event"] & model["expected_bat_runs"].notna()], "striker_id", "bat_exposure", "bat_residual"),
        ("bowler", bowl, "bowler_id", "bowl_exposure", "bowl_runs_saved"),
    ):
        frame = frame.copy()
        frame["segment"] = np.where(frame["high_pressure"], "high", "ordinary")
        frame["segment_exposure"] = frame["pressure_weight"] * frame[event_col]
        frame["segment_residual"] = frame["pressure_weight"] * frame[residual_col]
        grouped = frame.groupby(["competition_pool", player_col, "segment"], as_index=False, dropna=False).agg(
            exposure=("segment_exposure", "sum"), residual=("segment_residual", "sum"),
        )
        grouped["rate"] = 100 * grouped["residual"] / grouped["exposure"].replace(0, np.nan)
        exposure_wide = grouped.pivot(index=["competition_pool", player_col], columns="segment", values="exposure")
        rate_wide = grouped.pivot(index=["competition_pool", player_col], columns="segment", values="rate")
        part = pd.DataFrame(index=rate_wide.index).reset_index()
        part["high_pressure_weighted_balls"] = part.apply(lambda row: exposure_wide.loc[(row["competition_pool"], row[player_col]), "high"] if "high" in exposure_wide.columns else 0, axis=1)
        part["ordinary_pressure_weighted_balls"] = part.apply(lambda row: exposure_wide.loc[(row["competition_pool"], row[player_col]), "ordinary"] if "ordinary" in exposure_wide.columns else 0, axis=1)
        high = rate_wide["high"] if "high" in rate_wide.columns else pd.Series(dtype=float)
        ordinary = rate_wide["ordinary"] if "ordinary" in rate_wide.columns else pd.Series(dtype=float)
        index = pd.MultiIndex.from_frame(part[["competition_pool", player_col]])
        part["high_pressure_rate"] = index.map(high)
        part["ordinary_pressure_rate"] = index.map(ordinary)
        part["high_vs_ordinary_uplift"] = part["high_pressure_rate"] - part["ordinary_pressure_rate"]
        part["eligible_both_segments"] = (
            part["high_pressure_weighted_balls"].ge(uplift_minimum_balls)
            & part["ordinary_pressure_weighted_balls"].ge(uplift_minimum_balls)
        )
        part.loc[~part["eligible_both_segments"], "high_vs_ordinary_uplift"] = np.nan
        part["role_type"] = label
        uplift_rows.append(part.rename(columns={player_col: "player_id"}))
    uplift = pd.concat(uplift_rows, ignore_index=True) if uplift_rows else pd.DataFrame()
    return bat_match, bowl_match, uplift


def _posterior_rank(
    match_rates: pd.DataFrame,
    *,
    numerator_col: str,
    minimum_matches: int,
    minimum_balls: float,
    lower_quantile: float,
    minimum_prior_variance: float,
) -> pd.DataFrame:
    # Zero-pressure-exposure matches do not contribute to rates, uncertainty, or eligibility.
    match_rates = match_rates.loc[
        match_rates["pressure_balls"].gt(0) & match_rates["rate"].notna()
    ].copy()
    if match_rates.empty:
        return pd.DataFrame()

    id_cols = ["player_id", "player_name", "display_name", "role"]
    player = match_rates.groupby(id_cols, as_index=False, dropna=False).agg(
        weighted_residual_sum=(numerator_col, "sum"),
        pressure_weighted_balls=("pressure_balls", "sum"),
        pressure_matches=("match_id", "nunique"),
        consistency_sd=("rate", "std"),
        match_equal_mean_rate=("rate", "mean"),
        median_match_rate=("rate", "median"),
        positive_match_share=("rate", lambda values: float((values > 0).mean())),
    )
    player["consistency_sd"] = player["consistency_sd"].fillna(0.0)
    player["raw_rate"] = 100 * player["weighted_residual_sum"] / player["pressure_weighted_balls"]

    # Match-cluster robust uncertainty for a weighted ratio estimator. This
    # preserves unequal exposure and does not treat deliveries within a match
    # as independent Bernoulli/normal observations.
    means = player.set_index(id_cols)["raw_rate"] / 100.0
    match_rates = match_rates.merge(means.rename("player_mean_per_ball"), left_on=id_cols, right_index=True, validate="many_to_one")
    match_rates["cluster_score"] = match_rates[numerator_col] - match_rates["player_mean_per_ball"] * match_rates["pressure_balls"]
    cluster = match_rates.groupby(id_cols, as_index=False, dropna=False).agg(
        cluster_score_sq_sum=("cluster_score", lambda values: float(np.square(values).sum())),
        cluster_count=("match_id", "nunique"),
    )
    player = player.merge(cluster, on=id_cols, how="left", validate="one_to_one")
    cluster_correction = player["cluster_count"] / (player["cluster_count"] - 1).clip(lower=1)
    player["sampling_variance"] = (
        10000.0 * cluster_correction * player["cluster_score_sq_sum"]
        / player["pressure_weighted_balls"].pow(2)
    )

    # Method-of-moments role variance: observed player-level dispersion less
    # average sampling variance, bounded below by the frozen variance floor.
    raw_var = player.groupby("role", dropna=False)["raw_rate"].var().rename("observed_player_variance")
    mean_noise = player.groupby("role", dropna=False)["sampling_variance"].mean().rename("mean_sampling_variance")
    prior = pd.concat([raw_var, mean_noise], axis=1).reset_index()
    role_mean = player.groupby("role", dropna=False)["raw_rate"].mean().rename("role_mean").reset_index()
    prior = prior.merge(role_mean, on="role", how="left", validate="one_to_one")
    prior["role_variance"] = (
        prior["observed_player_variance"].fillna(0) - prior["mean_sampling_variance"].fillna(0)
    ).clip(lower=float(minimum_prior_variance))
    player = player.merge(prior[["role", "role_mean", "role_variance"]], on="role", how="left", validate="many_to_one")
    player["shrinkage"] = player["role_variance"] / (player["role_variance"] + player["sampling_variance"])
    player["posterior_rate"] = player["role_mean"] + player["shrinkage"] * (player["raw_rate"] - player["role_mean"])
    posterior_variance = player["role_variance"] * player["sampling_variance"] / (
        player["role_variance"] + player["sampling_variance"]
    )
    player["posterior_sd"] = np.sqrt(posterior_variance.clip(lower=0))
    z_lower = NormalDist().inv_cdf(lower_quantile)
    lower_col = f"lower_bound_{int(lower_quantile * 100)}"
    player[lower_col] = player["posterior_rate"] + z_lower * player["posterior_sd"]
    player["eligible"] = player["pressure_matches"].ge(minimum_matches) & player["pressure_weighted_balls"].ge(minimum_balls)
    player = player.loc[player["eligible"]].sort_values(
        [lower_col, "posterior_rate", "pressure_matches"], ascending=[False, False, False]
    ).reset_index(drop=True)
    player["rank"] = np.arange(1, len(player) + 1)
    return player


def score_players(data: pd.DataFrame, predictions: pd.DataFrame, config: dict):
    phase = config.get("phase3", {})
    bat_match, bowl_match, uplift = _player_match_rates(
        data, predictions, uplift_minimum_balls=float(phase.get("uplift_min_pressure_weighted_balls", 50))
    )
    batting = bat_match.rename(columns={"striker_id": "player_id"})
    bowling = bowl_match.rename(columns={"bowler_id": "player_id"})
    lower_quantile = float(phase.get("lower_posterior_quantile", 0.20))
    prior_variance = float(phase.get("minimum_prior_variance", 0.01))
    pools = phase.get("train_pools", ["IPL", "T20I"])
    ranking_options = phase.get("ranking_eligibility", {"min_pressure_matches": 10, "min_pressure_weighted_balls": 100})
    candidate_options = phase.get("candidate_eligibility", {"min_pressure_matches": 3, "min_pressure_weighted_balls": 30})
    all_bat_ranks: list[pd.DataFrame] = []
    all_bowl_ranks: list[pd.DataFrame] = []
    candidates_bat: list[pd.DataFrame] = []
    candidates_bowl: list[pd.DataFrame] = []

    def rank_pool(source: pd.DataFrame, pool: str, numerator: str, minimum: dict) -> pd.DataFrame:
        frame = source.loc[source["competition_pool"].eq(pool)].copy()
        frame = frame.rename(columns={"striker_id": "player_id", "bowler_id": "player_id"})
        return _posterior_rank(
            frame,
            numerator_col=numerator,
            minimum_matches=int(minimum["min_pressure_matches"]),
            minimum_balls=float(minimum["min_pressure_weighted_balls"]),
            lower_quantile=lower_quantile,
            minimum_prior_variance=prior_variance,
        )

    for pool in pools:
        b = rank_pool(batting, pool, "weighted_residual", ranking_options)
        w = rank_pool(bowling, pool, "weighted_residual", ranking_options)
        cb = rank_pool(batting, pool, "weighted_residual", candidate_options)
        cw = rank_pool(bowling, pool, "weighted_residual", candidate_options)
        for frame, target in ((b, all_bat_ranks), (w, all_bowl_ranks), (cb, candidates_bat), (cw, candidates_bowl)):
            if not frame.empty:
                frame.insert(0, "competition_pool", pool)
                target.append(frame)
        if not w.empty:
            wicket = bowl_match.loc[bowl_match["competition_pool"].eq(pool)].groupby("bowler_id").agg(
                wicket_residual_sum=("weighted_wicket_residual", "sum"), exposure=("pressure_balls", "sum")
            )
            w["wicket_uplift_per_100"] = w["player_id"].map(100 * wicket["wicket_residual_sum"] / wicket["exposure"].replace(0, np.nan))
        if not cw.empty:
            wicket = bowl_match.loc[bowl_match["competition_pool"].eq(pool)].groupby("bowler_id").agg(
                wicket_residual_sum=("weighted_wicket_residual", "sum"), exposure=("pressure_balls", "sum")
            )
            cw["wicket_uplift_per_100"] = cw["player_id"].map(100 * wicket["wicket_residual_sum"] / wicket["exposure"].replace(0, np.nan))

    bat_ranks = pd.concat(all_bat_ranks, ignore_index=True) if all_bat_ranks else pd.DataFrame()
    bowl_ranks = pd.concat(all_bowl_ranks, ignore_index=True) if all_bowl_ranks else pd.DataFrame()
    bat_candidates = pd.concat(candidates_bat, ignore_index=True) if candidates_bat else pd.DataFrame()
    bowl_candidates = pd.concat(candidates_bowl, ignore_index=True) if candidates_bowl else pd.DataFrame()
    if not uplift.empty:
        for ranks, kind in ((bat_ranks, "batter"), (bowl_ranks, "bowler"), (bat_candidates, "batter"), (bowl_candidates, "bowler")):
            if ranks.empty:
                continue
            values = uplift.loc[uplift["role_type"].eq(kind), ["competition_pool", "player_id", "high_vs_ordinary_uplift", "high_pressure_weighted_balls", "ordinary_pressure_weighted_balls"]]
            lookup = values.set_index(["competition_pool", "player_id"])
            idx = pd.MultiIndex.from_frame(ranks[["competition_pool", "player_id"]])
            ranks["high_vs_ordinary_uplift"] = idx.map(lookup["high_vs_ordinary_uplift"])
            ranks["high_pressure_weighted_balls"] = idx.map(lookup["high_pressure_weighted_balls"])
            ranks["ordinary_pressure_weighted_balls"] = idx.map(lookup["ordinary_pressure_weighted_balls"])

    return bat_ranks, bowl_ranks, bat_match, bowl_match, uplift, bat_candidates, bowl_candidates

"""Reconstruct pre-delivery match state without using the current outcome."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _season_year(value: object) -> int | None:
    match = re.search(r"(\d{4})", str(value))
    return int(match.group(1)) if match else None


def build_match_state(
    deliveries: pd.DataFrame,
    *,
    regular_innings_overs: int = 20,
    super_over_balls: int = 6,
) -> pd.DataFrame:
    """Add shifted score, wickets, legal-ball, chase and phase features."""
    data = deliveries.sort_values("delivery_id", kind="stable").copy()
    group_keys = ["match_id", "innings"]
    groups = data.groupby(group_keys, sort=False)

    data["score_before"] = groups["team_runs"].cumsum() - data["team_runs"]
    wicket_event = data["is_wicket"].fillna(False).astype("int64")
    legal_event = data["is_legal"].astype("int64")
    data["wickets_lost_before"] = wicket_event.groupby([data[k] for k in group_keys], sort=False).cumsum() - wicket_event
    data["legal_balls_before"] = legal_event.groupby([data[k] for k in group_keys], sort=False).cumsum() - legal_event
    data["wickets_remaining"] = (10 - data["wickets_lost_before"]).clip(lower=0)
    data["season_year"] = data["season"].map(_season_year).astype("Int64")

    innings_one = data.loc[data["innings"].eq(1)].groupby("match_id", as_index=False).agg(
        first_innings_total=("team_runs", "sum")
    )
    data = data.merge(innings_one, on="match_id", how="left", validate="many_to_one")

    scheduled_balls = pd.Series(int(regular_innings_overs) * 6, index=data.index, dtype="int64")
    revised_first = pd.to_numeric(data["first_inn_revised_overs"], errors="coerce")
    revised_second = pd.to_numeric(data["second_inn_revised_overs"], errors="coerce")
    scheduled_balls = scheduled_balls.where(data["innings"].gt(1), revised_first.fillna(scheduled_balls))
    scheduled_balls = scheduled_balls.where(~data["innings"].eq(2), revised_second.fillna(scheduled_balls))
    scheduled_balls = scheduled_balls.where(~data["is_super_over"], int(super_over_balls))
    if scheduled_balls.lt(0).any() or scheduled_balls.gt(int(regular_innings_overs) * 6).any():
        raise ValueError("Scheduled legal-ball limits exceed the configured T20 innings limit")
    data["scheduled_legal_balls"] = scheduled_balls.astype("int64")
    data["scheduled_overs"] = data["scheduled_legal_balls"] / 6.0
    data["balls_remaining"] = (data["scheduled_legal_balls"] - data["legal_balls_before"]).clip(lower=0)
    data["current_run_rate"] = data["score_before"] / (data["legal_balls_before"].replace(0, pd.NA) / 6.0)
    data["innings_phase"] = pd.cut(
        data["legal_balls_before"],
        bins=[0, 36, 90, 10_000],
        labels=["powerplay", "middle", "death"],
        right=False,
    ).astype("string")
    data["innings_type"] = data["innings"].map({1: "first_innings", 2: "chase"}).fillna("super_over")
    data["wickets_lost_bucket"] = pd.cut(
        data["wickets_lost_before"], bins=[-1, 0, 2, 5, 10], labels=["0", "1-2", "3-5", "6+"],
    ).astype("string")
    data["balls_remaining_bucket"] = pd.cut(
        data["balls_remaining"], bins=[-1, 6, 18, 30, 60, 10_000],
        labels=["0-6", "7-18", "19-30", "31-60", "61+"], right=True,
    ).astype("string")

    data["target"] = pd.NA
    chase = data["innings"].eq(2)
    revised_target = pd.to_numeric(data["revised_target"], errors="coerce")
    ordinary_target = pd.to_numeric(data["first_innings_total"], errors="coerce") + 1
    data.loc[chase & revised_target.notna(), "target"] = revised_target[chase & revised_target.notna()]
    standard_chase = chase & ~data["is_dls"].fillna(False)
    data.loc[standard_chase, "target"] = ordinary_target[standard_chase]
    data["target"] = pd.to_numeric(data["target"], errors="coerce")
    data["runs_needed"] = (data["target"] - data["score_before"]).clip(lower=0)
    data["required_run_rate"] = data["runs_needed"] / (data["balls_remaining"].replace(0, pd.NA) / 6.0)
    data.loc[~chase, ["target", "runs_needed", "required_run_rate"]] = pd.NA
    data["chase_target_available"] = chase & data["target"].notna()
    data["pressure_state_available"] = ~data["is_super_over"] & ~data["is_dls"] & (
        data["innings"].eq(1) | data["chase_target_available"]
    )
    return data

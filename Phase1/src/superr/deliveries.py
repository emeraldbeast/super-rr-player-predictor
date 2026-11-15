"""Delivery normalization and source-data attribution rules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import write_parquet


WIDE_TYPES = {"wide"}
NO_BALL_TYPES = {"overstep", "other"}
NON_BOWLER_EXTRA_TYPES = {"byes", "legbyes", "penalty"}
DEFAULT_BOWLER_WICKETS = {"bowled", "caught", "lbw", "stump", "hit wicket"}


def normalize_dismissal(value: object) -> object:
    """Normalize known spelling/format variants while preserving unknown labels."""
    if pd.isna(value):
        return pd.NA
    raw = str(value).strip()
    key = raw.casefold().replace("_", " ").replace("-", " ")
    key = " ".join(key.split())
    mapping = {
        "obstracting the field": "Obstructing the field",
        "obstructing the field": "Obstructing the field",
        "runout": "Run Out",
        "run out": "Run Out",
        "stump": "Stump",
        "hit wicket": "Hit Wicket",
        "retire out": "Retire Out",
        "lbw": "LBW",
        "bowled": "Bowled",
        "caught": "Caught",
    }
    return mapping.get(key, raw)


def build_deliveries(
    ball_by_ball: pd.DataFrame,
    matches: pd.DataFrame,
    competitions: pd.DataFrame,
    *,
    infer_other_as_no_ball: bool = True,
    infer_untyped_extras1_as_no_ball: bool = True,
    bowler_charged_extras2_assumption: bool = False,
    bowler_credited_dismissals: set[str] | None = None,
    dls_pattern: str = r"\bD/?L\b|DLS|duckworth",
    ipl_name_prefix: str = "IPL ",
    bowl_speed_min_valid: float = 0,
    bowl_speed_max_valid: float = 160,
) -> pd.DataFrame:
    """Join match context and add auditable delivery outcomes.

    `runs` is treated as striker runs. Team runs are the sum of all three run
    fields. `extras_2_runs` has no schema definition in the pack; it is excluded
    from bowler charges in the primary output and flagged for sensitivity.
    """
    credited = {v.casefold() for v in (bowler_credited_dismissals or DEFAULT_BOWLER_WICKETS)}
    match_cols = [
        "match_id", "comp_id", "match_date", "stage", "knock_out", "venue_country",
        "first_inn_revised_overs", "second_inn_revised_overs", "revised_target",
        "home_team_id", "home_team", "away_team_id", "away_team", "win_team_id", "result",
    ]
    match_context = matches[[c for c in match_cols if c in matches.columns]].copy()
    comp_context = competitions[["comp_id", "comp_name", "match_format", "season"]].copy()

    if matches["match_id"].duplicated().any():
        raise ValueError("matches.csv contains duplicate match_id values")
    if competitions["comp_id"].duplicated().any():
        raise ValueError("competitions.csv contains duplicate comp_id values")

    out = ball_by_ball.merge(match_context, on="match_id", how="left", validate="many_to_one", indicator="_match_join")
    orphan_matches = int(out["_match_join"].ne("both").sum())
    if orphan_matches:
        raise ValueError(f"{orphan_matches} delivery rows have no matching match_id")
    out = out.drop(columns="_match_join").merge(comp_context, on="comp_id", how="left", validate="many_to_one", indicator="_comp_join")
    orphan_comps = int(out["_comp_join"].ne("both").sum())
    if orphan_comps:
        raise ValueError(f"{orphan_comps} delivery rows have no matching comp_id")
    out = out.drop(columns="_comp_join")

    for col in ("runs", "extras1_runs", "extras_2_runs"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int64")
    et = out["extras_type"].fillna("").astype(str).str.strip().str.casefold()
    out["extras_type_normalized"] = et.mask(et.eq(""), pd.NA)
    out["extras_type_unresolved"] = et.eq("") & ((out["extras1_runs"] > 0) | (out["extras_2_runs"] > 0))

    out["is_wide"] = et.isin(WIDE_TYPES)
    out["is_no_ball"] = et.eq("overstep")
    if infer_other_as_no_ball:
        out["is_no_ball"] |= et.eq("other")
    if infer_untyped_extras1_as_no_ball:
        out["is_no_ball"] |= et.eq("") & out["extras1_runs"].gt(0)
    out["is_legal"] = ~(out["is_wide"] | out["is_no_ball"])

    out["bat_runs"] = out["runs"]
    out["team_runs"] = out["runs"] + out["extras1_runs"] + out["extras_2_runs"]
    out["extras2_attribution_unverified"] = out["extras_2_runs"].gt(0)

    # Byes, leg-byes and penalty extras are not charged to the bowler. The
    # `runs` field remains included if it records striker runs on that event.
    bowler_extras = out["extras1_runs"] + (out["extras_2_runs"] if bowler_charged_extras2_assumption else 0)
    nonbowler_type = et.isin(NON_BOWLER_EXTRA_TYPES)
    out["bowler_conceded"] = out["bat_runs"] + bowler_extras.where(~nonbowler_type, 0)
    # Preserve separate countable assumptions for later review.
    out["bowler_extras2_assumed_charged"] = out["extras_2_runs"].gt(0) & bowler_charged_extras2_assumption

    normalized = out["dismissal_type"].map(normalize_dismissal).astype("string")
    out["dismissal_type_normalized"] = normalized
    normalized_key = normalized.fillna("").str.casefold()
    out["bowler_wicket"] = out["is_wicket"].fillna(False).astype(bool) & normalized_key.isin(credited)
    dismissed_id = pd.to_numeric(out["dismissed_player_id"], errors="coerce")
    striker_id = pd.to_numeric(out["striker_id"], errors="coerce")
    non_striker_id = pd.to_numeric(out["non_striker_id"], errors="coerce")
    is_wicket = out["is_wicket"].fillna(False).astype(bool)
    out["striker_out"] = is_wicket & dismissed_id.notna() & striker_id.notna() & dismissed_id.eq(striker_id)
    out["nonstriker_out"] = is_wicket & dismissed_id.notna() & non_striker_id.notna() & dismissed_id.eq(non_striker_id)
    out["retired_not_out"] = ~is_wicket & dismissed_id.notna()
    out["batter_failure"] = out["striker_out"] & normalized_key.ne("retire out")

    out["is_super_over"] = out["innings"].gt(2)
    out["is_knockout"] = pd.to_numeric(out["knock_out"], errors="coerce").fillna(0).astype(int).eq(1)
    dls_columns = [c for c in ("revised_target", "first_inn_revised_overs", "second_inn_revised_overs") if c in out.columns]
    out["is_dls_fields"] = out[dls_columns].notna().any(axis=1) if dls_columns else False
    out["is_dls_text"] = out["result"].fillna("").astype(str).str.contains(dls_pattern, case=False, regex=True)
    out["is_dls"] = out["is_dls_fields"] | out["is_dls_text"]
    out["is_ipl"] = out["comp_name"].astype(str).str.startswith(ipl_name_prefix)
    out["is_t20i"] = out["match_format"].eq("T20I")
    out["is_other_t20"] = out["match_format"].eq("T20") & ~out["is_ipl"]
    out["competition_pool"] = "other T20"
    out.loc[out["is_ipl"], "competition_pool"] = "IPL"
    out.loc[out["is_t20i"], "competition_pool"] = "T20I"

    # The provider's innings_ball_num is a delivery ordinal (including illegal
    # balls), not a legal-ball number. Validate it and add deterministic IDs.
    order_cols = ["match_id", "innings", "innings_ball_num", "overid", "over_ball_num"]
    out = out.sort_values(order_cols, kind="stable").reset_index(drop=True)
    sequence_groups = out.groupby(["match_id", "innings"], sort=False, dropna=False)
    expected_seq = sequence_groups.cumcount().add(1)
    if not expected_seq.equals(pd.to_numeric(out["innings_ball_num"], errors="coerce").astype("int64")):
        raise ValueError("innings_ball_num is not a unique contiguous 1..N sequence within every innings")
    out["delivery_id"] = range(1, len(out) + 1)
    out["legal_ball_no"] = out["is_legal"].astype("int64").groupby([out["match_id"], out["innings"]], sort=False).cumsum()
    out = out.drop(columns=["bowl_speed", "bowl_speed_raw"], errors="ignore")
    return out


def write_deliveries(df: pd.DataFrame, output_path: str | Path) -> None:
    """Write normalized deliveries as Parquet."""
    write_parquet(df, output_path)

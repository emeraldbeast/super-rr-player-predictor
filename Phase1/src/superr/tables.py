"""Build player, innings, and player-match tables for Phase 1."""

from __future__ import annotations

import pandas as pd


def build_players(player_mapping: pd.DataFrame, competitions: pd.DataFrame) -> pd.DataFrame:
    mapping = player_mapping.merge(
        competitions[["comp_id", "season"]], on="comp_id", how="left", validate="many_to_one"
    ).copy()
    mapping["season_year"] = pd.to_numeric(
        mapping["season"].astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce"
    )
    rows: list[dict[str, object]] = []
    for player_id, group in mapping.groupby("player_id", sort=True):
        ordered = group.sort_values(["season_year", "comp_id"], na_position="first")
        latest = ordered.iloc[-1]
        names = sorted(group["player_name"].dropna().astype(str).unique().tolist())
        teams_recent_first = ordered.drop_duplicates("team_name", keep="last").sort_values(
            ["season_year", "comp_id"], ascending=False, na_position="last"
        )["team_name"].dropna().astype(str).tolist()
        team = teams_recent_first[0] if teams_recent_first else "Unknown team"
        name = str(latest["player_name"])
        rows.append({
            "player_id": int(player_id),
            "player_name": name,
            "name_variants": " | ".join(names),
            "display_team": team,
            "teams_seen": " | ".join(dict.fromkeys(teams_recent_first)),
            "display_name": f"{name} ({team}; ID {int(player_id)})",
            "latest_season": latest["season"],
        })
    return pd.DataFrame(rows).sort_values("player_id").reset_index(drop=True)


def build_innings_table(deliveries: pd.DataFrame, matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    keys = ["match_id", "comp_id", "innings"]
    base = deliveries.groupby(keys, as_index=False, sort=True).agg(
        batting_team_id=("batting_team_id", "first"),
        bowling_team_id=("bowling_team_id", "first"),
        batting_team_count=("batting_team_id", "nunique"),
        bowling_team_count=("bowling_team_id", "nunique"),
        delivery_rows=("delivery_id", "size"),
        total_runs=("team_runs", "sum"),
        wickets=("is_wicket", "sum"),
        legal_balls=("is_legal", "sum"),
        is_super_over=("is_super_over", "first"),
        is_knockout=("is_knockout", "first"),
        is_dls=("is_dls", "first"),
        is_dls_text=("is_dls_text", "first"),
        competition_pool=("competition_pool", "first"),
    )
    if (base["batting_team_count"].gt(1) | base["bowling_team_count"].gt(1)).any():
        raise ValueError("A match innings has more than one batting or bowling team")
    base = base.drop(columns=["batting_team_count", "bowling_team_count"])

    team_lookup = teams[["comp_id", "team_id", "team_name"]].drop_duplicates(["comp_id", "team_id"])
    base = base.merge(
        team_lookup.rename(columns={"team_id": "batting_team_id", "team_name": "batting_team"}),
        on=["comp_id", "batting_team_id"], how="left", validate="many_to_one",
    )
    base = base.merge(
        team_lookup.rename(columns={"team_id": "bowling_team_id", "team_name": "bowling_team"}),
        on=["comp_id", "bowling_team_id"], how="left", validate="many_to_one",
    )
    match_fields = [
        "match_id", "revised_target", "first_inn_revised_overs", "second_inn_revised_overs",
        "result", "win_team_id", "match_date", "stage",
    ]
    base = base.merge(matches[[c for c in match_fields if c in matches.columns]], on="match_id", how="left", validate="many_to_one")

    first_scores = base.loc[base["innings"].eq(1), ["match_id", "total_runs"]].rename(columns={"total_runs": "first_innings_total"})
    base = base.merge(first_scores, on="match_id", how="left", validate="many_to_one")
    base["target"] = pd.NA
    chase = base["innings"].eq(2)
    revised_target = pd.to_numeric(base["revised_target"], errors="coerce")
    ordinary_target = pd.to_numeric(base["first_innings_total"], errors="coerce") + 1
    base.loc[chase & revised_target.notna(), "target"] = revised_target[chase & revised_target.notna()]
    # If DLS is mentioned but no revised target is supplied, leave target null.
    normal_chase = chase & ~base["is_dls"]
    base.loc[normal_chase, "target"] = ordinary_target[normal_chase]
    base["target"] = pd.to_numeric(base["target"], errors="coerce")

    revised_balls = pd.to_numeric(base["first_inn_revised_overs"], errors="coerce").where(
        base["innings"].eq(1), pd.to_numeric(base["second_inn_revised_overs"], errors="coerce")
    )
    scheduled_balls = revised_balls.fillna(120).round().astype("int64")
    scheduled_balls = scheduled_balls.where(~base["is_super_over"], 6)
    if scheduled_balls.lt(0).any() or scheduled_balls.gt(120).any():
        raise ValueError("Scheduled legal-ball limits must be in 0..120 for T20 or 6 for Super Overs")
    base["scheduled_legal_balls"] = scheduled_balls
    base["scheduled_overs"] = base["scheduled_legal_balls"] / 6.0

    reached_by_state = chase & base["target"].notna() & base["total_runs"].ge(base["target"])
    reached_by_result = chase & pd.to_numeric(base["win_team_id"], errors="coerce").eq(
        pd.to_numeric(base["batting_team_id"], errors="coerce")
    )
    base["termination_reason"] = "truncated"
    base.loc[base["wickets"].ge(10), "termination_reason"] = "all out"
    base.loc[base["legal_balls"].ge(base["scheduled_legal_balls"]), "termination_reason"] = "overs complete"
    base.loc[reached_by_state | reached_by_result, "termination_reason"] = "target reached"

    columns = [
        "match_id", "comp_id", "match_date", "stage", "competition_pool", "innings",
        "batting_team_id", "batting_team", "bowling_team_id", "bowling_team",
        "total_runs", "wickets", "legal_balls", "scheduled_overs", "scheduled_legal_balls",
        "target", "is_dls", "is_dls_text", "is_knockout", "is_super_over", "termination_reason",
        "delivery_rows", "first_innings_total", "result", "win_team_id",
    ]
    return base[columns].sort_values(["match_id", "innings"]).reset_index(drop=True)


def build_player_match(
    playing_eleven: pd.DataFrame,
    deliveries: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    roster = playing_eleven.copy()
    roster["is_played"] = roster["is_played"].fillna(False).astype(bool)
    roster = roster.drop_duplicates(["match_id", "team_id", "player_id"])
    roster["roster_row"] = True

    bat_rows = deliveries.assign(
        balls_faced_row=~deliveries["is_wide"],
        striker_seen=True,
        nonstriker_seen=False,
        bowler_seen=False,
        balls_bowled_row=False,
    )[["match_id", "batting_team_id", "striker_id", "balls_faced_row", "striker_seen", "nonstriker_seen", "bowler_seen", "balls_bowled_row"]].rename(
        columns={"batting_team_id": "team_id", "striker_id": "player_id"}
    )
    non_rows = deliveries.assign(
        balls_faced_row=False,
        striker_seen=False,
        nonstriker_seen=True,
        bowler_seen=False,
        balls_bowled_row=False,
    )[["match_id", "batting_team_id", "non_striker_id", "balls_faced_row", "striker_seen", "nonstriker_seen", "bowler_seen", "balls_bowled_row"]].rename(
        columns={"batting_team_id": "team_id", "non_striker_id": "player_id"}
    )
    bowl_rows = deliveries.assign(
        balls_faced_row=False,
        striker_seen=False,
        nonstriker_seen=False,
        bowler_seen=True,
        balls_bowled_row=deliveries["is_legal"],
    )[["match_id", "bowling_team_id", "bowler_id", "balls_faced_row", "striker_seen", "nonstriker_seen", "bowler_seen", "balls_bowled_row"]].rename(
        columns={"bowling_team_id": "team_id", "bowler_id": "player_id"}
    )
    events = pd.concat([bat_rows, non_rows, bowl_rows], ignore_index=True)
    events = events.loc[events["player_id"].notna()].copy()
    events["balls_faced_row"] = events["balls_faced_row"].astype(int)
    events["balls_bowled_row"] = events["balls_bowled_row"].astype(int)
    event_stats = events.groupby(["match_id", "team_id", "player_id"], as_index=False).agg(
        balls_faced=("balls_faced_row", "sum"),
        balls_bowled=("balls_bowled_row", "sum"),
        appeared_as_striker=("striker_seen", "any"),
        appeared_as_nonstriker=("nonstriker_seen", "any"),
        bowled=("bowler_seen", "any"),
    )
    event_stats["batted"] = event_stats["appeared_as_striker"] | event_stats["appeared_as_nonstriker"]
    event_stats["event_participant"] = True

    base = roster.merge(event_stats, on=["match_id", "team_id", "player_id"], how="outer", validate="one_to_one")
    base["roster_row"] = base["roster_row"].astype("boolean").fillna(False).astype(bool)
    base["is_played"] = base["is_played"].astype("boolean").fillna(False).astype(bool)
    base["event_participant"] = base["event_participant"].astype("boolean").fillna(False).astype(bool)
    for col in ("balls_faced", "balls_bowled"):
        base[col] = base[col].fillna(0).astype("int64")
    for col in ("batted", "appeared_as_striker", "appeared_as_nonstriker", "bowled"):
        base[col] = base[col].astype("boolean").fillna(False).astype(bool)
    base["not_in_played_xi"] = base["event_participant"] & ~base["is_played"]
    base = base.merge(players[["player_id", "player_name", "display_name"]], on="player_id", how="left", validate="many_to_one")
    return base.sort_values(["match_id", "team_id", "player_id"]).reset_index(drop=True)

"""Coverage, result reconciliation, delivery structure, and Phase 1 audit."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .deliveries import build_deliveries
from .io import verify_checksums, write_parquet
from .tables import build_innings_table, build_player_match, build_players


def _pool_name(comp_name: object, match_format: object, ipl_prefix: str) -> str:
    if str(comp_name).startswith(ipl_prefix):
        return "IPL"
    if str(match_format) == "T20I":
        return "T20I"
    return "other T20"


def _delivery_settings(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    competitions = config.get("competitions", {})
    return {
        "infer_other_as_no_ball": bool(data.get("infer_other_as_no_ball", True)),
        "infer_untyped_extras1_as_no_ball": bool(data.get("infer_untyped_extras1_as_no_ball", True)),
        "bowler_charged_extras2_assumption": bool(data.get("bowler_charged_extras2_assumption", False)),
        "bowler_credited_dismissals": set(data.get("bowler_credited_dismissals", [])) or None,
        "dls_pattern": str(data.get("result_dls_pattern", r"\bD/?L\b|DLS|duckworth")),
        "ipl_name_prefix": str(competitions.get("ipl_name_prefix", "IPL ")),
        "bowl_speed_min_valid": float(data.get("bowl_speed_min_valid", 0)),
        "bowl_speed_max_valid": float(data.get("bowl_speed_max_valid", 160)),
    }


def _result_reconciliation(deliveries: pd.DataFrame, innings: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"\bwon\s+by\s*(\d+)\s*(runs?|wickets?|wkts?)\b", flags=re.IGNORECASE)
    scorecards = innings.loc[~innings["is_super_over"]].copy()
    rows: list[dict[str, object]] = []
    by_match = {mid: group for mid, group in scorecards.groupby("match_id", sort=False)}
    team_inn = deliveries.loc[~deliveries["is_super_over"]].groupby(["match_id", "innings"], sort=False)["batting_team_id"].first()
    for match in matches.itertuples(index=False):
        result = "" if pd.isna(match.result) else str(match.result)
        parsed = pattern.search(result)
        if not parsed:
            continue
        data = by_match.get(match.match_id)
        if data is None or data["innings"].nunique() < 2 or pd.isna(match.win_team_id):
            continue
        reported = int(parsed.group(1))
        unit_raw = parsed.group(2).casefold()
        unit = "wickets" if unit_raw.startswith("wicket") or unit_raw.startswith("wkt") else "runs"
        winner_id = int(match.win_team_id)
        winner_innings = None
        for inn_num in (1, 2):
            key = (match.match_id, inn_num)
            if key in team_inn.index and int(team_inn.loc[key]) == winner_id:
                winner_innings = inn_num
                break
        actual: int | None = None
        status = "unresolved_winner_team"
        if winner_innings is not None:
            if unit == "wickets":
                score = data.loc[data["innings"].eq(winner_innings)]
                if not score.empty:
                    actual = 10 - int(score.iloc[0]["wickets"])
            else:
                winner = data.loc[data["innings"].eq(winner_innings)]
                loser = data.loc[data["innings"].ne(winner_innings)]
                if not winner.empty and not loser.empty:
                    actual = abs(int(winner.iloc[0]["total_runs"]) - int(loser.iloc[0]["total_runs"]))
            if actual is not None:
                status = "reconciles" if actual == reported else "margin_mismatch"
        is_dls = bool(deliveries.loc[deliveries["match_id"].eq(match.match_id), "is_dls"].any())
        # DLS results are compared to an adjusted target/margin that is not
        # always provided. Keep them visible but out of the direct-margin rate.
        if is_dls:
            status = "dls_adjusted_not_directly_comparable"
        rows.append({
            "match_id": int(match.match_id),
            "comp_id": int(match.comp_id),
            "result": result,
            "margin_unit": unit,
            "reported_margin": reported,
            "derived_margin": actual,
            "is_dls": is_dls,
            "status": status,
        })
    return pd.DataFrame(rows)


def _over_structure(deliveries: pd.DataFrame, matches: pd.DataFrame, expected_legal_balls: int) -> pd.DataFrame:
    regular = deliveries.loc[~deliveries["is_super_over"]].copy()
    over = regular.groupby(["match_id", "innings", "overid"], as_index=False).agg(
        legal_balls=("is_legal", "sum"),
        delivery_rows=("delivery_id", "size"),
        max_over_ball_num=("over_ball_num", "max"),
        batting_team_id=("batting_team_id", "first"),
    )
    over["final_over"] = over["overid"].eq(over.groupby(["match_id", "innings"])["overid"].transform("max"))
    over["expected_legal_balls"] = expected_legal_balls
    over["structure_exception"] = ~over["final_over"] & over["legal_balls"].ne(expected_legal_balls)
    meta = deliveries[["match_id", "comp_id", "home_team", "away_team"]].drop_duplicates("match_id")
    return over.merge(meta, on="match_id", how="left", validate="many_to_one").sort_values(
        ["match_id", "innings", "overid"]
    ).reset_index(drop=True)


def _column_policy(columns: list[str]) -> pd.DataFrame:
    pre_ball = {
        "match_id", "comp_id", "comp_name", "match_format", "season", "match_date", "match_number",
        "stage", "venue_id", "venue_country", "home_team_id", "home_team", "away_team_id", "away_team",
        "home_team_captain_id", "away_team_captain_id", "toss_win_team_id", "toss_opted", "knock_out",
        "innings", "overid", "over_ball_num", "innings_ball_num", "batting_team_id", "bowling_team_id",
        "striker_id", "striker_name", "non_striker_id", "non_striker_name", "bowler_id", "bowler_name",
        "batting_type", "bowling_type", "batting_order", "pace_or_spin", "first_inn_revised_overs",
        "second_inn_revised_overs", "revised_target", "is_super_over", "is_knockout", "is_dls_fields",
        "is_dls_text", "is_dls", "is_ipl", "is_t20i", "is_other_t20", "competition_pool", "delivery_id",
    }
    outcomes = {
        "runs", "is_four", "is_six", "is_wicket", "dismissal_type", "dismissal_type_normalized",
        "dismissal_type_id", "dismissed_player_id", "extras_type", "extras_type_normalized", "extras1_runs",
        "extras_2_runs", "bat_runs", "team_runs", "bowler_conceded", "bowler_wicket", "striker_out",
        "nonstriker_out", "retired_not_out", "batter_failure", "is_legal", "is_wide", "is_no_ball",
        "extras_type_unresolved", "extras2_attribution_unverified", "bowler_extras2_assumed_charged",
        "legal_ball_no",
    }
    post_ball = {
        "result", "win_team_id", "man_of_the_match", "bowl_speed", "bowl_speed_raw", "stroke",
        "delivery_style", "over_or_around", "on_or_off", "delivery_ist_time",
    }
    notes = {
        "pre-ball safe": "Available before outcome or stable match/player context.",
        "outcome": "Observed during/after delivery; never use as a pre-delivery predictor.",
        "post-ball": "Measured or annotated after delivery, or match outcome; exclude from pre-ball predictors.",
    }
    rows = []
    for col in columns:
        if col in pre_ball:
            policy = "pre-ball safe"
        elif col in outcomes:
            policy = "outcome"
        elif col in post_ball:
            policy = "post-ball"
        else:
            policy = "post-ball"
        rows.append({"column": col, "policy": policy, "use_note": notes[policy]})
    return pd.DataFrame(rows)


def write_run_manifest(raw_dir: str | Path, output_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    checksums = verify_checksums(raw_dir)
    versions: dict[str, str] = {"python": sys.version.split()[0], "pandas": pd.__version__}
    for module_name in ("pyarrow", "yaml"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = "not installed"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "input_sha256": checksums,
        "package_versions": versions,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def run_audit(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
    interim_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    b, m, c = tables["ball_by_ball"], tables["matches"], tables["competitions"]
    pe, player_mapping, team_table = tables["playing_eleven"], tables["player_mapping"], tables["teams"]
    settings = _delivery_settings(config)

    match_context = m.merge(c[["comp_id", "comp_name", "match_format", "season"]], on="comp_id", how="left", validate="many_to_one")
    match_context["competition_pool"] = match_context.apply(
        lambda x: _pool_name(x["comp_name"], x["match_format"], settings["ipl_name_prefix"]), axis=1
    )
    delivery_counts = b.groupby("match_id").size().rename("delivery_rows")
    match_context = match_context.join(delivery_counts, on="match_id")
    match_context["has_delivery_data"] = match_context["delivery_rows"].notna()
    coverage_by_competition = match_context.groupby(
        ["competition_pool", "match_format", "comp_name", "season"], dropna=False
    ).agg(
        matches=("match_id", "nunique"),
        matches_with_deliveries=("has_delivery_data", "sum"),
        delivery_rows=("delivery_rows", "sum"),
    ).reset_index()
    coverage_by_competition["matches_without_deliveries"] = coverage_by_competition["matches"] - coverage_by_competition["matches_with_deliveries"]
    coverage_by_pool = match_context.groupby("competition_pool", dropna=False).agg(
        matches=("match_id", "nunique"),
        matches_with_deliveries=("has_delivery_data", "sum"),
        delivery_rows=("delivery_rows", "sum"),
    ).reset_index()
    coverage_by_pool["matches_without_deliveries"] = coverage_by_pool["matches"] - coverage_by_pool["matches_with_deliveries"]
    missing_matches = match_context.loc[~match_context["has_delivery_data"], [
        "match_id", "comp_id", "competition_pool", "comp_name", "match_format", "season", "match_date", "home_team", "away_team", "result"
    ]].copy()

    bcopy = b.copy()
    for col in ("runs", "extras1_runs", "extras_2_runs"):
        bcopy[col] = pd.to_numeric(bcopy[col], errors="coerce").fillna(0)
    bcopy["team_runs_from_components"] = bcopy["runs"] + bcopy["extras1_runs"] + bcopy["extras_2_runs"]
    extras = bcopy.assign(extras_type=bcopy["extras_type"].fillna("<none>")).groupby("extras_type", dropna=False).agg(
        deliveries=("match_id", "size"), runs_sum=("runs", "sum"), extras1_sum=("extras1_runs", "sum"),
        extras2_sum=("extras_2_runs", "sum"), team_runs_from_components=("team_runs_from_components", "sum"),
        extras1_min=("extras1_runs", "min"), extras1_max=("extras1_runs", "max"),
        extras2_nonzero=("extras_2_runs", lambda s: int(s.gt(0).sum())),
    ).reset_index()

    deliveries = build_deliveries(b, m, c, **settings)
    players = build_players(player_mapping, c)
    innings = build_innings_table(deliveries, m, team_table)
    player_match = build_player_match(pe, deliveries, players)
    normal_innings_counts = innings.loc[~innings["is_super_over"]].groupby("match_id")["innings"].nunique().rename("normal_innings_with_delivery_data")
    incomplete_innings_matches = match_context.join(normal_innings_counts, on="match_id")
    incomplete_innings_matches["normal_innings_with_delivery_data"] = incomplete_innings_matches["normal_innings_with_delivery_data"].fillna(0).astype(int)
    incomplete_innings_matches = incomplete_innings_matches.loc[
        incomplete_innings_matches["normal_innings_with_delivery_data"].eq(1),
        ["match_id", "competition_pool", "comp_name", "match_date", "home_team", "away_team", "result", "normal_innings_with_delivery_data"],
    ].copy()

    # Component-level per-innings reconciliation (source scorecard totals are
    # not included in matches.csv; result margins below provide an independent check).
    innings_components = deliveries.groupby(["match_id", "innings"], as_index=False).agg(
        legal_balls=("is_legal", "sum"),
        total_runs=("team_runs", "sum"),
        wickets=("is_wicket", "sum"),
        missing_dismissal_type_on_wicket=("wicket_missing_dismissal_type", "sum"),
    ) if "wicket_missing_dismissal_type" in deliveries else deliveries.groupby(["match_id", "innings"], as_index=False).agg(
        legal_balls=("is_legal", "sum"), total_runs=("team_runs", "sum"), wickets=("is_wicket", "sum")
    )
    innings_components = innings_components.merge(
        innings[["match_id", "innings", "scheduled_legal_balls", "termination_reason", "batting_team", "bowling_team"]],
        on=["match_id", "innings"], how="left", validate="one_to_one",
    )
    innings_components["legal_balls_within_limit"] = innings_components["legal_balls"].le(innings_components["scheduled_legal_balls"])
    innings_components["wickets_within_limit"] = innings_components["wickets"].le(10)

    over_structure = _over_structure(
        deliveries, m, int(config.get("validation", {}).get("expected_nonfinal_over_legal_balls", 6))
    )
    result_reconciliation = _result_reconciliation(deliveries, innings, m)

    duplicates_by_name = players.groupby("player_name").agg(
        player_ids=("player_id", "nunique"), display_names=("display_name", lambda s: " | ".join(s)),
    ).reset_index()
    duplicate_player_names = duplicates_by_name.loc[duplicates_by_name["player_ids"].gt(1)].copy()
    players_name_conflicts = players.loc[players["name_variants"].str.contains(r"\|", regex=True), [
        "player_id", "player_name", "name_variants", "display_name"
    ]].copy()
    participants_not_in_played_xi = player_match.loc[player_match["not_in_played_xi"], [
        "match_id", "team_id", "player_id", "player_name", "display_name", "appeared_as_striker",
        "appeared_as_nonstriker", "bowled", "balls_faced", "balls_bowled",
    ]].copy()
    role_exceptions: list[pd.DataFrame] = []
    for role, flag in (
        ("striker", "appeared_as_striker"),
        ("non_striker", "appeared_as_nonstriker"),
        ("bowler", "bowled"),
    ):
        part = player_match.loc[player_match["not_in_played_xi"] & player_match[flag]].copy()
        part["role"] = role
        role_exceptions.append(part[["match_id", "team_id", "player_id", "player_name", "display_name", "role"]])
    participant_role_exceptions = pd.concat(role_exceptions, ignore_index=True)
    impact_usage = pe[["is_impact", "is_impact_off"]].apply(pd.to_numeric, errors="coerce").fillna(0).sum().rename(
        "flagged_rows"
    ).rename_axis("field").reset_index()
    impact_usage["total_rows"] = len(pe)
    duplicate_roster_rows = pe.loc[pe.duplicated(["match_id", "team_id", "player_id"], keep=False)].sort_values(
        ["match_id", "team_id", "player_id"]
    ).copy()
    unique_roster = pe.assign(is_played=pe["is_played"].fillna(False).astype(bool)).groupby(
        ["match_id", "team_id", "player_id"], as_index=False
    ).agg(is_played=("is_played", "max"))
    played_counts = unique_roster.groupby(["match_id", "team_id"], as_index=False).agg(
        played_slots=("is_played", "sum"), roster_slots=("player_id", "nunique")
    )
    played_counts["more_than_11_played_slots"] = played_counts["played_slots"].gt(11)

    speed_raw = pd.to_numeric(b["bowl_speed"], errors="coerce")
    speed_audit = pd.DataFrame([
        {"measure": "rows", "value": int(speed_raw.notna().sum())},
        {"measure": "zero_missing_candidates", "value": int(speed_raw.eq(0).sum())},
        {"measure": "negative_or_sentinel", "value": int(speed_raw.lt(0).sum())},
        {"measure": "above_160", "value": int(speed_raw.gt(160).sum())},
        {"measure": "maximum", "value": float(speed_raw.max())},
    ])
    dls_field_matches = int(m[["revised_target", "first_inn_revised_overs", "second_inn_revised_overs"]].notna().any(axis=1).sum())
    dls_text_matches = int(m["result"].fillna("").astype(str).str.contains(settings["dls_pattern"], case=False, regex=True).sum())
    dls_missing_fields = m.loc[
        m["result"].fillna("").astype(str).str.contains(settings["dls_pattern"], case=False, regex=True)
        & m[["revised_target", "first_inn_revised_overs", "second_inn_revised_overs"]].isna().all(axis=1),
        ["match_id", "comp_id", "match_date", "home_team", "away_team", "result"],
    ].copy()

    manifest = _column_policy(deliveries.columns.tolist())
    result_standard = result_reconciliation.loc[
        ~result_reconciliation["is_dls"] & result_reconciliation["status"].isin(["reconciles", "margin_mismatch"])
    ]
    result_exact_rate = float(result_standard["status"].eq("reconciles").mean()) if len(result_standard) else None
    result_threshold = float(config.get("validation", {}).get("result_margin_exact_match_threshold", 0.99))
    role_exception_counts = participant_role_exceptions.groupby("role").size().to_dict()
    summary = {
        "matches": int(m["match_id"].nunique()),
        "matches_with_deliveries": int(b["match_id"].nunique()),
        "deliveries": int(len(b)),
        "pool_coverage": coverage_by_pool.to_dict("records"),
        "missing_delivery_matches": int(len(missing_matches)),
        "matches_with_one_recorded_normal_innings": int(len(incomplete_innings_matches)),
        "one_innings_result_counts": {str(k): int(v) for k, v in incomplete_innings_matches["result"].value_counts(dropna=False).items()},
        "is_dls_field_matches": dls_field_matches,
        "is_dls_text_matches": dls_text_matches,
        "dls_text_without_revision_fields": int(len(dls_missing_fields)),
        "innings": int(len(innings)),
        "team_run_sum": int(deliveries["team_runs"].sum()),
        "innings_legal_balls_within_limit": bool(innings_components["legal_balls_within_limit"].all()),
        "innings_wickets_within_limit": bool(innings_components["wickets_within_limit"].all()),
        "nonfinal_over_structure_exceptions": int(over_structure["structure_exception"].sum()),
        "direct_result_margins_checked": int(len(result_standard)),
        "direct_result_margins_exact_rate": result_exact_rate,
        "direct_result_margin_threshold": result_threshold,
        "direct_result_margin_threshold_met": bool(result_exact_rate is not None and result_exact_rate >= result_threshold),
        "result_margin_exceptions": result_reconciliation.loc[result_reconciliation["status"].eq("margin_mismatch"), "match_id"].astype(int).tolist(),
        "players": int(len(players)),
        "duplicate_player_names": int(len(duplicate_player_names)),
        "event_participant_keys_not_in_played_xi": int(len(participants_not_in_played_xi)),
        "event_role_slots_not_in_played_xi": int(len(participant_role_exceptions)),
        "event_role_slot_counts_not_in_played_xi": {k: int(v) for k, v in role_exception_counts.items()},
        "duplicate_playing_eleven_keys": int(len(duplicate_roster_rows)),
        "max_unique_played_slots_per_team": int(played_counts["played_slots"].max()),
        "bowler_extras2_assumption": "not charged in primary; sensitivity option in config",
    }

    outputs = {
        "coverage_by_pool": coverage_by_pool,
        "coverage_by_competition": coverage_by_competition,
        "matches_missing_deliveries": missing_matches,
        "matches_with_incomplete_innings": incomplete_innings_matches,
        "extras_reconciliation": extras,
        "innings_reconciliation": innings_components,
        "innings_table": innings,
        "result_margin_reconciliation": result_reconciliation,
        "over_structure": over_structure,
        "dls_text_without_revision_fields": dls_missing_fields,
        "players": players,
        "duplicate_player_names": duplicate_player_names,
        "players_with_name_variants": players_name_conflicts,
        "player_match": player_match,
        "participants_not_in_played_xi": participants_not_in_played_xi,
        "participant_role_exceptions": participant_role_exceptions,
        "duplicate_playing_eleven_rows": duplicate_roster_rows,
        "played_slots_by_match_team": played_counts,
        "impact_player_usage": impact_usage,
        "bowl_speed_audit": speed_audit,
        "delivery_column_policy": manifest,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    output_path = Path(interim_dir) if interim_dir else output_dir.parent.parent / "data" / "interim"
    output_path.mkdir(parents=True, exist_ok=True)
    write_parquet(deliveries, output_path / "deliveries.parquet")
    write_parquet(innings, output_path / "innings.parquet")
    write_parquet(player_match, output_path / "player_match.parquet")
    write_parquet(players, output_path / "players.parquet")

    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return outputs

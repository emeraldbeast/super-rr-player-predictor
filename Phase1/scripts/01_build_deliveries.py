from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superr.audit import write_run_manifest
from superr.deliveries import build_deliveries, write_deliveries
from superr.io import load_config, load_csvs
from superr.tables import build_innings_table, build_player_match, build_players


def main() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    tables = load_csvs(ROOT / "data" / "raw", verify=True)
    data_config = config.get("data", {})
    competitions_config = config.get("competitions", {})
    deliveries = build_deliveries(
        tables["ball_by_ball"],
        tables["matches"],
        tables["competitions"],
        infer_other_as_no_ball=bool(data_config.get("infer_other_as_no_ball", True)),
        infer_untyped_extras1_as_no_ball=bool(data_config.get("infer_untyped_extras1_as_no_ball", True)),
        bowler_charged_extras2_assumption=bool(data_config.get("bowler_charged_extras2_assumption", False)),
        bowler_credited_dismissals=set(data_config.get("bowler_credited_dismissals", [])) or None,
        dls_pattern=str(data_config.get("result_dls_pattern", r"\bD/?L\b|DLS|duckworth")),
        ipl_name_prefix=str(competitions_config.get("ipl_name_prefix", "IPL ")),
        bowl_speed_min_valid=float(data_config.get("bowl_speed_min_valid", 0)),
        bowl_speed_max_valid=float(data_config.get("bowl_speed_max_valid", 160)),
    )
    interim = ROOT / "data" / "interim"
    output = interim / "deliveries.parquet"
    write_deliveries(deliveries, output)
    players = build_players(tables["player_mapping"], tables["competitions"])
    innings = build_innings_table(deliveries, tables["matches"], tables["teams"])
    player_match = build_player_match(tables["playing_eleven"], deliveries, players)
    from superr.io import write_parquet
    write_parquet(innings, interim / "innings.parquet")
    write_parquet(player_match, interim / "player_match.parquet")
    write_parquet(players, interim / "players.parquet")
    write_run_manifest(ROOT / "data" / "raw", ROOT / "outputs" / "audit" / "run_manifest.json", config)
    print(f"Wrote {len(deliveries):,} delivery rows to {output}")
    print(f"Wrote {len(innings):,} innings, {len(player_match):,} player-match, and {len(players):,} player rows")
    print(f"Legal balls: {int(deliveries['is_legal'].sum()):,}; bowler wickets: {int(deliveries['bowler_wicket'].sum()):,}")
    print(f"Rows with unresolved extras type/attribution: {int(deliveries['extras_type_unresolved'].sum()):,} / {int(deliveries['extras2_attribution_unverified'].sum()):,}")


if __name__ == "__main__":
    main()

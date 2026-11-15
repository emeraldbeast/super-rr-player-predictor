"""Build pressure-adjusted player rankings from cross-fitted residuals."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from superr.io import load_config
from superr.scoring import score_players


def main() -> None:
    config = load_config(ROOT / "configs" / "config.yaml")
    table = pd.read_parquet(ROOT / "data" / "processed" / "model_table.parquet")
    predictions = pd.read_parquet(ROOT / "data" / "processed" / "expected_outcomes_oof.parquet")
    batting, bowling, batting_matches, bowling_matches, uplift, bat_candidates, bowl_candidates = score_players(table, predictions, config)
    out = ROOT / "outputs"
    batting.to_csv(out / "phase3_batting_rankings_all_pools.csv", index=False)
    bowling.to_csv(out / "phase3_bowling_rankings_all_pools.csv", index=False)
    batting_matches.to_csv(out / "phase3_batting_player_match.csv", index=False)
    bowling_matches.to_csv(out / "phase3_bowling_player_match.csv", index=False)
    uplift.to_csv(out / "phase3_high_pressure_uplift.csv", index=False)
    bat_candidates.to_csv(out / "phase3_batting_candidate_rankings.csv", index=False)
    bowl_candidates.to_csv(out / "phase3_bowling_candidate_rankings.csv", index=False)
    primary = config["phase3"]["primary_ranking_pool"]
    batting.loc[batting["competition_pool"].eq(primary)].head(10).to_csv(out / "top10_batters.csv", index=False)
    bowling.loc[bowling["competition_pool"].eq(primary)].head(10).to_csv(out / "top10_bowlers.csv", index=False)
    summary = {
        "primary_pool": primary,
        "eligible_batters_primary": int(batting["competition_pool"].eq(primary).sum()),
        "eligible_bowlers_primary": int(bowling["competition_pool"].eq(primary).sum()),
        "top10_batters_written": int(min(10, batting["competition_pool"].eq(primary).sum())),
        "top10_bowlers_written": int(min(10, bowling["competition_pool"].eq(primary).sum())),
    }
    (out / "phase3_scoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nIPL batting top 10")
    print(batting.loc[batting["competition_pool"].eq(primary), ["rank", "player_name", "role", "posterior_rate", "lower_bound_20", "pressure_matches", "pressure_weighted_balls"]].head(10).to_string(index=False))
    print("\nIPL bowling top 10")
    print(bowling.loc[bowling["competition_pool"].eq(primary), ["rank", "player_name", "role", "posterior_rate", "lower_bound_20", "wicket_uplift_per_100", "pressure_matches", "pressure_weighted_balls"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

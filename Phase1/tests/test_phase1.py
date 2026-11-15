from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superr.deliveries import build_deliveries, normalize_dismissal


def fixture_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    balls = pd.DataFrame([
        # wide: illegal, one run charged to bowler
        {"match_id": 1, "innings": 1, "overid": 1, "over_ball_num": 1, "batting_team_id": 10, "bowling_team_id": 20,
         "striker_id": 100, "striker_name": "A", "non_striker_id": 101, "non_striker_name": "B", "bowler_id": 200,
         "bowler_name": "X", "runs": 0, "extras_type": "wide", "extras1_runs": 1, "extras_2_runs": 0,
         "is_wicket": False, "dismissal_type": None, "dismissed_player_id": None},
        # bye: legal, not charged to bowler
        {"match_id": 1, "innings": 1, "overid": 1, "over_ball_num": 2, "batting_team_id": 10, "bowling_team_id": 20,
         "striker_id": 100, "striker_name": "A", "non_striker_id": 101, "non_striker_name": "B", "bowler_id": 200,
         "bowler_name": "X", "runs": 0, "extras_type": "byes", "extras1_runs": 4, "extras_2_runs": 0,
         "is_wicket": False, "dismissal_type": None, "dismissed_player_id": None},
        # Other is treated as a no-ball in the supplied feed; runs and no-ball extras charged.
        {"match_id": 1, "innings": 1, "overid": 1, "over_ball_num": 3, "batting_team_id": 10, "bowling_team_id": 20,
         "striker_id": 100, "striker_name": "A", "non_striker_id": 101, "non_striker_name": "B", "bowler_id": 200,
         "bowler_name": "X", "runs": 4, "extras_type": "other", "extras1_runs": 1, "extras_2_runs": 0,
         "is_wicket": False, "dismissal_type": None, "dismissed_player_id": None},
        # Bowler-credited wicket
        {"match_id": 1, "innings": 1, "overid": 1, "over_ball_num": 4, "batting_team_id": 10, "bowling_team_id": 20,
         "striker_id": 100, "striker_name": "A", "non_striker_id": 101, "non_striker_name": "B", "bowler_id": 200,
         "bowler_name": "X", "runs": 0, "extras_type": None, "extras1_runs": 0, "extras_2_runs": 0,
         "is_wicket": True, "dismissal_type": "Bowled", "dismissed_player_id": 100},
        # Run out is a batter dismissal but not a bowler wicket.
        {"match_id": 1, "innings": 1, "overid": 1, "over_ball_num": 5, "batting_team_id": 10, "bowling_team_id": 20,
         "striker_id": 100, "striker_name": "A", "non_striker_id": 101, "non_striker_name": "B", "bowler_id": 200,
         "bowler_name": "X", "runs": 0, "extras_type": None, "extras1_runs": 0, "extras_2_runs": 0,
         "is_wicket": True, "dismissal_type": "RunOut", "dismissed_player_id": 101},
        # Penalty extras are not charged to the bowler.
        {"match_id": 1, "innings": 1, "overid": 1, "over_ball_num": 6, "batting_team_id": 10, "bowling_team_id": 20,
         "striker_id": 100, "striker_name": "A", "non_striker_id": 101, "non_striker_name": "B", "bowler_id": 200,
         "bowler_name": "X", "runs": 0, "extras_type": "Penalty", "extras1_runs": 5, "extras_2_runs": 0,
         "is_wicket": False, "dismissal_type": None, "dismissed_player_id": None},
        # extras_2_runs stays in team runs but is not bowler-charged by default.
        {"match_id": 1, "innings": 1, "overid": 2, "over_ball_num": 1, "batting_team_id": 10, "bowling_team_id": 20,
         "striker_id": 100, "striker_name": "A", "non_striker_id": 101, "non_striker_name": "B", "bowler_id": 200,
         "bowler_name": "X", "runs": 0, "extras_type": "Overstep", "extras1_runs": 1, "extras_2_runs": 4,
         "is_wicket": False, "dismissal_type": None, "dismissed_player_id": None},
        # Unresolved non-wicket dismissed-player rows are flagged, not batter failures.
        {"match_id": 1, "innings": 1, "overid": 2, "over_ball_num": 2, "batting_team_id": 10, "bowling_team_id": 20,
         "striker_id": 100, "striker_name": "A", "non_striker_id": 101, "non_striker_name": "B", "bowler_id": 200,
         "bowler_name": "X", "runs": 0, "extras_type": None, "extras1_runs": 0, "extras_2_runs": 0,
         "is_wicket": False, "dismissal_type": None, "dismissed_player_id": 100},
        # Retire Out is an event, but not counted as a batter failure.
        {"match_id": 1, "innings": 1, "overid": 2, "over_ball_num": 3, "batting_team_id": 10, "bowling_team_id": 20,
         "striker_id": 100, "striker_name": "A", "non_striker_id": 101, "non_striker_name": "B", "bowler_id": 200,
         "bowler_name": "X", "runs": 0, "extras_type": None, "extras1_runs": 0, "extras_2_runs": 0,
         "is_wicket": True, "dismissal_type": "Retire Out", "dismissed_player_id": 100},
    ])
    balls["innings_ball_num"] = range(1, len(balls) + 1)
    balls["bowl_speed"] = 0.0
    matches = pd.DataFrame([{
        "match_id": 1, "comp_id": 9, "match_date": "2026-01-01", "stage": "League", "knock_out": 0,
        "venue_country": "India", "first_inn_revised_overs": None, "second_inn_revised_overs": None,
        "revised_target": None, "home_team_id": 10, "home_team": "A", "away_team_id": 20,
        "away_team": "B", "win_team_id": 10, "result": "A won",
    }])
    competitions = pd.DataFrame([{"comp_id": 9, "comp_name": "IPL 2026", "match_format": "T20", "season": "2026"}])
    return balls, matches, competitions


class Phase1Tests(unittest.TestCase):
    def test_legal_ball_classification(self) -> None:
        balls, matches, competitions = fixture_tables()
        result = build_deliveries(balls, matches, competitions)
        self.assertEqual(result["is_legal"].tolist(), [False, True, False, True, True, True, False, True, True])

    def test_run_and_wicket_attribution(self) -> None:
        balls, matches, competitions = fixture_tables()
        result = build_deliveries(balls, matches, competitions)
        self.assertEqual(result["team_runs"].tolist(), [1, 4, 5, 0, 0, 5, 5, 0, 0])
        self.assertEqual(result["bowler_conceded"].tolist(), [1, 0, 5, 0, 0, 0, 1, 0, 0])
        self.assertEqual(result["bowler_wicket"].tolist(), [False, False, False, True, False, False, False, False, False])
        self.assertEqual(result["striker_out"].tolist(), [False, False, False, True, False, False, False, False, True])
        self.assertEqual(result["nonstriker_out"].tolist(), [False, False, False, False, True, False, False, False, False])
        self.assertEqual(result["retired_not_out"].tolist(), [False, False, False, False, False, False, False, True, False])
        self.assertEqual(result["batter_failure"].tolist(), [False, False, False, True, False, False, False, False, False])

    def test_extras2_can_be_enabled_as_sensitivity(self) -> None:
        balls, matches, competitions = fixture_tables()
        primary = build_deliveries(balls, matches, competitions)
        sensitivity = build_deliveries(balls, matches, competitions, bowler_charged_extras2_assumption=True)
        self.assertEqual(primary.loc[6, "bowler_conceded"], 1)
        self.assertEqual(sensitivity.loc[6, "bowler_conceded"], 5)

    def test_dismissal_spelling_normalization(self) -> None:
        self.assertEqual(normalize_dismissal("Obstracting the field"), "Obstructing the field")


if __name__ == "__main__":
    unittest.main()

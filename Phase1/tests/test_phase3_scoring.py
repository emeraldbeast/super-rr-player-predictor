from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superr.scoring import _posterior_rank


class Phase3ScoringTests(unittest.TestCase):
    def test_player_rate_is_ratio_of_weighted_sums_and_drops_zero_exposure(self) -> None:
        frame = pd.DataFrame([
            {"player_id": 1, "player_name": "A", "display_name": "A", "role": "top", "match_id": 1,
             "pressure_balls": 1.0, "weighted_residual": 1.0, "rate": 100.0},
            {"player_id": 1, "player_name": "A", "display_name": "A", "role": "top", "match_id": 2,
             "pressure_balls": 99.0, "weighted_residual": 0.0, "rate": 0.0},
            {"player_id": 1, "player_name": "A", "display_name": "A", "role": "top", "match_id": 3,
             "pressure_balls": 0.0, "weighted_residual": 5.0, "rate": float("nan")},
            {"player_id": 2, "player_name": "B", "display_name": "B", "role": "top", "match_id": 1,
             "pressure_balls": 1.0, "weighted_residual": 1.0, "rate": 100.0},
            {"player_id": 2, "player_name": "B", "display_name": "B", "role": "top", "match_id": 2,
             "pressure_balls": 1.0, "weighted_residual": 0.0, "rate": 0.0},
        ])
        result = _posterior_rank(
            frame,
            numerator_col="weighted_residual",
            minimum_matches=2,
            minimum_balls=1,
            lower_quantile=0.20,
            minimum_prior_variance=0.01,
        ).set_index("player_id")
        self.assertAlmostEqual(result.loc[1, "raw_rate"], 1.0)
        self.assertEqual(int(result.loc[1, "pressure_matches"]), 2)
        self.assertEqual(float(result.loc[1, "pressure_weighted_balls"]), 100.0)
        self.assertTrue(result["sampling_variance"].ge(0).all())


if __name__ == "__main__":
    unittest.main()

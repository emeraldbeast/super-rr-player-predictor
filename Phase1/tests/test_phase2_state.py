from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superr.pressure import add_pressure
from superr.state import build_match_state
from superr.expected import FEATURES, _target_population, assert_predictor_allowlist


class Phase2StateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.deliveries = pd.read_parquet(ROOT / "data" / "interim" / "deliveries.parquet")
        cls.table = pd.read_parquet(ROOT / "data" / "processed" / "model_table.parquet")

    def test_pre_ball_state_is_invariant_to_current_delivery_outcome(self) -> None:
        normal_matches = self.deliveries.loc[~self.deliveries["is_dls"] & ~self.deliveries["is_super_over"], "match_id"].drop_duplicates()
        chosen = normal_matches.iloc[[0, len(normal_matches) // 2, -1]].tolist()
        for match_id in chosen:
            sample = self.deliveries.loc[self.deliveries["match_id"].eq(match_id)].copy()
            baseline = build_match_state(sample)
            for index in (sample.index[0], sample.index[len(sample) // 2], sample.index[-1]):
                changed = sample.copy()
                changed.loc[index, "team_runs"] += 17
                changed.loc[index, "is_wicket"] = not bool(changed.loc[index, "is_wicket"])
                changed.loc[index, "is_legal"] = not bool(changed.loc[index, "is_legal"])
                altered = build_match_state(changed)
                cols = ["score_before", "wickets_lost_before", "legal_balls_before", "balls_remaining", "runs_needed", "required_run_rate"]
                pd.testing.assert_series_equal(
                    baseline.loc[baseline["delivery_id"].eq(sample.loc[index, "delivery_id"]), cols].reset_index(drop=True).iloc[0],
                    altered.loc[altered["delivery_id"].eq(sample.loc[index, "delivery_id"]), cols].reset_index(drop=True).iloc[0],
                    check_names=False,
                )
                # Mutating a later delivery cannot change the anchored earlier state.
                later = sample.loc[
                    sample["delivery_id"].gt(sample.loc[index, "delivery_id"])
                    & sample["innings"].eq(sample.loc[index, "innings"])
                ]
                if not later.empty:
                    future_index = later.index[-1]
                    future_changed = sample.copy()
                    future_changed.loc[future_index, "team_runs"] += 11
                    future_changed.loc[future_index, "is_wicket"] = not bool(future_changed.loc[future_index, "is_wicket"])
                    future_changed.loc[future_index, "is_legal"] = not bool(future_changed.loc[future_index, "is_legal"])
                    future_state = build_match_state(future_changed)
                    cols = ["score_before", "wickets_lost_before", "legal_balls_before", "balls_remaining", "runs_needed", "required_run_rate"]
                    pd.testing.assert_series_equal(
                        baseline.loc[baseline["delivery_id"].eq(sample.loc[index, "delivery_id"]), cols].reset_index(drop=True).iloc[0],
                        future_state.loc[future_state["delivery_id"].eq(sample.loc[index, "delivery_id"]), cols].reset_index(drop=True).iloc[0],
                        check_names=False,
                    )

    def test_pressure_is_bounded_and_super_overs_are_excluded(self) -> None:
        pressure = self.table["pressure_weight"]
        self.assertTrue(pressure.dropna().between(0, 1).all())
        super_overs = self.table["is_super_over"]
        self.assertTrue(self.table.loc[super_overs, "pressure_weight"].isna().all())
        self.assertFalse(self.table.loc[super_overs, "high_pressure"].any())

    def test_high_pressure_flag_uses_configured_quantile_per_pool(self) -> None:
        q = 0.75
        eligible = self.table.loc[self.table["pressure_available"]]
        for _, group in eligible.groupby("competition_pool"):
            threshold = group["pressure_weight"].quantile(q)
            self.assertTrue(group.loc[group["high_pressure"], "pressure_weight"].ge(threshold).all())

    def test_par_reference_falls_back_to_prior_phase_before_cold_start(self) -> None:
        state = pd.DataFrame([
            {"delivery_id": delivery_id, "competition_pool": "IPL", "match_id": 1, "match_date": "2024-01-01",
             "innings": 1, "innings_phase": "middle", "wickets_lost_bucket": "0", "team_runs": runs,
             "is_legal": True, "is_super_over": False, "chase_target_available": False,
             "is_wicket": False, "is_dls": False,
             "required_run_rate": pd.NA, "wickets_lost_before": 0, "balls_remaining": 100,
             "scheduled_legal_balls": 120, "score_before": 0, "legal_balls_before": 0}
            for delivery_id, runs in [(1, 2), (2, 0), (3, 0)]
        ] + [
            {"delivery_id": 4, "competition_pool": "IPL", "match_id": 2, "match_date": "2024-01-02",
             "innings": 1, "innings_phase": "middle", "wickets_lost_bucket": "1-2", "team_runs": 6,
             "is_legal": True, "is_super_over": False, "chase_target_available": False,
             "is_wicket": False, "is_dls": False,
             "required_run_rate": pd.NA, "wickets_lost_before": 1, "balls_remaining": 100,
             "scheduled_legal_balls": 120, "score_before": 0, "legal_balls_before": 0},
        ])
        pressured, _, _, _ = add_pressure(state, cold_start_rate=8.0)
        self.assertEqual(pressured.loc[0, "par_reference_source"], "configured_cold_start")
        self.assertEqual(pressured.loc[3, "par_reference_source"], "prior_match_phase")
        self.assertEqual(pressured.loc[3, "par_runs_per_over"], 4.0)

    def test_dls_matches_are_not_pressure_scored_or_model_features(self) -> None:
        self.assertFalse(self.table.loc[self.table["is_dls"], "pressure_available"].any())
        self.assertFalse(self.table.loc[self.table["is_dls"], "high_pressure"].any())
        assert_predictor_allowlist()
        self.assertNotIn("is_dls", FEATURES)
        self.assertNotIn("is_dls_text", FEATURES)

    def test_batter_run_training_population_excludes_wides(self) -> None:
        sample = pd.DataFrame({"bat_runs": [0, 1, 0], "is_wide": [True, False, False]})
        mask = _target_population(sample, "bat_runs")
        self.assertEqual(mask.tolist(), [False, True, True])

    def test_oof_model_choices_are_rolling_origin(self) -> None:
        choices = pd.read_csv(ROOT / "outputs" / "phase3_model_choices.csv")
        self.assertTrue((choices["max_training_season"] < choices["held_out_season"]).all())
        selected = choices.loc[choices["max_selection_season"].notna()]
        self.assertTrue((selected["max_selection_season"] < selected["held_out_season"]).all())


if __name__ == "__main__":
    unittest.main()

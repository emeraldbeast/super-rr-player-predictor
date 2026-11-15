from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superr.state import build_match_state


class Phase1DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.deliveries = pd.read_parquet(ROOT / "data" / "interim" / "deliveries.parquet")
        cls.players = pd.read_parquet(ROOT / "data" / "interim" / "players.parquet")
        cls.player_match = pd.read_parquet(ROOT / "data" / "interim" / "player_match.parquet")
        cls.coverage = pd.read_csv(ROOT / "outputs" / "audit" / "coverage_by_pool.csv")
        cls.results = pd.read_csv(ROOT / "outputs" / "audit" / "result_margin_reconciliation.csv")
        cls.overs = pd.read_csv(ROOT / "outputs" / "audit" / "over_structure.csv")
        cls.manifest = json.loads((ROOT / "outputs" / "audit" / "run_manifest.json").read_text(encoding="utf-8"))

    def test_real_delivery_keys_are_unique(self) -> None:
        self.assertTrue(self.deliveries["delivery_id"].is_unique)
        self.assertFalse(self.deliveries.duplicated(["match_id", "innings", "innings_ball_num"]).any())

    def test_real_innings_delivery_sequence_is_contiguous(self) -> None:
        for _, group in self.deliveries.groupby(["match_id", "innings"], sort=False):
            sequence = group["innings_ball_num"].astype(int).to_numpy()
            self.assertEqual(sequence.min(), 1)
            self.assertEqual(sequence.max(), len(sequence))
            self.assertEqual(len(set(sequence)), len(sequence))
        expected = self.deliveries["is_legal"].astype(int).groupby(
            [self.deliveries["match_id"], self.deliveries["innings"]], sort=False
        ).cumsum()
        pd.testing.assert_series_equal(
            self.deliveries["legal_ball_no"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_result_margins_reconcile_at_configured_threshold(self) -> None:
        standard = self.results.loc[~self.results["is_dls"] & self.results["status"].isin(["reconciles", "margin_mismatch"])]
        exact_rate = standard["status"].eq("reconciles").mean()
        threshold = self.manifest["config"]["validation"]["result_margin_exact_match_threshold"]
        self.assertGreaterEqual(exact_rate, threshold)
        mismatches = standard.loc[standard["status"].eq("margin_mismatch"), "match_id"].tolist()
        self.assertEqual(mismatches, [20190])

    def test_over_structure_exception_count_and_named_rows(self) -> None:
        exceptions = self.overs.loc[self.overs["structure_exception"]]
        expected_count = self.manifest["config"]["validation"]["known_over_structure_exception_count"]
        self.assertEqual(len(exceptions), expected_count)
        actual_keys = set(zip(exceptions.match_id, exceptions.innings, exceptions.overid))
        expected_keys = {
            (14130, 1, 14), (14130, 2, 14), (14164, 1, 14),
            (36097, 2, 8), (41509, 1, 12),
        }
        self.assertEqual(actual_keys, expected_keys)

    def test_competition_pools_are_separate_and_correct(self) -> None:
        pools = self.coverage.set_index("competition_pool")
        expected = {
            "IPL": (371, 367),
            "T20I": (482, 470),
            "other T20": (30, 30),
        }
        for pool, (matches, with_balls) in expected.items():
            self.assertEqual(int(pools.loc[pool, "matches"]), matches)
            self.assertEqual(int(pools.loc[pool, "matches_with_deliveries"]), with_balls)

    def test_players_and_player_match_have_stable_ids(self) -> None:
        self.assertTrue(self.players["player_id"].is_unique)
        self.assertTrue(self.players["display_name"].is_unique)
        self.assertFalse(self.players["name_variants"].str.contains(r"\|", regex=True).any())
        self.assertFalse(self.player_match.duplicated(["match_id", "team_id", "player_id"]).any())

    def test_revised_limits_are_legal_ball_counts(self) -> None:
        innings = pd.read_parquet(ROOT / "data" / "interim" / "innings.parquet")
        deliveries = self.deliveries
        self.assertTrue(innings.loc[~innings["is_super_over"], "scheduled_legal_balls"].between(0, 120).all())
        for match_id, expected in ((13612, 114), (21586, 90)):
            row = innings.loc[innings["match_id"].eq(match_id) & innings["innings"].eq(2)].iloc[0]
            self.assertEqual(int(row["scheduled_legal_balls"]), expected)
            self.assertEqual(int(row["legal_balls"]), expected)
        target_reached = innings.loc[innings["match_id"].eq(21586) & innings["innings"].eq(2)].iloc[0]
        self.assertEqual(target_reached["termination_reason"], "target reached")
        state = build_match_state(deliveries)
        for match_id, expected in ((13612, 114), (21586, 90)):
            rows = state.loc[state["match_id"].eq(match_id) & state["innings"].eq(2)]
            self.assertTrue(rows["scheduled_legal_balls"].eq(expected).all())
        for source_col, innings_number in (("first_inn_revised_overs", 1), ("second_inn_revised_overs", 2)):
            for expected in (30, 42, 54, 72, 90, 114):
                matches = deliveries.loc[pd.to_numeric(deliveries[source_col], errors="coerce").eq(expected), "match_id"].unique()
                self.assertGreater(len(matches), 0, f"fixture data should cover revised count {expected}")
                rows = state.loc[state["match_id"].isin(matches) & state["innings"].eq(innings_number)]
                self.assertTrue(rows["scheduled_legal_balls"].eq(expected).all())

    def test_analytical_delivery_table_drops_bowl_speed(self) -> None:
        self.assertNotIn("bowl_speed", self.deliveries.columns)
        self.assertNotIn("bowl_speed_raw", self.deliveries.columns)

    def test_dls_result_text_flags_missing_revision_fields(self) -> None:
        dls_missing = pd.read_csv(ROOT / "outputs" / "audit" / "dls_text_without_revision_fields.csv")
        ids = set(dls_missing["match_id"].astype(int))
        self.assertTrue({20182, 23674, 24822, 28334}.issubset(ids))
        self.assertTrue(self.deliveries.loc[self.deliveries.match_id.isin(ids), "is_dls"].all())


if __name__ == "__main__":
    unittest.main()

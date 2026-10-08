import copy
from dataclasses import asdict, replace
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import genetic_algorithm_v2 as ga
from decision_tree_xai_v2 import build_training_data, train_xai_v2, xai_is_current
from scenario_evaluation import ScenarioConfig, HospitalPolicy, evaluate_current_policy


class GAV2Tests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"predicted_probability": [.8, .2],
                                  "treatment_time_min": [30, 30], "length_of_stay_hours": [2, 2]})
        self.config = ScenarioConfig(simulation_duration_hours=4, best_case_arrival_rate=5,
                                     average_case_arrival_rate=10, worst_case_arrival_rate=20,
                                     replications_per_scenario=3)
        self.current = HospitalPolicy(1, 1, 1, 1)
        self.bounds = dict(icu_beds=(1, 100), general_beds=(1, 100), doctors=(1, 50), nurses=(1, 50))
        self.options = ga.GAConfig(population_size=6, generations=3, ga_replications_per_scenario=1)

    def run_small(self, **kwargs):
        return ga.run_ga_v2(self.data, self.current, self.bounds, self.config, self.options, **kwargs)

    def test_current_is_first_generation_zero_and_optimization_improves(self):
        result = self.run_small()
        first = result["history"][0]
        self.assertEqual(first["generation"], 0)
        for key in ga.GENES:
            self.assertEqual(first[key], getattr(self.current, key))
        self.assertGreater(result["verified_evaluation"]["overall_robustness"], result["current_evaluation"]["overall_robustness"])
        for record in result["history"]:
            self.assertIn(record["scenario_verdict"], ("ACCEPTABLE", "UNACCEPTABLE"))
        # Elitism makes the observed best search fitness nondecreasing.
        scores = [r["best_fitness"] for r in result["generations"]]
        self.assertEqual(scores, sorted(scores))

    def test_genetic_operators_respect_integer_bounds_and_fixed_rf_threshold(self):
        rng = random.Random(12)
        for _ in range(500):
            a, b = ga.random_policy(rng, self.bounds), ga.random_policy(rng, self.bounds)
            child = ga.mutate_v2(ga.crossover(a, b, rng), rng, self.bounds, mutation_rate=1.)
            self.assertEqual(child.icu_risk_threshold, .5)
            for key in ga.GENES:
                self.assertIsInstance(getattr(child, key), int)
                self.assertTrue(self.bounds[key][0] <= getattr(child, key) <= self.bounds[key][1])

    def test_common_random_numbers_across_all_candidates_and_generations(self):
        result = ga.run_ga_v2(self.data, self.current, self.bounds, self.config,
                              replace(self.options, ga_replications_per_scenario=3))
        reference = result["evaluations"][0]["evaluation"]
        for observation in result["evaluations"]:
            evaluation = observation["evaluation"]
            for name in ga.SCENARIOS:
                a, b = evaluation["scenarios"][name]["runs"], reference["scenarios"][name]["runs"]
                self.assertEqual([r["seed"] for r in a], [r["seed"] for r in b])
                self.assertEqual([r["patients_arrived"] for r in a], [r["patients_arrived"] for r in b])
                self.assertEqual([r["high_risk_patients"] for r in a], [r["high_risk_patients"] for r in b])
                self.assertEqual(len({r["seed"] for r in a}), 3)

    def test_reproducible_experiment_and_no_dataframe_mutation(self):
        original = self.data.copy(deep=True)
        a, b = self.run_small(), self.run_small()
        a.pop("runtime_seconds")
        b.pop("runtime_seconds")
        self.assertEqual(a, b)
        pd.testing.assert_frame_equal(self.data, original)

    def test_severe_overload_cannot_win_on_cost_and_priorities_are_effective(self):
        config = replace(self.config, best_case_arrival_rate=20, average_case_arrival_rate=30, worst_case_arrival_rate=40)
        good_policy = HospitalPolicy(100, 100, 50, 50)
        good = evaluate_current_policy(self.data, good_policy, config)
        bad = evaluate_current_policy(self.data, self.current, config)
        self.assertEqual(good["overall_policy_status"], "ROBUST")
        self.assertEqual(bad["overall_policy_status"], "UNACCEPTABLE")
        for efficiency in (False, True):
            options = replace(self.options, prioritize_efficiency=efficiency)
            good_score, _ = ga.calculate_fitness_v2(good, good_policy, self.bounds, options)
            bad_score, _ = ga.calculate_fitness_v2(bad, self.current, self.bounds, options)
            self.assertGreater(good_score - bad_score, 10000)
        _, h = ga.calculate_fitness_v2(bad, self.current, self.bounds, self.options)
        _, normal = ga.calculate_fitness_v2(bad, self.current, self.bounds, replace(self.options, prioritize_high_risk=False, prioritize_efficiency=False))
        self.assertLess(h["delay_penalty"], normal["delay_penalty"])
        self.assertAlmostEqual(h["resource_cost_penalty"], 2 * normal["resource_cost_penalty"])

    def test_memoization_full_verification_once_and_early_stopping(self):
        bounds = {key: (1, 1) for key in ga.GENES}
        with patch.object(ga, "evaluate_current_policy", wraps=evaluate_current_policy) as evaluate:
            result = ga.run_ga_v2(self.data, self.current, bounds, self.config,
                                  replace(self.options, generations=10, patience=2))
        self.assertEqual(result["unique_candidate_evaluations"], 1)
        self.assertEqual(evaluate.call_count, 2)  # one cached search, one full verification
        self.assertTrue(result["early_stopping_occurred"])
        self.assertEqual(result["generations_completed"], 3)
        self.assertEqual(result["final_verification_replications"], 3)
        for scenario in result["search_evaluation"]["scenarios"].values():
            self.assertEqual(len(scenario["runs"]), 1)
        for scenario in result["verified_evaluation"]["scenarios"].values():
            self.assertEqual(len(scenario["runs"]), 3)

    def test_search_replication_limits_do_not_relax_part1_validation(self):
        replace(self.options, ga_replications_per_scenario=2).validate()
        with self.assertRaises(ValueError):
            replace(self.config, replications_per_scenario=2).validate()
        for n in (0, 11, 1.5):
            with self.assertRaises(ValueError):
                replace(self.options, ga_replications_per_scenario=n).validate()

    def test_invalid_bounds_reject_current_policy_without_changing_it(self):
        bounds = dict(self.bounds, icu_beds=(2, 100))
        with self.assertRaisesRegex(ValueError, "outside optimization bounds"):
            ga.validate_bounds(bounds, self.current)
        self.assertEqual(self.current.icu_beds, 1)
        with self.assertRaises(ValueError):
            ga.validate_bounds(dict(self.bounds, nurses=(100, 1)), self.current)

    def test_xai_labels_exactly_match_simulated_scenario_labels(self):
        result = self.run_small()
        data = build_training_data(result)
        self.assertEqual(len(data), 3 * len(result["evaluations"]))
        for observation in result["evaluations"]:
            policy_id = ga.canonical_hash(observation["policy"])
            for name, scenario in observation["evaluation"]["scenarios"].items():
                row = data[(data.policy_id == policy_id) & (data.scenario == name)].iloc[0]
                self.assertEqual(row.acceptable_policy, int(scenario["scenario_verdict"] == "ACCEPTABLE"))
        corrupted = copy.deepcopy(result)
        corrupted["evaluations"][0]["evaluation"]["scenarios"]["best"]["runs"][0]["policy_verdict"] = "ACCEPTABLE"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            build_training_data(corrupted)
        report, tree, training = train_xai_v2(result, min_samples_leaf=1)
        self.assertIsNotNone(tree)
        self.assertEqual(report["metrics"]["training_sample_count"], len(training))
        self.assertTrue(xai_is_current(report, result))
        for name, explanation in report["explanations"].items():
            self.assertEqual(explanation["authoritative_verdict"], result["verified_evaluation"]["scenarios"][name]["scenario_verdict"])

    def test_xai_single_class_is_unavailable_gracefully(self):
        config = replace(self.config, best_case_arrival_rate=0, average_case_arrival_rate=0, worst_case_arrival_rate=0)
        result = ga.run_ga_v2(self.data, self.current, self.bounds, config, self.options)
        report, tree, data = train_xai_v2(result)
        self.assertIsNone(tree)
        self.assertEqual(report["status"], "unavailable")
        self.assertIn("same deterministic class", report["message"])
        self.assertEqual(data.acceptable_policy.nunique(), 1)

    def test_configuration_changes_invalidate_recommendation_and_xai(self):
        result = self.run_small()
        self.assertTrue(ga.result_is_current(result, self.current, self.config, self.bounds, self.options))
        for field, value in (("simulation_duration_hours", 5), ("worst_case_arrival_rate", 21),
                             ("random_seed", 43), ("mean_wait_target_minutes", 19),
                             ("max_utilization_target", .8), ("replications_per_scenario", 4)):
            self.assertFalse(ga.result_is_current(result, self.current, replace(self.config, **{field: value}), self.bounds, self.options))
        for field, value in (("population_size", 7), ("generations", 4), ("ga_replications_per_scenario", 2),
                             ("prioritize_efficiency", False), ("prioritize_high_risk", False)):
            self.assertFalse(ga.result_is_current(result, self.current, self.config, self.bounds, replace(self.options, **{field: value})))
        self.assertFalse(ga.result_is_current(result, HospitalPolicy(2, 1, 1, 1), self.config, self.bounds, self.options))
        self.assertFalse(ga.result_is_current(result, self.current, self.config, dict(self.bounds, nurses=(1, 60)), self.options))
        report, _, _ = train_xai_v2(result)
        changed = copy.deepcopy(result)
        changed["run_id"] = "different-experiment"
        self.assertFalse(xai_is_current(report, changed))


if __name__ == "__main__":
    unittest.main()

import json
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scenario_evaluation import (
    ScenarioConfig, HospitalPolicy, evaluate_current_policy, generate_arrivals,
    simulate_time_window, policy_verdict,
)


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.profiles = pd.DataFrame({"predicted_probability": [.8, .2],
                                      "treatment_time_min": [30, 30],
                                      "length_of_stay_hours": [2, 2]})
        self.config = ScenarioConfig(replications_per_scenario=3,
                                     best_case_arrival_rate=5,
                                     average_case_arrival_rate=10,
                                     worst_case_arrival_rate=20)

    def test_24h_reproducibility_resource_response_and_seed(self):
        before = self.profiles.copy(deep=True)
        low = evaluate_current_policy(self.profiles, HospitalPolicy(1, 1, 1, 1), self.config)
        high = evaluate_current_policy(self.profiles, HospitalPolicy(200, 200, 100, 100), self.config)
        self.assertEqual(low["overall_policy_status"], "UNACCEPTABLE")
        self.assertEqual(high["overall_policy_status"], "ROBUST")
        self.assertGreater(high["overall_robustness"], low["overall_robustness"])
        repeat = evaluate_current_policy(self.profiles, HospitalPolicy(1, 1, 1, 1), self.config)
        self.assertEqual(json.dumps(low, sort_keys=True), json.dumps(repeat, sort_keys=True))
        other = evaluate_current_policy(self.profiles, HospitalPolicy(1, 1, 1, 1), replace(self.config, random_seed=43))
        self.assertNotEqual(low["scenarios"]["worst"]["metrics"]["patients_arrived"],
                            other["scenarios"]["worst"]["metrics"]["patients_arrived"])
        pd.testing.assert_frame_equal(self.profiles, before)
        for scenario in low["scenarios"].values():
            self.assertEqual(len(scenario["runs"]), 3)
            for run in scenario["runs"]:
                self.assertEqual(run["unfinished_patients"], run["patients_waiting_at_end"] + run["patients_in_treatment_at_end"])
                self.assertEqual(run["patients_arrived"], run["patients_completed"] + run["unfinished_patients"])

    def test_short_window_censoring_and_busy_time(self):
        # One long-stay patient and one queued patient. No completions, but busy
        # time must include the unfinished patient and queue wait must be capped.
        df = pd.DataFrame({"predicted_probability": [.8], "treatment_time_min": [120],
                           "length_of_stay_hours": [8]})
        config = replace(self.config, simulation_duration_hours=4)
        with patch("scenario_evaluation.generate_arrivals", return_value=[60., 120.]):
            metrics = simulate_time_window(df, HospitalPolicy(1, 1, 1, 1), 10, config)
        self.assertEqual(metrics["patients_arrived"], 2)
        self.assertEqual(metrics["patients_completed"], 0)
        self.assertEqual(metrics["patients_waiting_at_end"], 1)
        self.assertEqual(metrics["patients_in_treatment_at_end"], 1)
        self.assertEqual(metrics["mean_wait"], 60.)
        self.assertEqual(metrics["high_risk_mean_wait"], 60.)
        self.assertEqual(metrics["icu_utilization"], .75)
        self.assertEqual(metrics["doctor_utilization"], .5)
        self.assertEqual(metrics["nurse_utilization"], .5)

    def test_partial_staff_service_utilization(self):
        with patch("scenario_evaluation.generate_arrivals", return_value=[210.]):
            df = self.profiles.iloc[:1].assign(treatment_time_min=120)
            metrics = simulate_time_window(df, HospitalPolicy(1, 1, 1, 1), 5,
                                           replace(self.config, simulation_duration_hours=4))
        self.assertEqual(metrics["patients_completed"], 0)
        self.assertEqual(metrics["doctor_utilization"], 30 / 240)

    def test_validation(self):
        for config in [replace(self.config, best_case_arrival_rate=11),
                       replace(self.config, simulation_duration_hours=0),
                       replace(self.config, replications_per_scenario=2),
                       replace(self.config, max_utilization_target=90),
                       replace(self.config, worst_case_arrival_rate=float("nan"))]:
            with self.assertRaises(ValueError):
                config.validate()
        with self.assertRaises(ValueError):
            evaluate_current_policy(self.profiles, HospitalPolicy(0, 1, 1, 1), self.config)

    def test_poisson_arrival_formula(self):
        actual = generate_arrivals(10, 4, np.random.default_rng(42))
        rng = np.random.default_rng(42)
        expected, time = [], 0
        while True:
            time += rng.exponential(60 / 10)
            if time >= 4 * 60:
                break
            expected.append(time)
        self.assertEqual(actual, expected)
        self.assertGreater(np.std(np.diff(actual)), 0)

    def test_zero_arrivals_and_missing_high_risk(self):
        config = replace(self.config, best_case_arrival_rate=0, average_case_arrival_rate=0, worst_case_arrival_rate=0)
        result = evaluate_current_policy(self.profiles, HospitalPolicy(1, 1, 1, 1), config)
        self.assertEqual(result["overall_robustness"], 100)
        run = result["scenarios"]["best"]["runs"][0]
        self.assertEqual(run["mean_wait"], 0)
        self.assertEqual(run["patients_arrived"], 0)
        metrics = simulate_time_window(self.profiles.iloc[1:], HospitalPolicy(200, 200, 100, 100), 10, self.config)
        self.assertEqual(metrics["high_risk_mean_wait"], 0)

    def test_acceptance_uses_every_critical_resource(self):
        metrics = dict(mean_wait=20, high_risk_mean_wait=10,
                       icu_utilization=.9, general_utilization=.9,
                       doctor_utilization=.9, nurse_utilization=.9)
        self.assertEqual(policy_verdict(metrics, self.config)[0], "ACCEPTABLE")
        for resource in ["icu", "general", "doctor", "nurse"]:
            failed = dict(metrics, **{resource + "_utilization": .91})
            verdict, conditions = policy_verdict(failed, self.config)
            self.assertEqual(verdict, "UNACCEPTABLE")
            self.assertFalse(conditions[resource + "_util_pass"])

    def test_scenario_and_overall_threshold_rules(self):
        def fake_run(profiles, policy, rate, config, seed):
            scenario = (seed - config.random_seed * 1000) // 100
            index = seed % 100
            # Best 3/3, Average 2/3, Worst 1/3.
            return {"seed": seed, "mean_wait": float(index),
                    "policy_verdict": "ACCEPTABLE" if index < 3 - scenario else "UNACCEPTABLE"}
        with patch("scenario_evaluation._simulate", side_effect=fake_run):
            conditional = evaluate_current_policy(self.profiles, HospitalPolicy(1, 1, 1, 1),
                                                  replace(self.config, robustness_threshold=60))
            unacceptable = evaluate_current_policy(self.profiles, HospitalPolicy(1, 1, 1, 1), self.config)
            robust = evaluate_current_policy(self.profiles, HospitalPolicy(1, 1, 1, 1),
                                             replace(self.config, robustness_threshold=30))
        self.assertEqual(conditional["overall_policy_status"], "CONDITIONALLY ACCEPTABLE")
        self.assertEqual(unacceptable["overall_policy_status"], "UNACCEPTABLE")
        self.assertEqual(robust["overall_policy_status"], "ROBUST")
        self.assertAlmostEqual(conditional["overall_robustness"], 6 / 9 * 100)
        self.assertEqual(conditional["scenarios"]["average"]["scenario_verdict"], "ACCEPTABLE")
        self.assertEqual(conditional["scenarios"]["worst"]["scenario_verdict"], "UNACCEPTABLE")
        self.assertEqual(conditional["scenarios"]["best"]["metrics"]["mean_wait"], {"mean": 1., "min": 0., "max": 2.})


if __name__ == "__main__":
    unittest.main()

"""Availability overrides default ranges without changing simulation/fitness."""
import copy
from dataclasses import asdict, replace
import json
import os
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_part3 import interpretation
import feedback_interpreter as feedback
import genetic_algorithm_v2 as ga
from scenario_evaluation import HospitalPolicy, ScenarioConfig, evaluate_current_policy
import pandas as pd


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.current = HospitalPolicy(210, 280, 85, 100)
        self.auto = ga.automatic_bounds(self.current)
        self.config = ScenarioConfig(simulation_duration_hours=1, best_case_arrival_rate=1,
            average_case_arrival_rate=2, worst_case_arrival_rate=3, replications_per_scenario=3)
        self.options = ga.GAConfig(population_size=4, generations=2, ga_replications_per_scenario=1)
        self.data = pd.DataFrame({"predicted_probability": [.8, .2],
            "treatment_time_min": [30, 30], "length_of_stay_hours": [2, 2]})

    def apply(self, resource, **limits):
        parsed = interpretation()
        parsed["resource_constraints"][resource].update(limits)
        return feedback.apply_feedback_v2(parsed, self.auto, self.current, self.config, self.options)[0]

    def test_only_ten_nurses_are_available_allows_optimization(self):
        effective = self.apply("nurses", max=10)
        self.assertEqual(self.auto["nurses"], (70, 150))
        self.assertEqual(effective["nurses"], (1, 10))
        result = ga.run_ga_v2(self.data, self.current, effective, self.config, self.options,
                             allow_infeasible_current=True)
        self.assertLessEqual(result["best_policy"]["nurses"], 10)
        self.assertEqual(result["history"][0]["nurses"], 10)
        self.assertEqual(self.current.nurses, 100)

    def test_doctor_cap_below_old_current_is_valid(self):
        self.assertEqual(self.apply("doctors", max=60)["doctors"], (59, 60))

    def test_explicit_minimum_and_maximum_override_both_directions(self):
        self.assertEqual(self.apply("general_beds", min=50)["general_beds"], (50, 420))
        self.assertEqual(self.apply("general_beds", min=500)["general_beds"], (500, 500))
        self.assertEqual(self.apply("icu_beds", max=150)["icu_beds"], (147, 150))
        self.assertEqual(self.apply("nurses", max=180)["nurses"], (70, 180))

    def test_feasible_generation_zero_and_bounded_evolution(self):
        effective = self.apply("nurses", max=10)
        for policy in ga.initial_population(self.current, effective, 20, random.Random(42), allow_infeasible_current=True):
            for resource in ga.GENES:
                self.assertTrue(effective[resource][0] <= getattr(policy, resource) <= effective[resource][1])
        result = ga.run_ga_v2(self.data, self.current, effective, self.config, self.options,
                             allow_infeasible_current=True)
        for row in result["history"]:
            for resource in ga.GENES:
                self.assertTrue(effective[resource][0] <= row[resource] <= effective[resource][1])
        # No-feedback callers retain the original validation behavior.
        with self.assertRaisesRegex(ValueError, "outside optimization bounds"):
            ga.run_ga_v2(self.data, self.current, effective, self.config, self.options)

    def test_historical_reference_is_preserved_and_both_ranges_have_provenance(self):
        reference = evaluate_current_policy(self.data, self.current, self.config)
        saved = copy.deepcopy(reference)
        effective = self.apply("nurses", max=10)
        result = ga.run_ga_v2(self.data, self.current, effective, self.config, self.options,
                             allow_infeasible_current=True)
        self.assertEqual(reference, saved)
        self.assertEqual(result["current_evaluation"], saved)
        provenance = result["provenance"]
        self.assertEqual(provenance["automatic_bounds"]["nurses"], [70, 150])
        self.assertEqual(provenance["effective_bounds"]["nurses"], [1, 10])
        self.assertFalse(provenance["current_policy_feasible"])
        self.assertEqual(provenance["current_policy"]["nurses"], 100)
        self.assertIn("nearest feasible", provenance["initialization_strategy"])

    def test_preferences_have_no_hard_bounds(self):
        parsed = interpretation()
        parsed["priorities"]["resource_efficiency_priority"] = True
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-placeholder"}), patch.object(feedback, "Groq") as client:
            client.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(parsed)))])
            record = feedback.interpret_feedback_v2("try to use fewer nurses", self.auto, self.config)
            prompt = client.return_value.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("ALL resource bounds null", prompt)
        bounds, _, options = feedback.apply_feedback_v2(record["interpretation"], self.auto, self.current,
            self.config, replace(self.options, prioritize_efficiency=False))
        self.assertEqual(bounds, self.auto)
        self.assertTrue(options.prioritize_efficiency)

    def test_truly_impossible_and_unsupported_constraints_reject(self):
        with self.assertRaisesRegex(ValueError, "minimum 120 exceeds maximum 100"):
            self.apply("doctors", min=120, max=100)
        for value in (0, -1):
            with self.assertRaisesRegex(ValueError, "1 to 10000"):
                self.apply("nurses", max=value)
        parsed = interpretation()
        parsed["resource_constraints"]["surgeons"] = dict(min=None, max=10)
        with self.assertRaisesRegex(ValueError, "Unsupported or missing fields"):
            feedback.apply_feedback_v2(parsed, self.auto, self.current, self.config, self.options)

    def test_stale_protection_includes_effective_bounds_feedback_and_override_mode(self):
        effective = self.apply("nurses", max=10)
        dependencies = {"feedback": {"resource_constraints": {"nurses": {"max": 10}}}}
        result = ga.run_ga_v2(self.data, self.current, effective, self.config, self.options,
                             dependencies=dependencies, allow_infeasible_current=True)
        self.assertTrue(ga.result_is_current(result, self.current, self.config, effective, self.options,
                                           dependencies, allow_infeasible_current=True))
        self.assertFalse(ga.result_is_current(result, self.current, self.config, self.auto, self.options,
                                            dependencies, allow_infeasible_current=True))
        self.assertFalse(ga.result_is_current(result, self.current, self.config, effective, self.options,
                                            {"feedback": {}}, allow_infeasible_current=True))
        self.assertFalse(ga.result_is_current(result, self.current, self.config, effective, self.options, dependencies))


if __name__ == "__main__":
    unittest.main()

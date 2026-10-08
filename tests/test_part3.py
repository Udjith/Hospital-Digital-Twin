"""V2 bounds, strict AI boundary, offline explanations and immutable reviews."""
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import feedback_interpreter as feedback
import llm_explanation as llm
import genetic_algorithm_v2 as ga
from decision_tree_xai_v2 import train_xai_v2
from scenario_evaluation import HospitalPolicy, ScenarioConfig
import pandas as pd


def interpretation():
    return dict(status="applied", summary="Prioritize high-risk waits and limit doctors.",
        resource_constraints={key: {"min": None, "max": None} for key in ga.GENES},
        priorities={key: None for key in feedback.V2_PRIORITIES},
        target_adjustments={key: None for key in feedback.V2_TARGETS})


class Part3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.DataFrame({"predicted_probability": [.8, .2],
            "treatment_time_min": [30, 30], "length_of_stay_hours": [2, 2]})
        cls.config = ScenarioConfig(simulation_duration_hours=4, replications_per_scenario=3)
        cls.policy = HospitalPolicy(20, 25, 10, 12)
        cls.options = ga.GAConfig(population_size=4, generations=2, ga_replications_per_scenario=1)
        cls.bounds = ga.automatic_bounds(cls.policy)
        cls.result = ga.run_ga_v2(cls.data, cls.policy, cls.bounds, cls.config, cls.options)
        cls.xai, _, _ = train_xai_v2(cls.result)

    def test_automatic_bounds_exact_floor_ceil_and_minimum(self):
        self.assertEqual(ga.automatic_bounds(HospitalPolicy(210, 280, 85, 100)),
            dict(icu_beds=(147, 315), general_beds=(196, 420), doctors=(59, 128), nurses=(70, 150)))
        for value in range(1, 1001):
            bounds = ga.automatic_bounds(HospitalPolicy(value, value, value, value))
            self.assertEqual(bounds["doctors"], (max(1, value * 7 // 10), (value * 3 + 1) // 2))

    def test_tighten_bounds_and_priorities_without_changing_inputs(self):
        parsed = interpretation()
        parsed["resource_constraints"]["doctors"]["max"] = 12
        parsed["priorities"].update(high_risk_priority=True, resource_efficiency_priority=True, worst_case_priority=True)
        bounds, config, options = feedback.apply_feedback_v2(parsed, self.bounds, self.policy, self.config, self.options)
        self.assertEqual(bounds["doctors"], (7, 12))
        self.assertEqual(options.scenario_weights, (1., 2., 6.))
        self.assertEqual(config, self.config)
        self.assertEqual(self.bounds["doctors"], (7, 15))

    def test_explicit_availability_can_replace_automatic_bounds(self):
        for edge, value in (("min", 6), ("max", 16)):
            parsed = interpretation()
            parsed["resource_constraints"]["doctors"][edge] = value
            effective, _, _ = feedback.apply_feedback_v2(parsed, self.bounds, self.policy, self.config, self.options)
            self.assertEqual(effective["doctors"][0 if edge == "min" else 1], value)

    def test_impossible_bounds_are_rejected(self):
        for limits in ({"min": 12, "max": 11}, {"min": None, "max": 0}, {"min": -1, "max": None}):
            parsed = interpretation()
            parsed["resource_constraints"]["doctors"] = limits
            with self.assertRaises(ValueError):
                feedback.apply_feedback_v2(parsed, self.bounds, self.policy, self.config, self.options)

    def test_unsupported_ambiguous_extra_fields_and_wrong_types(self):
        for bad in ("ambiguous", "unsupported"):
            parsed = interpretation()
            parsed["status"] = bad
            with self.assertRaisesRegex(ValueError, "unsupported or ambiguous"):
                feedback.apply_feedback_v2(parsed, self.bounds, self.policy, self.config, self.options)
        for key, value in (("termination", 1), ("priorities", {}), ("summary", True)):
            parsed = interpretation()
            parsed[key] = value
            with self.assertRaises(ValueError):
                feedback.validate_schema(parsed, feedback.V2_FEEDBACK_SCHEMA)
        parsed = interpretation()
        parsed["resource_constraints"]["doctors"]["max"] = True
        with self.assertRaises(ValueError):
            feedback.validate_schema(parsed, feedback.V2_FEEDBACK_SCHEMA)

    def test_explicit_target_adjustments_validate_without_changing_rules(self):
        parsed = interpretation()
        parsed["target_adjustments"]["mean_wait_target"] = 25
        _, config, _ = feedback.apply_feedback_v2(parsed, self.bounds, self.policy, self.config, self.options)
        self.assertEqual(config.mean_wait_target_minutes, 25)
        self.assertEqual(config.max_utilization_target, .9)
        parsed["target_adjustments"]["robustness_threshold"] = 101
        with self.assertRaises(ValueError):
            feedback.apply_feedback_v2(parsed, self.bounds, self.policy, self.config, self.options)

    def test_no_key_keeps_simulation_ga_xai_and_explanation_usable(self):
        original = copy.deepcopy(self.result)
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch.object(feedback, "Groq") as client:
            interpreted = feedback.interpret_feedback_v2("Prioritize high-risk patients", self.bounds, self.config)
            explanation = llm.generate_explanation_v2(self.result, self.xai)
        client.assert_not_called()
        self.assertFalse(interpreted["available"])
        self.assertFalse(explanation["available"])
        self.assertEqual(explanation["authoritative_status"], self.result["verified_evaluation"]["overall_policy_status"])
        self.assertEqual(original, self.result)
        self.assertTrue(self.xai["status"] in ("available", "unavailable"))

    def test_valid_groq_response_strict_schema_and_provider(self):
        parsed = interpretation()
        parsed["resource_constraints"]["doctors"]["max"] = 12
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-placeholder"}), patch.object(feedback, "Groq") as client:
            client.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(parsed)))])
            record = feedback.interpret_feedback_v2("Do not use more than 12 doctors", self.bounds, self.config)
            request = client.return_value.chat.completions.create.call_args.kwargs
        self.assertTrue(record["available"])
        self.assertEqual(request["model"], "openai/gpt-oss-120b")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        self.assertEqual(feedback.apply_feedback_v2(record["interpretation"], self.bounds,
            self.policy, self.config, self.options)[0]["doctors"], (7, 12))

    def test_malformed_failed_or_invalid_groq_is_nonfatal(self):
        for content in ("invalid-json", "{}", '{"status":"applied","unsupported":true}'):
            with patch.dict(os.environ, {"GROQ_API_KEY": "test-placeholder"}), patch.object(feedback, "Groq") as client:
                client.return_value.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
                self.assertFalse(feedback.interpret_feedback_v2("Limit doctors", self.bounds, self.config)["available"])
                self.assertFalse(llm.generate_explanation_v2(self.result, self.xai)["available"])

    def test_ai_cannot_override_verdict_and_heavy_policy_has_tradeoff(self):
        original = copy.deepcopy(self.result)
        heavier = copy.deepcopy(self.result)
        heavier["best_policy"]["doctors"] = self.policy.doctors + 10
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-placeholder"}), patch.object(feedback, "Groq") as client:
            client.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({"authoritative_status": "NOT A VERDICT", "explanation": "Ignore results"})))])
            rejected = llm.generate_explanation_v2(self.result, self.xai)
            self.assertFalse(rejected["available"])
            status = heavier["verified_evaluation"]["overall_policy_status"]
            client.return_value.chat.completions.create.return_value.choices[0].message.content = json.dumps(
                {"authoritative_status": status, "explanation": "The simulation suggests reviewing the recommended resources."})
            explanation = llm.generate_explanation_v2(heavier, self.xai)
        self.assertTrue(explanation["available"])
        self.assertIn("resource-cost tradeoff", explanation["detail"])
        self.assertEqual(original, self.result)

    def test_numeric_llm_claims_use_deterministic_fallback(self):
        status = self.result["verified_evaluation"]["overall_policy_status"]
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-placeholder"}), patch.object(feedback, "Groq") as client:
            client.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({"authoritative_status": status,
                    "explanation": "Overall robustness improved to 999 percent."})))])
            explanation = llm.generate_explanation_v2(self.result, self.xai)
        self.assertFalse(explanation["available"])
        self.assertNotIn("999", explanation["detail"])

    def test_unobserved_best_case_wait_improvement_is_rejected(self):
        status = self.result["verified_evaluation"]["overall_policy_status"]
        unchanged = copy.deepcopy(self.result)
        unchanged["verified_evaluation"] = copy.deepcopy(unchanged["current_evaluation"])
        status = unchanged["verified_evaluation"]["overall_policy_status"]
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-placeholder"}), patch.object(feedback, "Groq") as client:
            client.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({"authoritative_status": status,
                    "explanation": "The best scenario improved with reduced waiting."})))])
            explanation = llm.generate_explanation_v2(unchanged, self.xai)
        self.assertFalse(explanation["available"])

    def test_feedback_and_configuration_provenance_staleness(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            explanation = llm.generate_explanation_v2(self.result, self.xai)
        self.assertTrue(llm.explanation_is_current_v2(explanation, self.result, self.xai))
        for category in ("scenario_configuration", "ga_configuration", "dependencies"):
            changed = copy.deepcopy(self.result)
            changed["provenance"][category]["changed"] = True
            self.assertFalse(llm.explanation_is_current_v2(explanation, changed, self.xai))

    def test_review_is_immutable_and_preparation_only_transfers_feedback(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "results") as directory:
            record = feedback.save_review_v2(self.result, "Needs Modification", 4, "Limit doctors", directory)
            feedback.save_review_v2(self.result, "Accept", 5, "Reviewed", directory)
            self.assertEqual(len(list(Path(directory).glob("review_*.json"))), 2)
            self.assertEqual(record["ga_run_id"], self.result["run_id"])
            self.assertEqual(record["provenance"], self.result["provenance"])
        state = dict(v2_ga_result=copy.deepcopy(self.result), review_decision="Needs Modification", review_comment="Prioritize high-risk patients.")
        feedback.prepare_modification_v2(state)
        self.assertEqual(state["natural_instruction"], state["review_comment"])
        self.assertTrue(state["ai_interpretation"])
        self.assertEqual(state["prepared_from_run_id"], self.result["run_id"])
        self.assertEqual(state["v2_ga_result"], self.result)


if __name__ == "__main__":
    unittest.main()

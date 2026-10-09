"""Hot-reload-safe hashing and grounded success/partial/failure explanations."""
from dataclasses import replace
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from test_live_policy_ga import hospital, waiting
from live_canonical import canonical_json, canonical_hash, live_hash_input
from live_hospital_state import LiveCapacity
from live_policy_ga import optimize_live_policy, optimization_is_current, LiveGAOptions
from live_policy_diagnosis import diagnose_live_result, diagnose_prediction, build_explanation_payload
from live_policy_explanation import explain_live_tree, explain_live_policy, explanation_is_current
from scenario_evaluation import ScenarioConfig, policy_verdict
from live_lookahead import handling_verdict
from live_pressure import LiveTargets, live_run_verdict, PressureIntegrator


def search(success=False):
    state = hospital(LiveCapacity(10, 10, 10, 1))
    waiting(state); waiting(state, high=True)
    targets = replace(LiveTargets(), mean_wait_target_minutes=100., max_utilization_target=1. if success else .9, critical_saturation_grace_minutes=30. if success else 15.)
    options = LiveGAOptions(population=4, generations=2, search_replications=1)
    result = optimize_live_policy(state, None, 60, 5, targets, options)
    return state, targets, options, result


class CanonicalTests(unittest.TestCase):
    def test_canonical_json_primitives_order_numpy_and_nonfinite(self):
        self.assertEqual(canonical_hash(dict(b=np.int64(2), a=np.array([.2, .3]))),
                         canonical_hash(dict(a=[.2, .3], b=2)))
        self.assertEqual(canonical_json(dict(b=2, a=1)), '{"a":1,"b":2}')
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                canonical_hash(dict(value=value))

    def test_state_hash_stable_clone_and_insertion_identity_independent(self):
        state = hospital(rate=1.)
        before = state.state_hash()
        self.assertEqual(before, state.state_hash())
        self.assertEqual(before, state.clone().state_hash())
        state.resources = dict(reversed(list(state.resources.items())))
        self.assertEqual(before, state.state_hash())
        json.loads(canonical_json(live_hash_input(state)))

    def test_hash_changes_for_clock_ownership_rng_queue_and_configuration(self):
        state = hospital(rate=1.)
        for mutate in (lambda s: setattr(s, "sim_time_minutes", 1.),
            lambda s: s._arrival_rng.random(), lambda s: waiting(s),
            lambda s: setattr(s.resources["doctors"]["DOC-001"], "patient_id", "P900000"),
            lambda s: setattr(s, "_arrival_rate", 2.),
            lambda s: s._schedule(2., 2, "ARRIVAL")):
            clone = state.clone()
            mutate(clone)
            self.assertNotEqual(state.state_hash(), clone.state_hash())

    def test_hot_reload_real_module_hash_provenance_result_loading(self):
        script = """
import sys, importlib, json
sys.path.insert(0, 'src')
import pandas as pd
import live_hospital_state as live
from live_policy_ga import optimize_live_policy, optimization_is_current, LiveGAOptions
from live_policy_explanation import explain_live_policy, explanation_is_current
from scenario_evaluation import ScenarioConfig
data=pd.DataFrame(dict(predicted_probability=[.1,.9], treatment_time_min=[15.,15.],length_of_stay_hours=[.5,.5]))
state=live.LiveHospitalState(data,live.LiveCapacity(3,3,3,3),1.)
state.start(); state.advance_simulation(20.)
old_policy=state.current_policy; original_hash=state.state_hash()
options=LiveGAOptions(population=4,generations=1,search_replications=1)
targets=ScenarioConfig()
result=optimize_live_policy(state,None,30,3,targets,options)
explanation=explain_live_policy(result)
importlib.reload(live)
assert type(old_policy) is not live.OperatingPolicy
assert state.state_hash()==original_hash
loaded=json.loads(json.dumps(result,allow_nan=False))
assert optimization_is_current(loaded,state,None,30,3,targets,options)
assert explanation_is_current(explanation,loaded)
assert optimize_live_policy(state,None,30,3,targets,options)['optimized_policy']==loaded['optimized_policy']
print('hot reload PASS')
"""
        completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_pickle_not_called_in_hash_or_provenance(self):
        with patch("pickle.dumps", side_effect=AssertionError("pickle forbidden")):
            state, targets, options, result = search()
            self.assertTrue(optimization_is_current(result, state, None, 60, 5, targets, options))
        self.assertNotIn("import pickle", (ROOT / "src/live_events.py").read_text())


class DiagnosisTests(unittest.TestCase):
    def test_staff_blocker_with_unchanged_robustness_and_better_wait(self):
        _, _, _, result = search()
        diagnosis = result["diagnosis"]
        self.assertEqual(result["current_verified"]["robustness"], 0.)
        self.assertEqual(result["optimized_verified"]["robustness"], 0.)
        self.assertEqual(diagnosis["likely_constraint_type"], "STAFF_BOTTLENECK")
        primary = diagnosis["primary_limiting_factor"]
        self.assertEqual(primary["metric"], "nurse_utilization")
        self.assertEqual((primary["observed"], primary["target"], primary["failed_replications"], primary["total_replications"]), (.5, .9, 5, 5))
        self.assertEqual(primary["severity"], "critical")
        self.assertTrue(diagnosis["secondary_improvements"]["high_risk_mean_wait_improved"])
        self.assertIn("Operational improvement without feasibility recovery", diagnosis["operational_result"])

    def test_actual_nurse_shortage_event_verifies_staff_bottleneck(self):
        state = hospital(LiveCapacity(10, 10, 10, 2))
        state.apply_event(state.propose_event("NURSE_SHORTAGE", 60, dict(count=1)))
        surge = state.propose_event("PATIENT_SURGE", 0, dict(count=4, high_risk_proportion=.5))
        result = optimize_live_policy(state, surge, 30, 5, options=LiveGAOptions(population=4, generations=1, search_replications=1))
        self.assertEqual(result["diagnosis"]["likely_constraint_type"], "STAFF_BOTTLENECK")
        self.assertEqual(result["diagnosis"]["primary_limiting_factor"]["observed"], 1.)

    def test_multiple_resource_failure_is_deterministic(self):
        state = hospital(LiveCapacity(1, 1, 1, 1))
        waiting(state); waiting(state, high=True)
        result = optimize_live_policy(state, None, 60, 3, options=LiveGAOptions(population=4, generations=1, search_replications=1))
        self.assertEqual(result["diagnosis"]["likely_constraint_type"], "MULTIPLE_RESOURCE_BOTTLENECK")
        failed = {c["metric"] for c in result["diagnosis"]["blocking_conditions"]}
        self.assertTrue({"doctor_utilization", "nurse_utilization", "robustness"} <= failed)

    def test_success_explanation_uses_same_payload_and_conditions(self):
        _, _, _, result = search(True)
        explanation = explain_live_policy(result)
        self.assertEqual(result["optimized_verified"]["robustness"], 100.)
        self.assertEqual(explanation["diagnosis"]["overall_result"], "VERIFIED_ACCEPTABLE_POLICY")
        self.assertEqual(explanation["diagnosis"]["blocking_conditions"], [])
        self.assertIn("All verified runs met", explanation["explanation"])
        self.assertIn("unchanged", explanation["explanation"])

    def test_no_key_fallback_contains_cause_improvement_limits_and_human_options(self):
        _, _, _, result = search()
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch("live_policy_explanation.Groq") as groq:
            explanation = explain_live_policy(result, True)
            groq.assert_not_called()
        self.assertIn("Nurse utilization", explanation["explanation"])
        self.assertIn("5/5", explanation["explanation"])
        self.assertIn("Finite GA", explanation["explanation"])
        self.assertIn("Human Review", explanation["operational_responses"]["title"])
        self.assertEqual(explanation["provider"], "deterministic fallback")

    def test_payload_rejects_altered_verified_values_before_groq(self):
        _, _, _, result = search()
        for alter in (lambda r: r["optimized_verified"]["aggregate"].update(nurse_utilization=.5),
            lambda r: r["optimized_verified"].update(verdict="HANDLED"),
            lambda r: r["optimized_verified"]["runs"][0].update(verdict="ACCEPTABLE")):
            changed = copy.deepcopy(result); alter(changed)
            with patch("live_policy_explanation.Groq") as groq, self.assertRaises(ValueError):
                explain_live_policy(changed, True)
            groq.assert_not_called()

    def test_groq_receives_validated_structured_inputs_and_cannot_change_verdict(self):
        _, _, _, result = search()
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch("live_policy_explanation.Groq") as groq:
            create = groq.return_value.chat.completions.create
            create.return_value.choices = [MagicMock(message=MagicMock(content='{"themes":["high_risk_wait"]}'))]
            explanation = explain_live_policy(result, True)
            payload = json.loads(create.call_args.kwargs["messages"][1]["content"])
            self.assertEqual(payload, explanation["structured_payload"])
            self.assertEqual(payload["optimized_policy"]["verdict"], "CANNOT HANDLE")
            self.assertEqual(payload["blocking_conditions"][0]["resource_pressure"]["peak_utilization"], 1.)
            self.assertTrue(payload["physical_capacity"]["remained_unchanged"])
            self.assertEqual(create.call_args.kwargs["model"], "openai/gpt-oss-120b")
            create.return_value.choices = [MagicMock(message=MagicMock(content='{"themes":["diagnosis"],"verdict":"HANDLED"}'))]
            invalid = explain_live_policy(result, True)
            self.assertEqual(invalid["provider"], "deterministic fallback")
            self.assertIn("CANNOT HANDLE", invalid["summary"])

    def test_actual_learned_rules_and_single_class_no_fabrication(self):
        _, _, _, result = search(True)
        xai = explain_live_tree(result)
        self.assertTrue(xai["available"])
        payload = build_explanation_payload(result, xai)
        self.assertEqual(payload["decision_tree_xai"]["rules"], xai["rules"])
        self.assertEqual(payload["decision_tree_xai"]["rule_path"], xai["rule_path"])
        self.assertAlmostEqual(sum(xai["class_probabilities"].values()), 1.)
        pattern = xai["failure_pattern"]
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern["prediction"], "UNACCEPTABLE")
        self.assertGreater(pattern["observed_unacceptable_replications"], 0)
        self.assertIn(pattern["policy"], [row["policy"] for row in result["evaluations"]])
        invented = copy.deepcopy(xai); invented["rules"] = ["invented rule"]
        with self.assertRaises(ValueError):
            build_explanation_payload(result, invented)
        _, _, _, failed = search()
        self.assertFalse(build_explanation_payload(failed)["decision_tree_xai"]["available"])

    def test_diagnosis_explanation_stale_for_state_metrics_xai_and_output(self):
        state, targets, options, result = search(True)
        explanation = explain_live_policy(result)
        self.assertTrue(explanation_is_current(explanation, result))
        changed = copy.deepcopy(result)
        changed["optimized_verified"]["aggregate"]["mean_wait"] += 1.
        self.assertFalse(explanation_is_current(explanation, changed))
        self.assertFalse(optimization_is_current(changed, state, None, 60, 5, targets, options))
        modified = copy.deepcopy(explanation); modified["explanation"] = "Invented clinical advice."
        self.assertFalse(explanation_is_current(modified, result))
        state.advance_simulation(1.)
        self.assertFalse(optimization_is_current(result, state, None, 60, 5, targets, options))

    def test_classification_precedence_beds_rare_capacity_waits_recovery_process_demand(self):
        targets = ScenarioConfig()
        def prediction(metrics):
            rows = []
            for values in metrics:
                row = dict(mean_wait=0., high_risk_mean_wait=0., icu_utilization=.2, general_utilization=.2,
                    doctor_utilization=.2, nurse_utilization=.2, queue_at_horizon_end=0,
                    additional_arrivals=0, completed_patients=0, recovery_time_minutes=1.)
                row.update(values)
                row.update(checkpoint_queue=0, unresolved_event_waiting=int(row["recovery_time_minutes"] is None))
                row["resource_pressure"] = {}
                for name in ("icu", "general", "doctor", "nurse"):
                    accumulator = PressureIntegrator(.9)
                    accumulator.integrate(row[name + "_utilization"], 60.)
                    row["resource_pressure"][name] = accumulator.result()
                verdict, conditions = live_run_verdict(row, targets)
                row.update(verdict=verdict, condition_breakdown=conditions)
                rows.append(row)
            accepted = sum(r["verdict"] == "HANDLED_RUN" for r in rows)
            robustness = 100 * accepted / len(rows)
            return dict(runs=rows, acceptable_runs=accepted, robustness=robustness,
                verdict=handling_verdict(robustness, np.mean([r["mean_wait"] for r in rows]),
                    np.mean([r["high_risk_mean_wait"] for r in rows]), targets))
        cases = [([dict(icu_utilization=1.)] * 3, "BED_CAPACITY_BOTTLENECK"),
            ([dict(nurse_utilization=1.), {}, {}], "PHYSICAL_CAPACITY_BOTTLENECK"),
            ([dict(mean_wait=30.)] * 3, "WAITING_TIME_TARGET_FAILURE"),
            ([dict(mean_wait=30., recovery_time_minutes=None)] * 3, "RECOVERY_FAILURE"),
            ([dict(mean_wait=30., queue_at_horizon_end=1)] * 3, "PROCESS_POLICY_LIMITATION"),
            ([dict(nurse_utilization=1., doctor_utilization=1., queue_at_horizon_end=10,
                additional_arrivals=10, completed_patients=0)] * 3, "DEMAND_OVERLOAD")]
        for values, category in cases:
            diagnosed = diagnose_prediction(prediction(values), targets, policy=dict(nurse_reserve_percentage=.1))
            self.assertEqual(diagnosed["likely_constraint_type"], category)

    def test_queue_nonrecovery_does_not_add_verdict_rules(self):
        state = hospital()
        event = state.propose_event("PATIENT_SURGE", 30, dict(count=1))
        result = optimize_live_policy(state, event, 10, 3,
            options=LiveGAOptions(population=4, generations=1, search_replications=1))
        self.assertEqual(result["diagnosis"]["overall_result"], "VERIFIED_ACCEPTABLE_POLICY")
        recovery = next(c for c in result["diagnosis"]["operational_observations"] if c["metric"] == "non_recovery")
        self.assertEqual(recovery["observed"], 1.)
        self.assertFalse(recovery["affects_acceptance"])
        self.assertEqual(result["optimized_verified"]["verdict"], "HANDLED")

    def test_initial_staff_peak_is_identified_without_interrupting_patients(self):
        state = hospital(LiveCapacity(10, 10, 10, 1))
        patient = waiting(state)
        state._dispatch()
        assignment = patient.assigned_nurse_id
        result = optimize_live_policy(state, None, 30, 3,
            options=LiveGAOptions(population=4, generations=1, search_replications=1))
        primary = result["diagnosis"]["primary_limiting_factor"]
        self.assertIsNone(primary)
        self.assertEqual(result["optimized_verified"]["verdict"], "HANDLED")
        self.assertEqual(patient.assigned_nurse_id, assignment)
        self.assertIn("did not constitute sustained", explain_live_policy(result)["explanation"])

    def test_unvalidated_extra_inputs_and_generation_changes_are_rejected(self):
        _, _, _, result = search()
        changed = copy.deepcopy(result)
        changed["provenance"]["unsupported_instruction"] = "Invented clinical conclusion"
        with patch("live_policy_explanation.Groq") as groq, self.assertRaises(ValueError):
            explain_live_policy(changed, True)
        groq.assert_not_called()
        explanation = explain_live_policy(result)
        explanation["explanation_provenance"]["generation_state"]["model"] = "invented"
        self.assertFalse(explanation_is_current(explanation, result))


if __name__ == "__main__":
    unittest.main()

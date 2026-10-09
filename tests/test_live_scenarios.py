"""Strict scenario parsing, read-only preview and compound human review."""
from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from test_live_policy_ga import hospital
from live_hospital_state import OperatingPolicy, LiveCapacity
from live_scenario_interpreter import SCENARIO_SCHEMA, interpret_scenario, validate_scenario, scenario_context, interpretation_is_current, audit_entry
from live_event_context import apply_event_context
from live_lookahead import simulate_lookahead_from_current_state
from live_policy_ga import optimize_live_policy, LiveGAOptions
from live_policy_explanation import explain_live_policy


def event(kind="PATIENT_SURGE", count=3, duration=20., delay=0., proportion=None, rate=None):
    return dict(event_type=kind, duration_minutes=duration, start_delay_minutes=delay,
        parameters=dict(count=count, arrival_rate=rate, high_risk_proportion=proportion))


def payload(*events):
    return dict(events=list(events), assumptions=[], warnings=[], clarification=None)


def mocked_interpret(text, state, data):
    with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch("live_scenario_interpreter.Groq") as groq:
        request = groq.return_value.chat.completions.create
        request.return_value.choices = [MagicMock(message=MagicMock(content=json.dumps(data)))]
        result = interpret_scenario(text, state)
        assert request.call_args.kwargs["model"] == "openai/gpt-oss-120b"
        assert request.call_args.kwargs["response_format"]["type"] == "json_schema"
        assert request.call_args.kwargs["messages"][1]["content"] == text
        return result


class ScenarioTests(unittest.TestCase):
    @patch("live_dashboard.wall_seconds", new=lambda: 0.)
    def test_documented_examples_raw_provider_json_through_ui_preview(self):
        from streamlit.testing.v1 import AppTest
        cases = [
            ("A bus accident sends 40 patients over 20 minutes", payload(event(count=40))),
            ("25 nurses are unavailable for two hours.", payload(event("NURSE_SHORTAGE", 25, 120.))),
            ("Patient arrivals increase to 10 per hour for 90 minutes.", payload(event("ARRIVAL_RATE_SPIKE", None, 90., rate=10.))),
            ("20 ICU beds become unavailable for one hour.", payload(event("ICU_BED_OUTAGE", 20, 60.))),
            ("30 patients arrive while 10 doctors are unavailable.", payload(event(count=30, duration=None), event("DOCTOR_SHORTAGE", 10, None))),
        ]
        def button(app, label):
            return next(w for w in app.button if w.label == label)
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch("live_scenario_interpreter.Groq") as groq:
            app = AppTest.from_file(str(ROOT / "src/dashboard.py"), default_timeout=60).run()
            button(app, "Start Live Simulation").click().run()
            state = app.session_state["v3_live_hospital"]
            before = state.state_hash()
            for text, data in cases:
                # Provider JSON goes through the actual textarea/button/decode/validator/preview path.
                groq.return_value.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content=json.dumps(data)))]
                app.text_area(key="v3_scenario_text").set_value(text).run()
                button(app, "Interpret Scenario").click().run()
                result = app.session_state["v3_scenario_interpretation"]
                if text == cases[-1][0]:
                    self.assertEqual(result["status"], "CLARIFICATION")
                    self.assertIn("duration of DOCTOR_SHORTAGE", result["message"])
                else:
                    self.assertEqual(result["status"], "VALID")
                    row = result["parsed_events"][0]
                    self.assertEqual(row["duration_minutes"], data["events"][0]["duration_minutes"])
                    self.assertEqual(row["parameters"], {k: v for k, v in data["events"][0]["parameters"].items() if v is not None})
                    scenario_context(result).validate(state)
                    if text == cases[0][0]:
                        self.assertTrue(any("+40 patients over 20m" in w.value for w in app.markdown))
                        button(app, "Run Look-Ahead").click().run()
                        self.assertEqual(app.session_state["v3_scenario_preview"]["aggregate"]["event_metrics"]["surge_patients_introduced"], 40.)
                        button(app, "Edit Scenario").click().run()
                        self.assertEqual(app.text_area(key="v3_scenario_text").value, text)
                        button(app, "Interpret Scenario").click().run()
                    button(app, "Cancel").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(before, state.state_hash())

    def test_provider_schema_forbids_derived_surge_rate_and_nulls_normalize(self):
        variants = SCENARIO_SCHEMA["properties"]["events"]["items"]["anyOf"]
        self.assertEqual(len(variants), 6)
        for variant in variants:
            kind = variant["properties"]["event_type"]["enum"][0]
            fields = variant["properties"]["parameters"]["properties"]
            self.assertEqual(set(fields), {"count", "arrival_rate", "high_risk_proportion"})
            if kind != "ARRIVAL_RATE_SPIKE":
                self.assertEqual(fields["arrival_rate"]["type"], "null")
            if kind != "PATIENT_SURGE":
                self.assertEqual(fields["high_risk_proportion"]["type"], "null")
        state = hospital(LiveCapacity(220, 550, 80, 140))
        result = mocked_interpret("A bus accident sends 40 patients over 20 minutes", state, payload(event(count=40)))
        self.assertEqual(result["parsed_events"][0]["parameters"], {"count": 40})
        self.assertTrue(any("RF-profile" in s for s in result["assumptions"]))
        self.assertTrue(any("current simulated time" in s for s in result["assumptions"]))
        # Exact invalid shape observed from the real provider; must not silently discard 120.
        invalid = mocked_interpret("A bus accident sends 40 patients over 20 minutes", state, payload(event(count=40, rate=120)))
        self.assertEqual(invalid["status"], "INVALID")
        self.assertEqual(invalid["technical_reason"], "PATIENT_SURGE received unsupported non-null parameter 'arrival_rate'.")
        for row in (event("DOCTOR_SHORTAGE", 10, rate=5), event("DOCTOR_SHORTAGE", 10, duration=None, rate=5),
                    event("NURSE_SHORTAGE", 10, proportion=0.), event(rate=0)):
            with self.assertRaisesRegex(ValueError, "unsupported non-null parameter"):
                validate_scenario(payload(row), state, "invalid")
        with self.assertRaisesRegex(ValueError, "requires integer parameter 'count'"):
            validate_scenario(payload(event(count=40.5)), state, "invalid")

    @patch("live_dashboard.wall_seconds", new=lambda: 0.)
    def test_precise_provider_validation_reason_is_visible_in_ui_details(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch("live_scenario_interpreter.Groq") as groq:
            groq.return_value.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content=json.dumps(payload(event(count=40, rate=120)))))]
            app = AppTest.from_file(str(ROOT / "src/dashboard.py"), default_timeout=60).run()
            next(b for b in app.button if b.label == "Start Live Simulation").click().run()
            app.text_area(key="v3_scenario_text").set_value("A bus accident sends 40 patients over 20 minutes").run()
            next(b for b in app.button if b.label == "Interpret Scenario").click().run()
            self.assertTrue(any(w.label == "Scenario Interpretation Details" for w in app.expander))
            self.assertTrue(any("unsupported non-null parameter 'arrival_rate'" in w.value for w in app.markdown))
            self.assertFalse(app.exception)

    def test_six_supported_text_cases_use_existing_groq_schema(self):
        state = hospital()
        cases = [("A bus accident sends 3 patients over 20 minutes", event()),
            ("2 nurses unavailable for an hour", event("NURSE_SHORTAGE", 2, 60.)),
            ("Arrivals rise to 10 per hour for 90 minutes", event("ARRIVAL_RATE_SPIKE", None, 90., rate=10.)),
            ("2 ICU beds unavailable for an hour", event("ICU_BED_OUTAGE", 2, 60.)),
            ("2 general beds unavailable for an hour", event("GENERAL_BED_OUTAGE", 2, 60.)),
            ("2 doctors unavailable for an hour", event("DOCTOR_SHORTAGE", 2, 60.))]
        before = state.state_hash()
        for text, row in cases:
            result = mocked_interpret(text, state, payload(row))
            self.assertEqual(result["status"], "VALID")
            self.assertEqual(result["parsed_events"][0]["event_type"], row["event_type"])
        self.assertEqual(before, state.state_hash())

    def test_compound_offsets_preview_and_apply_are_independent(self):
        state = hospital()
        result = mocked_interpret("3 patients arrive now; 1 doctor unavailable 20 minutes later", state,
            payload(event(duration=0.), event("DOCTOR_SHORTAGE", 1, 30., delay=20.)))
        context = scenario_context(result)
        self.assertEqual(context.events[1].start_sim_time - context.events[0].start_sim_time, 20.)
        before = state.state_hash()
        prediction = simulate_lookahead_from_current_state(state, context, 60., 3)
        self.assertEqual(state.state_hash(), before)
        self.assertEqual(prediction["aggregate"]["event_metrics"]["surge_patients_introduced"], 3.)
        apply_event_context(state, context)
        self.assertEqual(len(state.stress_events), 2)
        self.assertEqual(state.event_metrics(context.event_id)["surge_patients_introduced"], 3)
        self.assertEqual(state.stress_events[context.events[1].event_id].status, "SCHEDULED")

    def test_unknown_fields_types_negative_fraction_and_capacity_rejected(self):
        state = hospital()
        invalid = [event("UNSUPPORTED"), event(count=-1), event(count=0), event("NURSE_SHORTAGE", 100),
            event(proportion=1.5), event(count=True), event(duration=True), event(delay=-1),
            event("DOCTOR_SHORTAGE", 1, 0.), event("ARRIVAL_RATE_SPIKE", None, rate=-1.)]
        before = state.state_hash()
        for row in invalid:
            with self.assertRaises(ValueError):
                validate_scenario(payload(row), state, "invalid")
        data = payload(event()); data["verdict"] = "HANDLED"
        with self.assertRaises(ValueError):
            validate_scenario(data, state, "invalid")
        data = payload(event()); data["events"][0]["physical_capacity"] = 1000
        with self.assertRaises(ValueError):
            validate_scenario(data, state, "invalid")
        self.assertEqual(before, state.state_hash())

    def test_missing_values_disclose_defaults_or_request_clarification(self):
        state = hospital()
        result = validate_scenario(payload(event(duration=None, delay=None)), state, "3 patients suddenly arrive")
        self.assertEqual(result["parsed_events"][0]["duration_minutes"], 20.)
        self.assertTrue(any("20 minutes" in text for text in result["assumptions"]))
        self.assertTrue(any("RF-profile" in text for text in result["assumptions"]))
        result = validate_scenario(payload(event("NURSE_SHORTAGE", 1, None)), state, "1 nurse unavailable")
        self.assertEqual(result["status"], "CLARIFICATION")
        self.assertNotIn("parsed_events", result)
        self.assertEqual(validate_scenario(payload(event(count=None)), state, "patients arriving")["status"], "CLARIFICATION")
        data = payload(); data["clarification"] = "Specify how many patients."
        self.assertEqual(validate_scenario(data, state, "many patients")["status"], "CLARIFICATION")

    def test_missing_key_and_request_failure_preserve_manual_builder(self):
        state = hospital()
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch("live_scenario_interpreter.Groq") as groq:
            self.assertIn("Use Manual Event Builder", interpret_scenario("3 patients", state)["message"])
            groq.assert_not_called()
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch("live_scenario_interpreter.Groq", side_effect=RuntimeError("offline")):
            self.assertEqual(interpret_scenario("3 patients", state)["status"], "UNAVAILABLE")
        state.apply_event(state.propose_event("PATIENT_SURGE", 0., dict(count=3)))
        self.assertEqual(state._arrived, 3)

    def test_malformed_provider_output_is_nonfatal_and_has_no_authority(self):
        state = hospital()
        self.assertEqual(mocked_interpret("scenario", state, {"events": [], "verdict": "HANDLED"})["status"], "INVALID")
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch("live_scenario_interpreter.Groq") as groq:
            groq.return_value.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="not JSON"))]
            self.assertEqual(interpret_scenario("scenario", state)["status"], "INVALID")

    def test_conflicting_outages_are_atomic_and_counts_use_existing_usable_semantics(self):
        state = hospital()
        before = state.state_hash()
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            validate_scenario(payload(event("DOCTOR_SHORTAGE", 3), event("DOCTOR_SHORTAGE", 3)), state, "bad")
        self.assertEqual(before, state.state_hash())
        state.apply_event(state.propose_event("DOCTOR_SHORTAGE", 60., dict(count=3)))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            validate_scenario(payload(event("DOCTOR_SHORTAGE", 2)), state, "bad")

    def test_two_surges_aggregate_metrics_without_double_counting(self):
        state = hospital()
        result = validate_scenario(payload(event(count=2, proportion=1.), event(count=3, delay=10., proportion=0.)), state, "two surges")
        prediction = simulate_lookahead_from_current_state(state, scenario_context(result), 60., 2)
        self.assertEqual(prediction["aggregate"]["event_metrics"]["surge_patients_introduced"], 5.)

    def test_compound_ga_provenance_xai_explanation_and_temporary_recovery(self):
        state = hospital()
        parsed = validate_scenario(payload(event(count=2), event("NURSE_SHORTAGE", 1, 40., delay=10.)), state, "compound")
        context = scenario_context(parsed)
        capacity = asdict(state.capacity)
        before = state.state_hash()
        result = optimize_live_policy(state, context, 120., 3, options=LiveGAOptions(population=4, generations=1, search_replications=1))
        self.assertEqual(state.state_hash(), before)
        self.assertTrue(result["provenance"]["proposed_event"]["related_events"])
        self.assertIn("related_events", explain_live_policy(result)["structured_payload"]["event_context"]["event"])
        state.apply_operating_policy(OperatingPolicy(high_risk_priority_weight=4.), event=context, temporary=True,
            verification=result["optimized_verified"])
        self.assertEqual(len(state._temporary_policy["recovery_event_ids"]), 2)
        apply_event_context(state, context)
        state.advance_large_skip(120.)
        self.assertIsNone(state._temporary_policy)
        self.assertEqual(state.current_policy, OperatingPolicy())
        self.assertEqual(asdict(state.capacity), capacity)

    def test_hash_and_audit_are_bounded_and_stale_safe(self):
        state = hospital()
        result = validate_scenario(payload(event()), state, "3 patients")
        self.assertTrue(interpretation_is_current(result, state, "3 patients"))
        self.assertFalse(interpretation_is_current(result, state, "4 patients"))
        tampered = copy.deepcopy(result); tampered["assumptions"].append("changed")
        self.assertFalse(interpretation_is_current(tampered, state, "3 patients"))
        state.advance_simulation(1.)
        self.assertFalse(interpretation_is_current(result, state, "3 patients"))
        session = {}
        for index in range(250):
            audit_entry(session, state, "TEST", index=index)
        self.assertEqual(len(session["v3_scenario_audit"]), 200)
        json.dumps(list(session["v3_scenario_audit"]), allow_nan=False)

    @patch("live_dashboard.wall_seconds", new=lambda: 0.)
    def test_dashboard_interpret_cancel_edit_confirm_apply_and_stale(self):
        from streamlit.testing.v1 import AppTest
        def button(app, label):
            return next(w for w in app.button if w.label == label)
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch("live_scenario_interpreter.Groq") as groq:
            groq.return_value.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content=json.dumps(payload(event(count=3)))))]
            app = AppTest.from_file(str(ROOT / "src/dashboard.py"), default_timeout=60).run()
            button(app, "Start Live Simulation").click().run()
            state = app.session_state["v3_live_hospital"]
            before = state.state_hash()
            app.text_area(key="v3_scenario_text").set_value("3 patients over 20 minutes").run()
            button(app, "Interpret Scenario").click().run()
            self.assertEqual(state.state_hash(), before)
            button(app, "Cancel").click().run()
            self.assertEqual(state.state_hash(), before)
            button(app, "Interpret Scenario").click().run()
            button(app, "Edit Scenario").click().run()
            self.assertEqual(app.text_area(key="v3_scenario_text").value, "3 patients over 20 minutes")
            self.assertEqual(state.state_hash(), before)
            button(app, "Interpret Scenario").click().run()
            button(app, "Run Look-Ahead").click().run()
            self.assertEqual(state.state_hash(), before)
            self.assertFalse(app.exception)
            button(app, "Apply Scenario Events").click().run()
            self.assertEqual(len(state.stress_events), 1)
            self.assertFalse(app.exception)
            self.assertTrue(any("Applied interpreted scenario" in option for option in app.selectbox(key="v3_ga_context_name").options))
            self.assertTrue(any(row["action"] == "APPLY_SCENARIO_EVENTS" for row in app.session_state["v3_scenario_audit"]))

    @patch("live_dashboard.wall_seconds", new=lambda: 0.)
    def test_dashboard_compound_ga_reject_retry_apply_revalidate_and_recovery(self):
        from streamlit.testing.v1 import AppTest
        def button(app, label):
            return next(w for w in app.button if w.label == label)
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch("live_scenario_interpreter.Groq") as groq:
            groq.return_value.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content=json.dumps(
                payload(event(count=3), event("NURSE_SHORTAGE", 1, 60., delay=20.)))))]
            app = AppTest.from_file(str(ROOT / "src/dashboard.py"), default_timeout=60).run()
            button(app, "Start Live Simulation").click().run()
            app.number_input(key="v3_ga_population").set_value(4)
            app.number_input(key="v3_ga_generations").set_value(1)
            app.number_input(key="v3_ga_search_reps").set_value(1)
            app.text_area(key="v3_scenario_text").set_value("3 patients over 20 minutes; 1 nurse unavailable 20 minutes later for an hour").run()
            state = app.session_state["v3_live_hospital"]
            before = state.state_hash()
            button(app, "Interpret Scenario").click().run()
            button(app, "Run Look-Ahead").click().run()
            button(app, "Optimize Scenario Operating Policy").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(before, state.state_hash())
            button(app, "Reject Recommendation").click().run()
            self.assertEqual(before, state.state_hash())
            self.assertTrue(button(app, "Apply Optimized Operating Policy").disabled)
            button(app, "Modify / Retry").click().run()
            button(app, "Optimize Scenario Operating Policy").click().run()
            old_policy = state.current_policy
            capacity = asdict(state.capacity)
            button(app, "Apply Optimized Operating Policy").click().run()
            self.assertEqual(len(state._temporary_policy["recovery_event_ids"]), 2)
            self.assertTrue(any("STALE scenario" in w.value for w in app.warning))
            button(app, "Revalidate Unchanged Scenario").click().run()
            button(app, "Run Look-Ahead").click().run()
            button(app, "Apply Scenario Events").click().run()
            self.assertFalse(app.exception)
            state.advance_large_skip(240.)
            app.run()
            self.assertIsNone(state._temporary_policy)
            self.assertEqual(state.current_policy, old_policy)
            self.assertEqual(asdict(state.capacity), capacity)
            actions = {row["action"] for row in app.session_state["v3_scenario_audit"]}
            self.assertTrue({"HUMAN_REJECT", "HUMAN_MODIFY_RETRY", "POLICY_APPLIED", "SCENARIO_REVALIDATED", "APPLY_SCENARIO_EVENTS"} <= actions)
            # Revalidation and all simulations use no extra interpreter calls.
            self.assertEqual(groq.return_value.chat.completions.create.call_count, 1)


if __name__ == "__main__":
    unittest.main()

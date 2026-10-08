"""Real RF/simulator/GA/XAI with mocked network, Streamlit workflow checks."""
from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streamlit.testing.v1 import AppTest
from test_part3 import interpretation
from feedback_interpreter import prepare_modification_v2


def widget(app, group, label):
    return next(w for w in getattr(app, group) if w.label == label)


def main():
    started = time.perf_counter()
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch("feedback_interpreter.Groq") as client, \
         patch("genetic_algorithm_v2.save_ga_v2"), patch("decision_tree_xai_v2.save_xai_v2"), \
         patch("scenario_evaluation.save_current_policy_results"), patch("llm_explanation.Path.write_text"):
        app = AppTest.from_file(str(ROOT / "src/dashboard.py"), default_timeout=60)
        app.session_state["ga_population"] = 4
        app.session_state["ga_generations"] = 1
        app.run()
        assert not app.exception, [e.value for e in app.exception]
        labels = [w.label for w in app.number_input]
        assert not any(label.startswith(("Minimum ICU", "Maximum ICU", "Minimum General", "Maximum General", "Minimum Concurrent", "Maximum Concurrent")) for label in labels)
        widget(app, "number_input", "Simulation Duration (hours)").set_value(4.)
        widget(app, "number_input", "Replications Per Scenario").set_value(3)
        widget(app, "number_input", "GA Replications Per Scenario").set_value(1)
        app.run()
        widget(app, "button", "Evaluate Current Policy").click().run()
        assert not app.exception and not app.error
        # Toggle OFF ignores text and makes no provider calls.
        app.text_area(key="natural_instruction").set_value("Do not use more than 110 doctors.").run()
        widget(app, "button", "Run Optimization").click().run(timeout=60)
        assert not app.exception, [e.value for e in app.exception]
        assert not app.error, [e.value for e in app.error]
        assert app.session_state["optimization_run"]
        result = app.session_state["v2_ga_result"]
        assert result["provenance"]["bounds"]["doctors"] == [59, 128]
        assert result["final_verification_replications"] == 3
        assert app.session_state["v2_llm_result"]["authoritative_status"] == result["verified_evaluation"]["overall_policy_status"]
        assert not app.session_state["v2_llm_result"]["available"]
        assert any(e.label == "Detailed AI Explanation" and not e.proto.expanded for e in app.expander)
        client.assert_not_called()
        # Enabled, missing key: nonfatal feedback fallback still permits GA.
        app.toggle(key="ai_interpretation").set_value(True).run()
        assert not app.session_state["optimization_run"]
        assert any("stale" in w.value for w in app.warning)
        widget(app, "button", "Run Optimization").click().run(timeout=60)
        assert not app.exception and not app.error
        assert app.session_state["optimization_run"]
        client.assert_not_called()

        # Supported interpretation tightens bounds, raises priorities, and previews.
        parsed = interpretation()
        parsed["resource_constraints"]["doctors"]["max"] = 110
        parsed["priorities"]["high_risk_priority"] = True
        parsed["priorities"]["worst_case_priority"] = True
        with patch("feedback_interpreter.interpret_feedback_v2", return_value={"available": True, "interpretation": parsed}):
            widget(app, "button", "Preview AI Interpretation").click().run()
        assert not app.exception and not app.error
        widget(app, "button", "Run Optimization").click().run(timeout=60)
        assert not app.exception and not app.error
        result = app.session_state["v2_ga_result"]
        assert result["provenance"]["bounds"]["doctors"] == [59, 110]
        assert result["provenance"]["ga_configuration"]["scenario_weights"] == [1., 2., 6.]
        assert result["provenance"]["dependencies"]["feedback"]["interpretation"] == parsed

        # Explicit availability may exclude the unchanged current/reference policy.
        previous_reference = copy.deepcopy(app.session_state["current_policy_evaluation"])
        app.text_area(key="natural_instruction").set_value("only 10 nurses are available").run()
        limited = interpretation()
        limited["resource_constraints"]["nurses"]["max"] = 10
        with patch("feedback_interpreter.interpret_feedback_v2", return_value={"available": True, "interpretation": limited}):
            widget(app, "button", "Preview AI Interpretation").click().run()
        assert not app.exception and not app.error
        assert any("outside the newly stated availability" in warning.value for warning in app.warning)
        assert any("Effective GA range: 1–10" in text.value for text in app.markdown)
        widget(app, "button", "Run Optimization").click().run(timeout=60)
        assert not app.exception and not app.error
        result = app.session_state["v2_ga_result"]
        assert result["provenance"]["automatic_bounds"]["nurses"] == [70, 150]
        assert result["provenance"]["effective_bounds"]["nurses"] == [1, 10]
        assert not result["provenance"]["current_policy_feasible"]
        assert result["history"][0]["nurses"] == 10
        assert all(row["nurses"] <= 10 for row in result["history"])
        assert result["best_policy"]["nurses"] <= 10
        assert app.session_state["current_policy_evaluation"] == previous_reference

        # Review uses active V2 run, Prepare Modification transfers comments only.
        app.radio(key="review_decision").set_value("Needs Modification").run()
        app.text_area(key="review_comment").set_value("Prioritize resource efficiency.").run()
        with patch("feedback_interpreter.save_review_v2") as save_review:
            widget(app, "button", "Save Review").click().run()
            assert save_review.call_args.args[0]["run_id"] == result["run_id"]
        historical = json.dumps(result, sort_keys=True)
        widget(app, "button", "Prepare Modification").click().run()
        assert app.session_state["natural_instruction"] == "Prioritize resource efficiency."
        assert app.session_state["prepared_from_run_id"] == result["run_id"]
        assert json.dumps(app.session_state["v2_ga_result"], sort_keys=True) == historical
        assert not app.session_state["optimization_run"]

        # Contradictory feedback blocks run and retains historical result.
        bad = interpretation()
        bad["resource_constraints"]["doctors"] = {"min": 100, "max": 90}
        with patch("feedback_interpreter.interpret_feedback_v2", return_value={"available": True, "interpretation": bad}):
            widget(app, "button", "Preview AI Interpretation").click().run()
        assert app.error and not app.exception
        widget(app, "button", "Run Optimization").click().run()
        assert any("cannot run" in e.value for e in app.error)
        assert json.dumps(app.session_state["v2_ga_result"], sort_keys=True) == historical
        # Changing current resources derives fresh bounds, never manual cached limits.
        app.toggle(key="ai_interpretation").set_value(False).run()
        app.number_input(key="doctors_number").set_value(100).run()
        widget(app, "button", "Run Optimization").click().run(timeout=60)
        assert not app.exception and not app.error
        assert app.session_state["v2_ga_result"]["provenance"]["bounds"]["doctors"] == [70, 150]
    print(json.dumps({"dashboard_part3": "PASS", "real_RF_simulation_GA_XAI": True,
        "network_calls_without_key": 0, "runtime_seconds": time.perf_counter() - started,
        "verified_status": result["verified_evaluation"]["overall_policy_status"],
        "recommended_policy": result["best_policy"]}, indent=2))


if __name__ == "__main__":
    main()

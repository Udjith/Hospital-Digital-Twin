"""Streamlit interaction and actual main.py server smoke checks."""
from pathlib import Path
import os
import subprocess
import sys
import time
import urllib.request
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from streamlit.testing.v1 import AppTest
    with patch("scenario_evaluation.save_current_policy_results"):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/dashboard.py"), default_timeout=30).run()
        assert not app.exception, list(app.exception)
        for widget in app.number_input:
            if widget.label == "Simulation Duration (hours)":
                widget.set_value(4.)
            if widget.label == "Replications Per Scenario":
                widget.set_value(3)
        app.run()
        next(b for b in app.button if b.label == "Evaluate Current Policy").click().run(timeout=30)
        assert not app.exception, list(app.exception)
        assert not app.error, list(app.error)
        result = app.session_state["current_policy_evaluation"]
        print("DASHBOARD_EVALUATION", result["overall_policy_status"], result["overall_robustness"])
        assert all(len(s["runs"]) == 3 for s in result["scenarios"].values())
        next(w for w in app.number_input if w.label == "Best Arrival Rate (patients/hour)").set_value(30.)
        app.run()
        assert any("Best <= Average <= Worst" in e.value for e in app.error)
        assert not app.exception, list(app.exception)
        print("DASHBOARD_INVALID_ORDER: clear validation error")

    # main.py starts a child Streamlit process. Stop only this owned process tree.
    env = dict(os.environ, STREAMLIT_SERVER_HEADLESS="true", STREAMLIT_SERVER_PORT="8517",
               STREAMLIT_BROWSER_GATHER_USAGE_STATS="false", PYTHONDONTWRITEBYTECODE="1")
    log_path = Path("results/digital_twin/v2_launcher_validation.log")
    with log_path.open("w") as log:
        process = subprocess.Popen([sys.executable, "main.py"], env=env, stdout=log, stderr=log)
        try:
            healthy = False
            for _ in range(60):
                if process.poll() is not None:
                    raise AssertionError(f"Launcher exited: {process.returncode}")
                try:
                    with urllib.request.urlopen("http://localhost:8517/_stcore/health", timeout=1) as response:
                        healthy = response.status == 200
                    if healthy:
                        break
                except OSError:
                    time.sleep(.25)
            assert healthy, "Streamlit did not become healthy"
            print("MAIN_PY: Streamlit HTTP health 200 on port 8517")
        finally:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
            process.wait(timeout=15)


if __name__ == "__main__":
    main()

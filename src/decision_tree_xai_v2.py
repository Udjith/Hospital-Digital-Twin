"""Learned V2 explanations; deterministic scenario verdicts remain authoritative."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text

from genetic_algorithm_v2 import GENES, canonical_hash
from scenario_evaluation import ScenarioConfig, policy_verdict

VERSION = "2.0-scenario-xai"
FEATURES = [*GENES, "simulation_duration_hours", "arrival_rate"]
LABEL = "acceptable_policy"
SINGLE_CLASS_MESSAGE = "Decision Tree explanation unavailable because all evaluated policies received the same deterministic class."


def build_training_data(ga_result):
    """Use each actually evaluated policy/scenario once, validating its labels."""
    config = ScenarioConfig(**ga_result["provenance"]["scenario_configuration"])
    rows = []
    for observation in ga_result["evaluations"]:
        policy = observation["policy"]
        for name, scenario in observation["evaluation"]["scenarios"].items():
            accepted = 0
            for run in scenario["runs"]:
                verdict, conditions = policy_verdict(run, config)
                if verdict != run["policy_verdict"] or conditions != run["condition_breakdown"]:
                    raise ValueError("GA history contains a verdict inconsistent with Digital Twin conditions.")
                accepted += verdict == "ACCEPTABLE"
            robustness = accepted / len(scenario["runs"]) * 100
            verdict = "ACCEPTABLE" if robustness >= config.robustness_threshold else "UNACCEPTABLE"
            if verdict != scenario["scenario_verdict"] or abs(robustness - scenario["scenario_robustness"]) > 1e-9:
                raise ValueError("GA history scenario label does not match its observed replications.")
            rows.append({**{key: policy[key] for key in GENES},
                         "simulation_duration_hours": config.simulation_duration_hours,
                         "arrival_rate": scenario["arrival_rate"], "scenario": name,
                         "policy_id": canonical_hash(policy), LABEL: int(verdict == "ACCEPTABLE"),
                         "authoritative_verdict": verdict, "scenario_robustness": robustness})
    return pd.DataFrame(rows).drop_duplicates(["policy_id", "scenario"]).reset_index(drop=True) if rows else pd.DataFrame()


def xai_is_current(explanation, ga_result):
    return bool(explanation and ga_result and explanation.get("ga_run_id") == ga_result.get("run_id")
                and explanation.get("ga_provenance") == ga_result.get("provenance")
                and explanation.get("training_data_sha256") == canonical_hash(build_training_data(ga_result).to_dict("records")))


def _rules(tree, row):
    frame = pd.DataFrame([row], columns=FEATURES)
    leaf = tree.apply(frame)[0]
    rules = []
    for node in tree.decision_path(frame).indices:
        if node == leaf:
            continue
        key = FEATURES[tree.tree_.feature[node]]
        threshold = float(tree.tree_.threshold[node])
        actual = float(row[key])
        rules.append(f"{key} {'<=' if actual <= threshold else '>'} {threshold:.2f} (actual {actual:g})")
    return rules


def train_xai_v2(ga_result, max_depth=4, min_samples_leaf=2, seed=42):
    if int(max_depth) != max_depth or max_depth < 1 or int(min_samples_leaf) != min_samples_leaf or min_samples_leaf < 1:
        raise ValueError("Tree depth and minimum leaf samples must be positive integers.")
    data = build_training_data(ga_result)
    classes = data[LABEL].value_counts().to_dict() if not data.empty else {}
    report = dict(version=VERSION, ga_run_id=ga_result["run_id"], ga_provenance=ga_result["provenance"],
                  training_data_sha256=canonical_hash(data.to_dict("records")),
                  features=FEATURES, label_source="Actual GA-tested scenario verdict, validated against deterministic run conditions and robustness threshold",
                  status="unavailable", message=SINGLE_CLASS_MESSAGE,
                  metrics=dict(training_sample_count=len(data), unique_policy_count=int(data.policy_id.nunique()) if not data.empty else 0,
                               class_distribution={"ACCEPTABLE": int(classes.get(1, 0)), "UNACCEPTABLE": int(classes.get(0, 0))},
                               tree_depth=None, leaf_count=None, training_accuracy=None, validation_accuracy=None,
                               validation_train_samples=0, validation_test_samples=0,
                               validation_note="No valid split with both classes in train and validation across distinct policies."),
                  explanations={})
    if len(classes) < 2:
        if data.empty:
            report["message"] = "Decision Tree explanation unavailable because no evaluated policy/scenario observations were found."
        return report, None, data
    X, y = data[FEATURES], data[LABEL]
    def new_tree():
        return DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                      class_weight="balanced", random_state=seed)
    # Group by policy so elite/cache repetitions and scenarios from the same
    # resource configuration cannot leak between the validation partitions.
    if data.policy_id.nunique() >= 4:
        for train, test in GroupShuffleSplit(n_splits=20, test_size=.25, random_state=seed).split(X, y, groups=data.policy_id):
            if y.iloc[train].nunique() == y.iloc[test].nunique() == 2:
                validation_tree = new_tree().fit(X.iloc[train], y.iloc[train])
                report["metrics"].update(validation_accuracy=float(accuracy_score(y.iloc[test], validation_tree.predict(X.iloc[test]))),
                                         validation_train_samples=len(train), validation_test_samples=len(test),
                                         validation_note="Held-out resource policies within this GA experiment; common demand seeds are shared. This is limited validation, not evidence of generalization to new hospitals or demand experiments.")
                break
    tree = new_tree().fit(X, y)
    report["status"], report["message"] = "available", "Learned explanatory patterns; Digital Twin operational rules remain the authoritative verdict."
    report["metrics"].update(training_accuracy=float(accuracy_score(y, tree.predict(X))),
                             tree_depth=int(tree.get_depth()), leaf_count=int(tree.get_n_leaves()))
    report["rules_text"] = export_text(tree, feature_names=FEATURES, decimals=2)
    recommended = ga_result["best_policy"]
    for name, scenario in ga_result["verified_evaluation"]["scenarios"].items():
        row = {**{key: recommended[key] for key in GENES},
               "simulation_duration_hours": ga_result["verified_evaluation"]["configuration"]["simulation_duration_hours"],
               "arrival_rate": scenario["arrival_rate"]}
        frame = pd.DataFrame([row], columns=FEATURES)
        predicted = int(tree.predict(frame)[0])
        probability = dict(zip(tree.classes_, tree.predict_proba(frame)[0])).get(1, 0.)
        report["explanations"][name] = dict(authoritative_verdict=scenario["scenario_verdict"],
                                          authoritative_robustness=scenario["scenario_robustness"],
                                          surrogate_class="ACCEPTABLE" if predicted else "UNACCEPTABLE",
                                          surrogate_probability_acceptable=float(probability),
                                          learned_rules=_rules(tree, row))
    return report, tree, data


def save_xai_v2(report, tree, data, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "explanation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (directory / "decision_tree_rules.txt").write_text(report.get("rules_text", report["message"]), encoding="utf-8")
    data.to_csv(directory / "training_data.csv", index=False)
    # An unavailable explanation replaces the V2 bundle with tree=None so an
    # earlier trained model cannot accidentally be reused as the current tree.
    joblib.dump(dict(tree=tree, features=FEATURES, ga_run_id=report["ga_run_id"],
                     ga_provenance=report["ga_provenance"], training_data_sha256=report["training_data_sha256"]),
                directory / "decision_tree_v2.joblib")


def main():
    parser = argparse.ArgumentParser(description="V2 Decision Tree explanation of GA-tested scenarios")
    parser.add_argument("--ga-result", type=Path, default=Path("results/genetic_algorithm_v2/best_policy.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/decision_tree_xai_v2"))
    args = parser.parse_args()
    result = json.loads(args.ga_result.read_text(encoding="utf-8"))
    report, tree, data = train_xai_v2(result)
    save_xai_v2(report, tree, data, args.results_dir)
    print(json.dumps({"status": report["status"], "message": report["message"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()

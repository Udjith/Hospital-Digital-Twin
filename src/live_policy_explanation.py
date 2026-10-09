"""Live-policy surrogate rules and grounded prose; neither controls verdicts."""
from collections import Counter
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from live_policy_control import GENES
from live_canonical import canonical_hash
from live_policy_diagnosis import build_explanation_payload
from live_time_display import format_duration
from live_pressure import LiveTargets, live_run_verdict
from llm_explanation import Groq, DEFAULT_MODEL


def explain_live_tree(result):
    targets = LiveTargets(**result["provenance"]["targets"])
    rows = []
    for observation in result["evaluations"]:
        for run in observation["evaluation"]["runs"]:
            verdict, conditions = live_run_verdict(run, targets)
            if verdict != run["verdict"] or conditions != run["condition_breakdown"]:
                raise ValueError("Live GA labels disagree with deterministic look-ahead criteria.")
            rows.append({**observation["policy"], "label": int(verdict == "HANDLED_RUN"),
                         "group": tuple(observation["policy"][g] for g in GENES)})
    frame = pd.DataFrame(rows)
    distribution = Counter("ACCEPTABLE" if row["label"] else "UNACCEPTABLE" for row in rows)
    metadata = dict(run_id=result["run_id"], provenance=result["provenance"],
        training_sample_count=len(rows), class_distribution=dict(distribution),
        limitations="Learned operational patterns within this checkpoint/event experiment, not clinical triage or proof of generalization.")
    if len(distribution) < 2:
        return dict(metadata, available=False,
            message="Decision Tree explanation unavailable because all evaluated policies received the same deterministic class.")
    features, labels = frame[list(GENES)], frame["label"]
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(2, len(rows) // 20), random_state=42)
    validation = None
    groups = pd.factorize(frame["group"])[0]
    if len(set(groups)) >= 6 and min(distribution.values()) >= 2:
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(features, labels, groups))
        if len(set(labels.iloc[train])) == len(set(labels.iloc[test])) == 2:
            split_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
            split_tree.fit(features.iloc[train], labels.iloc[train])
            validation = float(accuracy_score(labels.iloc[test], split_tree.predict(features.iloc[test])))
    tree.fit(features, labels)
    row = pd.DataFrame([result["optimized_policy"]], columns=GENES)
    def learned_path(point):
        rules, path = [], []
        for node in tree.decision_path(point).indices:
            feature = tree.tree_.feature[node]
            if feature >= 0:
                gene = GENES[feature]
                threshold = float(tree.tree_.threshold[node])
                operator = "<=" if point.iloc[0, feature] <= threshold else ">"
                rules.append(f"{gene} {operator} {threshold:.3f}")
                path.append(dict(feature=gene, operator=operator, threshold=threshold))
        return rules, path
    rules, rule_path = learned_path(row)
    failure_pattern = None
    negative = features[labels == 0].drop_duplicates()
    if len(negative):
        negative = negative.loc[tree.predict(negative) == 0]
        if len(negative):
            point = negative.iloc[[0]]
            failed_rules, failed_path = learned_path(point)
            match = (features == point.iloc[0]).all(axis=1)
            failure_pattern = dict(policy={k: float(point.iloc[0][k]) for k in GENES},
                rules=failed_rules, rule_path=failed_path, prediction="UNACCEPTABLE",
                observed_unacceptable_replications=int((labels[match] == 0).sum()), tested_replications=int(match.sum()),
                class_probabilities={"ACCEPTABLE" if label else "UNACCEPTABLE": float(probability)
                    for label, probability in zip(tree.classes_, tree.predict_proba(point)[0])},
                limitations="Actual tested-policy path associated with unacceptable outcomes, not a causal explanation or acceptance rule.")
    return dict(metadata, available=True, rules=rules,
        rule_path=rule_path, class_probabilities={"ACCEPTABLE" if label else "UNACCEPTABLE": float(probability)
            for label, probability in zip(tree.classes_, tree.predict_proba(row)[0])},
        failure_pattern=failure_pattern,
        prediction="ACCEPTABLE" if tree.predict(row)[0] else "UNACCEPTABLE",
        tree_depth=int(tree.get_depth()), leaf_count=int(tree.get_n_leaves()),
        training_accuracy=float(accuracy_score(labels, tree.predict(features))), validation_accuracy=validation,
        message="XAI surrogate prediction does not determine the authoritative handling verdict.")


def factual_sentences(result, payload=None):
    payload = payload or build_explanation_payload(result)
    before, after = result["current_verified"], result["optimized_verified"]
    old, new = before["aggregate"], after["aggregate"]
    policy_changes = []
    for gene in GENES:
        a, b = result["current_policy"][gene], result["optimized_policy"][gene]
        if a != b:
            policy_changes.append(f"{gene.replace('_', ' ')} {a:.2f} -> {b:.2f}")
    diagnosis = payload["deterministic_diagnosis"]
    primary = diagnosis["primary_limiting_factor"]
    def display(metric, value):
        if value is None:
            return "not recovered within horizon"
        if "utilization" in metric or metric == "non_recovery":
            return f"{value:.1%}"
        if "wait" in metric or "minutes" in metric or "streak" in metric:
            return format_duration(value)
        if metric == "robustness":
            return f"{value:.1f}%"
        return f"{value:.1f}"
    blockers = []
    for condition in diagnosis["blocking_conditions"]:
        if condition["metric"] == "robustness":
            continue
        pressure = condition.get("resource_pressure")
        if pressure:
            limits = condition["thresholds"]
            blockers.append(f"{condition['label']}: time-weighted utilization {pressure['time_weighted_utilization']:.1%} (target {limits['utilization_target']:.1%}); peak {pressure['peak_utilization']:.1%}; above target {format_duration(pressure['minutes_above_target'])}, longest {format_duration(pressure['longest_above_target_streak'])} (grace {format_duration(limits['sustained_overload_grace_minutes'])}); fully saturated {format_duration(pressure['minutes_at_full_saturation'])}, longest {format_duration(pressure['longest_full_saturation_streak'])} (grace {format_duration(limits['critical_saturation_grace_minutes'])}); failed {condition['failed_replications']}/{condition['total_replications']} runs")
        else:
            blockers.append(f"{condition['label']}: observed {display(condition['metric'], condition['observed'])}, target {display(condition['metric'], condition['target'])}; failed {condition['failed_replications']}/{condition['total_replications']} runs")
    improved = [f"{metric.replace('_', ' ')} {display(metric, change['current'])} to {display(metric, change['optimized'])}"
        for metric, change in diagnosis["metric_changes"]["improved"].items()]
    unchanged = "; ".join(condition["label"] for condition in diagnosis["blocking_conditions"]
        if condition["metric"] in diagnosis["metric_changes"]["unchanged"])
    cause = ("All verified runs met the operational conditions." if after["robustness"] == 100 else
        "The optimized policy met the configured robustness threshold; residual failed runs remain documented." if after["verdict"] == "HANDLED" else
        "The fully verified recommendation did not reach the configured robustness threshold. " + "; ".join(blockers) + ".")
    for name, warning in diagnosis["transient_resource_warnings"].items():
        evidence = warning["evidence"]
        cause += f" {name.title()} briefly peaked above target (mean peak {evidence['peak_utilization']:.1%}); in {warning['warned_replications']}/{warning['total_replications']} runs this did not constitute sustained resource-overload failure."
    if after["verdict"] == "HANDLED" and before["verdict"] != "HANDLED":
        current_failures = diagnosis["current_policy_diagnosis"]["blocking_conditions"]
        cause = "The current policy failed " + "; ".join(f"{f['label']} in {f['failed_replications']}/{f['total_replications']} runs"
            for f in current_failures if f["metric"] != "robustness") + ". " + cause
    xai = payload["decision_tree_xai"]
    xai_text = ("Learned Decision Tree pattern: " + " AND ".join(xai["rules"]) + " -> " + xai["prediction"] +
        ". This is an association from tested policies, not the authoritative cause." if xai.get("available") else xai["message"])
    interpretation = {
        "STAFF_BOTTLENECK": "Repeated staff-utilization failures indicate a modeled staffing bottleneck under this event and these targets.",
        "BED_CAPACITY_BOTTLENECK": "Repeated bed-utilization failures indicate a modeled bed-capacity bottleneck.",
        "MULTIPLE_RESOURCE_BOTTLENECK": "More than one resource exceeded its configured utilization limit in verified runs.",
        "DEMAND_OVERLOAD": "Concurrent resource saturation and growing end-of-window queues indicate demand pressure beyond the tested operating performance.",
        "PHYSICAL_CAPACITY_BOTTLENECK": "Some verified runs exceeded a resource-utilization limit; availability and policy tradeoffs need human review.",
        "PROCESS_POLICY_LIMITATION": "Waiting and queue growth persisted below resource-utilization limits with reserve restrictions, indicating a possible process-policy limitation.",
        "WAITING_TIME_TARGET_FAILURE": "Waiting targets failed despite resource pressure meeting its operational limits.",
        "RECOVERY_FAILURE": "Recovery was not observed within the forecast horizon in some verified runs; recovery monitoring remains necessary.",
        "NO_BLOCKING_CONDITION": "The verified recommendation met the configured robustness requirement.",
    }[diagnosis["likely_constraint_type"]]
    return dict(
        high_risk_wait=f"Predicted high-risk wait changed from {format_duration(old['high_risk_mean_wait'])} to {format_duration(new['high_risk_mean_wait'])}.",
        mean_wait=f"Predicted mean wait changed from {format_duration(old['mean_wait'])} to {format_duration(new['mean_wait'])}.",
        p95_wait=f"Predicted P95 wait changed from {format_duration(old['p95_wait'])} to {format_duration(new['p95_wait'])}.",
        queue_peak=f"Mean predicted queue peak changed from {old['queue_peak']:.1f} to {new['queue_peak']:.1f}.",
        recovery=f"Recovery was observed in {old['recovered_replications']}/{len(before['runs'])} current-policy runs and {new['recovered_replications']}/{len(after['runs'])} optimized-policy runs.",
        reserves="Policy changes: " + ("; ".join(policy_changes) if policy_changes else "the current policy was retained after verification.") + " Reserve values are fractions.",
        diagnosis=cause + " " + interpretation,
        improvements=diagnosis["operational_result"] + ". " + ("Observed improvements: " + "; ".join(improved) + ". " if improved else "No verified metric improvement was observed. ") +
            ("Unchanged blocking conditions: " + unchanged + "." if unchanged else ""),
        xai=xai_text,
        fixed_capacity="Physical capacity remains unchanged: " + ", ".join(f"{value} {key.replace('_', ' ')}" for key, value in result['provenance']['physical_capacity'].items()) + ".",
        limitations="The simulation suggests an operational scheduling policy for human review, not a validated clinical triage protocol. Finite GA search cannot prove that no feasible policy exists.")


def explain_live_policy(result, use_groq=False, xai=None):
    validated = build_explanation_payload(result, xai)
    sentences = factual_sentences(result, validated)
    themes = ["diagnosis", "improvements", "reserves", "fixed_capacity", "limitations"]
    provider, message = "deterministic fallback", "AI prose unavailable or not requested; optimization results remain valid."
    if use_groq and os.environ.get("GROQ_API_KEY"):
        try:
            schema = dict(type="object", additionalProperties=False,
                properties=dict(themes=dict(type="array", minItems=1, maxItems=7,
                    items=dict(type="string", enum=list(sentences)))), required=["themes"])
            response = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=20., max_retries=0).chat.completions.create(
                model=DEFAULT_MODEL, temperature=0,
                messages=[dict(role="system", content="Select factual themes for a concise clinician/administrator explanation of these validated results. Explain the deterministic blocker and actual improvements, including unsuccessful and successful search. Return only the strict schema. Never override verdicts, invent values/rules, claim proof of infeasibility, or issue autonomous clinical decisions. Physical resources remained fixed. Human-review responses are options only."),
                    dict(role="user", content=json.dumps(validated, sort_keys=True, separators=(",", ":"), allow_nan=False))],
                response_format=dict(type="json_schema", json_schema=dict(name="live_policy_explanation", strict=True, schema=schema)))
            payload = json.loads(response.choices[0].message.content)
            if set(payload) != {"themes"} or not isinstance(payload["themes"], list) or not 1 <= len(payload["themes"]) <= 7 or any(
                    not isinstance(t, str) or t not in sentences for t in payload["themes"]):
                raise ValueError("Invalid explanation schema.")
            themes = list(dict.fromkeys(["diagnosis", "improvements"] + payload["themes"] + ["fixed_capacity", "limitations"]))
            provider, message = "Groq / " + DEFAULT_MODEL, "Groq selected factual themes; all displayed claims are generated from verified metrics."
        except Exception:
            message = "AI explanation unavailable. Simulation and optimization results remain valid; deterministic explanation used."
    header = (f"Authoritative verified result: {result['optimized_verified']['verdict']}. "
        f"Predictive robustness changed from {result['current_verified']['robustness']:.1f}% to {result['optimized_verified']['robustness']:.1f}%.")
    generation = dict(requested=bool(use_groq), provider=provider, model=DEFAULT_MODEL,
        schema_version="live-grounded-themes-v2")
    primary = validated["deterministic_diagnosis"]["primary_limiting_factor"]
    reason = "All configured operational targets met the verified robustness requirement." if primary is None else "Remaining limiting condition: " + primary["label"] + f", failing {primary['failed_replications']}/{primary['total_replications']} verified runs."
    wait_theme = "high_risk_wait" if abs(result["current_verified"]["aggregate"]["high_risk_mean_wait"] - result["optimized_verified"]["aggregate"]["high_risk_mean_wait"]) >= 1 / 60 else "mean_wait"
    changed = validated["deterministic_diagnosis"]["policy_changes"]
    policy_sentence = "Policy changes include " + ", ".join(gene.replace("_", " ") + (f" {values['current']:.0%} to {values['optimized']:.0%}" if "reserve" in gene else f" {values['current']:g} to {values['optimized']:g}") for gene, values in list(changed.items())[:2]) + "." if changed else "The current operating policy was retained."
    summary = header + " " + sentences[wait_theme] + " " + reason + " " + policy_sentence + " Physical resources remained unchanged; the finite scheduling-policy recommendation requires clinician/administrator review."
    explanation = header + " " + " ".join(sentences[t] for t in themes)
    explanation_provenance = dict(inputs=validated, input_sha256=canonical_hash(validated), generation_state=generation,
        output_sha256=canonical_hash(dict(summary=summary, explanation=explanation, provider=provider, message=message)))
    explanation_provenance["sha256"] = canonical_hash(explanation_provenance)
    return dict(run_id=result["run_id"], provenance=result["provenance"], provider=provider,
        explanation_provenance=explanation_provenance, structured_payload=validated,
        diagnosis=validated["deterministic_diagnosis"], operational_responses=validated["human_review_responses"],
        message=message, summary=summary, explanation=explanation)


def explanation_is_current(explanation, result, xai=None):
    try:
        provenance = explanation["explanation_provenance"]
        return (explanation["provenance"] == result["provenance"] and
            provenance["input_sha256"] == canonical_hash(build_explanation_payload(result, xai)) and
            provenance["output_sha256"] == canonical_hash({k: explanation[k] for k in ("summary", "explanation", "provider", "message")}) and
            provenance["sha256"] == canonical_hash({k: v for k, v in provenance.items() if k != "sha256"}))
    except (KeyError, ValueError, TypeError):
        return False

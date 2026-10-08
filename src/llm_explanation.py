
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_MODEL = "openai/gpt-oss-120b"

ASCII_PUNCTUATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u202f": " ",
        "\u00a0": " ",
        "\u2192": "->",
        "\u2248": "~",
        "\u2264": "<=",
        "\u2265": ">=",
    }
)


def normalize_ascii_punctuation(text: str) -> str:
    normalized = text.translate(ASCII_PUNCTUATION)
    return normalized.encode("ascii", errors="replace").decode("ascii")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompt(ga_result: dict, xai_result: dict) -> str:
    best_policy = ga_result.get("best_policy", {})
    metrics = ga_result.get("best_metrics", {})
    fitness = ga_result.get("best_fitness")
    xai_class = xai_result.get("decision_tree_class", "unknown")
    probability = xai_result.get("decision_tree_probability_acceptable")
    rules = xai_result.get("decision_path_rules", [])
    limits = xai_result.get("acceptability_definition", {})

    mean_wait = metrics.get("mean_waiting_time_min")
    high_risk_wait = metrics.get("high_risk_mean_waiting_time_min")
    mean_limit = limits.get("mean_waiting_time_min_max")
    high_risk_limit = limits.get("high_risk_mean_waiting_time_min_max")

    threshold_checks = {
        "mean_wait_within_target": (
            float(mean_wait) <= float(mean_limit)
            if mean_wait is not None and mean_limit is not None
            else None
        ),
        "high_risk_wait_within_target": (
            float(high_risk_wait) <= float(high_risk_limit)
            if high_risk_wait is not None and high_risk_limit is not None
            else None
        ),
    }

    rule_text = "\n".join(f"- {r}" for r in rules) or "- No rules available"

    return f"""
You are the explanation module of a healthcare operations Digital Twin project.

Explain the optimized hospital resource-allocation policy to a clinician or
hospital administrator.

Requirements:
- Operational decision support only, not diagnosis or treatment advice.
- Do not invent facts.
- Use "general beds" and "nurses" exactly; do not infer specialties.
- State that outputs come from simulation and optimization.
- GA fitness is relative: higher (less negative) is better for the current
  objective. Do not describe the GA as minimizing fitness or treat fitness as
  an intuitive absolute quality score.
- Explain the Decision Tree rule in plain English.
- Use the supplied threshold-check booleans exactly. Do not claim a metric
  exceeds a target when its boolean is true.
- An empty Decision Tree path means the tree used a constant leaf; do not invent
  an unreported threshold rule. Say only: "The tree used a constant leaf and
  provided no resource-threshold path."
- Format waiting times, throughput, and fitness to no more than 2 decimals.
  Format utilization as percentages with 1 decimal place.
- Mention limitations and that this is not clinically validated.
- Keep the response concise.
- Use ASCII punctuation only. Use '-' for hyphens, '<=' and '>=' for inequalities,
  and ordinary straight apostrophes. Do not use Unicode dashes or special spaces.
- Use these headings:
  1. Recommended Resource Policy
  2. Expected Operational Performance
  3. Why the System Recommends It
  4. Interpretation and Caution

Optimized policy:
{json.dumps(best_policy, indent=2)}

GA fitness:
{fitness}

Digital Twin metrics:
{json.dumps(metrics, indent=2)}

Decision Tree classification:
{xai_class}

Probability acceptable:
{probability}

Acceptability thresholds:
{json.dumps(limits, indent=2)}

Computed threshold checks:
{json.dumps(threshold_checks, indent=2)}

Decision path:
{rule_text}
""".strip()


def call_llm(prompt: str, model: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set in the project .env or process environment."
        )

    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Explain healthcare operations simulation outputs accurately "
                    "and conservatively. Use ASCII punctuation only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=1200,
    )

    text = (completion.choices[0].message.content or "").strip()

    if not text:
        raise RuntimeError("Groq returned an empty response.")

    return normalize_ascii_punctuation(text)


def safe_print(value=""):
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(
        text.encode(
            encoding,
            errors="replace",
        ).decode(
            encoding,
            errors="replace",
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ga-result",
        type=Path,
        default=Path("results/genetic_algorithm/best_policy.json"),
    )
    parser.add_argument(
        "--xai-result",
        type=Path,
        default=Path("results/decision_tree_xai/best_policy_explanation.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/llm/llm_explanation.txt"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("results/llm/llm_explanation_metadata.json"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)

    args = parser.parse_args()

    ga_result = load_json(args.ga_result)
    xai_result = load_json(args.xai_result)

    safe_print(f"Calling Groq model: {args.model}")
    explanation = call_llm(
        build_prompt(ga_result, xai_result),
        args.model,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(
        explanation,
        encoding="utf-8",
    )

    args.metadata_output.write_text(
        json.dumps(
            {
                "provider": "Groq",
                "model": args.model,
                "dynamic_api_call": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    safe_print("\nLLM EXPLANATION COMPLETE")
    safe_print(explanation)
    safe_print(f"\nExplanation saved to: {args.output}")


V2_PROMPT_VERSION = "3.2-grounded-prose"


def explanation_provenance_v2(result, xai):
    from genetic_algorithm_v2 import canonical_hash
    return {"version": V2_PROMPT_VERSION, "provider": "Groq", "model": DEFAULT_MODEL,
            "ga_run_id": result["run_id"], "ga_provenance": result["provenance"],
            "xai_sha256": canonical_hash(xai)}


def explanation_is_current_v2(explanation, result, xai):
    return bool(result and explanation and explanation.get("provenance") == explanation_provenance_v2(result, xai))


def build_payload_v2(result, xai):
    current = result["provenance"]["current_policy"]
    recommended = result["best_policy"]
    return {"current_policy": current,
        "resource_availability": {key: result["provenance"].get(key) for key in
            ("automatic_bounds", "effective_bounds", "current_policy_feasible", "initialization_strategy")},
        "demand_and_targets": result["provenance"]["scenario_configuration"],
        "current_policy_performance": result["current_evaluation"],
        "recommended_policy": recommended,
        "resource_differences": {key: recommended[key] - current[key] for key in
            ("icu_beds", "general_beds", "doctors", "nurses")},
        "ga_search_fitness": result["best_fitness"],
        "final_verified_performance": result["verified_evaluation"],
        "why_this_changed": result["change_summary"], "decision_tree_xai": xai,
        "limitations": "Simulation estimates, not clinical evidence. Sampled profiles, limited stochastic runs, fixed modeled resources. Staff are concurrent treatment resources, not employee headcounts. XAI is a learned surrogate; training accuracy is not generalization evidence. Human review is required."}


def _compact_evaluation(evaluation):
    return {"overall_robustness": evaluation["overall_robustness"],
            "overall_policy_status": evaluation["overall_policy_status"],
            "scenarios": {name: {key: value for key, value in scenario.items() if key != "runs"}
                          for name, scenario in evaluation["scenarios"].items()}}


def deterministic_explanation_v2(result, xai):
    from genetic_algorithm_v2 import GENES
    current, final = result["current_evaluation"], result["verified_evaluation"]
    before, after = result["provenance"]["current_policy"], result["best_policy"]
    failing = [name.title() for name, s in final["scenarios"].items() if s["scenario_verdict"] == "UNACCEPTABLE"]
    current_failing = [name.title() for name, s in current["scenarios"].items() if s["scenario_verdict"] == "UNACCEPTABLE"]
    changes = "; ".join(f"{key.replace('_', ' ')} {before[key]} -> {after[key]}" for key in GENES)
    summary = [f"Current status: {current['overall_policy_status']}.",
        "Main issue: " + (", ".join(current_failing) + " demand was below the robustness threshold." if current_failing else "All current scenarios met the robustness threshold."),
        f"GA change: {changes}.",
        f"Result: overall robustness {current['overall_robustness']:.1f}% -> {final['overall_robustness']:.1f}%; verified policy {final['overall_policy_status']}.",
        "Remaining concern: " + (", ".join(failing) + " remains below the robustness threshold." if failing else "No tested scenario failed; untested demand remains uncertain.")]
    if result["provenance"].get("current_policy_feasible") is False:
        summary.insert(1, "Availability: the current/reference policy is outside the newly stated constraints. GA searched only feasible configurations; historical current-policy results remain unchanged.")
    tradeoff = ("The recommendation increases modeled resource capacity to address demand. This carries a resource-cost tradeoff; the simulation does not establish real-world staffing or budget feasibility."
                if any(after[k] > before[k] for k in GENES) else
                "Resource savings must be balanced against operational performance and real-world feasibility.")
    rules = []
    for name, explanation in xai.get("explanations", {}).items():
        rules.append(f"{name.title()} learned pattern: " + " AND ".join(explanation.get("learned_rules", []))
                     + f" -> surrogate {explanation.get('surrogate_class', 'unavailable')}.")
    scenario_facts = []
    for name, scenario in final["scenarios"].items():
        old = current["scenarios"][name]
        metrics = scenario["metrics"]
        scenario_facts.append(f"{name.title()}: robustness {old['scenario_robustness']:.1f}% -> {scenario['scenario_robustness']:.1f}%, {scenario['scenario_verdict']}; "
            f"mean wait {old['metrics']['mean_wait']['mean']:.2f} -> {metrics['mean_wait']['mean']:.2f} min; "
            f"high-risk wait {old['metrics']['high_risk_mean_wait']['mean']:.2f} -> {metrics['high_risk_mean_wait']['mean']:.2f} min; "
            f"verified P95 {metrics['p95_wait']['mean']:.2f} min.")
    detail = "\n\n".join(["The simulation suggests the following resource configuration: " + changes + ".",
        f"Authoritative final policy status: {final['overall_policy_status']}. " + summary[-1],
        *scenario_facts, tradeoff, *rules,
        "Decision Tree rules are learned explanatory patterns, not the authoritative acceptance criteria. "
        "Limited replications and sampled clinical profiles restrict generalization. Concurrent staff capacities are not total employee counts. "
        "The recommendation should be reviewed before operational use."])
    return summary, detail, tradeoff


def generate_explanation_v2(result, xai):
    """Grounded facts remain deterministic; optional Groq prose cannot set a verdict."""
    from feedback_interpreter import _object, structured_groq
    summary, fallback, tradeoff = deterministic_explanation_v2(result, xai)
    final = result["verified_evaluation"]
    schema = _object({"authoritative_status": {"type": "string", "enum": [final["overall_policy_status"]]},
                      "explanation": {"type": "string"}})
    payload = build_payload_v2(result, xai)
    for key in ("current_policy_performance", "final_verified_performance"):
        payload[key] = _compact_evaluation(payload[key])
    prompt = ("Write a concise qualitative clinician/administrator interpretation (up to 150 words) of verified V2 simulation results. "
        "The application ALREADY supplies the exact resource changes, numeric metrics, status, failed scenarios and learned rules. "
        "Your explanation accompanies those facts: do NOT repeat ANY numbers, percentages, thresholds, counts or numeric rules. "
        "Explain problematic demand, ONLY the actually observed combined-policy changes (or explicitly say unchanged), "
        "remaining concerns, XAI limitations and human review. "
        "Never override the authoritative status or invent causes or clinical conclusions. Improvements reflect the combined policy, "
        "not isolated causal effects. Explicitly discuss resource-cost tradeoffs for larger configurations. "
        "Resource cost is a soft fitness penalty, NOT a budget/cost constraint. Do not claim resources were limited by a budget. "
        "Do not infer failure from aggregate utilization: the verdict uses per-run checks and robustness. "
        "If resources and robustness are unchanged, do not claim improvement, reduced waits or benefits from hypothetical additions. "
        "Compare each scenario separately: if Best wait was zero before and after, it did NOT improve. "
        "A scenario far below the robustness threshold is NOT marginal, borderline or nearly acceptable. "
        "XAI training samples are policy/scenario observations, not unique policies. "
        "Distinguish Digital Twin deterministic verdict, GA optimization, XAI learned surrogate and your non-authoritative prose. "
        "Use 'The simulation suggests' and recommend human review. No raw JSON in explanation. "
        "Use only supplied facts, and quote numeric results exactly.\n" + json.dumps(payload))
    available, message = False, "AI explanation unavailable. Simulation and optimization results remain valid. A deterministic explanation is shown."
    detail = fallback
    try:
        response = structured_groq(prompt, schema, "v2_policy_explanation")
        if not response["explanation"].strip():
            raise ValueError("Empty explanation")
        import re
        if re.search(r"\d", response["explanation"]):
            raise ValueError("Numeric claims must come from deterministic facts.")
        sentences = re.split(r"[.!?]\s*", response["explanation"].lower())
        for name, scenario in final["scenarios"].items():
            old = result["current_evaluation"]["scenarios"][name]
            for sentence in sentences:
                if name not in sentence:
                    continue
                if (old["metrics"]["mean_wait"]["mean"] <= scenario["metrics"]["mean_wait"]["mean"]
                    and re.search(r"improv\w*|reduc\w*|lower\w*", sentence)
                    and re.search(r"wait\w*", sentence)):
                    raise ValueError("Prose claims an unobserved scenario wait improvement.")
                if (scenario["scenario_robustness"] < payload["demand_and_targets"]["robustness_threshold"] - 20
                    and re.search(r"marginal\w*|borderline|nearly acceptable", sentence)):
                    raise ValueError("Prose understates a scenario robustness failure.")
        detail = fallback + "\n\nAI interpretation:\n" + response["explanation"]
        available, message = True, "AI explanation from Groq; authoritative results remain the Digital Twin outputs."
    except Exception:
        pass  # Never expose SDK errors or credentials; deterministic results remain usable.
    return {"schema_version": 2, "provenance": explanation_provenance_v2(result, xai),
            "available": available, "message": message, "summary": summary, "detail": detail,
            "authoritative_status": final["overall_policy_status"]}


if __name__ == "__main__":
    main()

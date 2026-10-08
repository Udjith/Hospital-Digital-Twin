
from __future__ import annotations

import argparse
import json
import os
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

RESOURCE_KEYS = ["icu_beds", "general_beds", "doctors", "nurses"]
OBJECTIVE_KEYS = [
    "acceptable_mean_wait_min",
    "acceptable_high_risk_wait_min",
    "mean_wait_weight",
    "high_risk_wait_weight",
    "utilization_weight_scale",
    "resource_cost_weight_scale",
    "throughput_reward_weight",
]

INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "apply_feedback": {"type": "boolean"},
        "summary": {"type": "string"},
        "constraints": {
            "type": "object",
            "properties": {
                resource: {
                    "type": "object",
                    "properties": {
                        "min": {"type": ["integer", "null"]},
                        "max": {"type": ["integer", "null"]},
                    },
                    "required": ["min", "max"],
                    "additionalProperties": False,
                }
                for resource in RESOURCE_KEYS
            },
            "required": RESOURCE_KEYS,
            "additionalProperties": False,
        },
        "objectives": {
            "type": "object",
            "properties": {
                key: {"type": ["number", "null"]}
                for key in OBJECTIVE_KEYS
            },
            "required": OBJECTIVE_KEYS,
            "additionalProperties": False,
        },
    },
    "required": [
        "apply_feedback",
        "summary",
        "constraints",
        "objectives",
    ],
    "additionalProperties": False,
}


def load_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompt(feedback: dict, current_result: dict) -> str:
    comment = str(feedback.get("comments", "") or "").strip()
    explicit_constraints = feedback.get("constraints", {})
    targets = feedback.get("targets", {})
    current_policy = current_result.get("best_policy", {})
    current_metrics = current_result.get("best_metrics", {})

    return f"""
Interpret hospital-operations feedback for a Genetic Algorithm.

You are ONLY a translator from natural language into supported numeric
constraints and objective settings. You do NOT choose the final policy.

SUPPORTED RESOURCE CONSTRAINTS:
- icu_beds: integer min/max
- general_beds: integer min/max
- doctors: integer min/max
- nurses: integer min/max

SUPPORTED OBJECTIVES:
- acceptable_mean_wait_min
- acceptable_high_risk_wait_min
- mean_wait_weight
- high_risk_wait_weight
- utilization_weight_scale
- resource_cost_weight_scale
- throughput_reward_weight

RULES:
1. Use null when no change was requested.
2. Never invent a numeric resource limit unless the user gave a number or
   an explicit dashboard control provides one.
3. "No more than X", "only X available", "maximum X" => max = X.
4. "At least X", "minimum X" => min = X.
5. "Between X and Y" => min = X and max = Y.
6. "Prioritize high-risk patients" with no number => high_risk_wait_weight = 3.0.
7. "Reduce unnecessary resources/cost" => resource_cost_weight_scale = 1.35.
8. "Prioritize utilization" => utilization_weight_scale = 1.25.
9. "Prioritize throughput" => throughput_reward_weight = 0.20.
10. Do not change the ICU-risk classification threshold.
11. Do not output unsupported fields.
12. Keep summary short and factual.

USER FEEDBACK:
{comment if comment else "(none)"}

EXPLICIT DASHBOARD CONSTRAINTS:
{json.dumps(explicit_constraints, indent=2)}

EXPLICIT DASHBOARD TARGETS:
{json.dumps(targets, indent=2)}

CURRENT POLICY:
{json.dumps(current_policy, indent=2)}

CURRENT METRICS:
{json.dumps(current_metrics, indent=2)}
""".strip()


def call_groq(prompt: str, model: str) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set in the project .env or process environment."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return only data conforming to the supplied JSON schema. "
                    "Translate hospital operations feedback into GA parameters."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        reasoning_effort="low",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ga_feedback_interpretation",
                "strict": True,
                "schema": INTERPRETATION_SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned an empty structured response.")

    return json.loads(content)


def validate_interpretation(data: dict) -> dict:
    summary = str(data.get("summary", "")).translate(ASCII_PUNCTUATION)
    data["summary"] = summary.encode(
        "ascii", errors="replace"
    ).decode("ascii")
    for resource in RESOURCE_KEYS:
        item = data["constraints"][resource]
        low = item["min"]
        high = item["max"]

        if low is not None and low < 1:
            raise ValueError(f"{resource}.min must be >= 1")
        if high is not None and high < 1:
            raise ValueError(f"{resource}.max must be >= 1")
        if low is not None and high is not None and low > high:
            raise ValueError(
                f"Invalid {resource} range: min {low} > max {high}"
            )

    obj = data["objectives"]

    for key in [
        "acceptable_mean_wait_min",
        "acceptable_high_risk_wait_min",
    ]:
        if obj[key] is not None and obj[key] <= 0:
            raise ValueError(f"{key} must be > 0")

    for key in [
        "mean_wait_weight",
        "high_risk_wait_weight",
        "utilization_weight_scale",
        "resource_cost_weight_scale",
        "throughput_reward_weight",
    ]:
        if obj[key] is not None and obj[key] < 0:
            raise ValueError(f"{key} must be >= 0")

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Interpret clinician feedback into strict GA parameters."
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        default=Path("results/feedback/latest_feedback.json"),
    )
    parser.add_argument(
        "--current-result",
        type=Path,
        default=Path("results/genetic_algorithm/best_policy.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/feedback/interpreted_feedback.json"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    feedback = load_json(args.feedback)
    current_result = load_json(args.current_result, {})

    print(f"Interpreting feedback with Groq model: {args.model}")

    interpreted = call_groq(
        build_prompt(feedback, current_result),
        args.model,
    )
    interpreted = validate_interpretation(interpreted)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(interpreted, indent=2),
        encoding="utf-8",
    )

    print("\nFEEDBACK INTERPRETATION COMPLETE")
    print(json.dumps(interpreted, indent=2))
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()

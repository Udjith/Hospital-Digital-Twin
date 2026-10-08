
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


if __name__ == "__main__":
    main()

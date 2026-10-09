# Intelligent Healthcare Digital Twin

An intelligent hospital operations Digital Twin that combines Machine Learning, simulation, optimization, explainable AI, and Large Language Models to support hospital resource-allocation decisions.

V3 Part 1 adds a separate **Live Twin** tab: a persistent simulated hospital state with fixed identifiable resources, progressive stochastic arrivals, visible allocation/release, queues, bounded events and live metrics. Start/pause/resume/stop/reset, manual stepping and optional 1x–60x playback are available. Live physical capacities lock until reset and are independent of V2 GA recommendations. This is a simulated feed, not a production EHR integration. See [V3 architecture, lifecycle and validation](docs/V3_PART1.md). All V2 tabs and workflows remain available.

V3 Part 2 adds manual/editable stress events, optional simulated-time random events (off by default), safe temporary outages and arrival spikes, and independent current-state look-ahead replications. Prepare an event, preview whether the current policy can handle it, then explicitly apply it. Physical totals remain fixed; stale predictions cannot be applied. See [event model, predictive rules and validation](docs/V3_PART2.md). This is a simulated operational stress-event feed, not a production emergency alert system.

V3 Part 3 adds **adaptive operating-policy GA** in Live Twin: risk/aging/surge queue weights and ICU/staff reserves are optimized from the current checkpoint using common future seeds, then fully verified. Physical bed/staff totals never change. Risky event previews offer optimization; policy and event application both require explicit user actions. Temporary policies restore after event/queue recovery, while persistent policies remain until manually changed. Learned Decision Tree rules and optional Groq explanations remain explanatory only. See [exact chromosome, reserves, fitness, seeds and validation](docs/V3_PART3.md).

Live provenance uses canonical JSON/SHA-256 hashing that survives Streamlit module reloads. Adaptive results explain verified blocking conditions, distinguish operational improvement from feasibility recovery, and show learned XAI patterns separately from deterministic causes. Groq receives validated structured evidence; a deterministic explanation remains available without it. Possible operational responses require human review. See [diagnosis rules, explanation schema and validation](docs/V3_DIAGNOSIS.md).

Live event acceptance now uses time-weighted utilization, continuous overload/full-saturation grace periods, waits and end-queue/event-wait recovery. Brief peaks remain warnings. GA returns the best fully verified tested policy, including partial improvements, for **Apply / Reject / Modify / Retry** human review. Applied and automatic synthetic events use the same prediction workflow. See [exact sustained-pressure criteria, fitness and validation](docs/V3_LIVE_ACCEPTANCE.md). V2 scenario criteria remain unchanged.

**Describe a Hospital Scenario** translates clinician/administrator text through the existing Groq model into strictly validated supported event proposals, including compound events and offsets. Interpretation never changes live state: users confirm look-ahead and explicitly apply events/policies. Missing Groq falls back to the collapsed manual builder. The main result view is concise; technical evidence remains in expanders. See [scenario schema, safeguards and final A–G lock validation](docs/V3_SCENARIOS_LOCK.md).

The Live Twin demonstration baseline is **220 ICU / 550 General / 80 doctors / 140 nurses, 4 patients/hour**, with normal policy weights 1.0 for high-risk and 0.1 for wait aging. Normal Demo Mode enables Low random-event frequency; random events default off. This is a project demonstration baseline, not clinically validated. Playback now supports 1x–3600x with bounded catch-up; manual skips support up to 30 simulated days with pause/cancel and bounded progress. The preserved V2 resource-optimization defaults below remain separate.

The preserved scenario/optimization workflow uses V2 Parts 1-3. Run `python main.py`, enter current resources, demand and targets, then **Evaluate Current Policy** or **Run Optimization**. The Genetic Algorithm performs simulation-based hospital process optimization; it never optimizes the Random Forest (threshold stays 0.50).

GA bounds refresh from current resources: `min = max(1, floor(current * 0.70))`, `max = ceil(current * 1.50)`. They are hidden in collapsed informational expanders, with no eight manual bound controls. These are default search ranges, not real-world limits. Explicit numeric availability constraints may override them. Generation 0 includes the current policy when feasible; otherwise it starts from its clearly labeled nearest feasible version.

Digital Twin operational criteria determine run verdicts. Scenario robustness is the percentage of replications passing all wait/utilization checks; overall robustness pools all three scenarios. The Decision Tree learns explanatory patterns from evaluated policies. Groq (`openai/gpt-oss-120b`) supplies optional prose and strict-schema feedback; neither AI component overrides the deterministic verdict.

Use **Preview AI Interpretation** to inspect supported resource caps/minima, priorities and explicitly requested target adjustments. Turn **AI Interpretation** off to ignore natural-language instructions without calling Groq. Missing keys or failed/invalid responses leave simulation, GA and XAI usable with a deterministic explanation. Contradictory, zero or negative resource constraints block optimization. Explicit availability may exclude the old current policy: the preview explains this, historical results remain unchanged, and GA searches feasible candidates only. Preferences such as "try to use fewer nurses" adjust resource-efficiency priority without forcing a hard bound.

Review a verified recommendation using **Accept**, **Needs Modification** or **Reject**, a rating and comments. **Save Review** creates a separate V2 record with the GA run ID and provenance. **Prepare Modification** transfers comments into the next instruction, enables interpretation and leaves historical results unchanged. Preview and rerun GA to evaluate the changes.

See the [availability-constraint correction](docs/V2_AVAILABILITY_CONSTRAINTS.md), [Part 1](docs/V2_PART1.md), [Part 2](docs/V2_PART2.md), and [Part 3 implementation/schema/validation](docs/V2_PART3.md). Demonstration resources remain 210 ICU beds, 280 general beds, 85 concurrent doctors and 100 concurrent nurses ([calibration](docs/V2_RESOURCE_CALIBRATION.md)). Keep base seed 42 for reproducibility; replications automatically derive distinct seeds. Legacy V1 CLI functions and their artifacts remain separate.

## Project Overview

The system creates a virtual model of hospital operations and uses it to test and optimize resource configurations such as:

- ICU beds
- General beds
- Doctors
- Nurses

The project combines patient data, machine learning predictions, a SimPy-based Digital Twin, a Genetic Algorithm, Decision Tree explainability, and LLM-based feedback interpretation.

## System Architecture

Data Input  
→ Data Preprocessing  
→ Synthetic Data Generation  
→ Augmented Dataset  
→ Random Forest Prediction  
→ Genetic Algorithm Optimization  
↔ Healthcare Digital Twin  
→ Optimized Policy  
→ Decision Tree Explainability  
→ LLM Explanation  
→ Dashboard / Human-in-the-Loop Feedback

Clinician or administrator feedback can be converted into structured constraints and sent back to the Genetic Algorithm for another optimization run.

## Main Modules

### 1. Data Preprocessing

Processes the eICU dataset and creates a cleaned hospital dataset containing patient and clinical information.

Output:

`data/processed/eicu_master.csv`

### 2. Synthetic Data Generation

Generates synthetic hospital encounters using patterns and distributions derived from the processed eICU data.

The current implementation generates approximately 7,500 synthetic encounters.

Output:

`data/synthetic/synthetic_hospital.csv`

### 3. Random Forest Prediction

A Random Forest model predicts whether a patient is likely to require higher hospital resources.

Important features include:

- APACHE score
- Acute physiology score
- Predicted ICU mortality
- SpO2
- Heart rate
- Respiratory rate
- Creatinine
- Age

Trained model:

`models/random_forest_pipeline.joblib`

### 4. Healthcare Digital Twin

The hospital Digital Twin is implemented using SimPy.

It simulates:

- Patient arrivals
- ICU beds
- General beds
- Doctors
- Nurses
- Treatment duration
- Waiting times
- Resource utilization
- Patient throughput

The Random Forest prediction is used to support patient resource assignment.

V2 observes only the selected 1-72 hour window (default 24). Arrivals are Poisson/exponential at Best/Average/Worst rates (12/22/35 patients/hour by default), with ten replications per scenario. Current resources are entered manually. Every patient holds exactly one bed type and requires both a doctor and a nurse for treatment. Utilization and censored waiting use only the observation window.

The original fixed-cohort/workload-derived baseline is available only through `python src/digital_twin.py --legacy-v1`; it is not the active V2 dashboard configuration.

### 5. Genetic Algorithm Optimization

A Genetic Algorithm searches for improved hospital resource configurations.

The Genetic Algorithm optimizes:

- ICU beds
- General beds
- Doctors
- Nurses

The ICU risk threshold remains fixed at:

`0.50`

The fitness function considers:

- Mean patient waiting time
- High-risk patient waiting time
- Resource utilization
- Resource cost
- Robustness and target violations (dominant); throughput is reported

### 6. Decision Tree Explainability

A shallow Decision Tree learns explanatory patterns from actually evaluated policy/scenario observations. It is a surrogate, not the authoritative verdict.

Its labels are the deterministic scenario verdicts: the percentage of runs passing both waiting-time targets and all four utilization targets must meet the robustness threshold. Single-class observations produce a clear unavailable message instead of a tree.

It generates simple rules such as:

`ICU beds > threshold AND doctors > threshold → acceptable policy`

### 7. LLM Explanation

The project uses Groq with:

`openai/gpt-oss-120b`

The LLM converts optimization results and Decision Tree rules into a more understandable explanation for hospital administrators.

The LLM does not perform the optimization itself.

### 8. Human-in-the-Loop Feedback

The dashboard allows users to modify optimization constraints using:

- Sliders
- Direct numeric input
- Operational targets
- Checkboxes
- Natural-language instructions

Example:

`Do not use more than 110 doctors and prioritize high-risk patients.`

The feedback interpreter converts this into structured Genetic Algorithm constraints and objectives.

The Genetic Algorithm then reruns using the updated constraints.

## Dashboard Features

The Streamlit dashboard includes:

- Light and dark monochrome themes
- Baseline hospital statistics
- Resource-allocation controls
- Sliders and direct numeric input
- Scenario presets
- Optimization progress
- Baseline vs recommended comparison
- Genetic Algorithm results
- Digital Twin performance metrics
- Decision Tree explainability
- LLM-generated explanations
- Human feedback and review
- Accept / reject / modify workflow
- Developer execution logs

## Project Structure

```text
healthcare_digital_twin/
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── models/
│   ├── random_forest_pipeline.joblib
│   └── decision_tree_xai.joblib
│
├── results/
│   ├── random_forest/
│   ├── digital_twin/
│   ├── genetic_algorithm/
│   ├── decision_tree_xai/
│   ├── llm/
│   └── feedback/
│
├── src/
│   ├── preprocessing.py
│   ├── synthetic_data.py
│   ├── random_forest.py
│   ├── digital_twin.py
│   ├── genetic_algorithm.py
│   ├── decision_tree_xai.py
│   ├── feedback_interpreter.py
│   ├── llm_explanation.py
│   └── dashboard.py
│
├── main.py
├── requirements.txt
└── README.md


Installation
Install the required packages:
pip install -r requirements.txt

Groq API Key
The LLM explanation and natural-language feedback interpreter require a Groq API key.
Create a `.env` file in the project root containing:

```text
GROQ_API_KEY=your_api_key_here
```

`python-dotenv` loads this file automatically when the launcher, dashboard,
feedback interpreter, or explanation module starts. A process-level
`GROQ_API_KEY` still takes precedence when one is explicitly supplied.

Do not store the API key directly inside source code or commit the `.env` file.
The repository `.gitignore` excludes `.env`.
Running the Project
From the project root:
python main.py

Or run the Streamlit dashboard directly:
python -m streamlit run src/dashboard.py

Individual Module Execution
The original GA/XAI/LLM/feedback CLI commands remain V1-compatible; the main dashboard uses the V2 Python entry points. V2 GA/XAI CLIs are `python src/genetic_algorithm_v2.py` and `python src/decision_tree_xai_v2.py`. The modules can also be run independently:
python src/preprocessing.py
python src/synthetic_data.py
python src/random_forest.py
python src/digital_twin.py
python src/genetic_algorithm.py
python src/decision_tree_xai.py
python src/llm_explanation.py

Current Random Forest Performance
The current trained Random Forest has produced approximately:
- Accuracy: 73.77%
- Precision: 68.95%
- Recall: 66.45%
- F1 Score: 67.68%
- ROC-AUC: 80.97%
These values may change if the dataset or model is retrained.
Current Development Notes
The core Digital Twin, Genetic Algorithm, Random Forest, Decision Tree XAI, LLM explanation, and feedback loop are functioning.
The dashboard is still being visually refined.
Current areas being improved include:
- Improving metric-card text sizing
- Fixing text overflow inside UI cards
- Improving light/dark theme contrast
- Improving dashboard layout and visual polish
Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- SimPy
- Streamlit
- Altair
- Joblib
- Groq API
- Random Forest
- Genetic Algorithm
- Decision Tree
Disclaimer
This project is an academic and research prototype.
It is designed for hospital operations and resource-allocation decision support.
It is not intended to provide medical diagnosis, treatment recommendations, or autonomous clinical decisions.
All recommendations are based on simulation, machine learning, and optimization assumptions and should be reviewed by a human before use.
```

# Intelligent Healthcare Digital Twin

An intelligent hospital operations Digital Twin that combines Machine Learning, simulation, optimization, explainable AI, and Large Language Models to support hospital resource-allocation decisions.

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

The reproducible dashboard baseline uses the earliest 500 synthetic arrivals in
a deterministic 3x large-hospital workload scenario. The implementation divides
each arrival offset by 3 while retaining the same patients, arrival order,
Random Forest predictions, treatment durations, and length-of-stay durations.
This produces 3x arrival demand without duplicating records or increasing the
SimPy event count.

Resource capacity is derived from the scaled cohort's offered concurrent load using:

`capacity = ceil(offered concurrent load / 0.75)`

The current scaled offered load and derived policy are:

- 113 ICU beds from 84.742 offered beds
- 130 general beds from 96.904 offered beds
- 13 modeled concurrent doctors from 9.235 concurrent treatment demand
- 13 modeled concurrent nurses from 9.235 concurrent treatment demand
- fixed Random Forest threshold of 0.50

The 75% target is the lower edge of the configured 75-80% utilization range.
It provides an integer-capacity safety margin for long treatment durations while
remaining entirely workload-derived. Doctor and nurse loads are equal because
the current simulation assigns one of each to every patient for the same
treatment duration.

The baseline result records hashes for the dataset, Random Forest model, and
Digital Twin implementation. The dashboard refuses to display it when those
dependencies no longer match. The capacity formula, offered loads, rounding
rule, assumptions, derived policy, and implied target utilization are retained
in the baseline metadata for auditing.

The Genetic Algorithm search envelope is derived independently for every
resource from the current baseline: the minimum is floor(70% of baseline) and
the maximum is ceil(130% of baseline). The dashboard reads these same dynamic
bounds, so a stale small-hospital search space cannot silently constrain the
large-hospital scenario.

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
- Patient throughput

### 6. Decision Tree Explainability

A shallow Decision Tree is used as an interpretable model to explain why an optimized resource policy is considered acceptable or unacceptable.

Its acceptability labels use the mean-wait and high-risk-wait targets saved by
the latest Genetic Algorithm run, including targets changed through feedback.

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

`Do not use more than 10 nurses and prioritize high-risk patients.`

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
The modules can also be run independently:
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

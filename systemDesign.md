# System Design

## Purpose
This document is the primary design authority for the repository. In this project, "system design" refers to the structure of an academic, notebook-led machine learning workflow rather than a production software architecture.

The repository exists to support a group coursework exercise in supervised classification for lung cancer prediction. Its purpose is to keep the notebook, datasets, and supporting docs organised around a clear analytical process: load data, justify preprocessing, explore patterns, train and evaluate models, interpret important features, and present conclusions with appropriate caution.

## Scope
The repository is in scope for:

- a notebook-centred end-to-end classification workflow
- dataset inspection, cleaning, encoding, and feature engineering
- exploratory analysis and visualisation
- model selection, training, and evaluation using classification metrics
- interpretation of predictive features and critical discussion of limitations
- supporting documentation that keeps the work aligned with the coursework brief and marking expectations

## Non-goals
The repository is not intended to provide:

- a clinical decision-support system
- a deployable model, web application, or inference API
- CI/CD, infrastructure automation, or MLOps workflows
- automated retraining or production data pipelines
- regulatory, medical, or causal claims beyond what the coursework analysis can support
- open-source governance material such as contributor policy or licensing boilerplate unless explicitly added later

## Project architecture and workflow
The project is organised around one primary analytical artefact and a small number of supporting components.

`datasets/*.csv` -> `blankTemplate.ipynb` -> figures, tables, model comparisons, feature interpretation, coursework conclusions

The notebook should remain the single place where the analytical story is visible end to end. Supporting docs explain how the repository is intended to work and where it currently diverges from that intended state.

## Major repository components and responsibilities

### `blankTemplate.ipynb`
This is the canonical coursework artefact. It should contain:

- the problem framing and project objective
- dataset overview and quality review
- preprocessing decisions and feature engineering
- exploratory data analysis and visual interpretation
- model-selection logic and training workflow
- evaluation results and error analysis
- feature-importance discussion
- final conclusion and reflection

### `datasets/`
This directory holds the raw tabular inputs for the project. The notebook currently reads `datasets/givenData.csv`, while `datasets/kaggleData.csv` is also present in the repo. Raw inputs should be preserved as source material rather than edited in place.

### `scripts/`
This directory is for minor helpers only. The current scripts support notebook presentation metadata and extraction of notebook code into a plain Python export for regression testing. They are not part of the analytical workflow and should not become a hidden modelling path.

### `tests/`
This directory holds regression tests for the notebook-led workflow. Tests should validate helper-generated exports of notebook code and check that key preprocessing and analysis behaviours remain stable without moving the canonical analysis out of the notebook. After changing notebook logic, helper scripts, or dependency definitions, contributors should rerun `python -m pytest` and treat the result as part of the normal verification step.

### `docs/`
This directory holds supporting documentation that clarifies project intent, repository boundaries, dataset handling, and active mismatches between intended design and current state.

## Notebook-centric workflow

### 1. Problem framing
Define the task as supervised classification on the `LUNG_CANCER` target and state what success means for the coursework. The framing should explain why classification metrics matter and why a simple accuracy-only reading would be insufficient.

### 2. Data loading and schema inspection
Load the chosen dataset, inspect the columns, identify the target, and explain the variable types at a practical level. At this stage the notebook should also surface issues such as column naming inconsistencies, binary encodings, class balance, and duplicated records.

### 3. Preprocessing and data preparation
Carry out only the transformations that can be justified. This may include duplicate handling, column cleanup, binary recoding, train/test-safe preprocessing, and modest feature engineering. Decisions should be documented in markdown and implemented reproducibly in code.

### 4. Analysis and visualisation
Use exploratory analysis to understand distributions, subgroup patterns, relationships to the target, and modelling risks such as imbalance or multicollinearity. Visuals should support later modelling decisions rather than exist as disconnected charts.

### 5. Modelling and evaluation
Train and compare candidate classifiers using an evaluation strategy appropriate for a classification task. Metrics should reflect the class distribution and the cost of different error types. Where preprocessing is learned from data, it should be contained within a safe modelling pipeline.

### 6. Interpretation and critical discussion
Interpret the final model through feature importance, error analysis, and limitations. The discussion should stay cautious: patterns in this dataset can inform coursework conclusions, but they do not justify claims of clinical validity or medical deployment.

### 7. Coursework-facing conclusion
Close the notebook by answering the brief directly, summarising the strongest findings, and reflecting on limitations and sensible future improvements.

## Design invariants
The following conditions should remain true as the repository evolves:

- The notebook remains aligned with the coursework marking rubric and preserves its section-based narrative.
- The repository is documentation-first and notebook-led, not a disguised production application.
- Preprocessing choices are justified in writing and implemented reproducibly.
- Raw datasets remain intact; cleaning and encoding happen in notebook or reproducible analytical code, not by overwriting source files.
- Evaluation remains appropriate for supervised classification and does not rely on a single headline metric.
- Leakage is actively prevented in any modelling workflow that learns transformations from data.
- Interpretation includes clinically relevant caution and avoids causal or deployment-level claims.
- Supporting docs describe the repository as it actually exists and are updated when reality changes.
- If multiple dataset files remain in the repo, their relationship is documented clearly so collaborators do not treat them as separate evidence sources by mistake.
- Relevant changes are followed by a regression-test run so notebook-supporting code and documented behaviour do not drift apart silently.

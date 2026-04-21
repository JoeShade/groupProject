<p align="center">
  <a href="https://www.lboro.ac.uk/">
    <img src="assets/LboroLogo.png" alt="Loughborough University logo">
  </a>
</p>

<h1 align="center">Machine Learning - Principles and Applications for Engineers Coursework</h1>

## Team Members

<div align="center">
  <a href="https://github.com/jamesbarlow1812"><img alt="James Barlow" src="https://img.shields.io/badge/%20-James%20Barlow-4A4A4A?style=flat-square&logo=github&logoColor=white&labelColor=7C3AED" /></a>
  <a href="https://github.com/JoeShade"><img alt="Joe Shade" src="https://img.shields.io/badge/%20-Joe%20Shade-4A4A4A?style=flat-square&logo=github&logoColor=white&labelColor=D97706" /></a>
  <a href="https://github.com/Nomo2001"><img alt="Lena Kraemer" src="https://img.shields.io/badge/%20-Lena%20Kraemer-4A4A4A?style=flat-square&logo=github&logoColor=white&labelColor=059669" /></a>
  <a href="https://github.com/SimonAndreou"><img alt="Simon Andreou" src="https://img.shields.io/badge/%20-Simon%20Andreou-4A4A4A?style=flat-square&logo=github&logoColor=white&labelColor=DC2626" /></a>
</div>

## Introduction
This repository supports a group machine learning coursework project built around a Jupyter notebook-led analysis of lung cancer prediction. The work is organised as an end-to-end supervised classification exercise: inspect the data, justify preprocessing choices, explore patterns, train and compare classifiers, and interpret the features that appear most predictive.

The repository is for group members, tutors, and markers reviewing the academic workflow. It is not intended as a reusable software package or deployable product.

## Disclaimer
This repo is for a four-day machine learning project for the 25WSP074 - Machine Learning - Principles and Applications for Engineers module at [Loughborough University](https://www.lboro.ac.uk/).

As such it will not be actively maintained or supported.

## Project objective
The objective is to develop and evaluate classification models that predict the `LUNG_CANCER` outcome from patient demographic, lifestyle, and symptom variables. The coursework emphasis is on:

- framing the classification problem clearly
- preparing the data in a justified and reproducible way
- comparing models with appropriate classification metrics
- identifying and discussing the most influential predictive features
- reflecting critically on limitations, bias, and uncertainty

## Setup
Use Python `3.10` for this repository. A clean virtual environment is recommended before installing dependencies.

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install the repository from the manifest.

```powershell
python -m pip install -e .
```

3. If you also want the regression-test tooling, install the `dev` extras.

```powershell
python -m pip install -e ".[dev]"
```

After installation, the repository can be used in three main ways:

1. Open the coursework notebook for the main analysis workflow.

```powershell
jupyter notebook blankTemplate.ipynb
```

2. Export notebook code cells into a plain Python file for regression testing support.

```powershell
python scripts\extract_notebook_code.py
```

This writes `tests/notebook_code.py` by default.

3. Run the regression tests that validate the helper-generated notebook export and the current notebook behaviour.

```powershell
python -m pytest
```

After changing notebook logic, helper scripts, test expectations, or dependency definitions, rerun `python -m pytest` before treating the work as complete.

## Repository structure
- `blankTemplate.ipynb`: main notebook template and primary coursework artefact. It contains the planned narrative from problem definition through conclusion.
- `datasets/givenData.csv`: tabular input data currently used by the notebook.
- `datasets/kaggleData.csv`: second copy of the same tabular dataset currently present in the repository.
- `pyproject.toml`: dependency manifest for the notebook workflow, helper scripts, and regression tests.
- `scripts/update_notebook_badges.py`: small helper script for refreshing the team badge cell in the notebook.
- `scripts/extract_notebook_code.py`: helper script for exporting notebook code cells into a plain Python file that regression tests can inspect.
- `tests/`: regression tests that exercise the helper-generated notebook code export and pin the current analysis behaviour.
- `docs/`: supporting documentation for workflow, design notes, dataset handling, and live deviations.

## Dataset overview
The repository currently includes two CSV files in `datasets/`. Both expose the same 16-column lung cancer classification schema and, in the current repository state, match at the file-content level.

- Target column: `LUNG_CANCER`
- Predictor types: one demographic field (`AGE`), one sex/gender field (`GENDER`), and a set of binary symptom or lifestyle indicators such as `SMOKING`, `WHEEZING`, `COUGHING`, `SHORTNESS OF BREATH`, and `CHEST PAIN`
- Current raw size: 309 rows x 16 columns
- Current raw class balance: 270 `YES` and 39 `NO`

The notebook already treats duplicate handling, column cleanup, and binary recoding as important preprocessing steps. Those decisions should remain explicit and justified in the final analysis.

## Notebook workflow
`blankTemplate.ipynb` is the centre of the repository. The expected workflow is:

1. define the supervised classification task and the coursework success criteria
2. inspect the dataset structure, column meanings, and data quality issues
3. carry out justified preprocessing, including duplicate handling, encoding, and any feature engineering
4. perform exploratory analysis and visualisation to understand class balance, patterns, and potential modelling issues
5. train and compare classification models using appropriate validation and evaluation metrics
6. interpret the final model through feature importance and error analysis
7. conclude against the brief and summarise limitations and next steps

The supporting documentation explains how the repo should be used, but the notebook remains the primary place where evidence, results, and narrative should live.

## Outputs / deliverables
This repository is set up to support the following coursework outputs:

- a completed notebook with narrative, code, figures, and interpretation
- classification performance summaries such as precision, recall, F1-score, confusion matrices, and model comparison tables
- discussion of the features that appear most important to prediction
- presentation-ready findings for a group submission

## Supporting docs
- [AGENTS.md](AGENTS.md)
- [systemDesign.md](systemDesign.md)
- [Architecture notes](docs/architecture-notes.md)
- [Dataset notes](docs/dataset-notes.md)
- [Active deviations](docs/deviations.md)

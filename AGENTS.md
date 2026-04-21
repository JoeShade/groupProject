# Repository Working Guide

## Repository purpose at a glance
This repository supports a group coursework project on supervised classification for lung cancer prediction. The main artefact is `blankTemplate.ipynb`, which is intended to tell the full analytical story from problem definition to conclusion. The surrounding files exist to support that notebook, not to replace it with a software package or engineering framework.

This guide is for both human collaborators and AI assistants working in the repo.

## Working safely in a notebook-based ML repository
- Treat `datasets/` as raw inputs. Do not overwrite or silently clean the CSV files in place.
- Keep the substantive analysis in `blankTemplate.ipynb`. Do not move core modelling logic into hidden scripts unless there is a clear coursework reason to do so.
- Keep the notebook section order aligned with the coursework flow: problem definition, data preparation, analysis, modelling, results, and conclusion.
- If rerunning cells changes counts, plots, metrics, or conclusions, update the surrounding markdown in the same pass.
- After changing notebook logic, helper scripts, test expectations, or dependency definitions, run the regression tests with `python -m pytest` before finalising the change. If tests cannot be run, record the reason clearly.
- Replace placeholder comments, informal notes, and unfinished prompts with submission-ready narrative before finalising work.
- Use `scripts/` only for small utilities, such as notebook presentation helpers. Do not turn it into a parallel analysis pipeline.

## Boundaries between exploration, modelling, and documentation
- Exploratory analysis belongs in the data preparation and EDA sections of the notebook. Use it to surface duplicates, encoding issues, class balance, subgroup patterns, and candidate relationships.
- Modelling belongs in the model selection, training, and results sections. Keep comparisons fair and traceable to the cleaned dataset used at that stage.
- Repository documentation belongs in `README.md`, `systemDesign.md`, and `docs/`. These files explain the repo structure, design intent, and current gaps, but they do not replace evidence that should appear in the notebook.
- If a change affects both the analysis and its explanation, update the notebook and the relevant docs together so they do not drift apart.

## ML-specific hygiene rules
- Avoid data leakage. Any scaler, encoder, selector, or other learned preprocessing step must be fitted only on training data or within a pipeline evaluated safely.
- Justify preprocessing choices. Duplicate handling, column renaming, binary recoding, outlier treatment, and feature engineering all need an explicit rationale.
- Preserve reproducibility where possible. Use explicit random states, stable train/test or cross-validation strategies, and clearly named intermediate datasets.
- Keep model iterations traceable. If the feature set, class-weighting approach, threshold, or hyperparameters change, record what changed and why.
- Use classification metrics appropriate to the task. Accuracy alone is not enough for an imbalanced health-related classification problem; precision, recall, F1-score, confusion matrices, and threshold trade-offs matter.
- Keep interpretation disciplined. Feature importance can support discussion of patterns in this dataset, but it does not establish causality or clinical significance.
- Do not overclaim medical value. This work is academic analysis and must not be framed as a deployable clinical tool.

## Expectations for keeping notebook, results, and docs aligned
- The notebook should remain the canonical record of the workflow and evidence.
- `README.md` should match the current repository contents and project framing.
- `systemDesign.md` should describe how the notebook-led workflow is supposed to operate.
- `docs/architecture-notes.md` should stay short, practical, and close to the coursework brief.
- `docs/deviations.md` should contain only active mismatches between intended design and current state. Remove an entry once it is resolved.
- `docs/dataset-notes.md` should explain how the two CSV files are being interpreted if both remain in the repo.
- `tests/` should stay aligned with the current notebook behaviour and should be rerun after relevant repository changes.

## Documentation update expectations
- Update the docs when the notebook structure changes materially.
- Update the docs when the canonical dataset source changes.
- Update the docs when model-selection strategy, evaluation priorities, or interpretation framing changes.
- Do not add documentation for deployment, APIs, CI/CD, licensing, or contributor governance unless the coursework scope changes and the repository actually gains those elements.

## Self-review checklist before finalising changes
- Does `blankTemplate.ipynb` still tell a coherent end-to-end coursework story?
- Are preprocessing choices explicit, justified, and reproducible?
- Is leakage avoided throughout modelling and evaluation?
- Are model comparisons based on appropriate classification metrics rather than a single headline score?
- Are feature-importance claims framed cautiously and linked to the dataset limits?
- Do the README and supporting docs still match the repository as it actually exists today?
- Have the regression tests been run after the change, and have any failures or skips been addressed or explained?
- Have placeholder prompts, open questions, and informal wording been removed from submission-facing material?

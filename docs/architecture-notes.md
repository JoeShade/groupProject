# Architecture Notes

## Repository boundaries
- `blankTemplate.ipynb` is the main analytical artefact and should contain the end-to-end coursework narrative.
- `datasets/` contains raw tabular inputs and should not be edited in place.
- `scripts/` is for small helper utilities only, not for hiding core preprocessing or modelling logic.
- `tests/` should validate helper-generated exports of notebook code rather than replace the notebook as the canonical analysis record.
- `docs/` supports the notebook by documenting structure, expectations, and live deviations.
- Changes to notebook logic, helper scripts, or dependency definitions should be followed by a `python -m pytest` run before the change is treated as complete.

## Notebook sections and rubric alignment
- Section 1 should define the problem, the classification objective, and why the task matters.
- Section 2 should explain dataset structure, quality checks, duplicate handling, encoding, and leakage-aware preprocessing decisions.
- Section 3 should use EDA and visuals to motivate modelling choices rather than just describe the dataset superficially.
- Section 4 should document candidate models, validation strategy, and iteration rationale.
- Section 5 should present metrics, comparison, feature importance, error analysis, and limitations.
- Section 6 should answer the brief directly and close with a measured reflection.

## Preprocessing, feature engineering, and evaluation
- Keep preprocessing modest, justified, and reproducible.
- Any duplicate-removal rule should be evidence-based and easy to rerun.
- Feature engineering should improve either interpretability, model utility, or both; avoid adding features that cannot be defended in the report.
- Use classification metrics beyond accuracy, especially because the current target distribution is imbalanced.
- Fit learned preprocessing inside the modelling workflow to avoid leakage.

## Experiment tracking expectations
- Record model iterations in notebook section 4.5 or another clearly linked location.
- For each iteration, note the feature set, model choice, tuning change, and impact on metrics.
- Remove stale claims when later runs change the conclusion.

## Presentation alignment
- Prefer plots and tables that can be reused in slides without rewriting the argument.
- Tie EDA findings to final model choice and to the feature-importance discussion.
- Keep healthcare framing cautious: this is coursework analysis, not a clinical recommendation system.

## Avoiding drift from the brief
- Do not add documentation for deployment, APIs, CI/CD, or package governance unless the coursework scope genuinely changes.
- Keep docs and notebook wording academic-professional rather than conversational draft notes.
- If the repo structure or data source changes, update the docs in the same change rather than later.

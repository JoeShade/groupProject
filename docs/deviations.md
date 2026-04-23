# Active Deviations

This file tracks only live mismatches between the intended project design and the current repository state.

## 1. The roles of `givenData.csv` and `kaggleData.csv` are not yet explicit
- Intended state: collaborators should be able to tell which dataset file is canonical for analysis and why any second copy is retained.
- Current state: both CSV files are present, they currently match at the file-content level, and the notebook only reads `datasets/givenData.csv`.
- Reason: source copies have been kept in the repo without an accompanying explanation of provenance or intended usage.
- Resolution path: either document the role of each file clearly or simplify the workflow around one canonical dataset reference.

# Regressor experiments: plan

**Status (2026-10-21): all experiments done.** Results: [README.md](README.md) and `notebooks/07_summary.ipynb`.

Task 2 from the meeting: train regressors on the current data **without** the main/side chain split.
Use tree-based models (random forest, XGBoost, …) and Bayesian models (Bayesian ridge, Gaussian
process), report MAPE, pairwise accuracy and RMSE, and use cross-validation.

Goal: beat the best models from `precursor2cof-capacitance` on the same data.

## Data

- `../cof_matched/data_substrate_cofs.csv`: 114 substrates, target `capacity_max` (2–768, median 234).
- `../cof_matched/rth_0.99/features/`: the feature sets precomputed in `precursor2cof-capacitance`
  (custom descriptors, custom patterns, ECFP counts, EState, functional groups, MACCS, and combinations).
  Rows are in the same order as the data file.
- The 8 lowest targets (2–16) are all pre-joined imine substrates. Small targets dominate MAPE.

## What was done before (in `precursor2cof-capacitance`)

- Models: KNN, XGBoost, random forest, Lasso, Bayesian ridge (BRR), MLP, TabPFN, Chemprop (CheMeleon).
  No Gaussian process.
- Greedy forward selection over feature sets. Best combinations:
  - BRR: descriptors + ECFP + EState + functional groups
  - RF: EState + functional groups
- Evaluation: one 10-fold stratified CV (seed 42, 10 target bins). In each fold, Optuna (QMC, 100 trials)
  minimises RMSE on one 70/30 split of the training part. Metrics are averaged over folds.
- No scores are saved in that repo, so the baseline has to be rerun here.

## Evaluation protocol used here

- **Repeated stratified CV**: 5 repeats × 10 folds (seeds 0–4, 10 target quantile bins). Every
  experiment uses exactly the same folds, so comparisons are paired.
- Metrics computed on the **pooled out-of-fold predictions** of each repeat (114 predictions), then
  mean ± std over the 5 repeats: MAPE, pairwise accuracy (PA), RMSE, plus SMAPE, MAE and R² for reference.
  Added during the work: MALE (mean absolute log error) and MAPE on y ≥ 20, because plain MAPE turned
  out to be dominated by the 10 smallest targets (a constant prediction of 6 has MAPE 0.94).
- Per-fold averaged metrics on the original single 10-fold split are also reported for the baseline,
  so the numbers can be compared with the old pipeline.
- Baseline to beat: training-set median, and the old best BRR and RF configurations.
- Any tuning happens inside the training part of each outer fold (no test data leaks into tuning).
- With 114 samples, differences smaller than about one std across repeats are treated as noise.

## Experiments

| # | Notebook | What | Expected effect |
|---|---|---|---|
| 0 | `00_baseline.ipynb` | Rerun the old best BRR and RF (and XGBoost) with the old protocol, then with the new one | Number to beat |
| 1 | `01_target_transform.ipynb` | Train on log(y) and sqrt(y) instead of raw y | Lower MAPE, maybe better PA |
| 2 | `02_features.ipynb` | Theoretical capacity (redox sites × 26801 / MW), full RDKit descriptors, feature-set comparison | Better accuracy from a physically meaningful feature |
| 3 | `03_bayesian_models.ipynb` | BRR, ARD, Gaussian process with Tanimoto kernel on fingerprints (+ RBF on descriptors), kernel ridge / SVR | GP is on the task list and strong on small molecule data |
| 4 | `04_tree_models.ipynb` | RF, ExtraTrees, XGBoost, LightGBM, CatBoost with small, regularised settings | Tree models from the task list |
| 5 | `05_pretrained.ipynb` (GPU) | Frozen embeddings (CheMeleon, others) + BRR/GP; TabPFN without tuning | Maybe gains from pretraining |
| 6 | `06_ensemble.ipynb` | Average the best 2–3 models | Small, steady gain |
| 7 | `07_summary.ipynb` | All results in one table and plots | Final comparison |

## Code layout

- `cofreg/`: shared code (data loading, folds, metrics copied from `precursor2cof-capacitance`, evaluation loop, models, plots).
- `notebooks/`: one notebook per experiment, with short descriptions and plots.
- `results/`: out-of-fold predictions and metrics for every run (CSV), read by the summary notebook.

## Changes from the plan

- 05 also fine-tunes CheMeleon end to end (as the old pipeline did), not only frozen embeddings.
- 04 compares three ways of setting hyperparameters: fixed, the old protocol, and nested inner-CV tuning
  (plan item "tuning" folded into 04 instead of a separate notebook).
- The ensemble in 06 is chosen in advance (RF + CatBoost + GP); all combinations are shown only as exploratory.

## Open questions

- Units and meaning of `capacity_max` (mAh/g, maximum over cycles?). Needed for the theoretical-capacity feature.
- Which metric matters most for the group: MAPE or pairwise accuracy (ranking)?

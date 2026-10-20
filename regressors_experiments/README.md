# regressors_experiments

Task 2: capacity regressors on the current substrate data, **without** the main/side chain split.
Tree-based and Bayesian models, MAPE / pairwise accuracy / RMSE, cross-validation.
The plan is in [PLAN.md](PLAN.md); the results are in the notebooks.

## Result in one table

Same 5 × 10-fold CV for every model, metrics on pooled out-of-fold predictions, mean over 5 repeats.

| model | MAPE | pairwise acc. | RMSE | MALE |
|---|---|---|---|---|
| old BRR (rerun of `precursor2cof-capacitance`) | 3.66 | 0.666 | **177.8** | 0.786 |
| old RF (rerun) | 2.69 | 0.652 | 182.4 | 0.754 |
| **RF, log target** (EState + functional groups) | **1.40** | 0.677 | 187.0 | 0.688 |
| RF, log target, inner-CV tuned | 1.40 | 0.679 | 180.5 | 0.679 |
| RF, log target (descriptors + ECFP + EState + fg) | 1.58 | **0.685** | 187.2 | 0.692 |
| ensemble RF + CatBoost + GP (log) | 1.44 | 0.681 | 188.7 | **0.673** |

- The **log target** is the change that matters: MAPE roughly halves and pairwise accuracy goes up
  (+0.025 vs old RF, better in 5/5 repeats).
- **MAPE is misleading on this data:** predicting the constant 6 for every substrate gives MAPE 0.94,
  better than any model, because 10 substrates with capacity < 20 dominate it. Look at pairwise accuracy,
  RMSE and MALE (mean absolute log error) too.
- Bayesian models (BRR, Tanimoto GP), boosting, tuning, new features (redox / theoretical capacity,
  RDKit) and pretrained models (CheMeleon, TabPFN) don't beat the simple log-target random forest.
  Everything ends at PA ≈ 0.68, R² ≈ 0.1: the monomer structure alone explains little of the capacity.

## Notebooks

| notebook | what |
|---|---|
| [00_baseline](notebooks/00_baseline.ipynb) | data, rerun of the old best models (old and new protocol), why MAPE misleads |
| [01_target_transform](notebooks/01_target_transform.ipynb) | raw / sqrt / log / rank target |
| [02_features](notebooks/02_features.ipynb) | 16 feature sets, redox features, feature importance |
| [03_bayesian_models](notebooks/03_bayesian_models.ipynb) | BRR, ARD, GP (RBF, Tanimoto), kernel ridge, SVR, GP uncertainty |
| [04_tree_models](notebooks/04_tree_models.ipynb) | RF, ExtraTrees, XGBoost, LightGBM, CatBoost; fixed vs tuned |
| [05_pretrained](notebooks/05_pretrained.ipynb) | CheMeleon (frozen, fine-tuned), TabPFN (GPU) |
| [06_ensemble](notebooks/06_ensemble.ipynb) | averaging the best models |
| [07_summary](notebooks/07_summary.ipynb) | all runs, paired comparison with the old models, conclusions |

## Run

```bash
uv sync                       # includes torch/chemprop/tabpfn; use --no-group gpu to skip them
# CheMeleon weights (needed only for notebook 05, not committed):
mkdir -p pretrained && cp ../../precursor2cof-capacitance/modules/training_and_evaluation/foundation_models/chemeleon_mp.pt pretrained/
cd notebooks
uv run jupyter lab            # or: uv run jupyter nbconvert --to notebook --execute --inplace 00_baseline.ipynb
```

Run the notebooks in order: 06 and 07 read the predictions saved by 00–05. Run times on 16 CPU cores +
RTX 3090: 04 ≈ 55 min (nested tuning), 05 ≈ 20 min plus ≈ 10 min for fine-tuning (cached after the first run),
the others under 2 min each.

## Layout

- `cofreg/`: shared code
  - `data.py`: loads `../cof_matched/` (data + precomputed feature sets), readable feature names
  - `cv.py`: folds (old single split and repeated CV), evaluation loop, saving out-of-fold predictions
  - `metrics.py`: MAPE, pairwise accuracy, RMSE, SMAPE, MAE, R², MALE, MAPE on y ≥ 20
  - `models.py`: old tuning protocol, median baseline, target transforms, inner-CV Optuna tuner
  - `gp.py`: Tanimoto and column-slice RBF kernels, Tanimoto GP, precomputed-kernel SVR / kernel ridge
  - `features.py`: redox / theoretical-capacity features, RDKit descriptors, count fingerprints
  - `pretrained.py`: CheMeleon embeddings and fine-tuning (chemprop)
  - `plots.py`: shared plots
- `results/<experiment>/`: out-of-fold predictions for every run (`<model>.csv`, one column per repeat)
  and `summary.csv` per experiment
- `pretrained/chemeleon_mp.pt`: CheMeleon weights, copied from `precursor2cof-capacitance` (not committed, see Run)

Copied from `precursor2cof-capacitance` (with a note in each file): the metrics, the fold logic, the old
model grids and tuning protocol, and the CheMeleon loading code. Nothing imports from that repo.

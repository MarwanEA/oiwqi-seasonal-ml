# oiwqi-seasonal-ml

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](#how-to-cite)
[![tests](https://github.com/MarwanEA/oiwqi-seasonal-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/MarwanEA/oiwqi-seasonal-ml/actions/workflows/tests.yml)

Python code to compute an additive irrigation water-quality index (OIWQI) from groundwater chemistry, train
machine-learning surrogates of the index for several hydrological phases, validate them with leave-one-out
cross-validation (LOOCV), and explain them with SHAP.

The code was developed for a small groundwater dataset (tens of wells, three hydrological phases) in a
semi-arid setting. The associated article and the software citation are listed under [How to cite](#how-to-cite).

Spatial interpolation and mapping are not included in this repository.

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `01_validate_data.py` | Checks every season table against the schema and plausible ranges. |
| 2 | `02_original_protocol.py` | Tunes 15 models once on a fixed split (grid search, unshuffled K-fold), then runs LOOCV with the selected hyperparameters. |
| 3 | `03_nested_cv.py` | Nested LOOCV: hyperparameters are tuned again inside every outer fold. Resumes from checkpoints. |
| 4 | `04_uncertainty.py` | Bootstrap intervals for R² and RMSE, share of resamples in which each model is best, paired comparisons with the leading model (bootstrap, and Wilcoxon with Holm correction). |
| 5 | `05_shap.py` | SHAP values of the explained model per season and of fixed single-class models, refitted on all wells; normalised importances, grouped SHAP, direction of effect, agreement between seasons. |
| 6 | `06_influence_and_learning_curves.py` | Influence of the most extreme well, errors on the index scale, learning curves. |
| 7 | `07_shap_bootstrap.py` | Bootstrap over wells of SHAP importances and ranks. |
| 8 | `08_index_decomposition.py` | Analytical decomposition of the additive index (covariance and variance shares, exact index Shapley values) and its agreement with model SHAP. |

`scripts/run_all.py` runs the eight steps in order with the same options and stops at the first failure.

The 15 models are XGBoost, RandomForest, ExtraTrees, GradientBoosting, LightGBM, CatBoost, Ridge, Lasso,
ElasticNet, SVR, KNN, a stacking and a voting ensemble (Lasso, ExtraTrees and SVR as base learners), XGBoost
tuned with Optuna, and Gaussian process regression.

## Installation

The code requires Python 3.14. `pyproject.toml` sets `requires-python = ">=3.14,<3.15"` because 3.14 is the
only version the pipeline was tested on, and the pinned wheels in `requirements.txt` were checked for it.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

For the tests and the linter: `pip install -r requirements-dev.txt`, then `pytest` and `ruff check src tests scripts`.

## Quick start on synthetic data

```bash
python scripts/run_all.py --synthetic
```

This writes a synthetic dataset (20 wells, three seasons) to `data/synthetic/`, then runs every step with a
quick profile: smaller grids, 5 Optuna trials, fewer bootstrap resamples and Monte Carlo samples, and nested
CV for Ridge, Lasso and ExtraTrees only. It takes about 3 minutes on one worker on a recent laptop. Results go
to `results/` and figures to `figures/`. Add `--full` to use the full settings on the synthetic data.

The synthetic values are random and carry no information about any real site.

## Running on your own data

```bash
python scripts/run_all.py --data-dir path/to/tables --workers 8
```

The expected files, columns and units are described in [data/README.md](data/README.md). File names, season
names and the target column can be changed in a configuration file (see below).

Options shared by every script:

| Option | Meaning |
|--------|---------|
| `--data-dir DIR` | Folder with one table per season. |
| `--synthetic` | Generate and use a synthetic dataset (quick profile unless `--full`). |
| `--config FILE` | YAML file whose keys override `configs/default.yaml`. |
| `--out DIR` | Output folder for `results/`, `figures/` and `checkpoints/` (default: current folder). |
| `--workers N` | Number of worker processes, one thread each. Reduced automatically if memory is short. |

## Configuration

All settings are in [configs/default.yaml](configs/default.yaml). A file given with `--config` only needs the
keys it changes.

| Key | Meaning |
|-----|---------|
| `seed` | Seed for splits, models, bootstrap and Monte Carlo sampling. |
| `data.seasons`, `data.files`, `data.season_labels` | Season names, their file names and plot labels. |
| `data.station_column`, `data.target`, `data.ec_raw_column` | Column names; `ec_raw_column` is renamed to `EC` on reading. |
| `features.all`, `features.measured` | Predictors in model order; the measured subset excludes derived indices. |
| `models.base_learners` | Base learners of the stacking and voting ensembles. |
| `models.optuna_trials` | Number of Optuna trials. |
| `models.explained` | `auto` (best original-protocol R² among models with an exact explainer) or `{season: model}`. |
| `grids` | Hyperparameter grids of the tuned models. |
| `validation.test_size`, `validation.original_cv_folds` | Fixed split and K-fold of the original protocol. |
| `validation.inner_cv_folds`, `validation.inner_shuffle` | Inner K-fold of nested CV. |
| `validation.scale_in_fold` | Fit a scaler inside every fold for scale-sensitive models (Ridge, Lasso, ElasticNet, SVR, KNN, GPR). |
| `validation.nested_models` | `all` or a list of models for nested CV. |
| `groups` | Feature groups for grouped SHAP. |
| `shap.corr_nsamples`, `shap.kernel_nsamples` | Monte Carlo samples of the correlation-dependent linear explainer and of KernelSHAP. |
| `shap.single_class_models` | Models explained in every season, with their explainer. |
| `shap.pairs` | Optional feature pairs, per season, for the probability that one ranks above the other. |
| `bootstrap.performance_B`, `bootstrap.shap_B` | Bootstrap resamples for performance and for SHAP. |
| `learning_curves.sizes`, `learning_curves.repeats` | Training sizes and repeats of the learning curves. |
| `profiles.quick` | Settings applied with `--synthetic` unless `--full` is given. |

## Reproducibility

- **Seeds.** One seed (`seed`, default 42) drives the train/test split, model seeds, the inner K-fold shuffle,
  the Optuna sampler, the bootstrap and the SHAP Monte Carlo sampling. Bootstrap replicate `b` of SHAP uses
  the explainer seed `seed + 1000 + b`, so the intervals include the sampling noise of the explainer.
- **Pinned versions.** `requirements.txt` pins every direct dependency to the versions used for testing.
- **One thread per worker.** `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are set to 1 and
  every estimator runs single-threaded. Parallel work uses separate processes, so results do not depend on the
  number of workers. Multi-threaded runs of the tree libraries can give slightly different numbers.
- **Monte Carlo settings.** The correlation-dependent linear explainer estimates its transform by sampling
  (`shap.corr_nsamples`, 20000 by default) from the global numpy generator, which is seeded just before each
  call. KernelSHAP uses `shap.kernel_nsamples` and `l1_reg=False`.
- **TreeSHAP.** The additivity error (base value plus SHAP sum minus prediction) is computed for every
  explanation and reported. Interventional TreeSHAP fails for XGBoost models with the pinned versions, so
  those models use KernelSHAP.
- **Nested CV runtime.** Nested CV repeats the hyperparameter search in every outer fold, so it costs about
  20 times the original protocol for 20 wells. With the full grids and three seasons of 20 wells, this is
  roughly 9 CPU hours (about 35 minutes on 16 workers), mostly for the XGBoost, RandomForest and CatBoost
  grids; the original protocol takes about 30 CPU minutes. Each (model, fold) result is
  written to `checkpoints/nested/<season>/<raw|scaled>/<model>/foldXX.json` as soon as it finishes, and an
  interrupted run resumes from these files. Delete them to recompute.
- **"As run" default and `scale_in_fold`.** By default (`validation.scale_in_fold: false`) the models receive
  unscaled features, as in the analysis the code was written for. Setting it to `true` puts a scaler inside
  every fold for the scale-sensitive models, so no information from the held-out well reaches the scaler.

## Layout

```
oiwqi-seasonal-ml/
├── configs/default.yaml        settings, grids and the quick profile
├── data/README.md              expected input tables (data are not distributed)
├── scripts/                    steps 01 to 08, run_all.py, shared options in _cli.py
├── src/oiwqi_ml/
│   ├── config.py               configuration loading
│   ├── data.py                 reading and validating the season tables
│   ├── index.py                additive OIWQI and its exact Shapley values
│   ├── synthetic.py            synthetic dataset generator
│   ├── models.py               model registry and grids
│   ├── validation.py           original protocol, LOOCV, nested CV, in-fold scaling, runtime helpers
│   ├── explain.py              SHAP explainers, grouped SHAP, bootstrap
│   ├── plotting.py             figure style and helpers
│   └── analysis/               uncertainty, influence, learning curves, decomposition, ranks
├── tests/                      unit tests and an end-to-end run on synthetic data
├── CITATION.cff, .zenodo.json  citation metadata
└── CHANGELOG.md
```

## Data availability

The original data are not distributed with this repository; they are available from the corresponding author
of the associated study on reasonable request. The synthetic generator lets every step run without them.

## How to cite

If you use this code, please cite the article and the software. Machine-readable metadata are in
[CITATION.cff](CITATION.cff).

**Article**

Ali El Mansour, Saida Ait Boughrous, Morad Chahid, Marouan El Aoumrani, Ahmed Mahmoud Ahmed Oubeid,
Mourad Aqnouy, Amine Assouguem, Ghadeer M. Albadrani, Hader Ibrahim Sakr, Abdelaaziz El Hibaoui and
Ali Ait Boughrous. Explainable machine learning and SHAP reveal seasonal hydrochemical controls governing the
Overall Irrigation Water Quality Index under tri-seasonal conditions. *Scientific Reports* (submitted).
DOI to be added on publication.

**Software**

Marouan El Aoumrani, Ali El Mansour, Abdelaaziz El Hibaoui and Ali Ait Boughrous. oiwqi-seasonal-ml v1.0.0.
Zenodo DOI to be added on release.

## Licence

MIT, see [LICENSE](LICENSE).

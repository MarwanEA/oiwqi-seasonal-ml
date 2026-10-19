# Changelog

## 1.0.0 (unreleased)

First public version.

- Additive OIWQI with exact index Shapley values, class-level decomposition and covariance shares.
- Model registry (15 models) with the hyperparameter grids, single-threaded estimators.
- Original protocol (one search on a fixed split, then LOOCV), nested LOOCV with checkpoints, optional
  in-fold scaling (`validation.scale_in_fold`).
- SHAP explainers (correlation-dependent and interventional linear, TreeSHAP with additivity check,
  KernelSHAP), grouped SHAP and bootstrap over wells.
- Bootstrap performance intervals, paired comparisons, influence of one well, learning curves, rank analyses.
- Pipeline scripts 01 to 08 and `run_all.py`, synthetic dataset generator and quick profile.
- Requirements for Python 3.14 pinned to exact versions. Compared with the environment the analysis was
  first run in, mapping, notebook and spreadsheet packages are not needed, and `psutil` is added (memory and
  power checks).

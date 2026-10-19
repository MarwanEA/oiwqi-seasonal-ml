import numpy as np

from oiwqi_ml import explain as E
from oiwqi_ml.data import load_season


def _season(synth_dir, cfg):
    return load_season(synth_dir, cfg["data"]["seasons"][0], cfg)


def test_correlation_dependent_explainer_is_reproducible_with_a_seed(synth_dir, cfg):
    st, X, y = _season(synth_dir, cfg)
    a, _ = E.full_data_shap("Lasso", {"alpha": 0.1}, X, y, cfg, kind="corr", seed=3, nsamples=300)
    b, _ = E.full_data_shap("Lasso", {"alpha": 0.1}, X, y, cfg, kind="corr", seed=3, nsamples=300)
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_treeshap_additivity_error_is_recorded_and_small(synth_dir, cfg):
    st, X, y = _season(synth_dir, cfg)
    sv, err = E.full_data_shap("ExtraTrees", {"n_estimators": 50}, X, y, cfg)
    assert sv.shape == X.shape and err < 1e-6


def test_grouped_shap_sums_within_group_before_absolute_value(synth_dir, cfg):
    st, X, y = _season(synth_dir, cfg)
    sv, _ = E.full_data_shap("Ridge", {"alpha": 1.0}, X, y, cfg, kind="interv")
    g = E.grouped(sv, cfg["groups"])
    assert np.isclose(g["Salinity"], sv[cfg["groups"]["Salinity"]].sum(axis=1).abs().mean())


def test_bootstrap_is_reproducible_with_a_fixed_seed(synth_dir, cfg):
    st, X, y = _season(synth_dir, cfg)
    a = E.bootstrap_shap("Ridge", {"alpha": 1.0}, X, y, cfg, B=6, seed=11, kind="interv", chunk=3)
    b = E.bootstrap_shap("Ridge", {"alpha": 1.0}, X, y, cfg, B=6, seed=11, kind="interv", chunk=3)
    c = E.bootstrap_shap("Ridge", {"alpha": 1.0}, X, y, cfg, B=6, seed=12, kind="interv", chunk=3)
    assert np.array_equal(a["feature_mean_abs"], b["feature_mean_abs"])
    assert not np.array_equal(a["feature_mean_abs"], c["feature_mean_abs"])
    assert E.intervals(a).shape[0] == len(X.columns) + len(cfg["groups"])

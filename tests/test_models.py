import numpy as np

from oiwqi_ml import models
from oiwqi_ml.data import load_season, r2


def test_config_grids_equal_registry_grids(cfg):
    assert cfg["grids"] == models.GRIDS
    assert cfg["lightgbm_small_sample_grid"] == models.LIGHTGBM_SMALL_SAMPLE_GRID
    assert cfg["models"]["base_learners"] == models.BASE_LEARNERS


def test_registry_builds_every_model():
    for m in models.TUNED + [models.GPR]:
        assert models.estimator(m) is not None
    assert len(models.ALL_MODELS) == 15


def test_lightgbm_default_is_constant_on_20_rows(synth_dir, cfg):
    st, X, y = load_season(synth_dir, cfg["data"]["seasons"][0], cfg)
    n = len(y)
    pred = np.empty(n)
    for i in range(n):
        tr = np.r_[0:i, i + 1:n]
        pred[i] = models.estimator("LightGBM").fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[[i]])[0]
    assert abs(r2(y, pred) - (1 - (n / (n - 1)) ** 2)) < 1e-6

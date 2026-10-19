import numpy as np
import pandas as pd

from oiwqi_ml.analysis import decomposition, influence, learning_curves, ranks, uncertainty
from oiwqi_ml.data import load_season
from oiwqi_ml.models import estimator


def _season(synth_dir, cfg):
    return load_season(synth_dir, cfg["data"]["seasons"][0], cfg)


def test_bootstrap_intervals_are_reproducible_and_contain_the_point_value(synth_dir, cfg):
    st, X, y = _season(synth_dir, cfg)
    rng = np.random.default_rng(0)
    preds = pd.DataFrame({"a": y + rng.normal(0, 0.5, len(y)), "b": y + rng.normal(0, 1.5, len(y))})
    c1 = uncertainty.confidence_intervals(y, preds, B=300, seed=1)
    c2 = uncertainty.confidence_intervals(y, preds, B=300, seed=1)
    assert c1.equals(c2)
    assert ((c1.R2_lo <= c1.R2) & (c1.R2 <= c1.R2_hi)).all()
    assert abs(uncertainty.p_best(y, preds, 300, 1).sum() - 1) < 1e-12
    assert uncertainty.paired_vs_leader(y, preds, 300, 1).leader.iloc[0] == "a"


def test_covariance_shares_sum_to_one(synth_dir, cfg):
    st, X, y = _season(synth_dir, cfg)
    p, c = decomposition.parameter_and_class_shares(X)
    assert abs(p.cov_share.sum() - 1) < 1e-10 and abs(c.cov_share.sum() - 1) < 1e-10


def test_rank_range_and_normalised_shares():
    imp = pd.DataFrame({"s1": [3.0, 2.0, 1.0], "s2": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])
    assert list(ranks.rank_range(imp)) == [2, 0, 2]
    sv = pd.DataFrame({"a": [1.0, -1.0], "b": [0.5, -0.5]})
    assert abs(ranks.normalised(sv, [1.0, 2.0], [1.2, 1.8]).share_pct.sum() - 100) < 1e-9


def test_influence_and_learning_curve_shapes(synth_dir, cfg):
    st, X, y = _season(synth_dir, cfg)
    w = influence.most_extreme_well(st, y)
    prof = influence.well_profile(st, y, w)
    assert 0 < prof["ss_share"] < 1
    lc = learning_curves.learning_curve(estimator("Ridge"), X, y, sizes=[10, 19], repeats=2, seed=1)
    assert set(lc.m_train) == {10, 19} and len(lc) == 4

import numpy as np
import pandas as pd

from oiwqi_ml import index


def _random_wells(n=30, seed=0):
    rng = np.random.default_rng(seed)
    lo_hi = {"EC": (300, 3500), "TDS": (200, 2500), "Na%": (5, 90), "SSP": (5, 90), "SAR": (0.5, 35), "PI": (20, 95),
             "K": (0.5, 40), "Na": (10, 450), "Cl": (10, 350), "pH": (6.2, 8.8), "Ca": (20, 200), "Mg": (5, 120),
             "NO3": (0.5, 80), "SO4": (5, 250), "HCO3": (30, 650), "RSBC": (-8, 18), "MAR": (5, 70)}
    return pd.DataFrame({p: rng.uniform(*lh, n) for p, lh in lo_hi.items()})


def test_index_equals_sum_of_parameter_and_class_terms():
    X = _random_wells()
    o = index.oiwqi(X)
    assert np.allclose(index.parameter_terms(X).sum(axis=1), o, atol=1e-10)
    assert np.allclose(index.class_terms(X).sum(axis=1), o, atol=1e-10)


def test_index_shapley_values_sum_to_prediction_minus_mean():
    X = _random_wells()
    o = index.oiwqi(X)
    assert np.allclose(index.index_shap(X).sum(axis=1), o - o.mean(), atol=1e-10)


def test_class_split_rule_sums_to_one_and_weights_sum_to_one():
    assert np.allclose(index.class_share().sum(axis=1), 1.0)
    assert abs(sum(index.WEIGHTS.values()) - 1) < 1e-12


def test_value_above_cmax_contributes_zero():
    assert index.q("EC", 5000) == 0.0

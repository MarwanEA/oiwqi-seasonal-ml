import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from oiwqi_ml import validation as V
from oiwqi_ml.data import load_season


class Recorder(BaseEstimator, RegressorMixin):
    """Records the column means of the training matrix it receives."""
    seen: list = []

    def fit(self, X, y):
        Recorder.seen.append((len(X), np.asarray(X).mean(axis=0)))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def _season(synth_dir, cfg):
    return load_season(synth_dir, cfg["data"]["seasons"][0], cfg)


def test_scale_in_fold_fits_the_scaler_on_the_training_fold_only(synth_dir, cfg):
    st, X, y = _season(synth_dir, cfg)
    Recorder.seen = []
    V.loocv_predict(Recorder(), X, y, scale=True)
    assert len(Recorder.seen) == len(y)
    for n_rows, means in Recorder.seen:
        assert n_rows == len(y) - 1
        assert np.allclose(means, 0.0, atol=1e-9)  # standardised with the fold's own mean


def test_loocv_returns_one_prediction_per_well(synth_dir, cfg):
    st, X, y = _season(synth_dir, cfg)
    out = V.original_protocol(X, y, cfg, ["Lasso", "Ridge"])
    assert all(len(r["pred"]) == len(y) for r in out.values())


def test_nested_returns_one_prediction_per_well_in_order(synth_dir, cfg, tmp_path):
    st, X, y = _season(synth_dir, cfg)
    df = V.nested_loocv(X, y, cfg, ["Lasso"], stations=st, checkpoint_dir=tmp_path)
    assert list(df.fold) == list(range(len(y))) and list(df.station) == list(st)
    again = V.nested_loocv(X, y, cfg, ["Lasso"], checkpoint_dir=tmp_path)  # resumes from checkpoints
    assert np.allclose(df.y_pred, again.y_pred)


def test_scaled_nested_search_uses_pipeline_parameters(synth_dir, cfg):
    st, X, y = _season(synth_dir, cfg)
    rec = V.nested_fold("Ridge", X, y, 0, cfg, scale=True)
    assert set(rec["params"]) == {"alpha"}

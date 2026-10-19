import numpy as np

from oiwqi_ml import index
from oiwqi_ml.data import load_all, read_table, validate
from oiwqi_ml.synthetic import generate


def test_synthetic_tables_pass_schema_validation(synth_dir, cfg):
    for s in cfg["data"]["seasons"]:
        df = read_table(synth_dir / cfg["data"]["files"][s])
        assert not [p for p in validate(df, cfg) if not p.startswith("warning")]


def test_synthetic_target_is_the_index_of_the_stored_predictors(synth_dir, cfg):
    for st, X, y in load_all(synth_dir, cfg).values():
        assert np.allclose(index.oiwqi(X), y, atol=1e-5)


def test_synthetic_generation_is_reproducible(cfg):
    a, b = generate(cfg, seed=7), generate(cfg, seed=7)
    assert all(a[s].equals(b[s]) for s in a)

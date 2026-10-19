import pandas as pd

from oiwqi_ml import plotting


def test_figures_are_written(tmp_path):
    ci = pd.DataFrame({"model": ["Ridge", "Lasso"], "R2": [0.5, -0.2], "R2_lo": [0.2, -1.5], "R2_hi": [0.7, 0.3]})
    s = pd.Series([1.0, 2.0], index=["a", "b"])
    lc = pd.DataFrame({"m_train": [10, 19], "R2_mean": [0.3, 0.6], "R2_p2.5": [0.1, 0.5], "R2_p97.5": [0.4, 0.7]})
    figs = {"forest.png": plotting.forest(ci, "t"), "bars.png": plotting.interval_bars(s, s - 0.5, s + 0.5, "t", "x"),
            "heat.png": plotting.heatmap(pd.DataFrame({"x": s, "y": s}), "t"),
            "lc.png": plotting.learning_curves({"Ridge": lc}, "t")}
    for name, fig in figs.items():
        assert plotting.save(fig, tmp_path / name).stat().st_size > 0

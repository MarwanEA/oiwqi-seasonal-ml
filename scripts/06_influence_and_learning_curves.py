"""Step 6: influence of the most extreme well, performance on the index scale and learning curves."""

import pandas as pd
from _cli import data, explained, label, load_params, require, save_csv, setup

from oiwqi_ml import plotting
from oiwqi_ml.analysis import influence as I
from oiwqi_ml.analysis import learning_curves as LC
from oiwqi_ml.validation import loocv_predict, with_params

run = setup(__doc__)
cfg = run.cfg
params = load_params(run)
loocv = pd.read_csv(require(run, "original_predictions.csv"))
chosen = explained(run)
lc_cfg = cfg["learning_curves"]
infl, curves, res = [], [], []
for s, (st, X, y) in data(run).items():
    well = I.well_profile(st, y, I.most_extreme_well(st, y))
    for m, g in loocv[loocv.season == s].groupby("model", sort=False):
        g = g.set_index("station").loc[st]
        row = {"season": s, "model": m, **well, **I.r2_with_without(st, g.y_true, g.y_pred, well["station"])}
        if m == chosen[s]:
            row["R2_refit_without"] = I.r2_refit_without(
                st, X, y, well["station"], lambda Xw, yw: loocv_predict(with_params(m, params[s][m], cfg), Xw, yw))
            res.append(I.residuals(st, g.y_true, g.y_pred).assign(season=s, model=m))
        infl.append(row)
    summary = {}
    for m in dict.fromkeys([chosen[s], *cfg["shap"]["single_class_models"]]):
        sizes = [k for k in lc_cfg["sizes"] if k < len(y)]
        lc = LC.learning_curve(with_params(m, params[s][m], cfg), X, y, sizes, lc_cfg["repeats"], cfg["seed"])
        summary[m] = LC.summarise_curve(lc)
        curves.append(summary[m].assign(season=s, model=m))
    plotting.save(plotting.learning_curves(summary, f"{label(run, s)}: learning curves"),
                  run.figures / f"learning_curves_{s}.png")
save_csv(pd.DataFrame(infl), run, "influence_extreme_well.csv")
save_csv(pd.concat(res), run, "residuals_explained_model.csv")
save_csv(pd.concat(curves), run, "learning_curves.csv")

"""Step 8: analytical decomposition of the index and its agreement with the model SHAP values from step 5."""

import pandas as pd
from _cli import data, label, require, save_csv, setup

from oiwqi_ml import index, plotting
from oiwqi_ml.analysis import decomposition as D
from oiwqi_ml.explain import mean_abs

run = setup(__doc__)
summary = pd.read_csv(require(run, "shap_summary.csv"))
shares, agree, cls = [], [], []
cov = {}
for s, (st, X, y) in data(run).items():
    p, c = D.parameter_and_class_shares(X)
    shares += [p.rename_axis("term").reset_index().assign(season=s, level="parameter"),
               c.rename_axis("term").reset_index().assign(season=s, level="class")]
    cov[label(run, s)] = p.cov_share
    for (set_name, m, kind), g in summary[summary.season == s].groupby(["set", "model", "kind"], sort=False):
        model_imp = g.set_index("feature").mean_abs_shap
        agree.append({"season": s, "set": set_name, "model": m, "kind": kind,
                      **D.agreement(p.index_shap, model_imp)})
        sv = pd.read_csv(run.results / f"shap_values_{s}_{m}_{kind}.csv", index_col="station")
        model_cls = mean_abs(D.class_level_shap(sv))
        index_cls = mean_abs(D.class_level_shap(index.index_shap(X)))
        cls.append(pd.DataFrame({"model_mean_abs": model_cls, "index_mean_abs": index_cls}).rename_axis("class")
                   .reset_index().assign(season=s, set=set_name, model=m, kind=kind))
save_csv(pd.concat(shares), run, "index_shares.csv")
save_csv(pd.DataFrame(agree), run, "index_vs_model_agreement.csv")
save_csv(pd.concat(cls), run, "index_vs_model_class_level.csv")
plotting.save(plotting.heatmap(pd.DataFrame(cov), "Index: covariance share per parameter", cmap="magma"),
              run.figures / "index_covariance_shares.png")

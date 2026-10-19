"""Step 5: SHAP values of the explained model per season and of the single-class models, refitted on all wells."""

import pandas as pd
from _cli import data, explained, label, load_params, require, save_csv, setup

from oiwqi_ml import plotting
from oiwqi_ml.analysis import ranks
from oiwqi_ml.explain import explainer_kind, full_data_shap, grouped, mean_abs

run = setup(__doc__)
cfg = run.cfg
params = load_params(run)
loocv = pd.read_csv(require(run, "original_predictions.csv"))
chosen = explained(run)
summary, groups, direction, add = [], [], [], []
imp = {}  # set -> {season: mean |SHAP|}
for s, (st, X, y) in data(run).items():
    X = X.set_axis(st.to_numpy())
    todo = {("explained", chosen[s]): explainer_kind(chosen[s])}
    todo.update({(m, m): k for m, k in cfg["shap"]["single_class_models"].items()})
    for (set_name, m), kind in todo.items():
        sv, err = full_data_shap(m, params[s][m], X, y.set_axis(X.index), cfg, kind)
        save_csv(sv.rename_axis("station"), run, f"shap_values_{s}_{m}_{kind}.csv", index=True)
        pred = loocv[(loocv.season == s) & (loocv.model == m)].y_pred
        n = ranks.normalised(sv, y, pred).assign(rank=lambda d: mean_abs(sv).rank(ascending=False, method="min"))
        summary.append(n.rename_axis("feature").reset_index().assign(season=s, set=set_name, model=m, kind=kind))
        groups.append(grouped(sv, cfg["groups"]).rename("grouped_mean_abs_shap").rename_axis("group").reset_index()
                      .assign(season=s, set=set_name, model=m, kind=kind))
        direction.append(ranks.direction_of_effect(X, sv).assign(season=s, set=set_name, model=m, kind=kind))
        add.append({"season": s, "set": set_name, "model": m, "kind": kind, "additivity_error": err})
        imp.setdefault(set_name, {})[s] = mean_abs(sv)
save_csv(pd.concat(summary), run, "shap_summary.csv")
save_csv(pd.concat(groups), run, "shap_grouped.csv")
save_csv(pd.concat(direction), run, "shap_direction.csv")
save_csv(pd.DataFrame(add), run, "shap_additivity.csv")
agree, rr = [], []
for set_name, by_season in imp.items():
    table = pd.DataFrame(by_season)
    agree.append(ranks.between_season_agreement(table).assign(set=set_name))
    rr.append(ranks.rank_range(table).rename("rank_range").rename_axis("feature").reset_index().assign(set=set_name))
    title = "Explained model per season" if set_name == "explained" else set_name
    plotting.save(plotting.heatmap(table.rename(columns=lambda c: label(run, c)), f"{title}: mean |SHAP|"),
                  run.figures / f"shap_heatmap_{set_name}.png")
save_csv(pd.concat(agree), run, "shap_between_season_agreement.csv")
save_csv(pd.concat(rr), run, "shap_rank_range.csv")

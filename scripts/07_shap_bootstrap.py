"""Step 7: bootstrap over wells of SHAP importances and ranks (refit on each resample, explain the original wells)."""

import numpy as np
import pandas as pd
from _cli import data, explained, label, load_params, save_csv, setup

from oiwqi_ml import plotting
from oiwqi_ml.analysis import ranks
from oiwqi_ml.explain import ADDITIVITY_TOL, bootstrap_shap, explainer_kind, intervals, pairwise_probability

run = setup(__doc__)
cfg = run.cfg
params = load_params(run)
chosen = explained(run)
B, seed = cfg["bootstrap"]["shap_B"], cfg["seed"]
ints, pairs, add, reps = [], [], [], {}
for s, (st, X, y) in data(run).items():
    todo = {"explained": (chosen[s], explainer_kind(chosen[s]))}
    todo.update({m: (m, k) for m, k in cfg["shap"]["single_class_models"].items()})
    for set_name, (m, kind) in todo.items():
        boot = bootstrap_shap(m, params[s][m], X, y, cfg, B, seed, kind, workers=run.workers)
        meta = {"season": s, "set": set_name, "model": m, "kind": kind}
        ints.append(intervals(boot).assign(**meta))
        if cfg["shap"]["pairs"].get(s):
            pairs.append(pairwise_probability(boot, [tuple(p) for p in cfg["shap"]["pairs"][s]]).assign(**meta))
        e = boot["additivity_error"]
        add.append({**meta, "B": B, "max_error": float(np.nanmax(e)) if np.isfinite(e).any() else np.nan,
                    "n_above_tolerance": int((e > ADDITIVITY_TOL).sum())})
        reps.setdefault(set_name, {})[s] = boot["feature_mean_abs"]
        if set_name == "explained":
            g = ints[-1][ints[-1].level == "group"].set_index("name")
            mean = pd.Series(boot["group_mean_abs"].mean(0), index=boot["groups"])
            plotting.save(plotting.interval_bars(mean, g.ci_lo, g.ci_hi, f"{label(run, s)}: {m}",
                                                 "grouped mean |SHAP| (95% bootstrap interval)"),
                          run.figures / f"shap_grouped_bootstrap_{s}.png")
save_csv(pd.concat(ints), run, "shap_bootstrap_intervals.csv")
if pairs:
    save_csv(pd.concat(pairs), run, "shap_pairwise_probability.csv")
save_csv(pd.DataFrame(add), run, "shap_bootstrap_additivity.csv")
rr = [ranks.rank_range_bootstrap(r).assign(feature=X.columns, set=k) for k, r in reps.items()]
save_csv(pd.concat(rr), run, "shap_rank_range_bootstrap.csv")

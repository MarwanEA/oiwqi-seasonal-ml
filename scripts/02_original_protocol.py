"""Step 2: original protocol. Hyperparameters chosen once on a fixed split, then LOOCV with fixed values."""

import pandas as pd
from _cli import data, save_csv, save_params, setup

from oiwqi_ml import models as M
from oiwqi_ml.analysis.learning_curves import scale_metrics
from oiwqi_ml.validation import original_protocol, run_parallel

run = setup(__doc__)
cfg, scale = run.cfg, run.cfg["validation"]["scale_in_fold"]
seasons = data(run)
# one task per model (Stacking and Voting together) so that seasons and models run in parallel
groups = [[m] for m in M.TUNED] + [M.ENSEMBLES, [M.OPTUNA], [M.GPR]]


def task(s, X, y, names):
    return s, original_protocol(X, y, cfg, names, scale)


parts = run_parallel(task, [(s, X, y, g) for s, (_, X, y) in seasons.items() for g in groups], run.workers)
preds, metrics, params = [], [], {s: {} for s in seasons}
for s, res in parts:
    st, X, y = seasons[s]
    for m, r in res.items():
        params[s][m] = r["params"]
        preds.append(pd.DataFrame({"season": s, "station": st.to_numpy(), "model": m, "y_true": y.to_numpy(),
                                   "y_pred": r["pred"]}))
        metrics.append({"season": s, "model": m, **scale_metrics(y, r["pred"])})
order = {m: i for i, m in enumerate(M.ALL_MODELS)}
pred = pd.concat(preds).sort_values(["season", "model"], key=lambda c: c.map(order) if c.name == "model" else c)
save_csv(pred, run, "original_predictions.csv")
save_csv(pd.DataFrame(metrics).sort_values(["season", "R2"], ascending=[True, False]), run, "original_metrics.csv")
save_params(params, run, "original_params.json")

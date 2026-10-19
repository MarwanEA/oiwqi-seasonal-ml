"""Step 3: nested LOOCV. Hyperparameters re-tuned inside every outer fold; resumes from checkpoints."""

import pandas as pd
from _cli import data, save_csv, setup

from oiwqi_ml import models as M
from oiwqi_ml.analysis.learning_curves import scale_metrics
from oiwqi_ml.validation import battery_ok, nested_loocv

run = setup(__doc__)
cfg = run.cfg
if not battery_ok():
    print("note: running on battery; the run resumes from checkpoints if the machine sleeps")
names = cfg["validation"]["nested_models"]
names = M.ALL_MODELS if names == "all" else names
frames, metrics = [], []
for s, (st, X, y) in data(run).items():
    df = nested_loocv(X, y, cfg, names, cfg["validation"]["scale_in_fold"], run.workers, run.checkpoints / "nested" / s,
                      st)
    df.insert(0, "season", s)
    frames.append(df)
    metrics += [{"season": s, "model": m, **scale_metrics(g.y_true, g.y_pred)} for m, g in df.groupby("model")]
    print(f"{s}: {len(df)} fold predictions")
out = pd.concat(frames)
out["params"] = out.params.map(str)
save_csv(out, run, "nested_predictions.csv")
save_csv(pd.DataFrame(metrics).sort_values(["season", "R2"], ascending=[True, False]), run, "nested_metrics.csv")

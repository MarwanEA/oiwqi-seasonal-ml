"""Step 4: bootstrap intervals, share of resamples won and paired comparisons with the leading model."""

import pandas as pd
from _cli import label, require, save_csv, setup

from oiwqi_ml import plotting
from oiwqi_ml.analysis import uncertainty as U

run = setup(__doc__)
cfg = run.cfg
B, seed = cfg["bootstrap"]["performance_B"], cfg["seed"]
ci_all, paired_all = [], []
for protocol in ("original", "nested"):
    path = run.results / f"{protocol}_predictions.csv"
    if protocol == "original":
        require(run, path.name)
    elif not path.exists():
        print("no nested predictions; skipping the nested protocol")
        continue
    pred = pd.read_csv(path)
    for s, g in pred.groupby("season", sort=False):
        wide = g.pivot(index="station", columns="model", values="y_pred")
        y = g.drop_duplicates("station").set_index("station").y_true.loc[wide.index]
        ci = U.confidence_intervals(y, wide, B, seed)
        ci["P_best"] = ci.model.map(U.p_best(y, wide, B, seed))
        paired = U.paired_vs_leader(y, wide, B, seed)
        for t in (ci, paired):
            t.insert(0, "protocol", protocol)
            t.insert(0, "season", s)
        ci_all.append(ci)
        paired_all.append(paired)
        plotting.save(plotting.forest(ci, f"{label(run, s)}: {protocol} protocol"),
                      run.figures / f"forest_{protocol}_{s}.png")
save_csv(pd.concat(ci_all), run, "performance_intervals.csv")
save_csv(pd.concat(paired_all), run, "paired_comparisons.csv")

"""Step 1: check every season table against the configured schema and plausible ranges."""

import sys

import pandas as pd
from _cli import save_csv, setup

from oiwqi_ml.data import read_table, validate

run = setup(__doc__)
cfg, errors, rows = run.cfg, 0, []
for season in cfg["data"]["seasons"]:
    df = read_table(run.data_dir / cfg["data"]["files"][season], cfg["data"]["ec_raw_column"])
    problems = validate(df, cfg)
    for p in problems:
        print(f"{season}: {p}")
    errors += sum(not p.startswith("warning:") for p in problems)
    rows.append({"season": season, "n_wells": len(df), "n_problems": len(problems),
                 "target_mean": df[cfg["data"]["target"]].mean(), "target_sd": df[cfg["data"]["target"]].std()})
save_csv(pd.DataFrame(rows), run, "data_summary.csv")
if errors:
    sys.exit(f"{errors} schema problem(s); fix the data before running the next steps")
print("data valid")

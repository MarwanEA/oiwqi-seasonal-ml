import pandas as pd
from oiwqi_ml.config import load_config
from oiwqi_ml.data import load_season, read_table, validate

CFG = load_config()
FEATS = CFG["features"]["all"]


def _table(n=4, ec_name="CE"):
    df = pd.DataFrame({f: [1.0 + i for i in range(n)] for f in FEATS})
    df = df.rename(columns={"EC": ec_name})
    df.insert(0, "Station", [f"S{i:02d}" for i in range(1, n + 1)])
    df["O_IWQI"] = [10.0 + i for i in range(n)]
    return df


def test_ce_renamed_and_whitespace_stripped(tmp_path):
    df = _table().rename(columns={"TDS": "TDS "})
    p = tmp_path / "dry.csv"
    df.to_csv(p, index=False)
    t = read_table(p)
    assert "EC" in t.columns and "CE" not in t.columns and "TDS" in t.columns


def test_station_labels_unchanged(tmp_path):
    df = _table()
    df.loc[0, "Station"] = "X17"
    df.to_csv(tmp_path / "dry.csv", index=False)
    st, X, y = load_season(tmp_path, "dry", CFG)
    assert st.iloc[0] == "X17" and list(X.columns) == FEATS and len(y) == 4


def test_missing_column_is_reported():
    assert any("missing column: SO4" in p for p in validate(_table().drop(columns="SO4"), CFG))


def test_user_file_overrides_profile(tmp_path):
    from oiwqi_ml.config import load_config
    f = tmp_path / "u.yaml"
    f.write_text("bootstrap: {shap_B: 4}\n")
    c = load_config(f, profile="quick")
    assert c["bootstrap"]["shap_B"] == 4 and c["bootstrap"]["performance_B"] == 500
    assert c["grids"]["Ridge"] == load_config()["grids"]["Ridge"]

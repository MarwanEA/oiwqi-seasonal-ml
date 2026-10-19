import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TINY = """
models: {optuna_trials: 2}
validation: {nested_models: [Ridge]}
shap: {corr_nsamples: 500, kernel_nsamples: 200}
bootstrap: {performance_B: 100, shap_B: 4}
learning_curves: {sizes: [19], repeats: 1}
"""


def test_run_all_on_synthetic_data(tmp_path):
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(TINY)
    r = subprocess.run([sys.executable, str(SCRIPTS / "run_all.py"), "--synthetic", "--config", str(cfg),
                        "--out", str(tmp_path), "--workers", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for name in ("original_predictions.csv", "nested_predictions.csv", "performance_intervals.csv", "shap_summary.csv",
                 "shap_bootstrap_intervals.csv", "learning_curves.csv", "index_vs_model_agreement.csv"):
        assert (tmp_path / "results" / name).stat().st_size > 0

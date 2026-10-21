"""Small synthetic-data checks; no private ADNI data is required."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def test_synthetic_end_to_end(tmp_path):
    rng = np.random.default_rng(42)
    ids = [f"subject_{i:03d}" for i in range(40)]
    labels = np.array([0, 1] * 20)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    specifications = [
        ("c.csv", 9, True),
        ("a.csv", 10, False),
        ("b.csv", 8, False),
        ("d.csv", 11, False),
    ]
    for filename, n_features, is_snp in specifications:
        if is_snp:
            matrix = rng.integers(0, 3, size=(len(ids), n_features)).astype(float)
        else:
            matrix = rng.normal(
                loc=labels[:, None] * 0.25,
                scale=1.0,
                size=(len(ids), n_features),
            )
        matrix[0, -1] = np.nan  # exercise fold-local imputation
        frame = pd.DataFrame(
            matrix, index=ids,
            columns=[f"feature_{j}" for j in range(n_features)],
        )
        frame.index.name = "sample_id"
        frame.to_csv(data_dir / filename)
    pd.DataFrame({"Label": labels}, index=ids).rename_axis("sample_id").to_csv(
        data_dir / "label.csv"
    )

    output = tmp_path / "results"
    repo = Path(__file__).resolve().parent
    args = [
        sys.executable, str(repo / "run.py"),
        "--mode", "holdout", "--data-dir", str(data_dir),
        "--folds", "2", "--epochs", "2", "--patience", "2",
        "--threads", "1", "--latent-dim", "6", "--hidden-dim", "12",
        "--top-n", "3", "--max-hub-features", "9",
        "--association-top-n", "3", "--expected-n", "0",
        "--out", str(output),
    ]
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib_cache")
    result = subprocess.run(
        args, cwd=repo, env=env, text=True,
        capture_output=True, timeout=150,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    held = json.loads((output / "holdout_evaluation" /
                       "summary.json").read_text())
    assert held["development_n"] + held["test_n"] == 40
    assert held["test_n"] == 8
    assert 1 <= held["module"] <= 6
    assert 0 <= held["heldout_auc"] <= 1
    preds = pd.read_csv(output / "holdout_evaluation" /
                        "heldout_predictions.csv")
    assert len(preds) == 8
    assert preds["row"].nunique() == 8
    assert preds["predicted_probability"].between(0, 1).all()
    assert (output / "holdout_evaluation" / "training_hubs.csv").exists()
    assert (output / "holdout_evaluation" / "split_manifest_hashed.csv").exists()


def test_feature_selection_and_seed():
    from features import select_hub_features
    from model import build_laplacian
    weights = np.array([0.56, 0.25, 0.14, 0.05], dtype=float)
    selected, _ = select_hub_features(
        weights, strategy="cumsum", cum_thresh=0.8,
        min_n=1, max_n=4,
    )
    assert selected.tolist() == [0, 1]
    assert np.allclose(build_laplacian(5), build_laplacian(5))


def test_optuna_objective_respects_epoch_limit(monkeypatch):
    import optuna
    import training

    seen_epochs = []

    def fake_training(**kwargs):
        seen_epochs.append(kwargs["epochs"])
        return None, 0.1, {}

    monkeypatch.setattr(training, "run_training", fake_training)
    objective = training.make_objective([], [], [4, 4], epochs=3)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)
    assert seen_epochs == [3]

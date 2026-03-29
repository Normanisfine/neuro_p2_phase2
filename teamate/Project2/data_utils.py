"""
Utilities for Project 2 Phase 1 data loading, masking simulation, evaluation,
and Kaggle submission file generation.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# Default to local extracted dataset path; can always be overridden by --data-dir.
DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "kaggle_data",
)

N_CHANNELS = 96
N_MASKED_CHANNELS = 30


def _session_file(data_dir: str, split: str, session_id: str, suffix: str) -> str:
    return os.path.join(data_dir, split, f"{session_id}_{suffix}")


def load_metadata(data_dir: str = DATA_DIR) -> pd.DataFrame:
    path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metadata.csv not found at: {path}")
    return pd.read_csv(path)


def load_session(data_dir: str, session_id: str, is_test: bool = False) -> Dict[str, np.ndarray]:
    split = "test" if is_test else "train"
    trial_path = _session_file(data_dir, split, session_id, "trial_info.npz")
    if not os.path.exists(trial_path):
        raise FileNotFoundError(f"Missing trial info: {trial_path}")
    trial = np.load(trial_path)
    start_bins = trial["start_bins"].astype(np.int64)
    end_bins = trial["end_bins"].astype(np.int64)
    n_trials = int(trial["n_trials"])

    kin = np.load(_session_file(data_dir, split, session_id, "kinematics.npy")).astype(np.float32)
    if is_test:
        sbp_m = np.load(_session_file(data_dir, split, session_id, "sbp_masked.npy")).astype(np.float32)
        mask = np.load(_session_file(data_dir, split, session_id, "mask.npy")).astype(bool)
        return {
            "session_id": session_id,
            "sbp_masked": sbp_m,
            "mask": mask,
            "kinematics": kin,
            "start_bins": start_bins,
            "end_bins": end_bins,
            "n_trials": n_trials,
        }

    sbp = np.load(_session_file(data_dir, split, session_id, "sbp.npy")).astype(np.float32)
    return {
        "session_id": session_id,
        "sbp": sbp,
        "kinematics": kin,
        "start_bins": start_bins,
        "end_bins": end_bins,
        "n_trials": n_trials,
    }


def get_validation_sessions(data_dir: str = DATA_DIR, n_val: int = 20) -> Tuple[List[str], List[str]]:
    """
    Deterministic split by chronology:
    - train_ids: all train sessions except last n_val
    - val_ids:   last n_val train sessions
    """
    meta = load_metadata(data_dir)
    tr = meta[meta["split"] == "train"].copy().sort_values("day")
    all_train = tr["session_id"].tolist()
    if len(all_train) <= n_val:
        raise ValueError(f"Need more than {n_val} train sessions, got {len(all_train)}")
    val_ids = all_train[-n_val:]
    train_ids = all_train[:-n_val]
    return train_ids, val_ids


def get_val_difficulty_labels(data_dir: str = DATA_DIR) -> Dict[str, str]:
    """
    Compatibility helper for older scripts. For train sessions difficulty is empty;
    return 'train' labels.
    """
    meta = load_metadata(data_dir)
    tr = meta[meta["split"] == "train"][["session_id"]]
    return {sid: "train" for sid in tr["session_id"].tolist()}


def simulate_masking(
    sbp: np.ndarray,
    start_bins: Sequence[int],
    end_bins: Sequence[int],
    n_trials: int,
    seed: int = 42,
    n_masked_trials: int = 10,
    n_masked_channels: int = N_MASKED_CHANNELS,
) -> np.ndarray:
    """
    Simulate test-time masking protocol on a train session:
    - randomly choose n_masked_trials trials
    - for each selected trial and each time bin, mask n_masked_channels / 96
    """
    rng = np.random.RandomState(seed)
    n_bins = sbp.shape[0]
    mask = np.zeros((n_bins, N_CHANNELS), dtype=bool)

    trial_indices = np.arange(n_trials)
    if n_trials <= n_masked_trials:
        chosen = trial_indices
    else:
        chosen = rng.choice(trial_indices, size=n_masked_trials, replace=False)

    for ti in chosen:
        s = int(start_bins[ti])
        e = int(end_bins[ti])
        for t in range(s, e):
            ch = rng.choice(N_CHANNELS, size=n_masked_channels, replace=False)
            mask[t, ch] = True
    return mask


def session_zscore_params(sbp: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-channel mean/std.
    If mask is given, estimate stats only on unmasked entries.
    """
    if mask is None:
        mean = sbp.mean(axis=0, keepdims=True).astype(np.float32)
        std = sbp.std(axis=0, keepdims=True).astype(np.float32)
    else:
        mean = np.zeros((1, sbp.shape[1]), dtype=np.float32)
        std = np.ones((1, sbp.shape[1]), dtype=np.float32)
        for c in range(sbp.shape[1]):
            obs = ~mask[:, c]
            if obs.any():
                vc = sbp[obs, c]
                mean[0, c] = float(vc.mean())
                std[0, c] = float(vc.std())
            else:
                mean[0, c] = 0.0
                std[0, c] = 1.0
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def zscore_normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def zscore_denormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x * std + mean).astype(np.float32)


def compute_nmse_multi(results: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]) -> float:
    """
    results: iterable of (y_true, y_pred, mask, session_id)
    NMSE is averaged equally over (session, channel) groups.
    """
    group_nmse: List[float] = []
    for y_true, y_pred, mask, _sid in results:
        # Variance per channel across all time bins in the session (competition definition)
        var_c = y_true.var(axis=0)
        var_c = np.maximum(var_c, 1e-10)
        for c in range(y_true.shape[1]):
            m = mask[:, c]
            if not np.any(m):
                continue
            mse = np.mean((y_pred[m, c] - y_true[m, c]) ** 2)
            group_nmse.append(float(mse / var_c[c]))
    if not group_nmse:
        return float("inf")
    return float(np.mean(group_nmse))


def build_submission(predictions: Dict[str, np.ndarray], data_dir: str = DATA_DIR, output_path: str = "submission.csv") -> None:
    """
    Fill sample_submission.csv with predicted_sbp values from full session matrices.
    """
    sample_path = os.path.join(data_dir, "sample_submission.csv")
    sub = pd.read_csv(sample_path)

    vals = np.zeros(len(sub), dtype=np.float32)
    for i, row in sub.iterrows():
        sid = row["session_id"]
        t = int(row["time_bin"])
        c = int(row["channel"])
        vals[i] = float(predictions[sid][t, c])

    out = sub.copy()
    out["predicted_sbp"] = vals
    out.to_csv(output_path, index=False)
    print(f"Saved submission: {output_path} ({len(out)} rows)")

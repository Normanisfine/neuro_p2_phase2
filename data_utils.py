"""
Shared utilities for Project 2 Phase 2 — Finger Position Decoding.

Data loading, train/val split, local R² evaluation, submission building.
Import from method directories via:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data_utils import ...
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

DATA_DIR    = os.path.join(os.path.dirname(__file__), 'kaggle_data')
P1_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'kaggle_data')  # Phase 1 data
N_CHANNELS  = 96
N_KIN       = 2  # index_pos, mrp_pos (columns 0 and 1 of kinematics)


# ---------------------------------------------------------------------------
# Session enumeration
# ---------------------------------------------------------------------------

def list_session_ids(data_dir: str = DATA_DIR, split: str = 'train') -> List[str]:
    """List session IDs sorted chronologically by D-number."""
    split_dir = os.path.join(data_dir, split)
    files = os.listdir(split_dir)
    sids  = sorted(
        set(f.split('_')[0] for f in files if f.endswith('.npy')),
        key=lambda s: int(s[1:])
    )
    return sids


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session(data_dir: str, session_id: str, is_test: bool = False) -> dict:
    """Load a single session's arrays.

    Returns dict with keys:
      sbp          : (N, 96) float32 — permanently-zeroed channels per session
      kinematics   : (N, 4) float32 [index_pos, mrp_pos, index_vel, mrp_vel]
                     (train only; NOT available for test)
      start_bins   : (n_trials,) int
      end_bins     : (n_trials,) int
      n_trials     : int
      active_mask  : (96,) bool — True for channels with non-zero activity
      dropout_ind  : (96,) float32 — 1.0 for permanently-zeroed channels
    """
    split_dir = os.path.join(data_dir, 'test' if is_test else 'train')
    result    = {}

    sbp = np.load(os.path.join(split_dir, f'{session_id}_sbp.npy'))
    result['sbp'] = sbp

    if not is_test:
        result['kinematics'] = np.load(
            os.path.join(split_dir, f'{session_id}_kinematics.npy')
        )

    tri = np.load(os.path.join(split_dir, f'{session_id}_trial_info.npz'))
    result['start_bins'] = tri['start_bins']
    result['end_bins']   = tri['end_bins']
    result['n_trials']   = int(tri['n_trials'])

    # Detect permanently-zeroed channels (channel dropout)
    active_mask           = (sbp != 0).any(axis=0)          # (96,) bool
    result['active_mask'] = active_mask
    result['dropout_ind'] = (~active_mask).astype(np.float32)  # 1=zeroed

    return result


# ---------------------------------------------------------------------------
# Normalisation (operates only on active channels; zeroed channels stay 0)
# ---------------------------------------------------------------------------

def session_zscore_params(sbp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std computed from non-zero timepoints only."""
    mean = np.zeros(N_CHANNELS, dtype=np.float64)
    std  = np.ones(N_CHANNELS,  dtype=np.float64)
    for c in range(N_CHANNELS):
        col     = sbp[:, c]
        nonzero = col[col != 0.0]
        if len(nonzero) > 1:
            mean[c] = nonzero.mean()
            std[c]  = nonzero.std()
    std = np.clip(std, 1e-8, None)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_normalize(sbp: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (sbp - mean[None, :]) / std[None, :]
    out[:, mean == 0] = 0.0   # keep permanently-zeroed channels at 0
    return out


def zscore_denormalize(sbp_z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return sbp_z * std[None, :] + mean[None, :]


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def get_validation_sessions(
    data_dir: str = DATA_DIR,
    n_hard: int   = 20,
    n_easy: int   = 8,
    seed: int     = 42,
) -> Tuple[List[str], List[str]]:
    """Chronological train/val split mimicking the test difficulty structure.

    - n_hard sessions: the last chronological training sessions (largest temporal gap)
    - n_easy sessions: randomly sampled from the earlier sessions

    Returns (train_session_ids, val_session_ids).
    """
    all_ids  = list_session_ids(data_dir, split='train')
    hard_val = all_ids[-n_hard:]
    remaining = all_ids[:-n_hard]

    rng      = np.random.RandomState(seed)
    easy_idx = rng.choice(len(remaining), size=n_easy, replace=False)
    easy_val = [remaining[i] for i in sorted(easy_idx)]

    val_ids   = set(hard_val + easy_val)
    train_ids = [s for s in all_ids if s not in val_ids]

    return train_ids, sorted(val_ids, key=lambda s: int(s[1:]))


# ---------------------------------------------------------------------------
# Local R² evaluation (mirrors kaggle_data/metric.py)
# ---------------------------------------------------------------------------

def compute_r2(pred: np.ndarray, true: np.ndarray) -> float:
    """R² for a single (session, position_channel) group."""
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-10)


def compute_r2_multi(
    results: List[Tuple[np.ndarray, np.ndarray, str]]
) -> float:
    """Mean R² across all (session, position_channel) groups.

    results: list of (pred_kin, true_kin, session_id)
             where kin arrays have shape (N, 2) — [index_pos, mrp_pos].
    """
    r2_values = []
    for pred_kin, true_kin, sid in results:
        for c in range(N_KIN):
            r2_values.append(compute_r2(pred_kin[:, c], true_kin[:, c]))
    return float(np.mean(r2_values)) if r2_values else 0.0


# ---------------------------------------------------------------------------
# Submission CSV building
# ---------------------------------------------------------------------------

def build_submission(
    predictions: Dict[str, np.ndarray],
    data_dir: str    = DATA_DIR,
    output_path: str = 'submission.csv',
) -> pd.DataFrame:
    """Build submission CSV from per-session prediction arrays.

    predictions: dict[session_id -> (N_bins, 2) array of [index_pos, mrp_pos]]
    Aligns with test_index.csv to produce the required submission format.
    """
    test_index = pd.read_csv(os.path.join(data_dir, 'test_index.csv'))

    index_pos_vals, mrp_pos_vals = [], []
    for _, row in test_index.iterrows():
        sid  = row['session_id']
        t    = int(row['time_bin'])
        pred = predictions[sid]
        index_pos_vals.append(float(np.clip(pred[t, 0], 0.0, 1.0)))
        mrp_pos_vals.append(  float(np.clip(pred[t, 1], 0.0, 1.0)))

    submission = pd.DataFrame({
        'sample_id': test_index['sample_id'],
        'index_pos': index_pos_vals,
        'mrp_pos':   mrp_pos_vals,
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission)} rows)")
    return submission


# ---------------------------------------------------------------------------
# Phase 1 data loading (for cross-phase augmentation)
# ---------------------------------------------------------------------------

def list_p1_session_ids(p1_data_dir: str = P1_DATA_DIR, split: str = 'train') -> List[str]:
    """List Phase 1 session IDs (S-prefix) sorted by number.

    Works for both train (_sbp.npy) and test (_sbp_masked.npy) splits.
    Only returns sessions that also have a kinematics file.
    """
    split_dir = os.path.join(p1_data_dir, split)
    files = os.listdir(split_dir)
    # Collect all session IDs that have kinematics (train always does; test also has it published)
    sids = sorted(
        set(f.split('_')[0] for f in files if f.endswith('_kinematics.npy')),
        key=lambda s: int(s[1:])
    )
    return sids


def load_p1_session_as_p2(
    p1_data_dir: str,
    session_id: str,
    n_dropout: int = 28,
    split: str = 'train',
) -> dict:
    """Load a Phase 1 session and simulate Phase 2-style permanent channel dropout.

    Phase 1 train sessions have all 96 channels active; Phase 2 has ~28 channels
    permanently zeroed per session.  This function simulates that dropout so Phase 1
    data can be mixed into Phase 2 training.  The dropout mask is deterministic per
    session_id so datasets are reproducible.

    Returns a dict with the same keys as load_session():
      sbp, kinematics, start_bins, end_bins, n_trials, active_mask, dropout_ind
    """
    split_dir = os.path.join(p1_data_dir, split)

    if split == 'test':
        # Phase 1 test has sbp_masked (with random per-timebin masking).
        # Use it as-is; the random zeros will look like extra dropout noise.
        sbp_file = os.path.join(split_dir, f'{session_id}_sbp_masked.npy')
    else:
        sbp_file = os.path.join(split_dir, f'{session_id}_sbp.npy')

    sbp = np.load(sbp_file).astype(np.float32)
    kin = np.load(os.path.join(split_dir, f'{session_id}_kinematics.npy')).astype(np.float32)
    tri = np.load(os.path.join(split_dir, f'{session_id}_trial_info.npz'))

    # Deterministic per-session dropout (seed = hash of session_id)
    seed = int.from_bytes(session_id.encode(), 'little') % (2**31)
    rng = np.random.RandomState(seed)
    dropout_channels = rng.choice(N_CHANNELS, size=n_dropout, replace=False)
    sbp[:, dropout_channels] = 0.0

    active_mask  = (sbp != 0).any(axis=0)
    dropout_ind  = (~active_mask).astype(np.float32)

    return {
        'sbp':          sbp,
        'kinematics':   kin,
        'start_bins':   tri['start_bins'],
        'end_bins':     tri['end_bins'],
        'n_trials':     int(tri['n_trials']),
        'active_mask':  active_mask,
        'dropout_ind':  dropout_ind,
    }

"""
Ridge baseline for masked SBP reconstruction.

Interface intentionally matches the original caller:
    ridge_predict_session(sbp_masked, mask, alpha=...)

Implementation:
  - For each target channel c, fit ridge using the other 95 channels as features
    on rows where channel c is observed (mask[:, c] == False).
  - Predict all rows, then fill only masked locations for channel c.
  - No kinematics dependency.
"""

from __future__ import annotations

import numpy as np


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    # Do not regularize bias term.
    xtx = X.T @ X
    reg = np.eye(xtx.shape[0], dtype=np.float64) * float(alpha)
    reg[-1, -1] = 0.0
    a = xtx + reg
    b = X.T @ y
    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(a, b, rcond=None)[0]
    return w


def ridge_predict_session(
    sbp_masked: np.ndarray,
    mask: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Reconstruct masked SBP with per-channel ridge from other channels.

    Args:
      sbp_masked: (N, C), masked entries set to 0 in the input.
      mask: (N, C), True where target value is hidden and must be predicted.
      alpha: L2 regularization strength.
    """
    x_raw = sbp_masked.astype(np.float64, copy=False)
    m = mask.astype(bool, copy=False)
    n, c = x_raw.shape
    pred = x_raw.copy()

    # Impute masked feature entries using per-channel observed means so that
    # feature zeros do not encode "missing" implicitly.
    feat = x_raw.copy()
    for j in range(c):
        obs_j = ~m[:, j]
        mu_j = float(x_raw[obs_j, j].mean()) if np.any(obs_j) else 0.0
        feat[m[:, j], j] = mu_j

    for ch in range(c):
        obs = ~m[:, ch]
        miss = m[:, ch]
        if not np.any(miss):
            continue
        if not np.any(obs):
            pred[miss, ch] = 0.0
            continue

        idx = np.arange(c) != ch
        X_all = feat[:, idx]
        Xtr = X_all[obs]
        ytr = x_raw[obs, ch]

        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        Xtr_n = (Xtr - mu) / sd
        Xall_n = (X_all - mu) / sd

        Xtr_b = np.concatenate([Xtr_n, np.ones((Xtr_n.shape[0], 1), dtype=np.float64)], axis=1)
        Xall_b = np.concatenate([Xall_n, np.ones((Xall_n.shape[0], 1), dtype=np.float64)], axis=1)

        w = _fit_ridge(Xtr_b, ytr, alpha=alpha)
        yhat = Xall_b @ w
        pred[miss, ch] = yhat[miss]

    return pred.astype(np.float32)

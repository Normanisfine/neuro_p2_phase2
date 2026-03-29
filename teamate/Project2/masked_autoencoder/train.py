"""
Method 4: Masked Autoencoder (Transformer)
===========================================
Transformer encoder over temporal sequences of SBP bins within trials.
Dynamic masking re-randomized every epoch. Warmup + cosine LR.

Usage:
    python train.py --quick                     # sanity check
    python train.py --epochs 40 --wandb         # standard run
    python train.py --lr 3e-4 --dropout 0.15    # tuning
"""

import sys
import os
import argparse
import math
import time
import json
import random
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_utils import (
    DATA_DIR, N_CHANNELS, N_MASKED_CHANNELS,
    load_session, load_metadata, simulate_masking,
    compute_nmse_multi, get_validation_sessions, get_val_difficulty_labels,
    session_zscore_params, zscore_normalize, zscore_denormalize,
)

WANDB_PROJECT = "neuro-p2-phase1"

MAX_SEQ_LEN = 128
INPUT_DIM = N_CHANNELS + N_CHANNELS + 4  # legacy default (mask indicator mode)


def input_dim_for_mode(use_mask_token: bool) -> int:
    return (N_CHANNELS + 4) if use_mask_token else (N_CHANNELS + N_CHANNELS + 4)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model (unchanged architecture — compatible with existing checkpoints)
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim=None, d_model=256, nhead=4,
                 num_layers=4, d_ff=512, dropout=0.1,
                 use_mask_token=False):
        super().__init__()
        self.use_mask_token = bool(use_mask_token)
        if input_dim is None:
            input_dim = input_dim_for_mode(self.use_mask_token)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPE(d_model, max_len=1024)
        self.dropout = nn.Dropout(dropout)
        if self.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(N_CHANNELS))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, N_CHANNELS),
        )

    def forward(self, x, padding_mask=None, mask_indicator=None):
        if self.use_mask_token:
            if mask_indicator is None:
                raise ValueError("mask_indicator is required when use_mask_token=True")
            m = mask_indicator.bool()
            sbp_in = x[..., :N_CHANNELS]
            rest = x[..., N_CHANNELS:]
            token = self.mask_token.view(1, 1, -1)
            sbp_in = torch.where(m, token, sbp_in)
            x = torch.cat([sbp_in, rest], dim=-1)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.dropout(h)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.output_head(h)


class ExponentialMovingAverage:
    """Simple EMA tracker over model state_dict tensors."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if torch.is_floating_point(v):
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow

    @contextmanager
    def apply_to(self, model: nn.Module):
        backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)


# ---------------------------------------------------------------------------
# Dataset — dynamic masking (re-randomized every epoch)
# ---------------------------------------------------------------------------

class DynamicTrialDataset(Dataset):
    def __init__(self, session_ids, data_dir, max_seq_len=MAX_SEQ_LEN,
                 n_masked_ch=N_MASKED_CHANNELS, seed=42,
                 use_mask_token=False, include_test_unmasked_trials=False):
        self.max_seq_len = max_seq_len
        self.n_masked_ch = n_masked_ch
        self.epoch_seed = seed
        self.use_mask_token = bool(use_mask_token)
        self.include_test_unmasked_trials = bool(include_test_unmasked_trials)

        self.trials = []
        self.source_counts = {"train_trials": 0, "test_unmasked_trials": 0}
        for sid in session_ids:
            sess = load_session(data_dir, sid, is_test=False)
            sbp = sess['sbp']
            kin = sess['kinematics']
            mean, std = session_zscore_params(sbp)
            sbp_z = zscore_normalize(sbp, mean, std)

            for ti in range(sess['n_trials']):
                s, e = int(sess['start_bins'][ti]), int(sess['end_bins'][ti])
                if e - s < 5:
                    continue
                self.trials.append((
                    sbp_z[s:e].astype(np.float32),
                    kin[s:e].astype(np.float32),
                    e - s,
                ))
                self.source_counts["train_trials"] += 1

        if self.include_test_unmasked_trials:
            meta = load_metadata(data_dir)
            test_ids = meta[meta['split'] == 'test']['session_id'].tolist()
            for sid in test_ids:
                sess = load_session(data_dir, sid, is_test=True)
                sbp_m = sess['sbp_masked']
                kin = sess['kinematics']
                full_mask = sess['mask']
                mean, std = session_zscore_params(sbp_m, full_mask)
                sbp_z = zscore_normalize(sbp_m, mean, std)
                for ti in range(sess['n_trials']):
                    s, e = int(sess['start_bins'][ti]), int(sess['end_bins'][ti])
                    if e - s < 5:
                        continue
                    # Use only fully observed trials from official test split.
                    if full_mask[s:e].any():
                        continue
                    self.trials.append((
                        sbp_z[s:e].astype(np.float32),
                        kin[s:e].astype(np.float32),
                        e - s,
                    ))
                    self.source_counts["test_unmasked_trials"] += 1

    def set_epoch(self, epoch):
        self.epoch_seed = epoch * 1000 + 42

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        sbp, kin, tlen = self.trials[idx]
        T = self.max_seq_len
        rng = np.random.RandomState(self.epoch_seed + idx)

        if tlen > T:
            off = rng.randint(0, tlen - T)
            sbp, kin = sbp[off:off+T], kin[off:off+T]
            sl = T
        else:
            sl = tlen

        mind = np.zeros((sl, N_CHANNELS), dtype=np.float32)
        for t in range(sl):
            ch = rng.choice(N_CHANNELS, size=self.n_masked_ch, replace=False)
            mind[t, ch] = 1.0

        sbp_in = sbp[:sl].copy()
        if not self.use_mask_token:
            sbp_in[mind.astype(bool)] = 0.0
        target = sbp[:sl].copy()

        def pad(a):
            if a.shape[0] >= T:
                return a[:T]
            return np.pad(a, [(0, T - a.shape[0])] + [(0, 0)] * (a.ndim - 1))

        if self.use_mask_token:
            feat = pad(np.concatenate([sbp_in, kin[:sl]], axis=1))
        else:
            feat = pad(np.concatenate([sbp_in, mind, kin[:sl]], axis=1))
        target, mind = pad(target), pad(mind)
        pmask = np.ones(T, dtype=bool)
        pmask[:sl] = False
        return (torch.from_numpy(feat), torch.from_numpy(target),
                torch.from_numpy(mind), torch.from_numpy(pmask))


# ---------------------------------------------------------------------------
# Scheduler — warmup + cosine
# ---------------------------------------------------------------------------

def make_scheduler(optimizer, warmup, total, min_ratio=0.01):
    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        p = (ep - warmup) / max(total - warmup, 1)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, recon_weight=0.0, loss_mode="mse",
                ema=None):
    model.train()
    total_loss = 0
    n = 0
    for feat, tgt, tmask, pmask in loader:
        feat, tgt = feat.to(device), tgt.to(device)
        tmask, pmask = tmask.to(device), pmask.to(device)

        pred = model(feat, padding_mask=pmask, mask_indicator=tmask)
        diff2 = (pred - tgt) ** 2

        masked_loss = (diff2 * tmask).sum() / tmask.sum().clamp(min=1)
        if loss_mode == "nmse_equiv":
            valid = (~pmask).unsqueeze(-1).float()
            mask = tmask.float()
            mask_count = mask.sum(dim=1)  # [B, C]
            mse_c = (diff2 * mask).sum(dim=1) / mask_count.clamp(min=1.0)

            mean_c = (tgt * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            var_num = ((tgt - mean_c.unsqueeze(1)) ** 2 * valid).sum(dim=1)
            var_den = valid.sum(dim=1).clamp(min=1.0)
            var_c = (var_num / var_den).clamp(min=1e-6)

            nmse_c = mse_c / var_c
            active = (mask_count > 0).float()
            nmse_per_item = (nmse_c * active).sum(dim=1) / active.sum(dim=1).clamp(min=1.0)
            main_loss = nmse_per_item.mean()
        else:
            main_loss = masked_loss

        if recon_weight > 0:
            valid = (~pmask).unsqueeze(-1).float()
            recon_loss = (diff2 * valid).sum() / valid.sum().clamp(min=1)
            loss = main_loss + recon_weight * recon_loss
        else:
            loss = main_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += masked_loss.item()
        n += 1
    return total_loss / max(n, 1)


def validate_model(model, val_ids, data_dir, device, max_seq_len=MAX_SEQ_LEN):
    model.eval()
    results = []
    with torch.no_grad():
        for sid in val_ids:
            sess = load_session(data_dir, sid, is_test=False)
            sbp = sess['sbp']
            kin = sess['kinematics']
            mean, std = session_zscore_params(sbp)
            sbp_z = zscore_normalize(sbp, mean, std)
            sim_mask = simulate_masking(sbp, sess['start_bins'], sess['end_bins'],
                                        sess['n_trials'], seed=42)
            sbp_mz = sbp_z.copy()
            sbp_mz[sim_mask] = 0.0
            pred_z = sbp_z.copy()

            for ti in range(sess['n_trials']):
                s, e = int(sess['start_bins'][ti]), int(sess['end_bins'][ti])
                tmask = sim_mask[s:e]
                if not tmask.any():
                    continue
                tlen = e - s
                mind = tmask.astype(np.float32)

                for cs in range(0, tlen, max_seq_len):
                    ce = min(cs + max_seq_len, tlen)
                    cl = ce - cs
                    mind_chunk = mind[cs:ce]
                    if model.use_mask_token:
                        f = np.concatenate([sbp_mz[s+cs:s+ce], kin[s+cs:s+ce]], axis=1).astype(np.float32)
                    else:
                        f = np.concatenate([sbp_mz[s+cs:s+ce], mind_chunk,
                                            kin[s+cs:s+ce]], axis=1).astype(np.float32)
                    T = max_seq_len
                    if cl < T:
                        f = np.pad(f, ((0, T - cl), (0, 0)))
                    ft = torch.from_numpy(f).unsqueeze(0).to(device)
                    mt = torch.from_numpy(np.pad(mind_chunk, ((0, T - cl), (0, 0))) if cl < T else mind_chunk).unsqueeze(0).to(device)
                    pm = torch.ones(1, T, dtype=torch.bool, device=device)
                    pm[0, :cl] = False
                    out = model(ft, padding_mask=pm, mask_indicator=mt).cpu().numpy()[0, :cl]
                    for lt in range(cl):
                        t = s + cs + lt
                        mc = np.where(sim_mask[t])[0]
                        if len(mc):
                            pred_z[t, mc] = out[lt, mc]

            pred = zscore_denormalize(pred_z, mean, std)
            results.append((sbp, pred, sim_mask, sid))
    return compute_nmse_multi(results)


def validate_model_eval(model, ema, val_ids, data_dir, device, max_seq_len=MAX_SEQ_LEN):
    if ema is None:
        return validate_model(model, val_ids, data_dir, device, max_seq_len=max_seq_len)
    with ema.apply_to(model):
        return validate_model(model, val_ids, data_dir, device, max_seq_len=max_seq_len)


def _predict_masked_trial(model, sbp_in_z, kin, trial_mask, device, max_seq_len):
    """Predict masked channels for one trial in z-space."""
    pred_z = sbp_in_z.copy()
    tlen = sbp_in_z.shape[0]
    with torch.no_grad():
        for cs in range(0, tlen, max_seq_len):
            ce = min(cs + max_seq_len, tlen)
            cl = ce - cs
            mask_chunk = trial_mask[cs:ce].astype(np.float32)
            if model.use_mask_token:
                feat = np.concatenate([sbp_in_z[cs:ce], kin[cs:ce]], axis=1).astype(np.float32)
            else:
                feat = np.concatenate([sbp_in_z[cs:ce], mask_chunk, kin[cs:ce]], axis=1).astype(np.float32)
            if cl < max_seq_len:
                feat = np.pad(feat, ((0, max_seq_len - cl), (0, 0)), mode='constant')
                mask_chunk = np.pad(mask_chunk, ((0, max_seq_len - cl), (0, 0)), mode='constant')
            ft = torch.from_numpy(feat).unsqueeze(0).to(device)
            mt = torch.from_numpy(mask_chunk).unsqueeze(0).to(device)
            pm = torch.ones(1, max_seq_len, dtype=torch.bool, device=device)
            pm[0, :cl] = False
            out = model(ft, padding_mask=pm, mask_indicator=mt).cpu().numpy()[0, :cl]
            for lt in range(cl):
                mc = np.where(trial_mask[cs + lt])[0]
                if len(mc):
                    pred_z[cs + lt, mc] = out[lt, mc]
    return pred_z


def _pick_masked_trials(mask, start_bins, end_bins, n_sections):
    candidates = []
    for ti in range(len(start_bins)):
        s, e = int(start_bins[ti]), int(end_bins[ti])
        if mask[s:e].any():
            candidates.append(ti)
    if not candidates:
        candidates = list(range(len(start_bins)))
    if not candidates:
        return []
    if len(candidates) >= n_sections:
        idx = np.linspace(0, len(candidates) - 1, n_sections).round().astype(int)
        return [candidates[i] for i in idx]
    out = candidates[:]
    while len(out) < n_sections:
        out.append(candidates[len(out) % len(candidates)])
    return out


def _center_crop_time(arr, max_bins):
    if max_bins <= 0 or arr.shape[0] <= max_bins:
        return arr
    s = (arr.shape[0] - max_bins) // 2
    e = s + max_bins
    return arr[s:e]


def _collect_sections_known_gt(model, data_dir, sid, device, epoch, n_sections,
                               max_seq_len, plot_bins):
    """
    Build sample sections with known GT:
      - input (masked), mask, gt, pred
    """
    sess = load_session(data_dir, sid, is_test=False)
    sbp = sess['sbp']
    kin = sess['kinematics']
    start_bins, end_bins = sess['start_bins'], sess['end_bins']
    sim_mask = simulate_masking(
        sbp, start_bins, end_bins, sess['n_trials'], seed=42 + int(epoch)
    )
    mean, std = session_zscore_params(sbp)
    sbp_z = zscore_normalize(sbp, mean, std)

    sbp_in_raw = sbp.copy()
    sbp_in_raw[sim_mask] = 0.0

    sections = []
    for ti in _pick_masked_trials(sim_mask, start_bins, end_bins, n_sections):
        s, e = int(start_bins[ti]), int(end_bins[ti])
        tmask = sim_mask[s:e]
        sbp_in_z = sbp_z[s:e].copy()
        if not model.use_mask_token:
            sbp_in_z[tmask] = 0.0
        pred_z = _predict_masked_trial(
            model, sbp_in_z, kin[s:e], tmask, device, max_seq_len=max_seq_len
        )
        pred = zscore_denormalize(pred_z, mean, std)

        sec = {
            "session_id": sid,
            "trial_idx": int(ti),
            "input": _center_crop_time(sbp_in_raw[s:e], plot_bins),
            "mask": _center_crop_time(tmask, plot_bins),
            "gt": _center_crop_time(sbp[s:e], plot_bins),
            "pred": _center_crop_time(pred, plot_bins),
        }
        sections.append(sec)
    return sections


def _collect_sections_no_gt(model, data_dir, sid, device, seed, n_sections,
                            max_seq_len, plot_bins, is_test):
    """
    Build sample sections without GT:
      - input (masked), mask, pred
    """
    sess = load_session(data_dir, sid, is_test=is_test)
    kin = sess['kinematics']
    start_bins, end_bins = sess['start_bins'], sess['end_bins']
    if is_test:
        sbp_obs = sess['sbp_masked'].astype(np.float32)
        mask = sess['mask'].astype(bool)
        mean, std = session_zscore_params(sbp_obs, mask)
    else:
        sbp = sess['sbp']
        mask = simulate_masking(sbp, start_bins, end_bins, sess['n_trials'], seed=int(seed))
        sbp_obs = sbp.copy()
        sbp_obs[mask] = 0.0
        mean, std = session_zscore_params(sbp)

    sbp_z = zscore_normalize(sbp_obs, mean, std)
    if not model.use_mask_token:
        sbp_z[mask] = 0.0

    sections = []
    for ti in _pick_masked_trials(mask, start_bins, end_bins, n_sections):
        s, e = int(start_bins[ti]), int(end_bins[ti])
        tmask = mask[s:e]
        sbp_in_z = sbp_z[s:e].copy()
        pred_z = _predict_masked_trial(
            model, sbp_in_z, kin[s:e], tmask, device, max_seq_len=max_seq_len
        )
        pred = zscore_denormalize(pred_z, mean, std)
        sec = {
            "session_id": sid,
            "trial_idx": int(ti),
            "input": _center_crop_time(sbp_obs[s:e], plot_bins),
            "mask": _center_crop_time(tmask, plot_bins),
            "gt": None,
            "pred": _center_crop_time(pred, plot_bins),
        }
        sections.append(sec)
    return sections


def _robust_limits(x):
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(vals, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        m = float(np.mean(vals))
        s = float(np.std(vals)) + 1e-6
        lo, hi = m - 2.5 * s, m + 2.5 * s
    return float(lo), float(hi)


def _plot_sections_figure(sections, split_name, epoch, out_path, include_gt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    n = len(sections)
    if n == 0:
        return None
    ncols = 5 if include_gt else 3
    fig, axes = plt.subplots(
        n, ncols,
        figsize=(4.2 * ncols, 2.25 * n),
        dpi=150,
        constrained_layout=True,
    )
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    sig_cmap = plt.cm.viridis.copy()
    sig_cmap.set_bad(color="#f2f2f2")
    err_cmap = plt.cm.magma.copy()
    err_cmap.set_bad(color="#f2f2f2")
    mask_cmap = ListedColormap(["#f7f7f7", "#d62728"])

    for r, sec in enumerate(sections):
        inp = sec["input"]
        mask = sec["mask"].astype(bool)
        pred = sec["pred"]
        gt = sec["gt"]

        if gt is not None:
            vmin, vmax = _robust_limits(gt)
        else:
            observed = inp[~mask]
            if observed.size < 10:
                observed = inp.reshape(-1)
            vmin, vmax = _robust_limits(observed)

        ax = axes[r, 0]
        ax.imshow(inp.T, origin='lower', aspect='auto', cmap=sig_cmap, vmin=vmin, vmax=vmax)
        overlay = np.zeros((mask.T.shape[0], mask.T.shape[1], 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 3] = mask.T.astype(np.float32) * 0.28
        ax.imshow(overlay, origin='lower', aspect='auto')
        ax.set_title(
            f"{split_name} | {sec['session_id']} T{sec['trial_idx']}\nInput (mask in red)",
            fontsize=9,
        )
        ax.set_ylabel("Channel")

        ax = axes[r, 1]
        ax.imshow(mask.T.astype(np.float32), origin='lower', aspect='auto',
                  cmap=mask_cmap, vmin=0, vmax=1)
        ax.set_title("Mask map", fontsize=9)

        if include_gt:
            gt_m = np.where(mask, gt, np.nan)
            pred_m = np.where(mask, pred, np.nan)
            err_m = np.where(mask, np.abs(pred - gt), np.nan)
            evmin, evmax = _robust_limits(err_m)

            ax = axes[r, 2]
            ax.imshow(gt_m.T, origin='lower', aspect='auto', cmap=sig_cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"GT @ masked\n[{vmin:.2f}, {vmax:.2f}]", fontsize=9)

            ax = axes[r, 3]
            ax.imshow(pred_m.T, origin='lower', aspect='auto', cmap=sig_cmap, vmin=vmin, vmax=vmax)
            ax.set_title("Pred @ masked", fontsize=9)

            ax = axes[r, 4]
            ax.imshow(err_m.T, origin='lower', aspect='auto', cmap=err_cmap, vmin=evmin, vmax=evmax)
            ax.set_title(f"|Pred-GT| @ masked\n[0, {evmax:.2f}]", fontsize=9)
        else:
            pred_m = np.where(mask, pred, np.nan)
            ax = axes[r, 2]
            ax.imshow(pred_m.T, origin='lower', aspect='auto', cmap=sig_cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"Pred @ masked\n[{vmin:.2f}, {vmax:.2f}]", fontsize=9)

        for c in range(ncols):
            a = axes[r, c]
            if r == n - 1:
                a.set_xlabel("Time Bin (trial-local)")
            else:
                a.set_xticklabels([])
            a.tick_params(axis='both', labelsize=7, length=2)

    fig.suptitle(
        f"Epoch {epoch} Sample Sections ({split_name})",
        fontsize=12,
        fontweight='bold',
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def log_sample_sections(model, data_dir, train_ids, val_ids, test_ids,
                        device, epoch, max_seq_len, n_sections, plot_bins,
                        out_dir):
    """
    Create three visualization figures every logging interval:
      - training (GT + Pred on masked)
      - val (held-out train session: GT + Pred on masked)
      - testing (official test session: no GT + Pred on masked)
    """
    slot = max(epoch // 10 - 1, 0)
    train_sid = train_ids[slot % len(train_ids)]
    val_sid = val_ids[slot % len(val_ids)]
    test_sid = test_ids[slot % len(test_ids)] if test_ids else val_sid

    model.eval()
    tr_secs = _collect_sections_known_gt(
        model, data_dir, train_sid, device, epoch, n_sections, max_seq_len, plot_bins
    )
    val_secs = _collect_sections_known_gt(
        model, data_dir, val_sid, device, epoch + 509, n_sections, max_seq_len, plot_bins
    )
    te_secs = _collect_sections_no_gt(
        model, data_dir, test_sid, device, epoch + 911, n_sections, max_seq_len, plot_bins,
        is_test=True
    )

    out_train = os.path.join(out_dir, f"epoch_{epoch:03d}_training.png")
    out_val = os.path.join(out_dir, f"epoch_{epoch:03d}_val.png")
    out_test = os.path.join(out_dir, f"epoch_{epoch:03d}_testing.png")

    p1 = _plot_sections_figure(tr_secs, "training", epoch, out_train, include_gt=True)
    p2 = _plot_sections_figure(val_secs, "val", epoch, out_val, include_gt=True)
    p3 = _plot_sections_figure(te_secs, "testing", epoch, out_test, include_gt=False)
    return {"training": p1, "val": p2, "testing": p3}


def main():
    pa = argparse.ArgumentParser(description='Masked Autoencoder training')
    pa.add_argument('--data-dir', type=str, default=DATA_DIR)
    pa.add_argument('--epochs', type=int, default=40)
    pa.add_argument('--batch-size', type=int, default=64)
    pa.add_argument('--lr', type=float, default=5e-4)
    pa.add_argument('--warmup-epochs', type=int, default=5)
    pa.add_argument('--dropout', type=float, default=0.1)
    pa.add_argument('--recon-weight', type=float, default=0.0,
                    help='Auxiliary reconstruction loss on all channels')
    pa.add_argument('--loss-mode', type=str, default='mse',
                    choices=['mse', 'nmse_equiv'],
                    help='Training loss: masked MSE or differentiable NMSE-equivalent surrogate')
    pa.add_argument('--use-mask-token', action='store_true',
                    help='Use learnable mask token instead of zeroing masked channels and feeding mask indicator')
    pa.add_argument('--n-masked-channels', type=int, default=N_MASKED_CHANNELS,
                    help='Number of channels masked at each time step during training/TTA-style masking')
    pa.add_argument('--include-test-unmasked-trials', action='store_true',
                    help='Include official test split trials with no masked points into pretraining data')
    pa.add_argument('--d-model', type=int, default=256)
    pa.add_argument('--nhead', type=int, default=4)
    pa.add_argument('--num-layers', type=int, default=4)
    pa.add_argument('--d-ff', type=int, default=512)
    pa.add_argument('--max-seq-len', type=int, default=MAX_SEQ_LEN)
    pa.add_argument('--save-every', type=int, default=10)
    pa.add_argument('--plot-every', type=int, default=10,
                    help='Log sample section plots every N epochs')
    pa.add_argument('--sample-sections', type=int, default=5,
                    help='Number of trial sections to plot per split')
    pa.add_argument('--sample-plot-bins', type=int, default=160,
                    help='Max time bins shown per trial section')
    pa.add_argument('--sample-plot-dir', type=str, default='sample_plots')
    pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--use-ema', action='store_true',
                    help='Track exponential moving average weights and use EMA for validation/best checkpoint')
    pa.add_argument('--ema-decay', type=float, default=0.999)
    pa.add_argument('--quick', action='store_true')
    pa.add_argument('--wandb', action='store_true')
    pa.add_argument('--wandb-project', type=str, default=WANDB_PROJECT)
    pa.add_argument('--wandb-name', type=str, default=None)
    args = pa.parse_args()
    if args.n_masked_channels <= 0 or args.n_masked_channels > N_CHANNELS:
        raise ValueError(f"--n-masked-channels must be in [1, {N_CHANNELS}]")
    if not (0.0 < args.ema_decay < 1.0):
        raise ValueError("--ema-decay must be in (0,1)")

    set_global_seed(args.seed)

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        name = args.wandb_name or (
            f"mae_lr{args.lr}_do{args.dropout}_ep{args.epochs}_rw{args.recon_weight}"
        )
        wandb.init(project=args.wandb_project, name=name, config=vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ids, val_ids = get_validation_sessions(args.data_dir)
    meta = load_metadata(args.data_dir)
    test_ids = meta[meta['split'] == 'test']['session_id'].tolist()
    if args.quick:
        train_ids, val_ids = train_ids[:5], val_ids[:3]
        args.epochs = 2

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    print(f"Epochs: {args.epochs}  BS: {args.batch_size}  LR: {args.lr}  "
          f"Dropout: {args.dropout}  Warmup: {args.warmup_epochs}  "
          f"recon_w: {args.recon_weight}  loss_mode: {args.loss_mode}  "
          f"mask_token: {args.use_mask_token}  n_masked_channels: {args.n_masked_channels}  "
          f"include_test_unmasked: {args.include_test_unmasked_trials}  "
          f"seed: {args.seed}  use_ema: {args.use_ema}  ema_decay: {args.ema_decay}")

    print("Building dataset...")
    t0 = time.time()
    ds = DynamicTrialDataset(train_ids, args.data_dir,
                             max_seq_len=args.max_seq_len,
                             n_masked_ch=args.n_masked_channels,
                             seed=args.seed,
                             use_mask_token=args.use_mask_token,
                             include_test_unmasked_trials=args.include_test_unmasked_trials)
    print(f"  {len(ds)} trials ({time.time()-t0:.1f}s)")
    print(f"  source_counts={ds.source_counts}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = MaskedAutoencoder(
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, d_ff=args.d_ff, dropout=args.dropout,
        use_mask_token=args.use_mask_token,
    ).to(device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {npar:,}")

    if use_wandb:
        wandb.config.update({"n_params": npar, "n_examples": len(ds)})

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = make_scheduler(opt, args.warmup_epochs, args.epochs)
    ema = ExponentialMovingAverage(model, decay=args.ema_decay) if args.use_ema else None

    def save_ckpt(path, ep, vnmse, best, model_state_dict=None):
        state = model_state_dict if model_state_dict is not None else model.state_dict()
        torch.save({'epoch': ep, 'model_state_dict': state,
                     'optimizer_state_dict': opt.state_dict(),
                     'scheduler_state_dict': sched.state_dict(),
                     'val_nmse': vnmse, 'best_nmse': best}, path)

    best = float('inf')
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        ds.set_epoch(ep)
        loss = train_epoch(model, loader, opt, device,
                           recon_weight=args.recon_weight, loss_mode=args.loss_mode,
                           ema=ema)
        sched.step()
        vnmse = validate_model_eval(model, ema, val_ids[:3], args.data_dir, device,
                                    max_seq_len=args.max_seq_len)
        dt = time.time() - t0
        is_best = vnmse < best
        if is_best:
            best = vnmse
            if ema is not None:
                save_ckpt('best_model.pt', ep, vnmse, best, model_state_dict=ema.state_dict())
                save_ckpt('best_model_ema.pt', ep, vnmse, best, model_state_dict=ema.state_dict())
                save_ckpt('best_model_raw.pt', ep, vnmse, best, model_state_dict=model.state_dict())
            else:
                save_ckpt('best_model.pt', ep, vnmse, best)
        if args.save_every and ep % args.save_every == 0:
            save_ckpt(f'checkpoint_epoch{ep}.pt', ep, vnmse, best)
        save_ckpt('last_model.pt', ep, vnmse, best)

        lr_now = opt.param_groups[0]['lr']
        print(f"Ep {ep:3d}/{args.epochs} | loss={loss:.4f} | "
              f"val={vnmse:.4f} {'*' if is_best else ''} | "
              f"lr={lr_now:.2e} | {dt:.0f}s")
        if use_wandb:
            wandb.log({"epoch": ep, "train/loss": loss, "val/nmse": vnmse,
                        "val/best": best, "lr": lr_now, "time_s": dt})
            if args.plot_every > 0 and ep % args.plot_every == 0:
                try:
                    if ema is None:
                        plot_paths = log_sample_sections(
                            model=model,
                            data_dir=args.data_dir,
                            train_ids=train_ids,
                            val_ids=val_ids,
                            test_ids=test_ids,
                            device=device,
                            epoch=ep,
                            max_seq_len=args.max_seq_len,
                            n_sections=args.sample_sections,
                            plot_bins=args.sample_plot_bins,
                            out_dir=args.sample_plot_dir,
                        )
                    else:
                        with ema.apply_to(model):
                            plot_paths = log_sample_sections(
                                model=model,
                                data_dir=args.data_dir,
                                train_ids=train_ids,
                                val_ids=val_ids,
                                test_ids=test_ids,
                                device=device,
                                epoch=ep,
                                max_seq_len=args.max_seq_len,
                                n_sections=args.sample_sections,
                                plot_bins=args.sample_plot_bins,
                                out_dir=args.sample_plot_dir,
                            )
                    payload = {"epoch": ep}
                    if plot_paths.get("training"):
                        payload["samples/training"] = wandb.Image(plot_paths["training"])
                    if plot_paths.get("val"):
                        payload["samples/val"] = wandb.Image(plot_paths["val"])
                    if plot_paths.get("testing"):
                        payload["samples/testing"] = wandb.Image(plot_paths["testing"])
                    wandb.log(payload)
                except Exception as e:
                    print(f"[warn] sample plotting failed at epoch {ep}: {e}")

    print("\nFull validation with best model...")
    ckpt = torch.load('best_model.pt', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    full = validate_model(model, val_ids, args.data_dir, device,
                          max_seq_len=args.max_seq_len)
    print(f"Full val NMSE: {full:.4f} (best epoch {ckpt['epoch']})")

    cfg = {k: getattr(args, k) for k in
           ['d_model', 'nhead', 'num_layers', 'd_ff', 'max_seq_len',
            'dropout', 'recon_weight', 'warmup_epochs',
            'epochs', 'batch_size', 'lr',
            'plot_every', 'sample_sections', 'sample_plot_bins',
            'loss_mode', 'use_mask_token', 'n_masked_channels',
            'include_test_unmasked_trials',
            'seed', 'use_ema', 'ema_decay']}
    cfg.update({'best_val_nmse': best, 'best_epoch': ckpt['epoch'],
                'full_val_nmse': full,
                'eval_checkpoint_type': 'ema' if args.use_ema else 'raw'})
    with open('config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print("Saved config.json, best_model.pt, last_model.pt")

    if use_wandb:
        wandb.summary.update({"nmse/best_val": best, "nmse/full_val": full,
                              "best_epoch": ckpt['epoch']})
        wandb.finish()


if __name__ == '__main__':
    main()

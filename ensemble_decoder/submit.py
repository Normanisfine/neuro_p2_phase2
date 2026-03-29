"""
Ensemble Decoder Submission — Phase 2 Finger Position Decoding
==============================================================
Loads multiple Phase 2 checkpoints (each trained from a different Phase 1
initialisation) and averages their predictions per test session.

Why ensemble works: each checkpoint (user's Phase 1, teammate's m1/m2/m5)
captures different representations of the SBP→position relationship, giving
lower variance predictions that generalise better to shifted test sessions.

TTA note: self-supervised reconstruction TTA *hurts* Phase 2 performance
(plain 0.474 > tta5 0.422 on public leaderboard).  The default is no TTA.
Use --tta-epochs 1 or 2 only if you want to experiment.

Usage:
    # Plain ensemble (4 models, no TTA):
    python submit.py \\
        --checkpoints models/user_ft/best_model.pt \\
                      models/m1_ft/best_model.pt \\
                      models/m2_ft/best_model.pt \\
                      models/m5_ft/best_model.pt \\
        --output submission_ensemble.csv

    # With minimal TTA (use with caution):
    python submit.py \\
        --checkpoints models/user_ft/best_model.pt ... \\
        --tta-epochs 2 --tta-lr 7e-5 \\
        --output submission_ensemble_tta2.csv
"""

import sys
import os
import argparse
import copy
import json
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, '..'))
from data_utils import (
    DATA_DIR, N_CHANNELS, N_KIN,
    load_session, list_session_ids,
    session_zscore_params, zscore_normalize,
    build_submission,
)

# Reuse model definition from this package's train.py
from train import MAEFinetuneDecoder, MAX_SEQ_LEN

N_MASK_TTA = 30   # synthetic masking channels during TTA


# ---------------------------------------------------------------------------
# TTA dataset
# ---------------------------------------------------------------------------

class TTADataset(Dataset):
    """Self-supervised reconstruction dataset for a single test session."""
    def __init__(self, sbp_z, d_ind, start_bins, end_bins, n_trials,
                 max_seq_len=MAX_SEQ_LEN, seed=0):
        self.max_seq_len = max_seq_len
        self.d_ind       = d_ind
        self.seed        = seed
        self.active_ch   = np.where(d_ind == 0.0)[0]
        self.examples    = []
        for ti in range(n_trials):
            s, e = int(start_bins[ti]), int(end_bins[ti])
            if e - s < 5:
                continue
            self.examples.append((sbp_z[s:e].astype(np.float32), e - s))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sbp_t, tlen = self.examples[idx]
        T   = self.max_seq_len
        rng = np.random.RandomState(self.seed + idx * 7919)

        if tlen > T:
            off   = rng.randint(0, tlen - T)
            sbp_t = sbp_t[off:off + T]
            sl    = T
        else:
            sl = tlen

        n_mask = min(N_MASK_TTA, len(self.active_ch))
        tmask  = np.zeros((sl, N_CHANNELS), dtype=np.float32)
        for t in range(sl):
            ch = rng.choice(self.active_ch, size=n_mask, replace=False)
            tmask[t, ch] = 1.0

        sbp_in  = sbp_t[:sl].copy()
        sbp_in[tmask.astype(bool)] = 0.0
        combined = np.maximum(np.tile(self.d_ind, (sl, 1)), tmask)
        zeros4   = np.zeros((sl, 4), dtype=np.float32)

        def pad(a):
            if a.shape[0] >= T:
                return a[:T]
            return np.pad(a, [(0, T - a.shape[0])] + [(0, 0)] * (a.ndim - 1))

        feat   = pad(np.concatenate([sbp_in, combined, zeros4], axis=1))
        tgt    = pad(sbp_t[:sl])
        tmask_ = pad(tmask)
        pmask  = np.ones(T, dtype=bool)
        pmask[:sl] = False

        return (
            torch.from_numpy(feat),
            torch.from_numpy(tgt),
            torch.from_numpy(tmask_),
            torch.from_numpy(pmask),
        )


def tta_finetune(model, dataset, device, epochs=2, lr=7e-5, batch_size=32):
    """Minimal self-supervised TTA — freeze decode_head, adapt encoder + recon_head."""
    adapted = copy.deepcopy(model)
    adapted.train()
    for p in adapted.decode_head.parameters():
        p.requires_grad_(False)

    opt    = torch.optim.AdamW(
        [p for p in adapted.parameters() if p.requires_grad],
        lr=lr, weight_decay=0,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    for _ in range(epochs):
        for feat, tgt, tmask, pmask in loader:
            feat, tgt    = feat.to(device), tgt.to(device)
            tmask, pmask = tmask.to(device), pmask.to(device)
            pred = adapted.reconstruct(feat, padding_mask=pmask)
            loss = ((pred - tgt) ** 2 * tmask).sum() / tmask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), 1.0)
            opt.step()

    for p in adapted.decode_head.parameters():
        p.requires_grad_(True)
    adapted.eval()
    return adapted


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_session(model, sess, device, max_seq_len=MAX_SEQ_LEN):
    sbp    = sess['sbp']
    d_ind  = sess['dropout_ind']
    mean, std = session_zscore_params(sbp)
    sbp_z  = zscore_normalize(sbp, mean, std).astype(np.float32)
    N      = sbp_z.shape[0]
    d_t    = np.tile(d_ind, (N, 1))
    zeros4 = np.zeros((N, 4), dtype=np.float32)
    feat   = np.concatenate([sbp_z, d_t, zeros4], axis=1).astype(np.float32)

    pred_all = np.full((N, N_KIN), 0.5, dtype=np.float32)
    model.eval()

    for ti in range(sess['n_trials']):
        s, e = int(sess['start_bins'][ti]), int(sess['end_bins'][ti])
        tlen = e - s
        for cs in range(0, tlen, max_seq_len):
            ce  = min(cs + max_seq_len, tlen)
            cl  = ce - cs
            T   = max_seq_len
            ch  = feat[s + cs:s + ce]
            if cl < T:
                ch = np.pad(ch, ((0, T - cl), (0, 0)))
            ft  = torch.from_numpy(ch).unsqueeze(0).to(device)
            pm  = torch.ones(1, T, dtype=torch.bool, device=device)
            pm[0, :cl] = False
            out = model(ft, padding_mask=pm).cpu().numpy()[0, :cl]
            pred_all[s + cs:s + ce] = out

    return np.clip(pred_all, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path, device):
    """Load a trained Phase 2 MAEFinetuneDecoder from a checkpoint."""
    ckpt_dir    = os.path.dirname(os.path.abspath(ckpt_path))
    config_path = os.path.join(ckpt_dir, 'config.json')

    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        model = MAEFinetuneDecoder(
            d_model     = cfg.get('d_model',     256),
            nhead       = cfg.get('nhead',        4),
            num_layers  = cfg.get('num_layers',   4),
            d_ff        = cfg.get('d_ff',         512),
            dropout     = cfg.get('dropout',      0.1),
        )
        max_seq_len = cfg.get('max_seq_len', MAX_SEQ_LEN)
    else:
        model       = MAEFinetuneDecoder()
        max_seq_len = MAX_SEQ_LEN

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        val_r2 = ckpt.get('val_r2', '?')
        ep     = ckpt.get('epoch',  '?')
        if isinstance(val_r2, float):
            print(f"  {ckpt_path}  (epoch={ep}, val_r2={val_r2:.4f})")
        else:
            print(f"  {ckpt_path}  (epoch={ep}, val_r2={val_r2})")
    else:
        model.load_state_dict(ckpt)
        print(f"  {ckpt_path}")

    return model.to(device).eval(), max_seq_len


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',    type=str,   default=DATA_DIR)
    parser.add_argument('--checkpoints', nargs='+',  required=True,
                        help='Phase 2 checkpoint paths to ensemble (≥2 recommended)')
    parser.add_argument('--output',      type=str,   default='submission_ensemble.csv')
    parser.add_argument('--tta-epochs',  type=int,   default=0,
                        help='Self-supervised TTA epochs per session per model (0=off, '
                             'WARNING: TTA hurt Phase 2 performance in experiments)')
    parser.add_argument('--tta-lr',      type=float, default=7e-5)
    parser.add_argument('--tta-bs',      type=int,   default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading {len(args.checkpoints)} checkpoints:")
    models = []
    for ckpt_path in args.checkpoints:
        m, seq_len = load_checkpoint(ckpt_path, device)
        models.append((m, seq_len))
    print(f"Ensemble size: {len(models)} models")

    if args.tta_epochs > 0:
        print(f"TTA: {args.tta_epochs} epochs @ lr={args.tta_lr} "
              f"(WARNING: may hurt — check plain ensemble first)")
    else:
        print("TTA: disabled (plain ensemble)")

    test_ids = list_session_ids(args.data_dir, split='test')
    print(f"Test sessions: {len(test_ids)}\n")

    predictions = {}
    for i, sid in enumerate(test_ids):
        t0   = time.time()
        sess = load_session(args.data_dir, sid, is_test=True)

        all_preds = []
        for model, max_seq_len in models:
            if args.tta_epochs > 0:
                sbp    = sess['sbp']
                d_ind  = sess['dropout_ind']
                mean, std = session_zscore_params(sbp)
                sbp_z  = zscore_normalize(sbp, mean, std).astype(np.float32)
                tta_ds = TTADataset(sbp_z, d_ind, sess['start_bins'],
                                    sess['end_bins'], sess['n_trials'],
                                    max_seq_len=max_seq_len, seed=i * 100)
                adapted = tta_finetune(model, tta_ds, device,
                                       epochs=args.tta_epochs, lr=args.tta_lr,
                                       batch_size=args.tta_bs)
                pred = predict_session(adapted, sess, device, max_seq_len=max_seq_len)
                del adapted
                torch.cuda.empty_cache()
            else:
                pred = predict_session(model, sess, device, max_seq_len=max_seq_len)
            all_preds.append(pred)

        # Average across all ensemble members
        ensemble_pred   = np.mean(all_preds, axis=0)
        predictions[sid] = ensemble_pred
        dt = time.time() - t0
        print(f"  [{i+1:3d}/{len(test_ids)}] {sid}: {ensemble_pred.shape[0]} bins  "
              f"({dt:.1f}s, {len(all_preds)} models avg)")

    build_submission(predictions, args.data_dir, args.output)


if __name__ == '__main__':
    main()

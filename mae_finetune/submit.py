"""
Method 2: MAE Fine-tune — Submission
======================================
Supports plain inference and self-supervised test-time adaptation (TTA).

TTA fine-tunes the encoder on each test session's SBP using a masked-channel
reconstruction objective (no kinematics labels required). The decode head is
then used for final position prediction.

Usage:
    python submit.py                                       # plain inference
    python submit.py --tta-epochs 5 --tta-lr 1e-5         # TTA per session
    python submit.py --checkpoint sweep_X/best_model.pt   # specific checkpoint
    python submit.py --tta-epochs 5 --output sub_tta5.csv
"""

import sys
import os
import argparse
import copy
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_utils import (
    DATA_DIR, N_CHANNELS, N_KIN,
    load_session, list_session_ids,
    session_zscore_params, zscore_normalize,
    build_submission,
)
from train import MAEFinetuneDecoder, MAX_SEQ_LEN, INPUT_DIM_P1

N_MASK_TTA = 30   # synthetic mask channels for self-supervised TTA


# ---------------------------------------------------------------------------
# TTA dataset: self-supervised reconstruction on test-session SBP
# ---------------------------------------------------------------------------

class TTADataset(Dataset):
    """Synthetic masked-SBP examples from a test session (no labels needed).

    Randomly masks N_MASK_TTA active channels per timestep per trial.
    The model is fine-tuned to reconstruct those channels via recon_head.
    """
    def __init__(self, sbp_z, d_ind, start_bins, end_bins, n_trials,
                 max_seq_len=MAX_SEQ_LEN, seed=0):
        self.max_seq_len = max_seq_len
        self.d_ind       = d_ind
        self.seed        = seed
        self.active_ch   = np.where(d_ind == 0.0)[0]   # channels to synthetically mask
        self.examples    = []

        for ti in range(n_trials):
            s, e = int(start_bins[ti]), int(end_bins[ti])
            if e - s < 5:
                continue
            self.examples.append((sbp_z[s:e].astype(np.float32), e - s))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sbp_trial, tlen = self.examples[idx]
        T   = self.max_seq_len
        rng = np.random.RandomState(self.seed + idx * 7919)

        if tlen > T:
            off       = rng.randint(0, tlen - T)
            sbp_trial = sbp_trial[off:off + T]
            sl = T
        else:
            sl = tlen

        n_mask = min(N_MASK_TTA, len(self.active_ch))
        tmask  = np.zeros((sl, N_CHANNELS), dtype=np.float32)
        for t in range(sl):
            ch = rng.choice(self.active_ch, size=n_mask, replace=False)
            tmask[t, ch] = 1.0

        sbp_in = sbp_trial[:sl].copy()
        sbp_in[tmask.astype(bool)] = 0.0

        # Combined indicator: permanent dropout + synthetic mask
        combined = np.maximum(np.tile(self.d_ind, (sl, 1)), tmask)
        zeros4   = np.zeros((sl, 4), dtype=np.float32)

        def pad(a):
            if a.shape[0] >= T:
                return a[:T]
            return np.pad(a, [(0, T - a.shape[0])] + [(0, 0)] * (a.ndim - 1))

        feat  = pad(np.concatenate([sbp_in, combined, zeros4], axis=1))
        tgt   = pad(sbp_trial[:sl])
        tmask = pad(tmask)
        pmask = np.ones(T, dtype=bool)
        pmask[:sl] = False

        return (
            torch.from_numpy(feat),
            torch.from_numpy(tgt),
            torch.from_numpy(tmask),
            torch.from_numpy(pmask),
        )


def tta_finetune(model: MAEFinetuneDecoder, dataset: TTADataset,
                 device, epochs=5, lr=1e-5, batch_size=32):
    """Self-supervised TTA: fine-tune encoder + recon_head on reconstruction.

    decode_head is frozen during TTA so position decoding capacity is preserved.
    """
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
    sbp     = sess['sbp']
    d_ind   = sess['dropout_ind']
    mean, std = session_zscore_params(sbp)
    sbp_z   = zscore_normalize(sbp, mean, std).astype(np.float32)
    N       = sbp_z.shape[0]
    d_tiled = np.tile(d_ind, (N, 1))
    zeros4  = np.zeros((N, 4), dtype=np.float32)
    feat    = np.concatenate([sbp_z, d_tiled, zeros4], axis=1).astype(np.float32)

    pred_all = np.full((N, N_KIN), 0.5, dtype=np.float32)
    model.eval()

    for ti in range(sess['n_trials']):
        s, e = int(sess['start_bins'][ti]), int(sess['end_bins'][ti])
        tlen = e - s
        for cs in range(0, tlen, max_seq_len):
            ce  = min(cs + max_seq_len, tlen)
            cl  = ce - cs
            T   = max_seq_len
            chunk = feat[s + cs:s + ce]
            if cl < T:
                chunk = np.pad(chunk, ((0, T - cl), (0, 0)))
            ft  = torch.from_numpy(chunk).unsqueeze(0).to(device)
            pm  = torch.ones(1, T, dtype=torch.bool, device=device)
            pm[0, :cl] = False
            out = model(ft, padding_mask=pm).cpu().numpy()[0, :cl]
            pred_all[s + cs:s + ce] = out

    return np.clip(pred_all, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',   type=str,   default=DATA_DIR)
    parser.add_argument('--checkpoint', type=str,   default='best_model.pt')
    parser.add_argument('--output',     type=str,   default='submission.csv')
    parser.add_argument('--tta-epochs', type=int,   default=0,
                        help='Self-supervised TTA epochs per test session (0=off)')
    parser.add_argument('--tta-lr',     type=float, default=1e-5)
    parser.add_argument('--tta-bs',     type=int,   default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_dir    = os.path.dirname(os.path.abspath(args.checkpoint))
    config_path = os.path.join(ckpt_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        base_model = MAEFinetuneDecoder(
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 4),
            d_ff=config.get('d_ff', 512),
            dropout=config.get('dropout', 0.1),
        )
        max_seq_len = config.get('max_seq_len', MAX_SEQ_LEN)
    else:
        base_model  = MAEFinetuneDecoder()
        max_seq_len = MAX_SEQ_LEN

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        base_model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded: {args.checkpoint} (epoch {ckpt.get('epoch','?')}, "
              f"val_r2={ckpt.get('val_r2','?')})")
    else:
        base_model.load_state_dict(ckpt)
    base_model = base_model.to(device).eval()

    print(f"TTA: {args.tta_epochs} epochs @ lr={args.tta_lr}" if args.tta_epochs
          else "TTA: disabled")

    test_ids = list_session_ids(args.data_dir, split='test')
    print(f"Test sessions: {len(test_ids)}")

    predictions = {}
    for i, sid in enumerate(test_ids):
        t0   = time.time()
        sess = load_session(args.data_dir, sid, is_test=True)

        if args.tta_epochs > 0:
            sbp     = sess['sbp']
            d_ind   = sess['dropout_ind']
            mean, std = session_zscore_params(sbp)
            sbp_z   = zscore_normalize(sbp, mean, std).astype(np.float32)
            tta_ds  = TTADataset(sbp_z, d_ind, sess['start_bins'],
                                 sess['end_bins'], sess['n_trials'],
                                 max_seq_len=max_seq_len, seed=i * 100)
            model    = tta_finetune(base_model, tta_ds, device,
                                    epochs=args.tta_epochs, lr=args.tta_lr,
                                    batch_size=args.tta_bs)
            tta_info = f", TTA {args.tta_epochs}ep on {len(tta_ds)} trials"
        else:
            model    = base_model
            tta_info = ""

        pred = predict_session(model, sess, device, max_seq_len=max_seq_len)
        predictions[sid] = pred
        print(f"  [{i+1}/{len(test_ids)}] {sid}: {pred.shape[0]} bins "
              f"({time.time()-t0:.1f}s{tta_info})")

        if args.tta_epochs > 0:
            del model
            torch.cuda.empty_cache()

    build_submission(predictions, args.data_dir, args.output)


if __name__ == '__main__':
    main()

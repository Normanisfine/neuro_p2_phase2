"""
Method 4: Masked Autoencoder — Submission
==========================================
Supports test-time fine-tuning (TTA) and ridge blending.

Usage:
    python submit.py                                      # plain inference
    python submit.py --tta-epochs 5 --tta-lr 1e-4         # TTA per session
    python submit.py --ridge-blend 0.2                    # ridge ensemble
    python submit.py --tta-epochs 5 --ridge-blend 0.2     # TTA + ridge
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_utils import (
    DATA_DIR, N_CHANNELS, N_MASKED_CHANNELS, load_metadata, load_session,
    build_submission, session_zscore_params, zscore_normalize, zscore_denormalize,
)
from train import MaskedAutoencoder, MAX_SEQ_LEN

import importlib.util
_ridge_spec = importlib.util.spec_from_file_location(
    "ridge_train",
    os.path.join(os.path.dirname(__file__), '..', 'ridge_regression', 'train.py'),
)
_ridge_mod = importlib.util.module_from_spec(_ridge_spec)
_ridge_spec.loader.exec_module(_ridge_mod)
ridge_predict_session = _ridge_mod.ridge_predict_session


# ---------------------------------------------------------------------------
# TTA dataset: synthetic masking on a test session's unmasked trials
# ---------------------------------------------------------------------------

class SessionFTDataset(Dataset):
    def __init__(self, sbp_z, kin, start_bins, end_bins, n_trials, mask,
                 max_seq_len=MAX_SEQ_LEN, n_masked_ch=N_MASKED_CHANNELS, seed=0,
                 use_mask_token=False):
        self.examples = []
        self.max_seq_len = max_seq_len
        self.n_masked_ch = n_masked_ch
        self.seed = seed
        self.use_mask_token = bool(use_mask_token)
        for ti in range(n_trials):
            s, e = int(start_bins[ti]), int(end_bins[ti])
            if mask[s:e].any():
                continue
            if e - s < 5:
                continue
            self.examples.append((
                sbp_z[s:e].astype(np.float32),
                kin[s:e].astype(np.float32),
                e - s,
            ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sbp, kin, tlen = self.examples[idx]
        T = self.max_seq_len
        rng = np.random.RandomState(self.seed + idx * 31337)
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


def finetune_on_session(model, dataset, device, epochs=5, lr=1e-4, batch_size=32):
    adapted = copy.deepcopy(model)
    adapted.train()
    opt = torch.optim.AdamW(adapted.parameters(), lr=lr, weight_decay=0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    for ep in range(epochs):
        for feat, tgt, tmask, pmask in loader:
            feat, tgt = feat.to(device), tgt.to(device)
            tmask, pmask = tmask.to(device), pmask.to(device)
            pred = adapted(feat, padding_mask=pmask, mask_indicator=tmask)
            loss = ((pred - tgt) ** 2 * tmask).sum() / tmask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), 1.0)
            opt.step()
    adapted.eval()
    return adapted


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_test_session(model, sess, device, max_seq_len=MAX_SEQ_LEN):
    sbp_masked = sess['sbp_masked']
    mask = sess['mask']
    kin = sess['kinematics']

    mean, std = session_zscore_params(sbp_masked, mask)
    sbp_z = zscore_normalize(sbp_masked, mean, std)
    if not model.use_mask_token:
        sbp_z[mask] = 0.0

    pred_z = zscore_normalize(sbp_masked, mean, std)
    pred_z[mask] = 0.0

    model.eval()
    with torch.no_grad():
        for trial_idx in range(sess['n_trials']):
            start = int(sess['start_bins'][trial_idx])
            end = int(sess['end_bins'][trial_idx])
            trial_mask = mask[start:end]
            if not trial_mask.any():
                continue
            trial_len = end - start
            trial_sbp_in = sbp_z[start:end]
            trial_kin = kin[start:end]
            mask_ind = trial_mask.astype(np.float32)

            for chunk_start in range(0, trial_len, max_seq_len):
                chunk_end = min(chunk_start + max_seq_len, trial_len)
                clen = chunk_end - chunk_start
                mask_chunk = mask_ind[chunk_start:chunk_end]
                if model.use_mask_token:
                    feat = np.concatenate([
                        trial_sbp_in[chunk_start:chunk_end],
                        trial_kin[chunk_start:chunk_end],
                    ], axis=1).astype(np.float32)
                else:
                    feat = np.concatenate([
                        trial_sbp_in[chunk_start:chunk_end],
                        mask_chunk,
                        trial_kin[chunk_start:chunk_end],
                    ], axis=1).astype(np.float32)
                T = max_seq_len
                if clen < T:
                    feat = np.pad(feat, ((0, T - clen), (0, 0)), mode='constant')
                    mask_chunk = np.pad(mask_chunk, ((0, T - clen), (0, 0)), mode='constant')
                feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask_chunk).unsqueeze(0).to(device)
                pad_mask = torch.ones(1, T, dtype=torch.bool, device=device)
                pad_mask[0, :clen] = False
                out = model(feat_t, padding_mask=pad_mask, mask_indicator=mask_t).cpu().numpy()[0, :clen]

                abs_start = start + chunk_start
                for local_t in range(clen):
                    t = abs_start + local_t
                    masked_ch = np.where(mask[t])[0]
                    if len(masked_ch) > 0:
                        pred_z[t, masked_ch] = out[local_t, masked_ch]

    pred = zscore_denormalize(pred_z, mean, std)
    return pred


def main():
    parser = argparse.ArgumentParser(description='Masked Autoencoder submission')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--checkpoint', type=str, default='best_model.pt')
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--tta-epochs', type=int, default=0,
                        help='Test-time fine-tuning epochs per session (0=off)')
    parser.add_argument('--tta-lr', type=float, default=1e-4)
    parser.add_argument('--tta-bs', type=int, default=32)
    parser.add_argument('--tta-seed', type=int, default=0,
                        help='Base random seed for TTA masking/cropping')
    parser.add_argument('--ridge-blend', type=float, default=0.0,
                        help='Blend weight for ridge (0=pure MAE)')
    parser.add_argument('--ridge-alpha', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config_path = os.path.join(os.path.dirname(args.checkpoint), 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        max_seq_len = config.get('max_seq_len', MAX_SEQ_LEN)
        use_mask_token = bool(config.get('use_mask_token', False))
        tta_n_masked = int(config.get('n_masked_channels', N_MASKED_CHANNELS))
        base_model = MaskedAutoencoder(
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 4),
            d_ff=config.get('d_ff', 512),
            dropout=config.get('dropout', 0.1),
            use_mask_token=use_mask_token,
        )
    else:
        max_seq_len = MAX_SEQ_LEN
        use_mask_token = False
        tta_n_masked = N_MASKED_CHANNELS
        base_model = MaskedAutoencoder()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        base_model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, "
              f"val_nmse={ckpt.get('val_nmse', '?')})")
    else:
        base_model.load_state_dict(ckpt)
    base_model = base_model.to(device).eval()

    print(f"TTA: {args.tta_epochs} epochs @ lr={args.tta_lr}, seed={args.tta_seed}" if args.tta_epochs
          else "TTA: disabled")
    print(f"use_mask_token: {use_mask_token}  tta_n_masked_channels: {tta_n_masked}")
    if args.ridge_blend > 0:
        print(f"Ridge blend: {args.ridge_blend:.2f}")

    meta = load_metadata(args.data_dir)
    test_sessions = meta[meta['split'] == 'test']['session_id'].tolist()

    predictions = {}
    for i, sid in enumerate(test_sessions):
        t0 = time.time()
        sess = load_session(args.data_dir, sid, is_test=True)

        if args.tta_epochs > 0:
            sbp_m = sess['sbp_masked']
            mask = sess['mask']
            kin = sess['kinematics']
            mean, std = session_zscore_params(sbp_m, mask)
            sbp_z = zscore_normalize(sbp_m, mean, std)
            sbp_z[mask] = 0.0
            ft_ds = SessionFTDataset(sbp_z, kin, sess['start_bins'],
                                     sess['end_bins'], sess['n_trials'],
                                     mask, max_seq_len=max_seq_len,
                                     n_masked_ch=tta_n_masked,
                                     seed=args.tta_seed + i * 100,
                                     use_mask_token=use_mask_token)
            model = finetune_on_session(base_model, ft_ds, device,
                                        epochs=args.tta_epochs, lr=args.tta_lr,
                                        batch_size=args.tta_bs)
            ft_info = f", TTA {args.tta_epochs}ep on {len(ft_ds)} trials"
        else:
            model = base_model
            ft_info = ""

        mae_pred = predict_test_session(model, sess, device, max_seq_len=max_seq_len)

        if args.ridge_blend > 0:
            r_pred = ridge_predict_session(sess['sbp_masked'], sess['mask'],
                                           alpha=args.ridge_alpha)
            final = (1 - args.ridge_blend) * mae_pred + args.ridge_blend * r_pred
        else:
            final = mae_pred

        predictions[sid] = final
        elapsed = time.time() - t0
        n_masked = sess['mask'].sum()
        print(f"  [{i+1}/{len(test_sessions)}] {sid}: {n_masked} masked ({elapsed:.1f}s{ft_info})")

        if args.tta_epochs > 0:
            del model
            torch.cuda.empty_cache()

    build_submission(predictions, args.data_dir, args.output)


if __name__ == '__main__':
    main()

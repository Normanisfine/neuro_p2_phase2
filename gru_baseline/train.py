"""
Method 1: GRU Baseline — Finger Position Decoding
==================================================
2-layer bidirectional GRU decoding finger positions from SBP.
Input per timestep: z-scored SBP (96) + channel-dropout indicator (96) = 192 dims.
Output per timestep: [index_pos, mrp_pos] in [0, 1].

Usage:
    python train.py --quick                      # sanity check
    python train.py --epochs 30 --wandb          # standard run
    python train.py --epochs 30 --lr 3e-4        # tuning
"""

import sys
import os
import argparse
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_utils import (
    DATA_DIR, N_CHANNELS, N_KIN,
    load_session, get_validation_sessions,
    session_zscore_params, zscore_normalize,
    compute_r2_multi,
)

WANDB_PROJECT = "neuro-p2-phase2"
MAX_SEQ_LEN   = 128
INPUT_DIM     = N_CHANNELS + N_CHANNELS  # 192: sbp_z + dropout_ind


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GRUDecoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=256, num_layers=2,
                 dropout=0.1, out_dim=N_KIN):
        super().__init__()
        self.bigru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, T, 192)
        out, _ = self.bigru(x)   # (B, T, hidden*2)
        return self.head(out)    # (B, T, 2)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrialDataset(Dataset):
    def __init__(self, session_ids, data_dir, max_seq_len=MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.trials = []

        for sid in session_ids:
            sess    = load_session(data_dir, sid, is_test=False)
            sbp     = sess['sbp']
            kin     = sess['kinematics']
            d_ind   = sess['dropout_ind']                          # (96,)
            mean, std = session_zscore_params(sbp)
            sbp_z   = zscore_normalize(sbp, mean, std).astype(np.float32)
            N       = sbp_z.shape[0]
            d_tiled = np.tile(d_ind, (N, 1))                      # (N, 96)
            feat    = np.concatenate([sbp_z, d_tiled], axis=1)    # (N, 192)
            kin_pos = kin[:, :N_KIN].astype(np.float32)           # (N, 2)

            for ti in range(sess['n_trials']):
                s, e = int(sess['start_bins'][ti]), int(sess['end_bins'][ti])
                if e - s < 5:
                    continue
                self.trials.append((feat[s:e], kin_pos[s:e], e - s))

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        feat, kin_pos, tlen = self.trials[idx]
        T = self.max_seq_len

        if tlen > T:
            off     = np.random.randint(0, tlen - T)
            feat    = feat[off:off + T]
            kin_pos = kin_pos[off:off + T]
            sl = T
        else:
            sl = tlen

        def pad(a):
            if a.shape[0] >= T:
                return a[:T]
            return np.pad(a, [(0, T - a.shape[0])] + [(0, 0)] * (a.ndim - 1))

        feat_p = pad(feat)
        kin_p  = pad(kin_pos)
        pmask  = np.ones(T, dtype=bool)
        pmask[:sl] = False   # True = padding position

        return (
            torch.from_numpy(feat_p),
            torch.from_numpy(kin_p),
            torch.from_numpy(pmask),
            sl,
        )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def make_scheduler(optimizer, warmup, total, min_ratio=0.01):
    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        p = (ep - warmup) / max(total - warmup, 1)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for feat, kin_pos, pmask, sl in loader:
        feat    = feat.to(device)      # (B, T, 192)
        kin_pos = kin_pos.to(device)   # (B, T, 2)
        pmask   = pmask.to(device)     # (B, T) bool

        pred  = model(feat)            # (B, T, 2)
        valid = (~pmask).unsqueeze(-1).float()
        loss  = ((pred - kin_pos) ** 2 * valid).sum() / valid.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate_model(model, val_ids, data_dir, device, max_seq_len=MAX_SEQ_LEN):
    model.eval()
    results = []
    for sid in val_ids:
        sess    = load_session(data_dir, sid, is_test=False)
        sbp     = sess['sbp']
        kin     = sess['kinematics']
        d_ind   = sess['dropout_ind']
        mean, std = session_zscore_params(sbp)
        sbp_z   = zscore_normalize(sbp, mean, std).astype(np.float32)
        N       = sbp_z.shape[0]
        d_tiled = np.tile(d_ind, (N, 1))
        feat    = np.concatenate([sbp_z, d_tiled], axis=1).astype(np.float32)
        kin_pos = kin[:, :N_KIN].astype(np.float32)
        pred_all = np.full((N, N_KIN), 0.5, dtype=np.float32)

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
                out = model(ft).cpu().numpy()[0, :cl]
                pred_all[s + cs:s + ce] = out

        results.append((pred_all, kin_pos, sid))
    return compute_r2_multi(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--data-dir',      type=str,   default=DATA_DIR)
    pa.add_argument('--epochs',        type=int,   default=30)
    pa.add_argument('--batch-size',    type=int,   default=128)
    pa.add_argument('--lr',            type=float, default=5e-4)
    pa.add_argument('--warmup-epochs', type=int,   default=3)
    pa.add_argument('--hidden-dim',    type=int,   default=256)
    pa.add_argument('--num-layers',    type=int,   default=2)
    pa.add_argument('--dropout',       type=float, default=0.1)
    pa.add_argument('--max-seq-len',   type=int,   default=MAX_SEQ_LEN)
    pa.add_argument('--save-every',    type=int,   default=10)
    pa.add_argument('--quick',         action='store_true')
    pa.add_argument('--wandb',         action='store_true')
    pa.add_argument('--wandb-project', type=str,   default=WANDB_PROJECT)
    pa.add_argument('--wandb-name',    type=str,   default=None)
    args = pa.parse_args()

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        name = args.wandb_name or (
            f"gru_lr{args.lr}_do{args.dropout}_ep{args.epochs}_h{args.hidden_dim}"
        )
        wandb.init(project=args.wandb_project, name=name, config=vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ids, val_ids = get_validation_sessions(args.data_dir)
    if args.quick:
        train_ids, val_ids = train_ids[:8], val_ids[:4]
        args.epochs = 2

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    print("Building dataset...")
    t0 = time.time()
    ds = TrialDataset(train_ids, args.data_dir, max_seq_len=args.max_seq_len)
    print(f"  {len(ds)} trials ({time.time() - t0:.1f}s)")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = GRUDecoder(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {npar:,}")

    if use_wandb:
        wandb.config.update({"n_params": npar, "n_trials": len(ds)})

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = make_scheduler(opt, args.warmup_epochs, args.epochs)

    def save_ckpt(path, ep, vr2, best):
        torch.save({
            'epoch': ep, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            'val_r2': vr2, 'best_r2': best,
        }, path)

    best = -float('inf')
    for ep in range(1, args.epochs + 1):
        t0      = time.time()
        loss    = train_epoch(model, loader, opt, device)
        sched.step()
        vr2     = validate_model(model, val_ids[:4], args.data_dir, device,
                                 max_seq_len=args.max_seq_len)
        dt      = time.time() - t0
        is_best = vr2 > best
        if is_best:
            best = vr2
            save_ckpt('best_model.pt', ep, vr2, best)
        if args.save_every and ep % args.save_every == 0:
            save_ckpt(f'checkpoint_epoch{ep}.pt', ep, vr2, best)
        save_ckpt('last_model.pt', ep, vr2, best)

        lr_now = opt.param_groups[0]['lr']
        print(f"Ep {ep:3d}/{args.epochs} | loss={loss:.4f} | "
              f"val_r2={vr2:.4f} {'*' if is_best else ''} | "
              f"lr={lr_now:.2e} | {dt:.0f}s")
        if use_wandb:
            wandb.log({"epoch": ep, "train/loss": loss, "val/r2": vr2,
                       "val/best_r2": best, "lr": lr_now, "time_s": dt})

    print("\nFull validation with best model...")
    ckpt = torch.load('best_model.pt', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    full_r2 = validate_model(model, val_ids, args.data_dir, device,
                             max_seq_len=args.max_seq_len)
    print(f"Full val R²: {full_r2:.4f} (best epoch {ckpt['epoch']})")

    cfg = {k: getattr(args, k) for k in
           ['hidden_dim', 'num_layers', 'dropout', 'max_seq_len',
            'warmup_epochs', 'epochs', 'batch_size', 'lr']}
    cfg.update({'best_val_r2': best, 'best_epoch': ckpt['epoch'],
                'full_val_r2': full_r2})
    with open('config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print("Saved config.json, best_model.pt, last_model.pt")

    if use_wandb:
        wandb.summary.update({"r2/best_val": best, "r2/full_val": full_r2,
                               "best_epoch": ckpt['epoch']})
        wandb.finish()


if __name__ == '__main__':
    main()

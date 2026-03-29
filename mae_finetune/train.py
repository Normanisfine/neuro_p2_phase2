"""
Method 2: MAE Pretrained Fine-tune — Finger Position Decoding
=============================================================
Fine-tunes the Phase 1 Masked Autoencoder encoder for position decoding.

Architecture:
  - Encoder: Phase 1 MAE Transformer (4-layer, d=256) loaded from pretrained weights
  - Input: sbp_z (96) + dropout_ind (96) + zeros (4) = 196 dims (matches Phase 1 contract)
  - decode_head: LayerNorm → Linear(256→128) → GELU → Dropout → Linear(128→2) → Sigmoid
  - recon_head: identical to Phase 1 output_head — used only for self-supervised TTA

Training uses differential LRs:
  encoder_lr (default 1e-5) for pretrained encoder layers
  head_lr    (default 5e-4) for new decode head

Usage:
    python train.py --quick
    python train.py --epochs 40 --pretrained-checkpoint <path/to/best_model.pt> --wandb
    python train.py --epochs 40 --encoder-lr 5e-5 --head-lr 1e-3
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

WANDB_PROJECT    = "neuro-p2-phase2"
MAX_SEQ_LEN      = 128
INPUT_DIM_P1     = 96 + 96 + 4   # 196 — matches Phase 1 MAE input exactly

PHASE1_MAE_DIR   = '/scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder'
DEFAULT_PRETRAINED = os.path.join(PHASE1_MAE_DIR, 'best_model.pt')


# ---------------------------------------------------------------------------
# Architecture  (encoder matches Phase 1 MaskedAutoencoder for weight loading)
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MAEFinetuneDecoder(nn.Module):
    """Phase 1 MAE encoder + new position decode head.

    Encoder weights (input_proj, pos_enc, encoder) are loaded from Phase 1.
    recon_head mirrors Phase 1 output_head and is used only for self-supervised TTA.
    decode_head is randomly initialised and trained for position regression.
    """
    def __init__(self, input_dim=INPUT_DIM_P1, d_model=256, nhead=4,
                 num_layers=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = SinusoidalPE(d_model, max_len=1024)
        self.drop       = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Reconstruction head — mirrors Phase 1 output_head (for TTA only)
        self.recon_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, N_CHANNELS),
        )
        # Position decode head (new; randomly initialised)
        self.decode_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_KIN),
            nn.Sigmoid(),
        )

    def encode(self, x, padding_mask=None):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.drop(h)
        return self.encoder(h, src_key_padding_mask=padding_mask)

    def forward(self, x, padding_mask=None):
        """Position decoding forward pass."""
        return self.decode_head(self.encode(x, padding_mask))  # (B, T, 2)

    def reconstruct(self, x, padding_mask=None):
        """Reconstruction forward pass — used only during self-supervised TTA."""
        return self.recon_head(self.encode(x, padding_mask))   # (B, T, 96)

    def encoder_params(self):
        return (list(self.input_proj.parameters()) +
                list(self.encoder.parameters()))

    def head_params(self):
        return (list(self.decode_head.parameters()) +
                list(self.pos_enc.parameters()))


def load_pretrained_encoder(model: MAEFinetuneDecoder, checkpoint_path: str,
                            device: torch.device) -> None:
    """Load Phase 1 MAE encoder weights into model.

    Copies: input_proj, pos_enc, encoder.
    Maps:   Phase 1 output_head → model recon_head (for TTA reconstruction).
    Skips:  decode_head (randomly initialised for Phase 2 decoding).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pretrained = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    ep    = ckpt.get('epoch', '?')    if isinstance(ckpt, dict) else '?'
    nmse  = ckpt.get('val_nmse', '?') if isinstance(ckpt, dict) else '?'
    print(f"  Phase 1 checkpoint: epoch={ep}, val_nmse={nmse}")

    model_dict  = model.state_dict()
    transferred, skipped = [], []

    for old_k, tensor in pretrained.items():
        if old_k.startswith('output_head.'):
            new_k = 'recon_head.' + old_k[len('output_head.'):]
        else:
            new_k = old_k  # input_proj, pos_enc, encoder.*

        if new_k in model_dict and model_dict[new_k].shape == tensor.shape:
            model_dict[new_k] = tensor
            transferred.append(old_k)
        else:
            skipped.append(old_k)

    model.load_state_dict(model_dict)
    print(f"  Transferred {len(transferred)} tensors, skipped {len(skipped)}")


# ---------------------------------------------------------------------------
# Dataset — 196-dim input matching Phase 1 (kin set to zeros at test time)
# ---------------------------------------------------------------------------

class TrialDataset(Dataset):
    def __init__(self, session_ids, data_dir, max_seq_len=MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.trials = []

        for sid in session_ids:
            sess    = load_session(data_dir, sid, is_test=False)
            sbp     = sess['sbp']
            kin     = sess['kinematics']
            d_ind   = sess['dropout_ind']                # (96,)
            mean, std = session_zscore_params(sbp)
            sbp_z   = zscore_normalize(sbp, mean, std).astype(np.float32)
            N       = sbp_z.shape[0]
            d_tiled = np.tile(d_ind, (N, 1))             # (N, 96)
            zeros4  = np.zeros((N, 4), dtype=np.float32)
            feat    = np.concatenate([sbp_z, d_tiled, zeros4], axis=1)  # (N, 196)
            kin_pos = kin[:, :N_KIN].astype(np.float32)

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
        pmask[:sl] = False

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
        feat    = feat.to(device)
        kin_pos = kin_pos.to(device)
        pmask   = pmask.to(device)

        pred  = model(feat, padding_mask=pmask)   # (B, T, 2)
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
        zeros4  = np.zeros((N, 4), dtype=np.float32)
        feat    = np.concatenate([sbp_z, d_tiled, zeros4], axis=1).astype(np.float32)
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
                pm  = torch.ones(1, T, dtype=torch.bool, device=device)
                pm[0, :cl] = False
                out = model(ft, padding_mask=pm).cpu().numpy()[0, :cl]
                pred_all[s + cs:s + ce] = out

        results.append((pred_all, kin_pos, sid))
    return compute_r2_multi(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--data-dir',               type=str,   default=DATA_DIR)
    pa.add_argument('--pretrained-checkpoint',  type=str,   default=DEFAULT_PRETRAINED,
                    help='Path to Phase 1 MAE best_model.pt')
    pa.add_argument('--no-pretrain',            action='store_true',
                    help='Train from scratch (ignore pretrained checkpoint)')
    pa.add_argument('--epochs',                 type=int,   default=40)
    pa.add_argument('--batch-size',             type=int,   default=64)
    pa.add_argument('--encoder-lr',             type=float, default=1e-5)
    pa.add_argument('--head-lr',                type=float, default=5e-4)
    pa.add_argument('--warmup-epochs',          type=int,   default=5)
    pa.add_argument('--d-model',                type=int,   default=256)
    pa.add_argument('--nhead',                  type=int,   default=4)
    pa.add_argument('--num-layers',             type=int,   default=4)
    pa.add_argument('--d-ff',                   type=int,   default=512)
    pa.add_argument('--dropout',                type=float, default=0.1)
    pa.add_argument('--max-seq-len',            type=int,   default=MAX_SEQ_LEN)
    pa.add_argument('--save-every',             type=int,   default=10)
    pa.add_argument('--quick',                  action='store_true')
    pa.add_argument('--wandb',                  action='store_true')
    pa.add_argument('--wandb-project',          type=str,   default=WANDB_PROJECT)
    pa.add_argument('--wandb-name',             type=str,   default=None)
    args = pa.parse_args()

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        name = args.wandb_name or (
            f"mae_ft_elr{args.encoder_lr}_hlr{args.head_lr}"
            f"_do{args.dropout}_ep{args.epochs}"
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

    model = MAEFinetuneDecoder(
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, d_ff=args.d_ff, dropout=args.dropout,
    ).to(device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {npar:,}")

    if not args.no_pretrain:
        print(f"Loading pretrained encoder from: {args.pretrained_checkpoint}")
        load_pretrained_encoder(model, args.pretrained_checkpoint, device)
    else:
        print("Training from scratch (no pretrained weights)")

    if use_wandb:
        wandb.config.update({"n_params": npar, "n_trials": len(ds)})

    # Differential learning rates
    opt = torch.optim.AdamW([
        {'params': model.encoder_params(),           'lr': args.encoder_lr},
        {'params': model.head_params(),              'lr': args.head_lr},
        {'params': model.recon_head.parameters(),    'lr': args.encoder_lr},
    ], weight_decay=1e-4)
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

        lr_enc = opt.param_groups[0]['lr']
        lr_hd  = opt.param_groups[1]['lr']
        print(f"Ep {ep:3d}/{args.epochs} | loss={loss:.4f} | "
              f"val_r2={vr2:.4f} {'*' if is_best else ''} | "
              f"lr_enc={lr_enc:.2e} lr_hd={lr_hd:.2e} | {dt:.0f}s")
        if use_wandb:
            wandb.log({"epoch": ep, "train/loss": loss, "val/r2": vr2,
                       "val/best_r2": best, "lr/encoder": lr_enc,
                       "lr/head": lr_hd, "time_s": dt})

    print("\nFull validation with best model...")
    ckpt = torch.load('best_model.pt', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    full_r2 = validate_model(model, val_ids, args.data_dir, device,
                             max_seq_len=args.max_seq_len)
    print(f"Full val R²: {full_r2:.4f} (best epoch {ckpt['epoch']})")

    cfg = {k: getattr(args, k) for k in
           ['d_model', 'nhead', 'num_layers', 'd_ff', 'max_seq_len',
            'dropout', 'warmup_epochs', 'epochs', 'batch_size',
            'encoder_lr', 'head_lr']}
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

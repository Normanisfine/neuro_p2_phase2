"""
MAE Multitask Decoder — Phase 2 finger position decoding
========================================================

Key features:
  - Long sliding-window training (5s / 250 bins, 75% overlap)
    instead of per-trial (~1.25s) — gives the model cross-trial context
  - Predicts positions and velocities jointly (multitask)
  - Temporal consistency loss: pos[t+1] - pos[t] ≈ vel[t]
  - Synthetic extra channel dropout augmentation during training
  - Balanced easy+hard validation subset for checkpoint selection
  - Hanning-weighted overlap-averaged inference across the full session

Usage:
    python train.py --quick
    python train.py --pretrained-checkpoint /path/to/phase1/best_model.pt --wandb
    python train.py --window-size 250  # 5s windows
"""

import sys
import os
import argparse
import json
import math
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, ".."))
from data_utils import (
    DATA_DIR,
    P1_DATA_DIR,
    N_CHANNELS,
    N_KIN,
    compute_r2_multi,
    get_validation_sessions,
    list_p1_session_ids,
    load_p1_session_as_p2,
    load_session,
    session_zscore_params,
    zscore_normalize,
)

WANDB_PROJECT = "neuro-p2-multitask"
WINDOW_SIZE   = 250   # 5 s at 50 Hz
WINDOW_STRIDE = 63    # 75% overlap (stride = 25% of window)
INPUT_DIM_P1  = 96 + 96 + 4
N_VEL = 2

PHASE1_MAE_DIR = "/scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder"
DEFAULT_PRETRAINED = os.path.join(PHASE1_MAE_DIR, "best_model.pt")


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MAEMultitaskDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM_P1,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc = SinusoidalPE(d_model, max_len=2048)
        self.drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.shared_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_KIN),
            nn.Sigmoid(),
        )
        self.vel_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_VEL),
        )

    def encode(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.pos_enc(h)
        h = self.drop(h)
        return self.encoder(h, src_key_padding_mask=padding_mask)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared_head(self.encode(x, padding_mask))
        return self.pos_head(h), self.vel_head(h)

    def encoder_params(self):
        return (
            list(self.input_proj.parameters())
            + list(self.input_norm.parameters())
            + list(self.encoder.parameters())
        )

    def head_params(self):
        return (
            list(self.shared_head.parameters())
            + list(self.pos_head.parameters())
            + list(self.vel_head.parameters())
            + list(self.pos_enc.parameters())
        )


def load_pretrained_encoder(model: MAEMultitaskDecoder, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pretrained = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ep = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    nmse = ckpt.get("val_nmse", "?") if isinstance(ckpt, dict) else "?"
    print(f"  Phase 1 checkpoint: epoch={ep}, val_nmse={nmse}")

    model_dict = model.state_dict()
    transferred, skipped = [], []
    for old_k, tensor in pretrained.items():
        if old_k.startswith("ema_") or old_k.startswith("output_head."):
            skipped.append(old_k)
            continue
        if old_k in model_dict and model_dict[old_k].shape == tensor.shape:
            model_dict[old_k] = tensor
            transferred.append(old_k)
        else:
            skipped.append(old_k)
    model.load_state_dict(model_dict)
    print(f"  Transferred {len(transferred)} tensors, skipped {len(skipped)}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_session_arrays(sess: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sbp = sess["sbp"]
    kin = sess["kinematics"]
    d_ind = sess["dropout_ind"].astype(np.float32)
    mean, std = session_zscore_params(sbp)
    sbp_z = zscore_normalize(sbp, mean, std).astype(np.float32)
    pos = kin[:, :N_KIN].astype(np.float32)
    vel = kin[:, 2:4].astype(np.float32)
    return sbp_z, d_ind, np.concatenate([pos, vel], axis=1)


class SlidingWindowDataset(Dataset):
    """Training dataset using long sliding windows across the full session.

    Windows of `window_size` bins (5 s = 250 bins at 50 Hz) are
    extracted with `window_stride` step (63 = 75% overlap) from
    each session's continuous SBP/kinematics arrays — NOT restricted to
    individual trial boundaries.  This gives the model cross-trial context
    and eliminates the ~1.25 s cap of the previous per-trial approach.
    """

    def __init__(
        self,
        session_ids: List[str],
        data_dir: str,
        window_size: int = WINDOW_SIZE,
        window_stride: int = WINDOW_STRIDE,
        extra_dropout_max: int = 8,
        training: bool = True,
        p1_data_dir: str = "",
        p1_n_dropout: int = 28,
        p1_splits: List[str] = None,
    ):
        self.window_size = window_size
        self.extra_dropout_max = extra_dropout_max
        self.training = training
        self.windows: List[tuple] = []  # (sbp_z, d_ind, kin_all, valid_len)

        def _add_session(sess: dict):
            sbp_z, d_ind, kin_all = prepare_session_arrays(sess)
            n = sbp_z.shape[0]
            W, S = window_size, window_stride
            starts = list(range(0, n, S))
            for cs in starts:
                ce = min(cs + W, n)
                cl = ce - cs
                if cl < 5:
                    continue
                self.windows.append((sbp_z[cs:ce], d_ind, kin_all[cs:ce], cl))

        for sid in session_ids:
            _add_session(load_session(data_dir, sid, is_test=False))

        if p1_data_dir:
            splits = p1_splits if p1_splits else ["train"]
            p1_count = 0
            for split in splits:
                for sid in list_p1_session_ids(p1_data_dir, split=split):
                    _add_session(load_p1_session_as_p2(p1_data_dir, sid, n_dropout=p1_n_dropout, split=split))
                    p1_count += 1
            print(f"  Added Phase 1 data: {p1_count} sessions ({splits})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sbp_z, d_ind, kin_all, cl = self.windows[idx]
        W = self.window_size

        sbp_chunk = sbp_z[:cl].copy()
        kin_chunk = kin_all[:cl].copy()
        drop_vec  = d_ind.copy()

        if self.training and self.extra_dropout_max > 0:
            active_ch = np.where(d_ind == 0.0)[0]
            max_extra = min(self.extra_dropout_max, len(active_ch))
            n_extra = np.random.randint(0, max_extra + 1) if max_extra > 0 else 0
            if n_extra > 0:
                extra_ch = np.random.choice(active_ch, size=n_extra, replace=False)
                sbp_chunk[:, extra_ch] = 0.0
                drop_vec[extra_ch] = 1.0

        d_chunk = np.tile(drop_vec, (cl, 1))
        zeros4  = np.zeros((cl, 4), dtype=np.float32)
        feat    = np.concatenate([sbp_chunk, d_chunk, zeros4], axis=1)

        def pad(a: np.ndarray) -> np.ndarray:
            if a.shape[0] >= W:
                return a[:W]
            return np.pad(a, [(0, W - a.shape[0])] + [(0, 0)] * (a.ndim - 1))

        pmask = np.ones(W, dtype=bool)
        pmask[:cl] = False

        return (
            torch.from_numpy(pad(feat)),
            torch.from_numpy(pad(kin_chunk[:, :N_KIN])),
            torch.from_numpy(pad(kin_chunk[:, 2:4])),
            torch.from_numpy(pmask),
        )


def make_scheduler(optimizer, warmup: int, total: int, min_ratio: float = 0.01):
    def lr_fn(ep: int):
        if ep < warmup:
            return (ep + 1) / max(warmup, 1)
        p = (ep - warmup) / max(total - warmup, 1)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def compute_losses(
    pred_pos: torch.Tensor,
    pred_vel: torch.Tensor,
    true_pos: torch.Tensor,
    true_vel: torch.Tensor,
    pmask: torch.Tensor,
    vel_weight: float,
    consistency_weight: float,
) -> tuple[torch.Tensor, dict]:
    valid = (~pmask).unsqueeze(-1).float()
    denom = valid.sum().clamp(min=1.0)

    pos_loss = ((pred_pos - true_pos) ** 2 * valid).sum() / denom
    vel_loss = ((pred_vel - true_vel) ** 2 * valid).sum() / denom

    pair_valid = ((~pmask[:, 1:]) & (~pmask[:, :-1])).unsqueeze(-1).float()
    pair_denom = pair_valid.sum().clamp(min=1.0)
    pos_diff = pred_pos[:, 1:] - pred_pos[:, :-1]
    cons_loss = ((pos_diff - pred_vel[:, :-1]) ** 2 * pair_valid).sum() / pair_denom

    loss = pos_loss + vel_weight * vel_loss + consistency_weight * cons_loss
    stats = {
        "pos_loss": float(pos_loss.item()),
        "vel_loss": float(vel_loss.item()),
        "cons_loss": float(cons_loss.item()),
    }
    return loss, stats


def train_epoch(
    model: MAEMultitaskDecoder,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    vel_weight: float,
    consistency_weight: float,
):
    model.train()
    total_loss = total_pos = total_vel = total_cons = 0.0
    n = 0
    for feat, pos, vel, pmask in loader:
        feat = feat.to(device)
        pos = pos.to(device)
        vel = vel.to(device)
        pmask = pmask.to(device)

        pred_pos, pred_vel = model(feat, padding_mask=pmask)
        loss, stats = compute_losses(
            pred_pos, pred_vel, pos, vel, pmask, vel_weight, consistency_weight
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_pos += stats["pos_loss"]
        total_vel += stats["vel_loss"]
        total_cons += stats["cons_loss"]
        n += 1

    n = max(n, 1)
    return {
        "loss": total_loss / n,
        "pos_loss": total_pos / n,
        "vel_loss": total_vel / n,
        "cons_loss": total_cons / n,
    }


def make_chunk_starts(length: int, window: int, stride: int) -> List[int]:
    if length <= window:
        return [0]
    starts = list(range(0, max(length - window + 1, 1), stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_pos_vel(pos_pred: np.ndarray, vel_pred: np.ndarray, position_blend: float) -> np.ndarray:
    integrated = np.empty_like(pos_pred)
    integrated[0] = pos_pred[0]
    if len(pos_pred) > 1:
        integrated[1:] = pos_pred[0] + np.cumsum(vel_pred[:-1], axis=0)
    hybrid = position_blend * pos_pred + (1.0 - position_blend) * integrated
    return np.clip(hybrid, 0.0, 1.0)


@torch.no_grad()
def predict_session_positions(
    model: MAEMultitaskDecoder,
    sess: dict,
    device: torch.device,
    window_size: int,
    window_stride: int,
    position_blend: float,
) -> np.ndarray:
    """Slide a window of `window_size` bins across the FULL session (no trial
    boundaries) with `window_stride` step.  Hanning-weighted overlap averaging
    yields smooth predictions at every time bin."""
    sbp = sess["sbp"]
    d_ind = sess["dropout_ind"]
    mean, std = session_zscore_params(sbp)
    sbp_z = zscore_normalize(sbp, mean, std).astype(np.float32)
    n_bins = sbp_z.shape[0]
    d_tiled = np.tile(d_ind, (n_bins, 1))
    zeros4  = np.zeros((n_bins, 4), dtype=np.float32)
    feat    = np.concatenate([sbp_z, d_tiled, zeros4], axis=1)

    pred_sum   = np.zeros((n_bins, N_KIN), dtype=np.float32)
    weight_sum = np.zeros(n_bins, dtype=np.float32)
    model.eval()

    for cs in make_chunk_starts(n_bins, window_size, window_stride):
        ce = min(cs + window_size, n_bins)
        cl = ce - cs
        chunk = feat[cs:ce]
        if cl < window_size:
            chunk = np.pad(chunk, ((0, window_size - cl), (0, 0)))

        ft = torch.from_numpy(chunk).unsqueeze(0).to(device)
        pm = torch.ones(1, window_size, dtype=torch.bool, device=device)
        pm[0, :cl] = False
        pos_pred, vel_pred = model(ft, padding_mask=pm)
        pos_np = pos_pred.cpu().numpy()[0, :cl]
        vel_np = vel_pred.cpu().numpy()[0, :cl]
        hybrid = blend_pos_vel(pos_np, vel_np, position_blend)

        weights = np.hanning(cl).astype(np.float32) if cl >= 3 else np.ones(cl, dtype=np.float32)
        weights = np.maximum(weights, 1e-3)
        pred_sum[cs:ce]   += hybrid * weights[:, None]
        weight_sum[cs:ce] += weights

    return np.clip(pred_sum / np.maximum(weight_sum[:, None], 1e-6), 0.0, 1.0)


@torch.no_grad()
def validate_model(
    model: MAEMultitaskDecoder,
    val_ids: List[str],
    data_dir: str,
    device: torch.device,
    window_size: int,
    window_stride: int,
    position_blend: float,
) -> float:
    results = []
    for sid in val_ids:
        sess = load_session(data_dir, sid, is_test=False)
        pred_pos = predict_session_positions(
            model, sess, device,
            window_size=window_size,
            window_stride=window_stride,
            position_blend=position_blend,
        )
        true_pos = sess["kinematics"][:, :N_KIN].astype(np.float32)
        results.append((pred_pos, true_pos, sid))
    return compute_r2_multi(results)


def make_balanced_val_subset(val_ids: List[str], n: int = 8) -> List[str]:
    n = min(n, len(val_ids))
    n_half = n // 2
    hard = val_ids[-n_half:]
    easy = val_ids[: n - n_half]
    return easy + hard


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data-dir", type=str, default=DATA_DIR)
    pa.add_argument("--p1-data-dir", type=str, default="",
                    help="Path to Phase 1 kaggle_data dir; enables cross-phase augmentation")
    pa.add_argument("--p1-splits", type=str, default="train",
                    help="Comma-separated Phase 1 splits to include: train,test")
    pa.add_argument("--p1-n-dropout", type=int, default=28,
                    help="Channels to zero per Phase 1 session (simulates Phase 2 dropout)")
    pa.add_argument("--pretrained-checkpoint", type=str, default=DEFAULT_PRETRAINED)
    pa.add_argument("--no-pretrain", action="store_true")
    pa.add_argument("--epochs", type=int, default=50)
    pa.add_argument("--batch-size", type=int, default=48)
    pa.add_argument("--encoder-lr", type=float, default=1e-5)
    pa.add_argument("--head-lr", type=float, default=5e-4)
    pa.add_argument("--warmup-epochs", type=int, default=5)
    pa.add_argument("--d-model", type=int, default=256)
    pa.add_argument("--nhead", type=int, default=4)
    pa.add_argument("--num-layers", type=int, default=4)
    pa.add_argument("--d-ff", type=int, default=512)
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                    help="Sliding window size in bins (250=5s at 50Hz)")
    pa.add_argument("--window-stride", type=int, default=WINDOW_STRIDE,
                    help="Sliding window stride in bins (default: 63 = 75%% overlap)")
    pa.add_argument("--position-blend", type=float, default=0.75)
    pa.add_argument("--vel-weight", type=float, default=0.5)
    pa.add_argument("--consistency-weight", type=float, default=0.25)
    pa.add_argument("--extra-dropout-max", type=int, default=8)
    pa.add_argument("--n-val-sessions", type=int, default=8)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--save-every", type=int, default=10)
    pa.add_argument("--quick", action="store_true")
    pa.add_argument("--wandb", action="store_true")
    pa.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    pa.add_argument("--wandb-name", type=str, default=None)
    args = pa.parse_args()

    set_seed(args.seed)
    use_wandb = args.wandb
    if use_wandb:
        import wandb

        name = args.wandb_name or f"mae_mt_s{args.seed}_vb{args.vel_weight}_cb{args.consistency_weight}"
        wandb.init(project=args.wandb_project, name=name, config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ids, val_ids = get_validation_sessions(args.data_dir)
    if args.quick:
        train_ids, val_ids = train_ids[:8], val_ids[:8]
        args.epochs = 3

    val_subset = make_balanced_val_subset(val_ids, n=args.n_val_sessions)
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Val subset: {len(val_subset)}")

    p1_splits = [s.strip() for s in args.p1_splits.split(",") if s.strip()]
    window_stride = args.window_stride if args.window_stride > 0 else args.window_size // 2
    print("Building dataset...")
    t0 = time.time()
    ds = SlidingWindowDataset(
        train_ids,
        args.data_dir,
        window_size=args.window_size,
        window_stride=window_stride,
        extra_dropout_max=args.extra_dropout_max,
        training=True,
        p1_data_dir=args.p1_data_dir,
        p1_n_dropout=args.p1_n_dropout,
        p1_splits=p1_splits,
    )
    print(f"  {len(ds)} windows ({time.time() - t0:.1f}s)")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = MAEMultitaskDecoder(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
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

    opt = torch.optim.AdamW(
        [
            {"params": model.encoder_params(), "lr": args.encoder_lr},
            {"params": model.head_params(), "lr": args.head_lr},
        ],
        weight_decay=1e-4,
    )
    sched = make_scheduler(opt, args.warmup_epochs, args.epochs)

    def save_ckpt(path: str, ep: int, vr2: float, best_r2: float):
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "val_r2": vr2,
                "best_r2": best_r2,
            },
            path,
        )

    best = -float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_epoch(
            model,
            loader,
            opt,
            device,
            vel_weight=args.vel_weight,
            consistency_weight=args.consistency_weight,
        )
        sched.step()
        vr2 = validate_model(
            model,
            val_subset,
            args.data_dir,
            device,
            window_size=args.window_size,
            window_stride=window_stride,
            position_blend=args.position_blend,
        )
        dt = time.time() - t0
        is_best = vr2 > best
        if is_best:
            best = vr2
            save_ckpt("best_model.pt", ep, vr2, best)
        if args.save_every and ep % args.save_every == 0:
            save_ckpt(f"checkpoint_epoch{ep}.pt", ep, vr2, best)
        save_ckpt("last_model.pt", ep, vr2, best)

        lr_enc = opt.param_groups[0]["lr"]
        lr_head = opt.param_groups[1]["lr"]
        print(
            f"Ep {ep:3d}/{args.epochs} | loss={train_stats['loss']:.4f} "
            f"(pos={train_stats['pos_loss']:.4f} vel={train_stats['vel_loss']:.4f} "
            f"cons={train_stats['cons_loss']:.4f}) | val_r2={vr2:.4f} "
            f"{'*' if is_best else ''} | lr_enc={lr_enc:.2e} lr_head={lr_head:.2e} | {dt:.0f}s"
        )
        if use_wandb:
            wandb.log(
                {
                    "epoch": ep,
                    "train/loss": train_stats["loss"],
                    "train/pos_loss": train_stats["pos_loss"],
                    "train/vel_loss": train_stats["vel_loss"],
                    "train/cons_loss": train_stats["cons_loss"],
                    "val/r2": vr2,
                    "val/best_r2": best,
                    "lr/encoder": lr_enc,
                    "lr/head": lr_head,
                    "time_s": dt,
                }
            )

    print("\nFull validation with best model...")
    ckpt = torch.load("best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    full_r2 = validate_model(
        model,
        val_ids,
        args.data_dir,
        device,
        window_size=args.window_size,
        window_stride=window_stride,
        position_blend=args.position_blend,
    )
    print(f"Full val R²: {full_r2:.4f} (best epoch {ckpt['epoch']})")

    cfg = {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "window_size": args.window_size,
        "window_stride": window_stride,
        "position_blend": args.position_blend,
        "vel_weight": args.vel_weight,
        "consistency_weight": args.consistency_weight,
        "extra_dropout_max": args.extra_dropout_max,
        "warmup_epochs": args.warmup_epochs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "seed": args.seed,
        "pretrained_checkpoint": "" if args.no_pretrain else args.pretrained_checkpoint,
        "best_val_r2": best,
        "best_epoch": ckpt["epoch"],
        "full_val_r2": full_r2,
    }
    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("Saved config.json, best_model.pt, last_model.pt")

    if use_wandb:
        wandb.summary.update(
            {"r2/best_val": best, "r2/full_val": full_r2, "best_epoch": ckpt["epoch"]}
        )
        wandb.finish()


if __name__ == "__main__":
    main()

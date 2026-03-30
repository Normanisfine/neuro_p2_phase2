"""
MAE Multitask Sweep — Phase 2 finger position decoding
=======================================================

Config-driven sweep over window size and overlap.
All runs use continuous sliding windows (no trial boundaries) for both
training and inference, and include Phase 1 data augmentation.

Usage:
    python train.py --config configs/5s_75ov.json
    python train.py --config configs/5s_75ov.json --quick
    python train.py --config configs/5s_75ov.json --wandb
"""

import argparse
import json
import math
import os
import random
import sys
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
    N_KIN,
    compute_r2_multi,
    get_validation_sessions,
    list_p1_session_ids,
    load_p1_session_as_p2,
    load_session,
    session_zscore_params,
    zscore_normalize,
)

INPUT_DIM_P1 = 96 + 96 + 4
N_VEL = 2

PHASE1_CKPT = "/scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/sweep_3325164_5/best_model.pt"

DEFAULTS = {
    "pretrained_checkpoint": PHASE1_CKPT,
    "epochs": 50,
    "batch_size": 32,
    "encoder_lr": 1e-5,
    "head_lr": 5e-4,
    "warmup_epochs": 5,
    "d_model": 256,
    "nhead": 4,
    "num_layers": 4,
    "d_ff": 512,
    "dropout": 0.1,
    "window_size": 250,
    "window_stride": 63,
    "position_blend": 0.75,
    "vel_weight": 0.5,
    "consistency_weight": 0.25,
    "extra_dropout_max": 8,
    "n_val_sessions": 8,
    "seed": 0,
    "save_every": 10,
    "p1_data_dir": P1_DATA_DIR,
    "p1_splits": "train",
    "p1_n_dropout": 28,
    "outdir": "",
    "wandb_project": "neuro-p2-mt-sweep",
    "wandb_name": None,
}


def load_config(config_path, cli_overrides):
    cfg = dict(DEFAULTS)
    if config_path:
        with open(config_path) as f:
            cfg.update(json.load(f))
        cfg["config_name"] = os.path.splitext(os.path.basename(config_path))[0]
    else:
        cfg["config_name"] = "adhoc"
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v
    if not cfg["outdir"]:
        cfg["outdir"] = os.path.join(THIS_DIR, "runs", cfg["config_name"])
    cfg["outdir"] = os.path.abspath(cfg["outdir"])
    return cfg


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

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MAEMultitaskDecoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM_P1, d_model=256, nhead=4, num_layers=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc = SinusoidalPE(d_model, max_len=2048)
        self.drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.shared_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_KIN), nn.Sigmoid(),
        )
        self.vel_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_VEL),
        )

    def encode(self, x, padding_mask=None):
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.pos_enc(h)
        h = self.drop(h)
        return self.encoder(h, src_key_padding_mask=padding_mask)

    def forward(self, x, padding_mask=None):
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


def load_pretrained_encoder(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pretrained = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ep = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    nmse = ckpt.get("val_nmse", "?") if isinstance(ckpt, dict) else "?"
    print(f"  Phase 1 checkpoint: epoch={ep}, val_nmse={nmse}")
    model_dict = model.state_dict()
    transferred, skipped = [], []
    for k, v in pretrained.items():
        if k.startswith("ema_") or k.startswith("output_head."):
            skipped.append(k)
            continue
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            transferred.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(model_dict)
    print(f"  Transferred {len(transferred)} tensors, skipped {len(skipped)}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_session_arrays(sess):
    sbp = sess["sbp"]
    kin = sess["kinematics"]
    d_ind = sess["dropout_ind"].astype(np.float32)
    mean, std = session_zscore_params(sbp)
    sbp_z = zscore_normalize(sbp, mean, std).astype(np.float32)
    pos = kin[:, :N_KIN].astype(np.float32)
    vel = kin[:, 2:4].astype(np.float32)
    return sbp_z, d_ind, np.concatenate([pos, vel], axis=1)


class SlidingWindowDataset(Dataset):
    """Continuous sliding windows across the full session — no trial boundaries."""

    def __init__(self, session_ids, data_dir, window_size, window_stride,
                 extra_dropout_max=8, training=True,
                 p1_data_dir="", p1_n_dropout=28, p1_splits=None):
        self.window_size = window_size
        self.extra_dropout_max = extra_dropout_max
        self.training = training
        self.windows: List[tuple] = []

        def _add_session(sess):
            sbp_z, d_ind, kin_all = prepare_session_arrays(sess)
            n = sbp_z.shape[0]
            for cs in range(0, n, window_stride):
                ce = min(cs + window_size, n)
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
        drop_vec = d_ind.copy()

        if self.training and self.extra_dropout_max > 0:
            active_ch = np.where(d_ind == 0.0)[0]
            max_extra = min(self.extra_dropout_max, len(active_ch))
            n_extra = np.random.randint(0, max_extra + 1) if max_extra > 0 else 0
            if n_extra > 0:
                extra_ch = np.random.choice(active_ch, size=n_extra, replace=False)
                sbp_chunk[:, extra_ch] = 0.0
                drop_vec[extra_ch] = 1.0

        d_chunk = np.tile(drop_vec, (cl, 1))
        zeros4 = np.zeros((cl, 4), dtype=np.float32)
        feat = np.concatenate([sbp_chunk, d_chunk, zeros4], axis=1)

        def pad(a):
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


def make_scheduler(optimizer, warmup, total, min_ratio=0.01):
    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / max(warmup, 1)
        p = (ep - warmup) / max(total - warmup, 1)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def compute_losses(pred_pos, pred_vel, true_pos, true_vel, pmask, vel_weight, consistency_weight):
    valid = (~pmask).unsqueeze(-1).float()
    denom = valid.sum().clamp(min=1.0)
    pos_loss = ((pred_pos - true_pos) ** 2 * valid).sum() / denom
    vel_loss = ((pred_vel - true_vel) ** 2 * valid).sum() / denom
    pair_valid = ((~pmask[:, 1:]) & (~pmask[:, :-1])).unsqueeze(-1).float()
    pair_denom = pair_valid.sum().clamp(min=1.0)
    pos_diff = pred_pos[:, 1:] - pred_pos[:, :-1]
    cons_loss = ((pos_diff - pred_vel[:, :-1]) ** 2 * pair_valid).sum() / pair_denom
    loss = pos_loss + vel_weight * vel_loss + consistency_weight * cons_loss
    return loss, {"pos_loss": float(pos_loss), "vel_loss": float(vel_loss), "cons_loss": float(cons_loss)}


def train_epoch(model, loader, optimizer, device, vel_weight, consistency_weight):
    model.train()
    total_loss = total_pos = total_vel = total_cons = 0.0
    n = 0
    for feat, pos, vel, pmask in loader:
        feat, pos, vel, pmask = feat.to(device), pos.to(device), vel.to(device), pmask.to(device)
        pred_pos, pred_vel = model(feat, padding_mask=pmask)
        loss, stats = compute_losses(pred_pos, pred_vel, pos, vel, pmask, vel_weight, consistency_weight)
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
    return {"loss": total_loss/n, "pos_loss": total_pos/n, "vel_loss": total_vel/n, "cons_loss": total_cons/n}


def make_chunk_starts(length, window, stride):
    if length <= window:
        return [0]
    starts = list(range(0, max(length - window + 1, 1), stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_pos_vel(pos_pred, vel_pred, position_blend):
    integrated = np.empty_like(pos_pred)
    integrated[0] = pos_pred[0]
    if len(pos_pred) > 1:
        integrated[1:] = pos_pred[0] + np.cumsum(vel_pred[:-1], axis=0)
    return np.clip(position_blend * pos_pred + (1.0 - position_blend) * integrated, 0.0, 1.0)


@torch.no_grad()
def predict_session_positions(model, sess, device, window_size, window_stride, position_blend):
    """Continuous overlap-averaged inference — no trial boundaries."""
    sbp = sess["sbp"]
    d_ind = sess["dropout_ind"]
    mean, std = session_zscore_params(sbp)
    sbp_z = zscore_normalize(sbp, mean, std).astype(np.float32)
    n_bins = sbp_z.shape[0]
    feat = np.concatenate([sbp_z, np.tile(d_ind, (n_bins, 1)), np.zeros((n_bins, 4), dtype=np.float32)], axis=1)

    pred_sum = np.zeros((n_bins, N_KIN), dtype=np.float32)
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
        hybrid = blend_pos_vel(pos_pred.cpu().numpy()[0, :cl], vel_pred.cpu().numpy()[0, :cl], position_blend)
        weights = np.hanning(cl).astype(np.float32) if cl >= 3 else np.ones(cl, dtype=np.float32)
        weights = np.maximum(weights, 1e-3)
        pred_sum[cs:ce] += hybrid * weights[:, None]
        weight_sum[cs:ce] += weights

    return np.clip(pred_sum / np.maximum(weight_sum[:, None], 1e-6), 0.0, 1.0)


@torch.no_grad()
def validate_model(model, val_ids, data_dir, device, window_size, window_stride, position_blend):
    results = []
    for sid in val_ids:
        sess = load_session(data_dir, sid, is_test=False)
        pred_pos = predict_session_positions(model, sess, device, window_size, window_stride, position_blend)
        results.append((pred_pos, sess["kinematics"][:, :N_KIN].astype(np.float32), sid))
    return compute_r2_multi(results)


def make_balanced_val_subset(val_ids, n=8):
    n = min(n, len(val_ids))
    n_half = n // 2
    return val_ids[: n - n_half] + val_ids[-n_half:]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", type=str, default=None)
    pa.add_argument("--data-dir", type=str, default=None)
    pa.add_argument("--outdir", type=str, default=None)
    pa.add_argument("--quick", action="store_true")
    pa.add_argument("--wandb", action="store_true")
    pa.add_argument("--no-pretrain", action="store_true")
    args = pa.parse_args()

    cli_overrides = {}
    if args.data_dir:
        cli_overrides["data_dir"] = args.data_dir
    if args.outdir:
        cli_overrides["outdir"] = args.outdir

    cfg = load_config(args.config, cli_overrides)
    os.makedirs(cfg["outdir"], exist_ok=True)

    set_seed(cfg["seed"])

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        name = cfg.get("wandb_name") or cfg["config_name"]
        wandb.init(project=cfg["wandb_project"], name=name, config=cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {cfg['config_name']}")
    print(f"Window: {cfg['window_size']} bins ({cfg['window_size']/50:.1f}s), stride: {cfg['window_stride']} ({100*(1 - cfg['window_stride']/cfg['window_size']):.0f}% overlap)")
    print(f"Outdir: {cfg['outdir']}")

    data_dir = cfg.get("data_dir", DATA_DIR)
    train_ids, val_ids = get_validation_sessions(data_dir)
    if args.quick:
        train_ids, val_ids = train_ids[:8], val_ids[:8]
        cfg["epochs"] = 3

    val_subset = make_balanced_val_subset(val_ids, n=cfg["n_val_sessions"])
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Val subset: {len(val_subset)}")

    p1_splits = [s.strip() for s in cfg["p1_splits"].split(",") if s.strip()]
    print("Building dataset...")
    t0 = time.time()
    ds = SlidingWindowDataset(
        train_ids, data_dir,
        window_size=cfg["window_size"],
        window_stride=cfg["window_stride"],
        extra_dropout_max=cfg["extra_dropout_max"],
        training=True,
        p1_data_dir=cfg["p1_data_dir"],
        p1_n_dropout=cfg["p1_n_dropout"],
        p1_splits=p1_splits,
    )
    print(f"  {len(ds)} windows ({time.time() - t0:.1f}s)")

    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = MAEMultitaskDecoder(
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"], dropout=cfg["dropout"],
    ).to(device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {npar:,}")

    if not args.no_pretrain and cfg.get("pretrained_checkpoint"):
        print(f"Loading pretrained encoder from: {cfg['pretrained_checkpoint']}")
        load_pretrained_encoder(model, cfg["pretrained_checkpoint"], device)

    if use_wandb:
        wandb.config.update({"n_params": npar, "n_windows": len(ds)})

    opt = torch.optim.AdamW(
        [{"params": model.encoder_params(), "lr": cfg["encoder_lr"]},
         {"params": model.head_params(), "lr": cfg["head_lr"]}],
        weight_decay=1e-4,
    )
    sched = make_scheduler(opt, cfg["warmup_epochs"], cfg["epochs"])

    outdir = cfg["outdir"]

    def save_ckpt(fname, ep, vr2, best_r2):
        torch.save({
            "epoch": ep, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "val_r2": vr2, "best_r2": best_r2,
        }, os.path.join(outdir, fname))

    best = -float("inf")
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_stats = train_epoch(model, loader, opt, device, cfg["vel_weight"], cfg["consistency_weight"])
        sched.step()
        vr2 = validate_model(model, val_subset, data_dir, device,
                              cfg["window_size"], cfg["window_stride"], cfg["position_blend"])
        dt = time.time() - t0
        is_best = vr2 > best
        if is_best:
            best = vr2
            save_ckpt("best_model.pt", ep, vr2, best)
        if cfg["save_every"] and ep % cfg["save_every"] == 0:
            save_ckpt(f"checkpoint_epoch{ep}.pt", ep, vr2, best)
        save_ckpt("last_model.pt", ep, vr2, best)

        lr_enc = opt.param_groups[0]["lr"]
        lr_head = opt.param_groups[1]["lr"]
        print(
            f"Ep {ep:3d}/{cfg['epochs']} | loss={train_stats['loss']:.4f} "
            f"(pos={train_stats['pos_loss']:.4f} vel={train_stats['vel_loss']:.4f} "
            f"cons={train_stats['cons_loss']:.4f}) | val_r2={vr2:.4f} "
            f"{'*' if is_best else ''} | lr_enc={lr_enc:.2e} lr_head={lr_head:.2e} | {dt:.0f}s"
        )
        if use_wandb:
            wandb.log({"epoch": ep, "train/loss": train_stats["loss"],
                       "train/pos_loss": train_stats["pos_loss"],
                       "train/vel_loss": train_stats["vel_loss"],
                       "train/cons_loss": train_stats["cons_loss"],
                       "val/r2": vr2, "val/best_r2": best,
                       "lr/encoder": lr_enc, "lr/head": lr_head, "time_s": dt})

    print("\nFull validation with best model...")
    ckpt = torch.load(os.path.join(outdir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    full_r2 = validate_model(model, val_ids, data_dir, device,
                             cfg["window_size"], cfg["window_stride"], cfg["position_blend"])
    print(f"Full val R²: {full_r2:.4f} (best epoch {ckpt['epoch']})")

    result_cfg = {**cfg,
                  "best_val_r2": best,
                  "best_epoch": ckpt["epoch"],
                  "full_val_r2": full_r2}
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(result_cfg, f, indent=2)
    print(f"Saved to {outdir}/")

    if use_wandb:
        wandb.summary.update({"r2/best_val": best, "r2/full_val": full_r2, "best_epoch": ckpt["epoch"]})
        wandb.finish()


if __name__ == "__main__":
    main()

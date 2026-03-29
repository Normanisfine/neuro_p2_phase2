"""
Paper Sweep Lab — unified Phase 2 trainer
=========================================

Config-driven experiment runner for Phase 2. Supports:
  - mae_multitask
  - mae_context
  - transformer_multitask
  - spint_like

The spint_like variant is a lightweight, paper-inspired channel-set encoder:
per-time-step channel tokens are pooled into a temporal sequence representation,
which is then decoded with a temporal Transformer.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, PHASE2_DIR)
from data_utils import (  # noqa: E402
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

N_CHANNELS = 96
N_VEL = 2
INPUT_DIM_P1 = 96 + 96 + 4
CTX_DIM = 96 * 3

DEFAULTS = {
    "variant": "mae_multitask",
    "data_dir": DATA_DIR,
    "pretrained_checkpoint": "/scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/sweep_3325164_5/best_model.pt",
    "epochs": 50,
    "batch_size": 48,
    "encoder_lr": 1e-5,
    "head_lr": 5e-4,
    "lr": 3e-4,
    "warmup_epochs": 5,
    "d_model": 256,
    "nhead": 4,
    "num_layers": 4,
    "d_ff": 512,
    "dropout": 0.1,
    "max_seq_len": 192,
    "inference_stride": 96,
    "position_blend": 0.75,
    "vel_weight": 1.0,
    "consistency_weight": 0.25,
    "extra_dropout_max": 8,
    "n_val_sessions": 8,
    "seed": 0,
    "save_every": 10,
    "unfreeze_last_n": 2,
    "train_input_proj": False,
    "spint_channel_dim": 128,
    "spint_channel_layers": 2,
    "outdir": "",
    "wandb": False,
    "wandb_project": "neuro-p2-paper-sweep",
    "wandb_name": None,
    "p1_data_dir": "",
    "p1_splits": "train",
    "p1_n_dropout": 28,
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_config(config_path, cli_overrides):
    cfg = dict(DEFAULTS)
    if config_path:
        cfg.update(load_json(config_path))
        cfg["config_path"] = os.path.abspath(config_path)
        cfg["config_name"] = os.path.splitext(os.path.basename(config_path))[0]
    else:
        cfg["config_path"] = ""
        cfg["config_name"] = "adhoc"

    for key, value in cli_overrides.items():
        if value is not None:
            cfg[key] = value

    if not cfg["outdir"]:
        cfg["outdir"] = os.path.join(THIS_DIR, "runs", cfg["config_name"])
    cfg["outdir"] = os.path.abspath(cfg["outdir"])
    return cfg


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
    cons_loss = (((pred_pos[:, 1:] - pred_pos[:, :-1]) - pred_vel[:, :-1]) ** 2 * pair_valid).sum() / pair_denom
    total = pos_loss + vel_weight * vel_loss + consistency_weight * cons_loss
    stats = {
        "pos_loss": float(pos_loss.item()),
        "vel_loss": float(vel_loss.item()),
        "cons_loss": float(cons_loss.item()),
        "w_pos": float(pos_loss.item()),
        "w_vel": float((vel_weight * vel_loss).item()),
        "w_cons": float((consistency_weight * cons_loss).item()),
    }
    return total, stats


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


def make_balanced_val_subset(val_ids, n=8):
    n = min(n, len(val_ids))
    n_half = n // 2
    return val_ids[: n - n_half] + val_ids[-n_half:]


def prepare_session_arrays(sess):
    sbp = sess["sbp"]
    kin = sess["kinematics"]
    base_dropout = sess["dropout_ind"].astype(np.float32)
    mean, std = session_zscore_params(sbp)
    sbp_z = zscore_normalize(sbp, mean, std).astype(np.float32)
    kin_all = kin[:, :4].astype(np.float32)
    channel_ctx = np.stack([base_dropout, mean.astype(np.float32), std.astype(np.float32)], axis=1)
    return sbp_z, base_dropout, channel_ctx, kin_all


class TrialDataset(Dataset):
    def __init__(
        self,
        session_ids,
        data_dir,
        max_seq_len,
        extra_dropout_max,
        training=True,
        p1_data_dir="",
        p1_n_dropout=28,
        p1_splits=None,
    ):
        self.max_seq_len = max_seq_len
        self.extra_dropout_max = extra_dropout_max
        self.training = training
        self.trials = []

        for sid in session_ids:
            sess = load_session(data_dir, sid, is_test=False)
            sbp_z, base_dropout, channel_ctx, kin_all = prepare_session_arrays(sess)
            for ti in range(sess["n_trials"]):
                s, e = int(sess["start_bins"][ti]), int(sess["end_bins"][ti])
                if e - s < 5:
                    continue
                self.trials.append((sbp_z[s:e], base_dropout, channel_ctx, kin_all[s:e], e - s))

        if p1_data_dir:
            splits = p1_splits if p1_splits else ["train"]
            p1_loaded = 0
            for split in splits:
                for sid in list_p1_session_ids(p1_data_dir, split=split):
                    sess = load_p1_session_as_p2(p1_data_dir, sid, n_dropout=p1_n_dropout, split=split)
                    sbp_z, base_dropout, channel_ctx, kin_all = prepare_session_arrays(sess)
                    for ti in range(sess["n_trials"]):
                        s, e = int(sess["start_bins"][ti]), int(sess["end_bins"][ti])
                        if e - s < 5:
                            continue
                        self.trials.append((sbp_z[s:e], base_dropout, channel_ctx, kin_all[s:e], e - s))
                        p1_loaded += 1
            print(f"  Added {p1_loaded} trials from Phase 1 data ({splits})")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        sbp_z, base_dropout, channel_ctx, kin_all, tlen = self.trials[idx]
        T = self.max_seq_len

        if tlen > T:
            off = np.random.randint(0, tlen - T + 1)
            sbp_z = sbp_z[off : off + T]
            kin_all = kin_all[off : off + T]
            sl = T
        else:
            sl = tlen

        sbp_chunk = sbp_z[:sl].copy()
        kin_chunk = kin_all[:sl].copy()
        drop_vec = base_dropout.copy()

        if self.training and self.extra_dropout_max > 0:
            active_ch = np.where(base_dropout == 0.0)[0]
            max_extra = min(self.extra_dropout_max, len(active_ch))
            n_extra = np.random.randint(0, max_extra + 1) if max_extra > 0 else 0
            if n_extra > 0:
                extra = np.random.choice(active_ch, size=n_extra, replace=False)
                sbp_chunk[:, extra] = 0.0
                drop_vec[extra] = 1.0

        dropout_t = np.tile(drop_vec, (sl, 1)).astype(np.float32)
        chan_ctx_cur = channel_ctx.copy()
        chan_ctx_cur[:, 0] = drop_vec

        def pad(a):
            if a.shape[0] >= T:
                return a[:T]
            return np.pad(a, [(0, T - a.shape[0])] + [(0, 0)] * (a.ndim - 1))

        pmask = np.ones(T, dtype=bool)
        pmask[:sl] = False
        return (
            torch.from_numpy(pad(sbp_chunk.astype(np.float32))),
            torch.from_numpy(pad(dropout_t)),
            torch.from_numpy(chan_ctx_cur.astype(np.float32)),
            torch.from_numpy(pad(kin_chunk[:, :2].astype(np.float32))),
            torch.from_numpy(pad(kin_chunk[:, 2:4].astype(np.float32))),
            torch.from_numpy(pmask),
        )


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MAEMultitaskDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM_P1, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc = SinusoidalPE(d_model)
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

    def encode(self, x, padding_mask=None):
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.pos_enc(h)
        h = self.drop(h)
        return self.encoder(h, src_key_padding_mask=padding_mask)

    def forward(self, feat, channel_ctx=None, padding_mask=None):
        h = self.shared_head(self.encode(feat, padding_mask))
        return self.pos_head(h), self.vel_head(h)

    def encoder_params(self):
        return list(self.input_proj.parameters()) + list(self.input_norm.parameters()) + list(self.encoder.parameters())

    def head_params(self):
        return list(self.shared_head.parameters()) + list(self.pos_head.parameters()) + list(self.vel_head.parameters()) + list(self.pos_enc.parameters())


class ContextConditionedMAEDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM_P1, d_model)
        self.pos_enc = SinusoidalPE(d_model)
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
        self.ctx_mlp = nn.Sequential(
            nn.Linear(CTX_DIM, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model * 2),
        )
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

    def forward(self, feat, channel_ctx, padding_mask=None):
        flat_ctx = channel_ctx.reshape(channel_ctx.shape[0], -1)
        h = self.input_proj(feat)
        h = self.pos_enc(h)
        h = self.drop(h)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        gamma, beta = torch.chunk(self.ctx_mlp(flat_ctx), 2, dim=-1)
        h = h * (1.0 + 0.1 * torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(1)
        h = self.shared_head(h)
        return self.pos_head(h), self.vel_head(h)


class TransformerMultitaskDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=6, d_ff=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(N_CHANNELS * 2, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc = SinusoidalPE(d_model)
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

    def forward(self, feat, channel_ctx=None, padding_mask=None):
        h = self.input_proj(feat)
        h = self.input_norm(h)
        h = self.pos_enc(h)
        h = self.drop(h)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        h = self.shared_head(h)
        return self.pos_head(h), self.vel_head(h)


class SPINTLikeDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4, d_ff=512, dropout=0.1, channel_dim=128, channel_layers=2):
        super().__init__()
        self.channel_proj = nn.Linear(4, channel_dim)
        self.channel_norm = nn.LayerNorm(channel_dim)
        ch_layer = nn.TransformerEncoderLayer(
            d_model=channel_dim,
            nhead=max(1, min(4, channel_dim // 32)),
            dim_feedforward=channel_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_encoder = nn.TransformerEncoder(ch_layer, num_layers=channel_layers)
        self.pool_score = nn.Linear(channel_dim, 1)
        self.time_proj = nn.Linear(channel_dim, d_model)
        self.time_norm = nn.LayerNorm(d_model)
        self.pos_enc = SinusoidalPE(d_model)
        self.drop = nn.Dropout(dropout)
        time_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.time_encoder = nn.TransformerEncoder(time_layer, num_layers=num_layers)
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

    def forward(self, sbp_z, dropout_t, channel_ctx, padding_mask=None):
        B, T, C = sbp_z.shape
        ctx = channel_ctx.unsqueeze(1).expand(B, T, C, 3)
        x = torch.cat([sbp_z.unsqueeze(-1), dropout_t.unsqueeze(-1), ctx[:, :, :, 1:]], dim=-1)
        x = x.reshape(B * T, C, 4)
        x = self.channel_proj(x)
        x = self.channel_norm(x)
        x = self.channel_encoder(x)
        scores = self.pool_score(x).squeeze(-1)
        dyn_drop = dropout_t.reshape(B * T, C)
        scores = scores.masked_fill(dyn_drop > 0.5, -1e9)
        attn = torch.softmax(scores, dim=-1)
        pooled = (x * attn.unsqueeze(-1)).sum(dim=1)
        pooled = pooled.reshape(B, T, -1)
        h = self.time_proj(pooled)
        h = self.time_norm(h)
        h = self.pos_enc(h)
        h = self.drop(h)
        h = self.time_encoder(h, src_key_padding_mask=padding_mask)
        h = self.shared_head(h)
        return self.pos_head(h), self.vel_head(h)


def load_pretrained_encoder(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pretrained = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
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


def freeze_context_model(model, unfreeze_last_n, train_input_proj=False):
    for p in model.input_proj.parameters():
        p.requires_grad_(train_input_proj)
    for p in model.pos_enc.parameters():
        p.requires_grad_(True)
    layers = list(model.encoder.layers)
    cutoff = len(layers) - min(max(unfreeze_last_n, 0), len(layers))
    for i, layer in enumerate(layers):
        for p in layer.parameters():
            p.requires_grad_(i >= cutoff)
    for name in ["ctx_mlp", "shared_head", "pos_head", "vel_head"]:
        for p in getattr(model, name).parameters():
            p.requires_grad_(True)


def build_model(cfg, device):
    variant = cfg["variant"]
    common_kwargs = dict(
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    )
    if variant == "mae_multitask":
        model = MAEMultitaskDecoder(**common_kwargs)
        if cfg.get("pretrained_checkpoint"):
            print(f"Loading pretrained encoder from: {cfg['pretrained_checkpoint']}")
            load_pretrained_encoder(model, cfg["pretrained_checkpoint"], device)
    elif variant == "mae_context":
        model = ContextConditionedMAEDecoder(**common_kwargs)
        print(f"Loading pretrained encoder from: {cfg['pretrained_checkpoint']}")
        load_pretrained_encoder(model, cfg["pretrained_checkpoint"], device)
        freeze_context_model(model, cfg["unfreeze_last_n"], train_input_proj=cfg["train_input_proj"])
    elif variant == "transformer_multitask":
        model = TransformerMultitaskDecoder(**common_kwargs)
    elif variant == "spint_like":
        model = SPINTLikeDecoder(
            **common_kwargs,
            channel_dim=cfg["spint_channel_dim"],
            channel_layers=cfg["spint_channel_layers"],
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return model.to(device)


def build_optimizer(model, cfg):
    variant = cfg["variant"]
    if variant == "mae_multitask":
        opt = torch.optim.AdamW(
            [
                {"params": model.encoder_params(), "lr": cfg["encoder_lr"]},
                {"params": model.head_params(), "lr": cfg["head_lr"]},
            ],
            weight_decay=1e-4,
        )
    elif variant == "mae_context":
        enc_params, head_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(("ctx_mlp", "shared_head", "pos_head", "vel_head")):
                head_params.append(p)
            else:
                enc_params.append(p)
        opt = torch.optim.AdamW(
            [
                {"params": enc_params, "lr": cfg["encoder_lr"]},
                {"params": head_params, "lr": cfg["head_lr"]},
            ],
            weight_decay=1e-4,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    return opt


def model_forward(model, cfg, sbp_z, dropout_t, channel_ctx, pmask):
    variant = cfg["variant"]
    if variant == "mae_multitask":
        zeros4 = torch.zeros(sbp_z.shape[0], sbp_z.shape[1], 4, device=sbp_z.device, dtype=sbp_z.dtype)
        feat = torch.cat([sbp_z, dropout_t, zeros4], dim=-1)
        return model(feat, padding_mask=pmask)
    if variant == "mae_context":
        zeros4 = torch.zeros(sbp_z.shape[0], sbp_z.shape[1], 4, device=sbp_z.device, dtype=sbp_z.dtype)
        feat = torch.cat([sbp_z, dropout_t, zeros4], dim=-1)
        return model(feat, channel_ctx, padding_mask=pmask)
    if variant == "transformer_multitask":
        feat = torch.cat([sbp_z, dropout_t], dim=-1)
        return model(feat, padding_mask=pmask)
    if variant == "spint_like":
        return model(sbp_z, dropout_t, channel_ctx, padding_mask=pmask)
    raise ValueError(cfg["variant"])


def train_epoch(model, loader, optimizer, device, cfg):
    model.train()
    totals = {"loss": 0.0, "pos_loss": 0.0, "vel_loss": 0.0, "cons_loss": 0.0, "w_pos": 0.0, "w_vel": 0.0, "w_cons": 0.0}
    n = 0
    for sbp_z, dropout_t, channel_ctx, pos, vel, pmask in loader:
        sbp_z = sbp_z.to(device)
        dropout_t = dropout_t.to(device)
        channel_ctx = channel_ctx.to(device)
        pos = pos.to(device)
        vel = vel.to(device)
        pmask = pmask.to(device)

        pred_pos, pred_vel = model_forward(model, cfg, sbp_z, dropout_t, channel_ctx, pmask)
        loss, stats = compute_losses(pred_pos, pred_vel, pos, vel, pmask, cfg["vel_weight"], cfg["consistency_weight"])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        totals["loss"] += loss.item()
        for k, v in stats.items():
            totals[k] += v
        n += 1

    n = max(n, 1)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def predict_session_positions(model, sess, device, cfg):
    sbp = sess["sbp"]
    base_dropout = sess["dropout_ind"].astype(np.float32)
    mean, std = session_zscore_params(sbp)
    sbp_z = zscore_normalize(sbp, mean, std).astype(np.float32)
    chan_ctx = np.stack([base_dropout, mean.astype(np.float32), std.astype(np.float32)], axis=1)
    n_bins = sbp_z.shape[0]
    dropout_t = np.tile(base_dropout, (n_bins, 1)).astype(np.float32)
    pred_all = np.full((n_bins, N_KIN), 0.5, dtype=np.float32)

    model.eval()
    for ti in range(sess["n_trials"]):
        s, e = int(sess["start_bins"][ti]), int(sess["end_bins"][ti])
        tlen = e - s
        if tlen <= 0:
            continue
        pred_sum = np.zeros((tlen, N_KIN), dtype=np.float32)
        weight_sum = np.zeros(tlen, dtype=np.float32)
        for cs in make_chunk_starts(tlen, cfg["max_seq_len"], cfg["inference_stride"]):
            ce = min(cs + cfg["max_seq_len"], tlen)
            cl = ce - cs
            sbp_chunk = sbp_z[s + cs : s + ce]
            drop_chunk = dropout_t[s + cs : s + ce]
            if cl < cfg["max_seq_len"]:
                pad = cfg["max_seq_len"] - cl
                sbp_chunk = np.pad(sbp_chunk, ((0, pad), (0, 0)))
                drop_chunk = np.pad(drop_chunk, ((0, pad), (0, 0)))
            sbp_t = torch.from_numpy(sbp_chunk).unsqueeze(0).to(device)
            drop_t = torch.from_numpy(drop_chunk).unsqueeze(0).to(device)
            ctx_t = torch.from_numpy(chan_ctx).unsqueeze(0).to(device)
            pm = torch.ones(1, cfg["max_seq_len"], dtype=torch.bool, device=device)
            pm[0, :cl] = False
            pos_pred, vel_pred = model_forward(model, cfg, sbp_t, drop_t, ctx_t, pm)
            pos_np = pos_pred.cpu().numpy()[0, :cl]
            vel_np = vel_pred.cpu().numpy()[0, :cl]
            hybrid = blend_pos_vel(pos_np, vel_np, cfg["position_blend"])
            weights = np.hanning(cl).astype(np.float32) if cl >= 3 else np.ones(cl, dtype=np.float32)
            weights = np.maximum(weights, 1e-3)
            pred_sum[cs:ce] += hybrid * weights[:, None]
            weight_sum[cs:ce] += weights
        pred_all[s:e] = pred_sum / np.maximum(weight_sum[:, None], 1e-6)
    return np.clip(pred_all, 0.0, 1.0)


@torch.no_grad()
def validate_model(model, val_ids, data_dir, device, cfg):
    results = []
    for sid in val_ids:
        sess = load_session(data_dir, sid, is_test=False)
        pred_pos = predict_session_positions(model, sess, device, cfg)
        true_pos = sess["kinematics"][:, :N_KIN].astype(np.float32)
        results.append((pred_pos, true_pos, sid))
    return compute_r2_multi(results)


def save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    cfg = resolve_config(
        args.config,
        {
            "outdir": args.outdir,
            "wandb": True if args.wandb else None,
        },
    )
    os.makedirs(cfg["outdir"], exist_ok=True)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config: {cfg['config_name']}  variant={cfg['variant']}")
    print(f"Outdir: {cfg['outdir']}")
    print(f"Device: {device}")

    use_wandb = cfg["wandb"]
    if use_wandb:
        import wandb

        name = cfg["wandb_name"] or cfg["config_name"]
        wandb.init(project=cfg["wandb_project"], name=name, config=cfg)

    train_ids, val_ids = get_validation_sessions(cfg["data_dir"])
    if args.quick:
        train_ids, val_ids = train_ids[:8], val_ids[:8]
        cfg["epochs"] = min(3, cfg["epochs"])
    val_subset = make_balanced_val_subset(val_ids, cfg["n_val_sessions"])
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Val subset: {len(val_subset)}")

    p1_splits = [s.strip() for s in cfg.get("p1_splits", "train").split(",") if s.strip()]
    print("Building dataset...")
    t0 = time.time()
    ds = TrialDataset(
        train_ids,
        cfg["data_dir"],
        cfg["max_seq_len"],
        cfg["extra_dropout_max"],
        training=True,
        p1_data_dir=cfg.get("p1_data_dir", ""),
        p1_n_dropout=cfg.get("p1_n_dropout", 28),
        p1_splits=p1_splits,
    )
    print(f"  {len(ds)} trials ({time.time() - t0:.1f}s)")
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    model = build_model(cfg, device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {npar:,}")

    if use_wandb:
        wandb.config.update({"n_params": npar, "n_trials": len(ds)}, allow_val_change=True)

    opt = build_optimizer(model, cfg)
    sched = make_scheduler(opt, cfg["warmup_epochs"], cfg["epochs"])

    best_path = os.path.join(cfg["outdir"], "best_model.pt")
    last_path = os.path.join(cfg["outdir"], "last_model.pt")

    def save_ckpt(path, ep, vr2, best_r2):
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "val_r2": vr2,
                "best_r2": best_r2,
                "config": cfg,
            },
            path,
        )

    best = -float("inf")
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_stats = train_epoch(model, loader, opt, device, cfg)
        sched.step()
        vr2 = validate_model(model, val_subset, cfg["data_dir"], device, cfg)
        dt = time.time() - t0
        is_best = vr2 > best
        if is_best:
            best = vr2
            save_ckpt(best_path, ep, vr2, best)
        if cfg["save_every"] and ep % cfg["save_every"] == 0:
            save_ckpt(os.path.join(cfg["outdir"], f"checkpoint_epoch{ep}.pt"), ep, vr2, best)
        save_ckpt(last_path, ep, vr2, best)

        lr0 = opt.param_groups[0]["lr"]
        lr1 = opt.param_groups[-1]["lr"]
        print(
            f"Ep {ep:3d}/{cfg['epochs']} | loss={train_stats['loss']:.4f} "
            f"(pos={train_stats['pos_loss']:.4f} vel={train_stats['vel_loss']:.4f} cons={train_stats['cons_loss']:.4f}; "
            f"wpos={train_stats['w_pos']:.4f} wvel={train_stats['w_vel']:.4f} wcons={train_stats['w_cons']:.4f}) "
            f"| val_r2={vr2:.4f} {'*' if is_best else ''} | "
            f"lr0={lr0:.2e} lr1={lr1:.2e} | {dt:.0f}s"
        )
        if use_wandb:
            wandb.log(
                {
                    "epoch": ep,
                    "train/loss": train_stats["loss"],
                    "train/pos_loss": train_stats["pos_loss"],
                    "train/vel_loss": train_stats["vel_loss"],
                    "train/cons_loss": train_stats["cons_loss"],
                    "train/w_pos": train_stats["w_pos"],
                    "train/w_vel": train_stats["w_vel"],
                    "train/w_cons": train_stats["w_cons"],
                    "val/r2": vr2,
                    "val/best_r2": best,
                    "lr/0": lr0,
                    "lr/1": lr1,
                    "time_s": dt,
                }
            )

    print("\nFull validation with best model...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    full_r2 = validate_model(model, val_ids, cfg["data_dir"], device, cfg)
    print(f"Full val R²: {full_r2:.4f} (best epoch {ckpt['epoch']})")

    final_cfg = dict(cfg)
    final_cfg.update({"best_val_r2": best, "best_epoch": ckpt["epoch"], "full_val_r2": full_r2})
    save_json(os.path.join(cfg["outdir"], "config.json"), final_cfg)
    print("Saved config.json, best_model.pt, last_model.pt")

    if use_wandb:
        wandb.summary.update({"r2/best_val": best, "r2/full_val": full_r2, "best_epoch": ckpt["epoch"]})
        wandb.finish()


if __name__ == "__main__":
    main()

"""
Evaluate submit-time variants on validation sessions with simulated masking.

This mirrors submit.py behavior (plain / TTA / ridge blend), but on train split
validation sessions where GT is available, so NMSE can be computed.
"""

import os
import json
import argparse
import time
import copy
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_utils import (
    DATA_DIR, load_session, get_validation_sessions, simulate_masking,
    session_zscore_params, zscore_normalize, zscore_denormalize, compute_nmse_multi,
)
from train import MaskedAutoencoder, MAX_SEQ_LEN
from submit import SessionFTDataset, finetune_on_session

import importlib.util
_ridge_spec = importlib.util.spec_from_file_location(
    "ridge_train",
    os.path.join(os.path.dirname(__file__), '..', 'ridge_regression', 'train.py'),
)
_ridge_mod = importlib.util.module_from_spec(_ridge_spec)
_ridge_spec.loader.exec_module(_ridge_mod)
ridge_predict_session = _ridge_mod.ridge_predict_session


def predict_masked_session(model, sbp_masked, mask, kin, trial_starts, trial_ends,
                           n_trials, device, max_seq_len):
    mean, std = session_zscore_params(sbp_masked, mask)
    sbp_z = zscore_normalize(sbp_masked, mean, std)
    sbp_z[mask] = 0.0

    pred_z = zscore_normalize(sbp_masked, mean, std)
    pred_z[mask] = 0.0

    model.eval()
    with torch.no_grad():
        for ti in range(n_trials):
            s = int(trial_starts[ti])
            e = int(trial_ends[ti])
            tmask = mask[s:e]
            if not tmask.any():
                continue
            tlen = e - s
            trial_sbp_in = sbp_z[s:e]
            trial_kin = kin[s:e]
            mask_ind = tmask.astype(np.float32)

            for cs in range(0, tlen, max_seq_len):
                ce = min(cs + max_seq_len, tlen)
                cl = ce - cs
                feat = np.concatenate([
                    trial_sbp_in[cs:ce],
                    mask_ind[cs:ce],
                    trial_kin[cs:ce],
                ], axis=1).astype(np.float32)
                if cl < max_seq_len:
                    feat = np.pad(feat, ((0, max_seq_len - cl), (0, 0)), mode='constant')
                ft = torch.from_numpy(feat).unsqueeze(0).to(device)
                pm = torch.ones(1, max_seq_len, dtype=torch.bool, device=device)
                pm[0, :cl] = False
                out = model(ft, padding_mask=pm).cpu().numpy()[0, :cl]

                for lt in range(cl):
                    t = s + cs + lt
                    mc = np.where(mask[t])[0]
                    if len(mc):
                        pred_z[t, mc] = out[lt, mc]

    return zscore_denormalize(pred_z, mean, std)


def build_base_model(checkpoint, device):
    cfg_path = os.path.join(os.path.dirname(checkpoint), "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        max_seq_len = cfg.get("max_seq_len", MAX_SEQ_LEN)
        model = MaskedAutoencoder(
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 4),
            d_ff=cfg.get("d_ff", 512),
            dropout=cfg.get("dropout", 0.1),
        )
    else:
        max_seq_len = MAX_SEQ_LEN
        model = MaskedAutoencoder()

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()
    return model, max_seq_len


def evaluate_variant(base_model, max_seq_len, val_ids, data_dir, device,
                     tta_epochs, tta_lr, tta_bs, ridge_blend, ridge_alpha, seed):
    results = []
    t0 = time.time()
    for i, sid in enumerate(val_ids):
        sess = load_session(data_dir, sid, is_test=False)
        gt = sess["sbp"]
        kin = sess["kinematics"]
        n_trials = sess["n_trials"]
        st = sess["start_bins"]
        ed = sess["end_bins"]
        sim_mask = simulate_masking(gt, st, ed, n_trials, seed=seed + i)
        sbp_masked = gt.copy()
        sbp_masked[sim_mask] = 0.0

        if tta_epochs > 0:
            mean, std = session_zscore_params(sbp_masked, sim_mask)
            sbp_z = zscore_normalize(sbp_masked, mean, std)
            sbp_z[sim_mask] = 0.0
            ft_ds = SessionFTDataset(
                sbp_z, kin, st, ed, n_trials, sim_mask,
                max_seq_len=max_seq_len, seed=1000 + i,
            )
            model = finetune_on_session(
                base_model, ft_ds, device, epochs=tta_epochs, lr=tta_lr, batch_size=tta_bs
            )
        else:
            model = base_model

        mae_pred = predict_masked_session(
            model, sbp_masked, sim_mask, kin, st, ed, n_trials, device, max_seq_len
        )

        if ridge_blend > 0:
            ridge_pred = ridge_predict_session(sbp_masked, sim_mask, alpha=ridge_alpha)
            pred = (1.0 - ridge_blend) * mae_pred + ridge_blend * ridge_pred
        else:
            pred = mae_pred

        results.append((gt, pred, sim_mask, sid))
        if tta_epochs > 0:
            del model
            torch.cuda.empty_cache()

    nmse = compute_nmse_multi(results)
    return nmse, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="Evaluate submit variants on val split")
    ap.add_argument("--data-dir", type=str, default=DATA_DIR)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--tta-bs", type=int, default=32)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-limit", type=int, default=0,
                    help="If >0, evaluate only first N val sessions")
    ap.add_argument("--output", type=str, default="val_variant_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, val_ids = get_validation_sessions(args.data_dir)
    if args.val_limit > 0:
        val_ids = val_ids[:args.val_limit]
    print(f"Val sessions: {len(val_ids)}")

    base_model, max_seq_len = build_base_model(args.checkpoint, device)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"max_seq_len: {max_seq_len}")

    variants = [
        ("sub_best_plain.csv",      0,   1e-4, 0.0),
        ("sub_best_tta3.csv",       3,   1e-4, 0.0),
        ("sub_best_tta5.csv",       5,   1e-4, 0.0),
        ("sub_best_tta10.csv",      10,  1e-4, 0.0),
        ("sub_best_tta5_lr5e5.csv", 5,   5e-5, 0.0),
        ("sub_best_tta5_r01.csv",   5,   1e-4, 0.1),
        ("sub_best_tta5_r02.csv",   5,   1e-4, 0.2),
        ("sub_best_tta5_r03.csv",   5,   1e-4, 0.3),
    ]

    rows = []
    for name, tta_ep, tta_lr, rblend in variants:
        print(f"\n=== {name} ===")
        nmse, sec = evaluate_variant(
            base_model=base_model,
            max_seq_len=max_seq_len,
            val_ids=val_ids,
            data_dir=args.data_dir,
            device=device,
            tta_epochs=tta_ep,
            tta_lr=tta_lr,
            tta_bs=args.tta_bs,
            ridge_blend=rblend,
            ridge_alpha=args.ridge_alpha,
            seed=args.seed,
        )
        print(f"NMSE={nmse:.6f}  time={sec/60:.1f} min")
        rows.append({
            "name": name,
            "tta_epochs": tta_ep,
            "tta_lr": tta_lr,
            "ridge_blend": rblend,
            "nmse": nmse,
            "seconds": sec,
        })

    rows = sorted(rows, key=lambda r: r["nmse"])
    print("\n=== Ranked by NMSE (lower is better) ===")
    for i, r in enumerate(rows, 1):
        print(f"{i:2d}. {r['name']:<24} nmse={r['nmse']:.6f}  "
              f"tta={r['tta_epochs']} lr={r['tta_lr']} rb={r['ridge_blend']}")

    with open(args.output, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "rows": rows}, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()


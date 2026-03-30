"""
MAE Multitask Sweep — submission
=================================

Usage:
    python submit.py --config configs/5s_75ov.json --checkpoint runs/5s_75ov/best_model.pt
"""

import argparse
import json
import os
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, ".."))
from data_utils import DATA_DIR, build_submission, list_session_ids, load_session
from train import (
    MAEMultitaskDecoder, DEFAULTS, load_config,
    predict_session_positions,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    cli = {}
    if args.data_dir:
        cli["data_dir"] = args.data_dir

    cfg = load_config(args.config, cli)
    outdir = cfg["outdir"]

    checkpoint = args.checkpoint or os.path.join(outdir, "best_model.pt")
    output = args.output or os.path.join(outdir, "submission.csv")

    # Load config.json from checkpoint dir (has final window params)
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint))
    saved_cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(saved_cfg_path):
        with open(saved_cfg_path) as f:
            saved = json.load(f)
        window_size = saved.get("window_size", cfg["window_size"])
        window_stride = saved.get("window_stride", cfg["window_stride"])
        position_blend = saved.get("position_blend", cfg["position_blend"])
        d_model = saved.get("d_model", cfg["d_model"])
        nhead = saved.get("nhead", cfg["nhead"])
        num_layers = saved.get("num_layers", cfg["num_layers"])
        d_ff = saved.get("d_ff", cfg["d_ff"])
        dropout = saved.get("dropout", cfg["dropout"])
    else:
        window_size = cfg["window_size"]
        window_stride = cfg["window_stride"]
        position_blend = cfg["position_blend"]
        d_model, nhead, num_layers, d_ff, dropout = (
            cfg["d_model"], cfg["nhead"], cfg["num_layers"], cfg["d_ff"], cfg["dropout"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MAEMultitaskDecoder(d_model=d_model, nhead=nhead, num_layers=num_layers, d_ff=d_ff, dropout=dropout)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded: {checkpoint} (epoch {ckpt.get('epoch','?')}, val_r2={ckpt.get('val_r2','?')})")
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()

    print(f"Inference: window_size={window_size} ({window_size/50:.1f}s), "
          f"stride={window_stride} ({100*(1-window_stride/window_size):.0f}% overlap), "
          f"blend={position_blend:.3f}")

    data_dir = cfg.get("data_dir", DATA_DIR)
    test_ids = list_session_ids(data_dir, split="test")
    print(f"Test sessions: {len(test_ids)}")

    predictions = {}
    for i, sid in enumerate(test_ids):
        t0 = time.time()
        sess = load_session(data_dir, sid, is_test=True)
        pred = predict_session_positions(model, sess, device, window_size, window_stride, position_blend)
        predictions[sid] = pred
        print(f"  [{i+1}/{len(test_ids)}] {sid}: {pred.shape[0]} bins ({time.time()-t0:.1f}s)")

    build_submission(predictions, data_dir, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

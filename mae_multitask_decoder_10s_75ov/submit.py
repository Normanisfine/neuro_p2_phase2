"""
MAE Multitask Decoder — submission
==================================

Loads a multitask checkpoint trained by train.py and generates a Phase 2
submission using overlap-averaged hybrid inference.

Usage:
    python submit.py --checkpoint best_model.pt --output submission.csv
"""

import sys
import os
import argparse
import json
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, ".."))
from data_utils import DATA_DIR, build_submission, list_session_ids, load_session
from train import MAEMultitaskDecoder, WINDOW_SIZE, WINDOW_STRIDE, predict_session_positions


def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    config_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        model = MAEMultitaskDecoder(
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 4),
            d_ff=cfg.get("d_ff", 512),
            dropout=cfg.get("dropout", 0.1),
        )
        window_size   = cfg.get("window_size",   cfg.get("max_seq_len", WINDOW_SIZE))
        window_stride = cfg.get("window_stride", cfg.get("inference_stride", window_size // 2))
        position_blend = cfg.get("position_blend", 0.75)
    else:
        model = MAEMultitaskDecoder()
        window_size   = WINDOW_SIZE
        window_stride = WINDOW_STRIDE
        position_blend = 0.75

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"Loaded: {checkpoint_path} (epoch {ckpt.get('epoch', '?')}, "
            f"val_r2={ckpt.get('val_r2', '?')})"
        )
    else:
        model.load_state_dict(ckpt)
        print(f"Loaded raw state dict: {checkpoint_path}")

    return model.to(device).eval(), window_size, window_stride, position_blend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--output", type=str, default="submission_multitask.csv")
    parser.add_argument("--window-size",    type=int,   default=None,
                        help="Override window size (bins) for inference")
    parser.add_argument("--window-stride",  type=int,   default=None,
                        help="Override window stride (bins) for inference")
    parser.add_argument("--position-blend", type=float, default=None,
                        help="Override direct/integrated position blend in [0, 1]")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, window_size, window_stride, position_blend = load_checkpoint(args.checkpoint, device)
    if args.window_size   is not None: window_size   = args.window_size
    if args.window_stride is not None: window_stride = args.window_stride
    if args.position_blend is not None: position_blend = args.position_blend
    position_blend = max(0.0, min(1.0, position_blend))

    print(
        f"Inference config: window_size={window_size}, window_stride={window_stride}, "
        f"position_blend={position_blend:.3f}"
    )

    test_ids = list_session_ids(args.data_dir, split="test")
    print(f"Test sessions: {len(test_ids)}")

    predictions = {}
    for i, sid in enumerate(test_ids):
        t0 = time.time()
        sess = load_session(args.data_dir, sid, is_test=True)
        pred = predict_session_positions(
            model, sess, device,
            window_size=window_size,
            window_stride=window_stride,
            position_blend=position_blend,
        )
        predictions[sid] = pred
        print(f"  [{i + 1}/{len(test_ids)}] {sid}: {pred.shape[0]} bins ({time.time() - t0:.1f}s)")

    build_submission(predictions, args.data_dir, args.output)


if __name__ == "__main__":
    main()

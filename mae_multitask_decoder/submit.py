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
from train import MAEMultitaskDecoder, MAX_SEQ_LEN, predict_session_positions


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
        max_seq_len = cfg.get("max_seq_len", MAX_SEQ_LEN)
        stride = cfg.get("inference_stride", max_seq_len // 2)
        position_blend = cfg.get("position_blend", 0.75)
    else:
        model = MAEMultitaskDecoder()
        max_seq_len = MAX_SEQ_LEN
        stride = max_seq_len // 2
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

    return model.to(device).eval(), max_seq_len, stride, position_blend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--output", type=str, default="submission_multitask.csv")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override config max_seq_len for inference",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override config overlap stride for inference",
    )
    parser.add_argument(
        "--position-blend",
        type=float,
        default=None,
        help="Override direct/integrated position blend in [0, 1]",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, max_seq_len, stride, position_blend = load_checkpoint(args.checkpoint, device)
    if args.max_seq_len is not None:
        max_seq_len = args.max_seq_len
    if args.stride is not None:
        stride = args.stride
    if args.position_blend is not None:
        position_blend = args.position_blend
    position_blend = max(0.0, min(1.0, position_blend))

    print(
        f"Inference config: max_seq_len={max_seq_len}, stride={stride}, "
        f"position_blend={position_blend:.3f}"
    )

    test_ids = list_session_ids(args.data_dir, split="test")
    print(f"Test sessions: {len(test_ids)}")

    predictions = {}
    for i, sid in enumerate(test_ids):
        t0 = time.time()
        sess = load_session(args.data_dir, sid, is_test=True)
        pred = predict_session_positions(
            model,
            sess,
            device,
            max_seq_len=max_seq_len,
            stride=stride,
            position_blend=position_blend,
        )
        predictions[sid] = pred
        print(f"  [{i + 1}/{len(test_ids)}] {sid}: {pred.shape[0]} bins ({time.time() - t0:.1f}s)")

    build_submission(predictions, args.data_dir, args.output)


if __name__ == "__main__":
    main()

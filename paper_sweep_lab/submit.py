"""
Paper Sweep Lab — submission helper
"""

import argparse
import json
import os
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from train import build_model, predict_session_positions, resolve_config  # noqa: E402

sys.path.insert(0, os.path.dirname(THIS_DIR))
from data_utils import DATA_DIR, build_submission, list_session_ids, load_session  # noqa: E402


def load_checkpoint(checkpoint_path, config_path, device):
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    cfg_path = config_path or os.path.join(ckpt_dir, "config.json")
    cfg = resolve_config(cfg_path, {"outdir": ckpt_dir}) if cfg_path else None
    model = build_model(cfg, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded: {checkpoint_path} (epoch {ckpt.get('epoch', '?')}, val_r2={ckpt.get('val_r2', '?')})")
    else:
        model.load_state_dict(ckpt)
    return model.to(device).eval(), cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--position-blend", type=float, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, cfg = load_checkpoint(args.checkpoint, args.config, device)

    if args.position_blend is not None:
        cfg["position_blend"] = args.position_blend
    if args.max_seq_len is not None:
        cfg["max_seq_len"] = args.max_seq_len
    if args.stride is not None:
        cfg["inference_stride"] = args.stride

    output = args.output or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "submission.csv")
    print(
        f"Inference config: variant={cfg['variant']} max_seq_len={cfg['max_seq_len']} "
        f"stride={cfg['inference_stride']} position_blend={cfg['position_blend']:.3f}"
    )

    predictions = {}
    test_ids = list_session_ids(args.data_dir, split="test")
    print(f"Test sessions: {len(test_ids)}")
    for i, sid in enumerate(test_ids):
        t0 = time.time()
        sess = load_session(args.data_dir, sid, is_test=True)
        pred = predict_session_positions(model, sess, device, cfg)
        predictions[sid] = pred
        print(f"  [{i + 1}/{len(test_ids)}] {sid}: {pred.shape[0]} bins ({time.time() - t0:.1f}s)")

    build_submission(predictions, args.data_dir, output)
    print(f"Submission path: {output}")


if __name__ == "__main__":
    main()

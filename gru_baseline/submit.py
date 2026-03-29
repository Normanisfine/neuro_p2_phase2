"""
Method 1: GRU Baseline — Submission
=====================================
Usage:
    python submit.py
    python submit.py --checkpoint best_model.pt --output submission.csv
"""

import sys
import os
import argparse
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_utils import (
    DATA_DIR, N_CHANNELS, N_KIN,
    load_session, list_session_ids,
    session_zscore_params, zscore_normalize,
    build_submission,
)
from train import GRUDecoder, MAX_SEQ_LEN


@torch.no_grad()
def predict_session(model, sess, device, max_seq_len=MAX_SEQ_LEN):
    sbp     = sess['sbp']
    d_ind   = sess['dropout_ind']
    mean, std = session_zscore_params(sbp)
    sbp_z   = zscore_normalize(sbp, mean, std).astype(np.float32)
    N       = sbp_z.shape[0]
    d_tiled = np.tile(d_ind, (N, 1))
    feat    = np.concatenate([sbp_z, d_tiled], axis=1).astype(np.float32)

    pred_all = np.full((N, N_KIN), 0.5, dtype=np.float32)
    model.eval()

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

    return np.clip(pred_all, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',   type=str, default=DATA_DIR)
    parser.add_argument('--checkpoint', type=str, default='best_model.pt')
    parser.add_argument('--output',     type=str, default='submission.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_dir    = os.path.dirname(os.path.abspath(args.checkpoint))
    config_path = os.path.join(ckpt_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        model = GRUDecoder(
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1),
        )
        max_seq_len = config.get('max_seq_len', MAX_SEQ_LEN)
    else:
        model       = GRUDecoder()
        max_seq_len = MAX_SEQ_LEN

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded: {args.checkpoint} (epoch {ckpt.get('epoch','?')}, "
              f"val_r2={ckpt.get('val_r2','?')})")
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()

    test_ids = list_session_ids(args.data_dir, split='test')
    print(f"Test sessions: {len(test_ids)}")

    predictions = {}
    for i, sid in enumerate(test_ids):
        t0   = time.time()
        sess = load_session(args.data_dir, sid, is_test=True)
        pred = predict_session(model, sess, device, max_seq_len=max_seq_len)
        predictions[sid] = pred
        print(f"  [{i+1}/{len(test_ids)}] {sid}: {pred.shape[0]} bins ({time.time()-t0:.1f}s)")

    build_submission(predictions, args.data_dir, args.output)


if __name__ == '__main__':
    main()

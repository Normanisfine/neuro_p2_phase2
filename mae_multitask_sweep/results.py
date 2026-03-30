"""
Show sweep results ranked by best_val_r2.

Usage:
    python results.py               # show all runs
    python results.py --submit best # generate submission for best run
    python results.py --submit 5s_75ov  # generate submission for specific run
"""

import argparse
import json
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(THIS_DIR, "runs")


def load_results():
    rows = []
    for name in sorted(os.listdir(RUNS_DIR)):
        cfg_path = os.path.join(RUNS_DIR, name, "config.json")
        ckpt_path = os.path.join(RUNS_DIR, name, "best_model.pt")
        if not os.path.exists(cfg_path):
            rows.append({"name": name, "status": "running", "best_val_r2": None, "full_val_r2": None})
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        rows.append({
            "name": name,
            "status": "done",
            "best_val_r2": cfg.get("best_val_r2"),
            "full_val_r2": cfg.get("full_val_r2"),
            "best_epoch": cfg.get("best_epoch"),
            "window_size": cfg.get("window_size"),
            "window_stride": cfg.get("window_stride"),
            "p1_data": bool(cfg.get("p1_data_dir", "")),
            "p1_n_dropout": cfg.get("p1_n_dropout", 28),
            "ckpt": ckpt_path,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", type=str, default=None,
                        help="Generate submission: 'best' or a run name")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: runs/{name}/submission.csv)")
    args = parser.parse_args()

    rows = load_results()
    done = [r for r in rows if r["status"] == "done"]
    running = [r for r in rows if r["status"] == "running"]

    done.sort(key=lambda r: r["best_val_r2"] or -999, reverse=True)

    print(f"\n{'Rank':<5} {'Run':<22} {'best_val_r2':>12} {'full_val_r2':>12} {'epoch':>6} {'win':>5} {'stride':>7} {'P1data':>7} {'dropout':>8}")
    print("-" * 95)
    for i, r in enumerate(done, 1):
        ws = r["window_size"]
        stride = r["window_stride"]
        overlap = int(100 * (1 - stride / ws))
        p1 = "yes" if r["p1_data"] else "no"
        drop = r["p1_n_dropout"] if r["p1_data"] else "-"
        print(f"{i:<5} {r['name']:<22} {r['best_val_r2']:>12.4f} {r['full_val_r2']:>12.4f} "
              f"{r['best_epoch']:>6} {ws/50:.0f}s{overlap}%{'':<2} {stride:>7} {p1:>7} {str(drop):>8}")

    if running:
        print(f"\nStill running ({len(running)}): {', '.join(r['name'] for r in running)}")

    if not args.submit:
        if done:
            print(f"\nBest: {done[0]['name']}  best_val_r2={done[0]['best_val_r2']:.4f}")
            print(f"\nTo generate submission:\n  python results.py --submit best")
        return

    # Generate submission
    if args.submit == "best":
        if not done:
            print("No completed runs yet.")
            sys.exit(1)
        target = done[0]
    else:
        matches = [r for r in done if r["name"] == args.submit]
        if not matches:
            print(f"Run '{args.submit}' not found or not done yet.")
            sys.exit(1)
        target = matches[0]

    ckpt = target["ckpt"]
    out_csv = args.output or os.path.join(RUNS_DIR, target["name"], "submission.csv")
    cfg_path = os.path.join(RUNS_DIR, target["name"], "config.json")

    print(f"\nGenerating submission for: {target['name']}")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Output:     {out_csv}")

    inner = (
        f"source /ext3/env.sh && conda activate neuroinformatics && "
        f"cd {THIS_DIR} && "
        f"python submit.py --config {cfg_path} --checkpoint {ckpt} --output {out_csv}"
    )
    cmd = [
        "singularity", "exec", "--nv",
        "--overlay", "/scratch/ml8347/neuroinformatics/neuro_overlay.ext3:ro",
        "/scratch/ml8347/container_volune/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif",
        "/bin/bash", "-c", inner,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

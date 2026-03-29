"""
Paper Sweep Lab — config runner
"""

import argparse
import glob
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(THIS_DIR, "configs")


def config_path_from_name(name):
    if name.endswith(".json"):
        return os.path.join(CONFIG_DIR, name)
    return os.path.join(CONFIG_DIR, f"{name}.json")


def list_configs():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(CONFIG_DIR, "*.json")))


def run_cmd(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", type=str, default=None, help="Config name or file stem to run")
    parser.add_argument("--all", action="store_true", help="Run all configs sequentially")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    cfgs = list_configs()
    if args.list or (not args.run and not args.all):
        print("Available configs:")
        for cfg in cfgs:
            print(" ", cfg)
        return

    targets = cfgs if args.all else [os.path.basename(config_path_from_name(args.run))]
    for target in targets:
        cfg_path = os.path.join(CONFIG_DIR, target)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(cfg_path)

        train_cmd = [sys.executable, os.path.join(THIS_DIR, "train.py"), "--config", cfg_path]
        if args.quick:
            train_cmd.append("--quick")
        if args.wandb:
            train_cmd.append("--wandb")

        submit_cmd = [
            sys.executable,
            os.path.join(THIS_DIR, "submit.py"),
            "--config",
            cfg_path,
            "--checkpoint",
            os.path.join(THIS_DIR, "runs", os.path.splitext(target)[0], "best_model.pt"),
        ]

        if not args.submit_only:
            run_cmd(train_cmd)
        if not args.train_only:
            run_cmd(submit_cmd)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run train -> holdout eval -> promotion as one retraining pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retraining pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--promotion-threshold",
        type=float,
        default=float(os.environ.get("PROMOTION_THRESHOLD", "0.35")),
        help="Holdout metric threshold for champion promotion",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("REGISTERED_MODEL_NAME", "cafa-go-model"),
        help="MLflow registered model name",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    _run(["python", "scripts/train.py", "--config", args.config], env)

    output_dir = Path("outputs")
    train_summary_path = output_dir / "train_run_summary.json"
    if not train_summary_path.exists():
        raise FileNotFoundError(f"Train summary not found: {train_summary_path}")
    train_summary = json.loads(train_summary_path.read_text())
    train_run_id = train_summary.get("train_run_id")
    if not train_run_id:
        raise RuntimeError("train_run_id missing in train summary")

    eval_env = env.copy()
    eval_env["TRAIN_RUN_ID"] = train_run_id
    _run(["python", "scripts/evaluate_holdout.py", "--config", args.config], eval_env)

    eval_summary_path = output_dir / "holdout_eval_summary.json"
    if not eval_summary_path.exists():
        raise FileNotFoundError(f"Eval summary not found: {eval_summary_path}")
    eval_summary = json.loads(eval_summary_path.read_text())
    eval_run_id = eval_summary.get("eval_run_id")
    if not eval_run_id:
        raise RuntimeError("eval_run_id missing in eval summary")

    _run(
        [
            "python",
            "scripts/promote_model.py",
            "--eval-run-id",
            eval_run_id,
            "--train-run-id",
            train_run_id,
            "--model-name",
            args.model_name,
            "--threshold",
            str(args.promotion_threshold),
        ],
        env,
    )


if __name__ == "__main__":
    main()

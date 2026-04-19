#!/usr/bin/env python
"""Promote a registered model version to champion alias based on eval metric."""

from __future__ import annotations

import argparse
import os
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


def _resolve_version_from_train_run(
    client: MlflowClient, model_name: str, train_run_id: str
) -> Optional[str]:
    target_source = f"runs:/{train_run_id}/model"
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.source == target_source:
            return str(mv.version)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote model alias from holdout evaluation")
    parser.add_argument("--eval-run-id", required=True, help="MLflow run id of holdout evaluation")
    parser.add_argument(
        "--train-run-id",
        default=None,
        help="MLflow run id of training run used for model registration",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("REGISTERED_MODEL_NAME", "cafa-go-model"),
        help="MLflow registered model name",
    )
    parser.add_argument(
        "--metric-name",
        default="holdout_f1_micro",
        help="Metric name on eval run used for promotion gate",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("PROMOTION_THRESHOLD", "0.35")),
        help="Minimum metric value required for champion promotion",
    )
    parser.add_argument("--alias", default="champion", help="Registry alias for promoted model")
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    eval_run = client.get_run(args.eval_run_id)
    metric_value = eval_run.data.metrics.get(args.metric_name)
    if metric_value is None:
        raise ValueError(f"Metric '{args.metric_name}' not found in eval run {args.eval_run_id}")

    train_run_id = args.train_run_id or eval_run.data.tags.get("train_run_id")
    if not train_run_id:
        raise ValueError("train_run_id is required (arg or eval run tag)")

    version = _resolve_version_from_train_run(client, args.model_name, train_run_id)
    if version is None:
        raise ValueError(
            f"No model version found for model='{args.model_name}' and train_run_id='{train_run_id}'"
        )

    client.set_model_version_tag(args.model_name, version, "promotion_metric", args.metric_name)
    client.set_model_version_tag(args.model_name, version, "promotion_value", str(metric_value))
    client.set_model_version_tag(args.model_name, version, "promotion_threshold", str(args.threshold))
    client.set_model_version_tag(args.model_name, version, "train_run_id", train_run_id)
    client.set_model_version_tag(args.model_name, version, "eval_run_id", args.eval_run_id)

    if metric_value >= args.threshold:
        client.set_registered_model_alias(args.model_name, args.alias, version)
        print(
            f"PROMOTED: model={args.model_name} version={version} "
            f"alias={args.alias} metric={args.metric_name} value={metric_value:.6f}"
        )
    else:
        client.set_model_version_tag(args.model_name, version, "promotion_decision", "challenger")
        print(
            f"CHALLENGER: model={args.model_name} version={version} "
            f"metric={args.metric_name} value={metric_value:.6f} threshold={args.threshold:.6f}"
        )


if __name__ == "__main__":
    main()

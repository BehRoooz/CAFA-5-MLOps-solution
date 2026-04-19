#!/usr/bin/env python
"""CLI: Train a protein function prediction model.

Usage:
    python scripts/train.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.preprocess.dataset import ProteinSequenceDataset
from src.preprocess.preprocessing import build_label_matrix, save_label_matrix
from src.models import build_model
from src.training.trainer import Trainer
from src.utils import set_seed, setup_logger


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CAFA protein function prediction model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    # --- Configuration ------------------------------------------------------
    config = load_config(args.config)

    # --- MLflow tracking ----------------------------------------------------
    # Set the MLflow tracking URI
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    # Set the MLflow experiment
    mlflow.set_experiment("cafa-train")

    # Start the MLflow run
    with mlflow.start_run(run_name="cafa-train"):
        train_run_id = mlflow.active_run().info.run_id
        registered_model_name = os.environ.get("REGISTERED_MODEL_NAME", "cafa-go-model")

        # Log the model parameters
        mlflow.log_params(
            {
                "embeddings_source": config.data.get("embeddings_source", "ESM2"),
                "embedding_dim": config.embedding_dim,
                "num_labels": config.num_labels,
                "batch_size": config.batch_size,
                "seed": config.seed,
                "model_type": config.model.get("type", "cnn1d"),
                "mlp_hidden_dims": config.model.get("mlp_hidden_dims", [864, 712]),
                "cnn_out_channels": config.model.get("cnn_out_channels", [3, 8]),
                "cnn_kernel_size": config.model.get("cnn_kernel_size", 3),
                "epochs": config.training.get("epochs", 20),
                "learning_rate": config.training.get("learning_rate", 0.001),
                "scheduler_factor": config.training.get("scheduler_factor", 0.1),
                "scheduler_patience": config.training.get("scheduler_patience", 1),
                "holdout_fraction": config.data.get("holdout_fraction", 0.1),
                "splits_dir": config.data.get("splits_dir", "outputs/splits"),
            }
        )
        # Log the model type
        mlflow.set_tag("model.type", config.model["type"])
        # Log the config file
        mlflow.log_artifact(args.config, artifact_path="config")

        # Log the phase and script tags
        mlflow.set_tag("phase", "training")
        mlflow.set_tag("script", "scripts/train.py")
        mlflow.set_tag("train_run_id", train_run_id)

        # --- Logging --------------------------------------------------------
        logger = setup_logger("cafa5", log_dir=config.output_dir)
        set_seed(config.seed)

        # --- Preprocessing (build label matrix if not already present) -----
        label_dir = config.output_dir / f"label_matrix_top{config.num_labels}"
        if not (label_dir / "label_matrix.npy").exists():
            logger.info("Label matrix not found — running preprocessing ...")
            lm, pids, terms = build_label_matrix(config)
            save_label_matrix(config, lm, pids, terms)
        else:
            logger.info("Using existing label matrix from %s", label_dir)

        embeddings_source = str(config.data.get("embeddings_source", "esm2")).lower()
        embeddings_dir = Path(config.data.get("embeddings_dir", "./data/embeddings"))
        embedding_subdir_map = {
            "esm2": "hf_esm2",
            "protbert": "hf_protbert",
            "t5": "hf_prot_t5",
        }
        embed_subdir = embeddings_dir / embedding_subdir_map.get(embeddings_source, "hf_esm2")
        train_ids_path = embed_subdir / "train_ids.npy"
        term_names_path = label_dir / "term_names.npy"

        dataset_snapshot_id = _sha256_file(train_ids_path) or "unknown"
        term_hash = _sha256_file(term_names_path) or "unknown"
        embedding_backend = str(config.embedding.get("backend", embeddings_source))
        embedding_version = str(config.embedding.get("version", os.environ.get("EMBEDDING_VERSION", "unknown")))

        mlflow.log_params(
            {
                "dataset_snapshot_id": dataset_snapshot_id,
                "embedding_backend": embedding_backend,
                "embedding_version": embedding_version,
                "term_hash": term_hash,
            }
        )

        # --- Dataset --------------------------------------------------------
        logger.info("Loading training dataset ...")

        # Load the training dataset
        dataset = ProteinSequenceDataset(
            config, datatype="train", label_matrix_dir=label_dir
        )

        # --- Model ----------------------------------------------------------
        # Build the model
        model = build_model(config)

        # Log the number of model parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model: %s — %s parameters", config.model["type"], f"{total_params:,}")
        mlflow.log_metric("model_param_count", float(total_params))

        # --- Training -------------------------------------------------------
        trainer = Trainer(config, model, dataset)
        history = trainer.train()

        # --- MLflow metric tracking ----------------------------------------
        # Log training and validation metrics for each epoch
        for i, epoch in enumerate(range(1, len(history["val_f1"]) + 1)):
            mlflow.log_metric("train_loss", history["train_loss"][i], step=epoch)
            mlflow.log_metric("val_loss", history["val_loss"][i], step=epoch)
            mlflow.log_metric("train_f1", history["train_f1"][i], step=epoch)
            mlflow.log_metric("val_f1", history["val_f1"][i], step=epoch)

        # Log the best validation F1 score
        mlflow.log_metric("best_val_f1", max(history["val_f1"]))

        # Save training history to file before logging as artifact
        history_path = Path(config.output_dir) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info("Training history saved → %s", history_path)


        mlflow.log_artifact(str(history_path), artifact_path="training_history")
        best_checkpoint = config.output_dir / "checkpoints" / "best_model.pt"
        if best_checkpoint.exists():
            mlflow.log_artifact(str(best_checkpoint), artifact_path="best_model")
            if term_names_path.exists():
                mlflow.log_artifact(str(term_names_path), artifact_path="label_artifacts")

            model_meta = {
                "model_type": config.model.get("type", "cnn1d"),
                "embedding_dim": int(config.embedding_dim),
                "num_labels": int(config.num_labels),
                "embedding_backend": embedding_backend,
                "embedding_version": embedding_version,
                "dataset_snapshot_id": dataset_snapshot_id,
                "term_hash": term_hash,
                "train_run_id": train_run_id,
            }
            model_meta_path = Path(config.output_dir) / "model_meta.json"
            model_meta_path.write_text(json.dumps(model_meta, indent=2))
            mlflow.log_artifact(str(model_meta_path), artifact_path="model_meta")

            checkpoint_obj = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
            state_dict = checkpoint_obj.get("model_state_dict") if isinstance(checkpoint_obj, dict) else None
            if state_dict:
                model.load_state_dict(state_dict, strict=False)

            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )

            client = MlflowClient()
            version = None
            for mv in client.search_model_versions(f"name='{registered_model_name}'"):
                if mv.source == f"runs:/{train_run_id}/model":
                    version = str(mv.version)
                    break

            run_summary = {
                "train_run_id": train_run_id,
                "registered_model_name": registered_model_name,
                "registered_model_version": version,
                "model_uri": model_info.model_uri,
                "dataset_snapshot_id": dataset_snapshot_id,
                "embedding_backend": embedding_backend,
                "embedding_version": embedding_version,
                "term_hash": term_hash,
            }
            summary_path = Path(config.output_dir) / "train_run_summary.json"
            summary_path.write_text(json.dumps(run_summary, indent=2))
            mlflow.log_artifact(str(summary_path), artifact_path="training_summary")
            logger.info("Training summary saved → %s", summary_path)


if __name__ == "__main__":
    main()

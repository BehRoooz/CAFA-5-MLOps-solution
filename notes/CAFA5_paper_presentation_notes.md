## CAFA-5 Protein Function Prediction (MLOps) - Paper and Presentation Notes

### Purpose
This document is a living summary of what the CAFA-5 MLOps project already implements, formatted so it can be reused to write a paper and make a presentation.
Status: initial draft created from the current repository state.

### Project Goal
Predict Gene Ontology (GO) terms for proteins as a multi-label classification task using protein language model embeddings (configurable sources: ESM2, ProtBERT, T5).

The pipeline trains a neural network to map an embedding vector to logits for a fixed set of GO terms, then applies a sigmoid at inference time to obtain confidences and exports a CAFA-5-style `submission.tsv`.

### What is Already Implemented (Code Modules)
1. Configuration
   - YAML-driven configuration via `configs/config.yaml` and loader in `src/config.py`.
   - Derived values include embedding dimension based on `data.embeddings_source`.

2. Label Preprocessing (top-N GO terms)
   - Reads `data/cafa-5-protein-function-prediction/Train/train_terms.tsv`.
   - Selects the top-N most frequent GO terms (current code: `num_labels`).
   - Constructs a binary label matrix `label_matrix` with shape `(n_proteins, num_labels)`.
   - Saves artifacts under:
     - `outputs/label_matrix_top{num_labels}/label_matrix.npy`
     - `outputs/label_matrix_top{num_labels}/protein_ids.npy`
     - `outputs/label_matrix_top{num_labels}/term_names.npy`
   - Key code: `src/preprocess/preprocessing.py` + CLI `scripts/preprocess.py`.

3. Data Loading
   - Loads protein embeddings directly from `.npy` files (no intermediate DataFrame).
   - Embedding sources are mapped by `embeddings_source` (ESM2/ProtBERT/T5) in `src/preprocess/dataset.py`.
   - For training, labels are aligned to the label-matrix protein IDs via a keep-mask.
   - Key code: `src/preprocess/dataset.py`.

4. Model Architectures (logits output; no sigmoid in the model)
   - `mlp`: configurable stack of `Linear + ReLU` layers, outputting raw logits.
     - Key file: `src/models/mlp.py`.
   - `cnn1d`: reshapes embedding to `(batch, 1, input_dim)` and applies:
     - `Conv1d -> tanh -> MaxPool1d`
     - `Conv1d -> tanh -> MaxPool1d`
     - `Flatten -> tanh(FC1) -> FC2`
     - Key file: `src/models/cnn1d.py`.

5. Training / Validation
   - Multi-label loss: `BCEWithLogitsLoss`.
   - Metric: `torchmetrics.classification.MultilabelF1Score` at `threshold=0.5` applied after `sigmoid`.
   - Train/val split: `random_split` with seeded generator and `train_val_split` (default `0.9`).
   - LR scheduler: `ReduceLROnPlateau` on validation loss.
   - Checkpointing: saves `outputs/checkpoints/best_model.pt` when validation F1 improves.
   - Key file: `src/training/trainer.py`.

6. Inference / Submission Export
   - Loads the checkpoint, runs the test dataset with `batch_size=1`.
   - For each test protein, computes `probs = sigmoid(logits)` producing confidences for all `num_labels` terms.
   - Exports a CAFA-5 style TSV with columns:
     - `Id` (protein EntryID)
     - `GO term` (term name)
     - `Confidence` (sigmoid probability)
   - Key files: `src/inference/predictor.py` + CLI `scripts/predict.py`.

### Current Experiment Snapshot (from repo artifacts)
Latest trained run snapshot stored in:
- `outputs/training_history.json`
- `outputs/train.log`

Configuration (from `configs/config.yaml` and `train.log`):
- Embeddings: `embeddings_source = ESM2` (embedding dim = 1280)
- Model: `cnn1d`
- Top GO terms: `num_labels = 500`
- Training: `epochs = 5`, `batch_size = 128`, `learning_rate = 0.001`, seed `42`
- Train/val split: `train_val_split = 0.9`

Label matrix artifacts:
- `outputs/label_matrix_top500/`
- Training label matrix size: `142246 proteins x 500 GO terms`

Validation performance (multi-label F1 at threshold=0.5):
Best validation F1: `0.1296983` at epoch `5`.

| Epoch | Train Loss | Val Loss | Train F1 | Val F1 |
|-------|-------------|----------|-----------|---------|
| 1     | 0.1563      | 0.1396   | 0.0191    | 0.0473  |
| 2     | 0.1361      | 0.1342   | 0.0693    | 0.0790  |
| 3     | 0.1315      | 0.1318   | 0.0971    | 0.1083  |
| 4     | 0.1285      | 0.1300   | 0.1170    | 0.1164  |
| 5     | 0.1258      | 0.1289   | 0.1370    | 0.1297  |

Epoch metrics above are taken from `outputs/training_history.json` for the current `cnn1d` configuration.

Test inference + submission export:
- Test dataset size in log: `141864` proteins
- Submission rows produced: `70932000` (141864 * 500)
- Output: `outputs/submission.tsv`

### Reproducibility (Run Commands)
1. Preprocess labels (build label matrix):
```bash
python scripts/preprocess.py --config configs/config.yaml
```

2. Train:
```bash
python scripts/train.py --config configs/config.yaml
```

3. Predict / export submission:
```bash
python scripts/predict.py --config configs/config.yaml
# optional:
python scripts/predict.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pt
```

### Biological / Methodological Notes (for Paper Writing)
1. Multi-label GO prediction
   - The task is formulated as multi-label classification over a selected GO vocabulary (top-N by frequency from training annotations).
   - The model outputs logits per GO term; probabilities come from sigmoid.

2. Label truncation to top-N terms
   - Restricting to top-N GO terms is a strong modeling assumption.
   - For the paper, specify that the label space is truncated and describe how it may affect rare GO terms and hierarchical structure.

3. Confounders / Limitations to Flag
   - Thresholding at `0.5` for F1 computation may not match the CAFA scoring protocol (CAFA uses its own evaluation scheme).
   - GO terms form a DAG; this pipeline does not explicitly encode GO hierarchy/constraints.
   - Embedding source controls representation quality; experiments should include embedding ablations.
   - Train/val split is random over proteins and may not reflect stratification by GO depth or GO frequency.

### Paper Draft To-Dos (Suggested Structure)
1. Abstract
2. Introduction
   - CAFA-5 problem framing; GO prediction; protein language model embeddings.
3. Methods
   - Embeddings, label preprocessing (top-N), model architectures, loss/metric, training details, checkpointing.
4. Experiments
   - Datasets and splits; reported metrics; training curves; inference and submission generation.
5. Results
   - Primary table: model vs embedding vs metric.
   - Include calibration/threshold sensitivity analysis if possible.
6. Discussion
   - Interpretability of embeddings, GO hierarchy limitation, failure modes for rare terms.
7. Conclusion and Future Work

### Presentation Draft To-Dos (Slide Outline)
1. Problem + Why it Matters (GO + CAFA-5)
2. Overview Diagram (preprocess -> train -> predict -> submission)
3. Label Construction (top-N terms, binary matrix)
4. Model Architecture (CNN1D and/or MLP)
5. Training Procedure (loss, metric, checkpointing)
6. Results (validation curves + best score)
7. Submission Output (example row / format)
8. Limitations + Next Steps

### Change Log (Keeps This Doc in Sync With the Project)
- 2026-03-25: Initial draft created from current repository code + current saved artifacts in `outputs/`.

---

# CAFA-5 MLOps Solution — Paper & Presentation Notes (Living Doc)

Last updated: 2026-03-26

## One-liner (talk opening)
- Predict Gene Ontology (GO) terms from protein sequences by generating protein language model embeddings (ESM-2 / ProtBERT / ProtT5) and training a lightweight multi-label classifier (MLP/CNN1D) in a reproducible, configurable pipeline.

## What this repo contributes (positioning)
- **Production-ready workflow**: config-driven CLI (`scripts/preprocess.py`, `scripts/split_train_holdout.py`, `scripts/embed_sequences.py`, `scripts/train.py`, `scripts/predict.py`) with checkpointing and output artifacts.
- **Embedding-first modeling**: decouple representation learning (PLM embeddings) from supervised GO classification to iterate quickly on model heads + training.
- **Reproducibility hooks**: centralized YAML config + dataclass validation (`src/config.py`), seeding, structured outputs (`outputs/training_history.json`, checkpoints, submission).

## Background slides (minimum)
- **GO is hierarchical**: DAG of terms across MF/BP/CC; evaluation can depend on hierarchy-aware propagation/metrics.
- **Task framing**: multi-label classification; labels are sparse and long-tailed; strong class imbalance is expected.

## Data & splits (paper Methods)
- **Inputs**: Kaggle CAFA-5 `train_sequences.fasta` and `train_terms.tsv`; embeddings stored under `data/embeddings/…`.
- **Label space**: configurable `data.num_labels` (default 500) → clarify how top-N terms are selected (frequency thresholding / top-k by count) and how ties are handled.
- **Splits**: `holdout_fraction` and `train_val_split`; document whether splits are ID-based and whether any leakage controls exist (sequence identity / homology clusters).
  - Confounder to flag: random splits can inflate performance due to homologous proteins across splits.

## Embedding generation (talk Methods + compute)
- **Backends**: HF models selected by `embedding.backend` (`esm2`, `protbert`, `t5`) with cache at `embedding.hf_cache_dir` (default `data/hf_cache`).
- **Tokenization nuance**: ProtBERT/ProtT5 use space-separated residues; ESM2 uses standard tokenization.
- **Sequence normalization**: uppercase + remap non-canonical residues (U/O/B/Z/J/unknown) to `X` (keeps embedding generation robust to FASTA quirks).
- **Pooling**:
  - `mean`: **residue-only mean pooling** (explicitly excludes special tokens via tokenizer `all_special_ids`)
  - `cls`: take first token embedding
- **Truncation**: `embedding.max_length` (default 1280) → note biological risk for long proteins (domain loss) and potential mitigation (chunking / sliding windows).
- **Precision/throughput**: `embedding.fp16=true` recommended; report GPU type + batch size used.
- **Artifacts**: save `*_embeddings.npy`, matching `*_ids.npy`, plus `embed_meta_<split>.json` (model ID, pooling, max_length, batch_size, fp16, embedding_dim, timestamps, FASTA path).

## Model heads (paper Methods)
- **MLP**: configurable hidden dims (default `[864, 712]`), ReLU, final linear layer outputs logits.
- **CNN1D**: 1D conv stack + pooling + FC output (logits); specify expected input shape and whether embeddings are treated as channels/sequence.
- **Loss**: `BCEWithLogitsLoss` (logits + stable sigmoid inside).
- **Imbalance handling (TODO if not implemented)**: consider `pos_weight`, focal loss, or class-balanced sampling; report what is used.

## Training protocol (paper Methods)
- **Optimization**: Adam with `learning_rate`; scheduler is `ReduceLROnPlateau` on **val loss** (factor/patience from config).
- **Model selection**: checkpoint `outputs/checkpoints/best_model.pt` saved by best **val F1**.
- **Thresholding (current)**: `torchmetrics.MultilabelF1Score(..., threshold=0.5)` computed on sigmoid(logits); note this is a single fixed threshold (not a CAFA \(F_{\max}\) sweep).
- **Seeds**: `training.seed=42`; document determinism caveats on GPU.

## Evaluation (must align with CAFA conventions)
- Current repo mentions **val F1**; CAFA typically reports **\(F_{\max}\)** and sometimes **\(S_{\min}\)** / AUPR variants, with hierarchy-aware propagation.
- **TODO (paper-critical)**:
  - Implement/report CAFA-style metrics (threshold sweep → \(F_{\max}\); ontology-aware propagation if required).
  - Clarify whether predictions are post-processed to respect GO DAG (propagate child → ancestors) before scoring/submission.
  - Add per-namespace reporting (MF/BP/CC) if labels span multiple namespaces.

## Submission generation
- Output format: `outputs/submission.tsv` with columns `(Id, GO term, Confidence)`.
- Clarify:
  - How confidences are calibrated (sigmoid probabilities, temperature scaling, Platt/isotonic).
  - How many terms per protein are output (top-k, above-threshold, fixed quota).
  - **Current behavior**: inference writes **all** `num_labels` terms per protein (rows = `n_proteins × num_labels`) with `Confidence = sigmoid(logit)`; no filtering/top-k yet.

## Results to capture (fill in after running)
- **Holdout performance**: F1 / \(F_{\max}\) with confidence intervals (bootstrap proteins) if feasible.
- **Ablations**:
  - Embedding backend: ESM-2 vs ProtBERT vs ProtT5
  - Pooling: mean vs CLS
  - Head: MLP vs CNN1D
  - Label space size: top-500 vs larger (sensitivity to long tail)
- **Compute cost**:
  - Embedding time per 10k proteins; peak GPU RAM; disk footprint of `.npy`.

## QC / sanity checks (talk appendix; paper rigor)
- Label matrix integrity: no all-zero rows in train; term frequency distribution; extreme imbalance.
- Embedding integrity: NaNs/Inf checks; consistent embedding dimensionality per backend.
- Leakage checks: duplicate IDs across splits; optional sequence identity clustering.
- Calibration checks: reliability diagrams / ECE for confidence values (if used competitively).

## Reproducibility & MLOps notes (paper “Implementation”)
- Config entrypoint: `configs/config.yaml` fully specifies data paths, embedding backend, model type, training settings, outputs.
- Artifacts and provenance:
  - Save resolved config alongside outputs (TODO if not already done).
  - Record package versions / git commit hash (TODO).
  - Record hardware info for embedding runs (GPU model, driver/CUDA).

## Limitations (explicitly acknowledge)
- Random holdout likely overestimates generalization without homology-aware splits.
- Truncation at `max_length` can bias against long, multi-domain proteins.
- Flat multi-label classification ignores GO hierarchy unless post-processing/metrics incorporate it.
- Top-N label restriction limits recall on rare terms (competition-relevant trade-off).

## Figures to include (paper + slides)
- Pipeline diagram: FASTA → embeddings → label matrix → training → thresholding → submission.
- Term frequency histogram + long-tail plot.
- PR curves or \(F_{\max}\) vs threshold sweep; calibration plot.
- Embedding backend comparison table (performance vs compute).

## Checklist before submission/talk (high priority)
- [ ] Confirm metric implementation matches CAFA evaluation (hierarchy, namespaces, threshold sweep).
- [ ] Document split strategy and leakage controls.
- [ ] Add run provenance: config snapshot + versions + git SHA.
- [ ] Summarize compute budget and embedding generation throughput.

## Integration Milestone (Embedding API <-> GO Prediction API)

### Milestone objective
- Move from offline-only prediction to an online inference flow where a user submits protein sequences (or FASTA) and receives GO term predictions end-to-end.
- Integrate two services:
  - `embedding-api`: sequence ingestion, embedding generation, artifact management, orchestration.
  - `go-prediction-api`: model loading and GO term inference from 1280-d embeddings.

### Final integrated request flow (demo-friendly)
1. Client calls `POST /api/v1/predict-go-from-sequences` on `embedding-api`.
2. `embedding-api` creates an internal embedding job and waits for completion.
3. Generated artifacts (`test_ids.npy`, `test_embeddings.npy`) are persisted in `outputs/service_artifacts/<job_id>/`.
4. For each selected sequence index, `embedding-api` calls `go-prediction-api /predict` with embedding vector + `top_k`.
5. `go-prediction-api` returns ranked GO terms with scores.
6. `embedding-api` aggregates per-sequence results (and failures, if any) and returns unified API response.

### Docker + deployment setup introduced
- Added a dedicated GO API Docker build path (`docker/docker_go_term/Dockerfile.api`) with explicit copy of required model artifacts:
  - `outputs/checkpoints/best_model.pt`
  - `outputs/label_matrix_top500/term_names.npy`
  - `outputs/splits/*`
- Compose orchestrates:
  - `embedding-api`
  - `go-prediction-api`
  - `mlflow` (tracking backend for inference logging)
- Environment links include `MLFLOW_TRACKING_URI` and internal service URL wiring for GO API calls.

### Incident during integration (excellent presentation story)

#### Symptom
- Public endpoint returned `502 Bad Gateway` when predicting GO from sequences.
- Error payload included:
  - `GO_API_UNREACHABLE` in one run
  - later `GO_API_HTTP_500: {"detail":"model artifacts could not be loaded"}` after partial fixes

#### Root-cause chain (what actually happened)
1. `go-prediction-api` initially crashed on startup due to `ModuleNotFoundError: No module named 'mlflow'`.
2. Because GO API was down, `embedding-api` could not reach upstream and returned `502`.
3. After making MLflow import non-fatal (or available), GO API started but `/predict` still returned 500.
4. Second root cause: loader expected `outputs/splits/model_meta.json`, but repository artifacts provided `split_meta.json` and checkpoint-contained config instead.
5. Model startup failed silently into `MODEL=None`, causing `/predict` to return `"model artifacts could not be loaded"` and propagate as `502` at gateway level.

### Fixes implemented
1. **Resilient MLflow integration**
   - `mlflow` import guarded so missing tracking dependency does not crash service startup.
   - Inference logging remains best-effort and non-blocking.
2. **Robust model metadata fallback**
   - In `services/go-prediction-api/model_loader.py`, added fallback metadata reconstruction from checkpoint `config` when `model_meta.json` is absent.
   - Fallback resolves:
     - `model_type`
     - `embedding_dim` (via backend mapping: ESM2/ProtBERT/T5)
     - `num_labels`
     - `model_version`
3. **Debug instrumentation loop (then cleaned)**
   - Added temporary runtime logs across service startup + upstream request path.
   - Confirmed hypotheses with runtime evidence.
   - Removed instrumentation after successful verification.

### Engineering lessons (good for “Challenges & Learnings” slide)
- Inference services must treat observability features (MLflow/logging) as optional, never hard startup dependencies.
- API gateway-style services (`embedding-api`) can mask upstream faults as 502; keep explicit upstream error typing for diagnosis.
- Artifact contracts between training and serving must be explicit and versioned (expected file names, schemas, dimensions).
- For biological ML systems, always validate:
  - embedding dimensionality consistency (1280 here),
  - metadata integrity (`num_labels`, term vocabulary),
  - prediction API schema stability across services.

### Production guidance decided in this milestone
- Keep checkpoints out of Git (`.gitignore`) to avoid repository bloat and model-version drift.
- Include checkpoints in Docker context only if using image-baked model artifacts.
- Preferred future direction: load model from registry/object storage (MLflow artifacts, S3, etc.) at startup with pinned version and health checks.

### Validation status after fix
- `go-prediction-api` starts successfully and serves `/predict`.
- `embedding-api` sequence-to-GO endpoint works end-to-end without 502 for this failure mode.
- Integration path now supports presentation demo of real-time GO prediction from sequence input.

### Presenter notes (talk track)
- “We intentionally converted a research pipeline into a service-oriented inference system.”
- “The key integration risk was not modeling accuracy, but artifact/runtime contract mismatches.”
- “We solved this by making the serving layer tolerant to optional components and by deriving missing metadata from checkpoint provenance.”
- “This milestone gave us a reproducible base for next milestone: full MLflow-enabled inference tracking and model registry loading.”


#!/usr/bin/env bash
# Smoke test for the Embedding API (local or Docker on port 8000).
# Usage:
#   ./scripts/smoke_embedding_api.sh
#   BASE_URL=http://127.0.0.1:8010 ./scripts/smoke_embedding_api.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "==> Health: GET ${BASE_URL}/api/v1/health"
curl -sS "${BASE_URL}/api/v1/health"
echo

echo "==> Submit job: POST ${BASE_URL}/api/v1/jobs"
RESP="$(curl -sS -X POST "${BASE_URL}/api/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "stage": "test",
    "backend": "esm2",
    "pooling": "mean",
    "batch_size": 2,
    "max_length": 1280,
    "sequences": [
      {"id": "smoke_P1", "sequence": "MKTAYIAKQRQISFVKSHFSRQ"},
      {"id": "smoke_P2", "sequence": "GAVLIPFYWSTCMNQDEKRH"}
    ]
  }')"
echo "$RESP"

JOB_ID="$(printf '%s' "$RESP" | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")"
echo "==> Job ID: ${JOB_ID}"

echo "==> Poll until succeeded (max ~120s)"
for _ in $(seq 1 60); do
  ST="$(curl -sS "${BASE_URL}/api/v1/jobs/${JOB_ID}")"
  STATUS="$(printf '%s' "$ST" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")"
  if [[ "$STATUS" == "succeeded" ]]; then
    echo "$ST" | python3 -m json.tool
    break
  fi
  if [[ "$STATUS" == "failed" ]]; then
    echo "$ST" | python3 -m json.tool
    exit 1
  fi
  sleep 2
done

if [[ "${STATUS:-}" != "succeeded" ]]; then
  echo "Timed out waiting for job ${JOB_ID}"
  exit 1
fi

OUT_DIR="$(mktemp -d)"
echo "==> Download artifacts to ${OUT_DIR}"
curl -sS -o "${OUT_DIR}/test_ids.npy" \
  "${BASE_URL}/api/v1/jobs/${JOB_ID}/artifacts/test_ids.npy"
curl -sS -o "${OUT_DIR}/test_embeddings.npy" \
  "${BASE_URL}/api/v1/jobs/${JOB_ID}/artifacts/test_embeddings.npy"

echo "==> Verify shapes (expect N=2, D=1280 for esm2, float32)"
python3 <<PY
import numpy as np
import pathlib
d = pathlib.Path("${OUT_DIR}")
emb = np.load(d / "test_embeddings.npy")
ids = np.load(d / "test_ids.npy", allow_pickle=True)
print("test_embeddings.npy:", emb.shape, emb.dtype)
print("test_ids.npy:", ids.shape, ids.dtype)
assert emb.ndim == 2 and emb.shape[0] == len(ids) == 2
assert emb.shape[1] == 1280
assert str(emb.dtype) == "float32"
print("OK")
PY

echo "==> Smoke test passed."

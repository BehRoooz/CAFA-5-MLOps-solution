from __future__ import annotations

import numpy as np

from model_loader import load_model, load_term_names
from predictor_service import predict_top_k

CHECKPOINT_PATH = "outputs/checkpoints/best_model.pt"
TERM_NAMES_PATH = "outputs/label_matrix_top500/term_names.npy"
META_PATH = "models/model_meta.json"


def main() -> None:
    model, meta = load_model(CHECKPOINT_PATH, META_PATH, device="cpu")
    term_names = load_term_names(TERM_NAMES_PATH)

    embedding_dim = int(meta["embedding_dim"])
    top_k_default = int(meta.get("top_k_default", 10))

    embedding = np.random.rand(embedding_dim).astype(np.float32)

    result = predict_top_k(
        model=model,
        embedding=embedding,
        term_names=term_names,
        top_k=top_k_default,
        apply_sigmoid=True,
        device="cpu",
    )

    print("model_version:", meta.get("model_version", "unknown"))
    print("top_k:", result["top_k"])
    print("predictions:")
    for pred in result["predictions"]:
        print(pred)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""CLI: Launch the CAFA-5 prediction API server.

Usage:
    python scripts/serve.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Start CAFA-5 API server")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    os.environ["CONFIG_PATH"] = args.config
    if args.checkpoint:
        os.environ["CHECKPOINT_PATH"] = args.checkpoint

    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

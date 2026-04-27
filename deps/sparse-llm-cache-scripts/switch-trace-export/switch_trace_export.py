#!/usr/bin/env python3
"""CLI: export Switch router behavior to train_predict_model contract."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export Switch Transformers router trace.")
    p.add_argument("--model-path", required=True, type=Path,
                   help="Local HF Switch model dir (with config.json).")
    p.add_argument("--prompt-file", required=True, type=Path,
                   help="Plain text file, one prompt per line.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Output dir; will create encoder/ and decoder/ subdirs.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify", action="store_true",
                   help="Run contract-level Verifier on both subdirs after writing.")
    p.add_argument("--smoke-train", action="store_true",
                   help="Run train_predict_model.py end-to-end smoke on both subdirs.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError("Wired in Task 8")


if __name__ == "__main__":
    raise SystemExit(main())

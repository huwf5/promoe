#!/usr/bin/env python3
"""CLI: export Switch router behavior to train_predict_model contract."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from utils import ContractWriter, SwitchRunner, Verifier  # noqa: E402


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
    prompts = [line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()]
    if not prompts:
        print("No prompts to process.", file=sys.stderr)
        return 1

    runner = SwitchRunner(model_path=str(args.model_path), device=args.device, seed=args.seed)
    t0 = time.time()
    enc_acc, dec_acc = runner.run(prompts, args.max_new_tokens, args.batch_size)
    elapsed = time.time() - t0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extra = {
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "model_path": str(args.model_path),
    }
    ContractWriter(args.output_dir / "encoder", stage="encoder").write(enc_acc, extra)
    ContractWriter(args.output_dir / "decoder", stage="decoder").write(dec_acc, extra)

    log = {
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "n_prompts": len(prompts),
        "N_enc": enc_acc.total_tokens(),
        "N_dec": dec_acc.total_tokens(),
        "elapsed_sec": round(elapsed, 3),
    }
    (args.output_dir / "run_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))

    if args.verify:
        Verifier.verify(args.output_dir / "encoder")
        Verifier.verify(args.output_dir / "decoder")
        print("Verifier: OK on both subdirs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

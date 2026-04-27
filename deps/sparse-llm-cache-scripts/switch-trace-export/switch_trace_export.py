#!/usr/bin/env python3
"""CLI: export Switch router behavior to train_predict_model contract."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from utils import ContractWriter, SwitchRunner, Verifier  # noqa: E402

TRAIN_SCRIPT_ENV = "PROMOE_SWITCH_TRACE_TRAIN_SCRIPT"


def _repo_train_script_path() -> Path:
    return THIS_DIR.parent / "train-predict-model" / "train_predict_model.py"


def _train_script_candidates(train_script_path: Path | None) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    if train_script_path is not None:
        candidates.append(("--train-script-path", train_script_path.expanduser()))
    env_path = os.environ.get(TRAIN_SCRIPT_ENV)
    if env_path:
        candidates.append((TRAIN_SCRIPT_ENV, Path(env_path).expanduser()))
    candidates.append(("repo-local default", _repo_train_script_path()))
    return candidates


def _resolve_train_script_path(train_script_path: Path | None) -> Path | None:
    for _, candidate in _train_script_candidates(train_script_path):
        if candidate.is_file():
            return candidate
    return None


def _smoke_train_cmd(trace_dir: Path, model_dir: Path, train_script: Path) -> list[str]:
    return [
        sys.executable, str(train_script),
        "--logits_path", str(trace_dir),
        "--predict_model_path", str(model_dir),
        "--predict_output", "freq",
        "--model_type", "split",
        "--window", "1",
        "--hidden_size", "16",
        "--n_layer", "1",
        "--batch_size", "16",
        "--lr", "0.001",
        "--threshold", "1.0",
        "--threshold_window", "1",
        "--model_index", "0",
    ]


def _run_smoke_train(output_dir: Path, train_script_path: Path | None = None) -> int:
    train_script = _resolve_train_script_path(train_script_path)
    if train_script is None:
        print("Could not find real train_predict_model.py for --smoke-train.", file=sys.stderr)
        print(
            f"Pass --train-script-path or set {TRAIN_SCRIPT_ENV}; checked:",
            file=sys.stderr,
        )
        for label, candidate in _train_script_candidates(train_script_path):
            print(f"  {label}: {candidate}", file=sys.stderr)
        return 1

    for stage in ("encoder", "decoder"):
        trace_dir = output_dir / stage
        model_dir = output_dir / "smoke_train" / stage
        print(f"Smoke training {stage} trace with {train_script}")
        r = subprocess.run(_smoke_train_cmd(trace_dir, model_dir, train_script))
        if r.returncode != 0:
            print(f"Smoke training failed for {stage} with exit code {r.returncode}", file=sys.stderr)
            return r.returncode
    return 0


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
    p.add_argument("--train-script-path", type=Path, default=None,
                   help=f"Path to real train_predict_model.py for --smoke-train; "
                        f"defaults to ${TRAIN_SCRIPT_ENV}, then repo-local train-predict-model/.")
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

    if args.smoke_train:
        smoke_rc = _run_smoke_train(args.output_dir, args.train_script_path)
        if smoke_rc != 0:
            return smoke_rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

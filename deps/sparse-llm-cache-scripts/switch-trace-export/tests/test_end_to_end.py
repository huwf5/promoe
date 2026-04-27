import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
SCRIPT = MODULE_DIR / "switch_trace_export.py"


def _resolve_train_script() -> Path:
    rel = Path("deps") / "sparse-llm-cache-scripts" / "train-predict-model" / "train_predict_model.py"
    for d in (THIS_DIR, *THIS_DIR.parents):
        candidate = d / rel
        if candidate.is_file():
            return candidate
    main_workspace = Path("/mnt/huwf5/promoe") / rel
    if main_workspace.is_file():
        return main_workspace
    raise FileNotFoundError(f"could not find train_predict_model.py via */{rel}")


TRAIN_SCRIPT = _resolve_train_script()


def _write_prompts(p: Path) -> Path:
    p.write_text(
        "\n".join(
            [
                "Translate: hello world",
                "Summarize: the quick brown fox jumps",
                "Question: What is the capital of France? Answer:",
                "Classify the sentiment of this review: I loved the food but hated the service.",
                "Rewrite this sentence in formal English: gonna head out soon",
                "List three colors and three animals in a compact sentence.",
            ]
        )
        + "\n"
    )
    return p


def _write_train_sitecustomize(p: Path) -> Path:
    p.write_text(
        """
import torch
import torch.utils.data

_OrigDataLoader = torch.utils.data.DataLoader


def _stack_tuple_batch(batch):
    if isinstance(batch, list) and batch and isinstance(batch[0], tuple):
        cols = list(zip(*batch))
        return tuple(torch.stack(list(col), dim=0) for col in cols)
    return batch


class _PatchedDataLoader(_OrigDataLoader):
    def __init__(self, *args, **kwargs):
        collate_fn = kwargs.get("collate_fn")
        if collate_fn is not None:
            kwargs["collate_fn"] = lambda batch: _stack_tuple_batch(collate_fn(batch))
        super().__init__(*args, **kwargs)


torch.utils.data.DataLoader = _PatchedDataLoader
"""
    )
    return p


def test_cli_smoke_writes_both_dirs(tmp_path, switch_model_path):
    prompts = _write_prompts(tmp_path / "prompts.txt")
    out = tmp_path / "out"
    cmd = [
        sys.executable, str(SCRIPT),
        "--model-path", str(switch_model_path),
        "--prompt-file", str(prompts),
        "--output-dir", str(out),
        "--max-new-tokens", "4",
        "--batch-size", "2",
        "--device", "cpu",
        "--seed", "42",
        "--verify",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert (out / "encoder/expert_selection.pt").exists()
    assert (out / "decoder/expert_selection.pt").exists()
    assert (out / "run_log.json").exists()
    log = json.loads((out / "run_log.json").read_text())
    assert log["seed"] == 42
    assert log["max_new_tokens"] == 4
    assert log["N_enc"] > 0 and log["N_dec"] > 0


def test_reproducibility_same_seed(tmp_path, switch_model_path):
    prompts = _write_prompts(tmp_path / "prompts.txt")
    outs = []
    for run in range(2):
        out = tmp_path / f"run{run}"
        subprocess.check_call([
            sys.executable, str(SCRIPT),
            "--model-path", str(switch_model_path),
            "--prompt-file", str(prompts),
            "--output-dir", str(out),
            "--max-new-tokens", "4",
            "--batch-size", "2",
            "--device", "cpu",
            "--seed", "42",
        ])
        outs.append(out)
    for sub in ("encoder", "decoder"):
        for fn in [
            "expert_selection.pt",
            "decode_stage_moe_layer_logits_per_token.pt",
            "decode_stage_expert_freq_per_token.pt",
            "decode_stage_token_ids_per_token.pt",
        ]:
            a = torch.load(outs[0] / sub / fn)
            b = torch.load(outs[1] / sub / fn)
            assert torch.equal(a, b), f"{sub}/{fn} differs across same-seed runs"


@pytest.mark.slow
def test_smoke_train_runs_on_both_subdirs(tmp_path, switch_model_path):
    prompts = _write_prompts(tmp_path / "prompts.txt")
    out = tmp_path / "out"
    subprocess.check_call([
        sys.executable, str(SCRIPT),
        "--model-path", str(switch_model_path),
        "--prompt-file", str(prompts),
        "--output-dir", str(out),
        "--max-new-tokens", "8",
        "--batch-size", "2",
        "--device", "cpu",
        "--seed", "42",
        "--verify",
    ])
    _write_train_sitecustomize(tmp_path / "sitecustomize.py")
    for sub in ("decoder", "encoder"):
        sd = tmp_path / f"smoke_{sub}"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), env.get("PYTHONPATH", "")])
        r = subprocess.run([
            sys.executable, str(TRAIN_SCRIPT),
            "--logits_path", str(out / sub),
            "--predict_model_path", str(sd),
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
        ], capture_output=True, text=True, env=env)
        assert r.returncode == 0, f"{sub} train failed:\n{r.stderr}"
        assert any(sd.glob("*.pt")), f"no .pt in {sd}"

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
SCRIPT = MODULE_DIR / "switch_trace_export.py"
TRAIN_SCRIPT = MODULE_DIR.parent / "train-predict-model" / "train_predict_model.py"


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
        pt_files = sorted(p.name for p in (outs[0] / sub).glob("*.pt"))
        assert pt_files
        assert pt_files == sorted(p.name for p in (outs[1] / sub).glob("*.pt"))
        for fn in pt_files:
            a = torch.load(outs[0] / sub / fn)
            b = torch.load(outs[1] / sub / fn)
            assert torch.equal(a, b), f"{sub}/{fn} differs across same-seed runs"


@pytest.mark.slow
def test_smoke_train_runs_on_both_subdirs(tmp_path, switch_model_path):
    assert TRAIN_SCRIPT.is_file(), f"missing train script at {TRAIN_SCRIPT}"
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
    for sub in ("decoder", "encoder"):
        sd = tmp_path / f"smoke_{sub}"
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
        ], capture_output=True, text=True)
        assert r.returncode == 0, f"{sub} train failed:\n{r.stderr}"
        assert any(sd.glob("*.pt")), f"no .pt in {sd}"

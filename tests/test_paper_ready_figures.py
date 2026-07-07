
import nbformat
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_paper_plotting_namespace():
    nb_path = Path("experiment/baseline/z_final_result/scripts/latency_data_parser.ipynb")
    nb = nbformat.read(nb_path, as_version=4)
    source = None
    for cell in nb.cells:
        text = "".join(cell.get("source", ""))
        if cell.get("cell_type") == "code" and "def draw_paper_latency_platform_comparison" in text:
            source = text
            break
    assert source is not None, "paper-ready plotting cell not found"

    namespace = {
        "Path": Path,
        "plt": plt,
        "np": np,
        "pd": pd,
        "FIGURE_DIR": Path("/tmp/paper-ready-figures"),
    }
    exec(source, namespace)
    return namespace


def _dummy_summary():
    rows = []
    for machine in ["4090d", "a6000"]:
        for model, scale in [("switch-base-128", 1), ("nllb", 100)]:
            gpu_settings = ["8gb", "16gb"] if model == "switch-base-128" else ["16gb"]
            for gpu_setting in gpu_settings:
                for runtime, factor in [("winmoe", 1.0), ("promoe", 1.4)]:
                    rows.append({
                        "runtime": runtime,
                        "machine": machine,
                        "model": model,
                        "gpu_setting": gpu_setting,
                        "benchmark_ttft_ms_avg": 10 * scale * factor,
                        "benchmark_decode_tpot_ms_avg": 2 * scale * factor,
                        "benchmark_e2e_ms_avg": 40 * scale * factor,
                    })
    return pd.DataFrame(rows)


def test_platform_comparison_uses_independent_axes_per_model():
    ns = _load_paper_plotting_namespace()
    captured = {}

    def capture_figure(fig, out_path):
        captured["axes"] = list(fig.axes)
        captured["titles"] = [ax.get_title() for ax in fig.axes]
        captured["texts"] = [text.get_text() for ax in fig.axes for text in ax.texts]
        captured["figure_legends"] = len(fig.legends)
        captured["axes_legends"] = sum(1 for ax in fig.axes if ax.get_legend() is not None)
        return Path(out_path)

    ns["_save_paper_figure"] = capture_figure
    summary = _dummy_summary()
    ns["draw_paper_latency_platform_comparison"](
        [
            {"summary": summary, "machine": "4090d", "title": "RTX 4090D"},
            {"summary": summary, "machine": "a6000", "title": "RTX A6000"},
        ],
        Path("/tmp/platform-comparison.pdf"),
        runtimes=["winmoe", "promoe"],
    )

    # 3 metrics x 2 platforms x 2 models. Each model panel gets its own y-axis.
    assert len(captured["axes"]) == 12
    assert "switch-base-128" in captured["titles"]
    assert "nllb" in captured["titles"]
    assert "RTX 4090D" not in captured["titles"]
    assert "RTX A6000" not in captured["titles"]



def test_single_platform_comparison_uses_compact_model_titles():
    ns = _load_paper_plotting_namespace()
    captured = {}

    def capture_figure(fig, out_path):
        captured["axes"] = list(fig.axes)
        captured["titles"] = [ax.get_title() for ax in fig.axes]
        captured["texts"] = [text.get_text() for ax in fig.axes for text in ax.texts]
        captured["figure_legends"] = len(fig.legends)
        captured["axes_legends"] = sum(1 for ax in fig.axes if ax.get_legend() is not None)
        return Path(out_path)

    ns["_save_paper_figure"] = capture_figure
    summary = _dummy_summary()
    ns["draw_paper_latency_platform_comparison"](
        [
            {"summary": summary, "machine": "4090d", "title": "RTX 4090D"},
        ],
        Path("/tmp/platform-4090d.pdf"),
        runtimes=["winmoe", "promoe"],
    )

    # 3 metrics x 1 platform x 2 models, with compact model labels inside panels.
    assert len(captured["axes"]) == 6
    assert "switch-base-128" in captured["titles"]
    assert "nllb" in captured["titles"]
    assert all("RTX" not in title for title in captured["titles"])
    assert captured["figure_legends"] == 1
    assert captured["axes_legends"] == 0



def test_single_platform_uses_inset_model_labels_and_uniform_x_slots():
    ns = _load_paper_plotting_namespace()
    captured = {}

    def capture_figure(fig, out_path):
        axes = list(fig.axes)
        captured["axes"] = axes
        captured["titles"] = [ax.get_title() for ax in axes]
        captured["texts"] = [text.get_text() for ax in axes for text in ax.texts]
        captured["xtick_counts"] = [len(ax.get_xticks()) for ax in axes[-2:]]
        captured["xlims"] = [ax.get_xlim() for ax in axes[-2:]]
        captured["figure_legends"] = len(fig.legends)
        captured["legend_ncols"] = [getattr(legend, "_ncols", None) for legend in fig.legends]
        captured["legend_anchor_y"] = [legend.get_bbox_to_anchor()._bbox.y1 for legend in fig.legends]
        captured["axes_legends"] = sum(1 for ax in axes if ax.get_legend() is not None)
        return Path(out_path)

    ns["_save_paper_figure"] = capture_figure
    summary = _dummy_summary()
    ns["draw_paper_latency_platform_comparison"](
        [
            {"summary": summary, "machine": "4090d", "title": "RTX 4090D"},
        ],
        Path("/tmp/platform-4090d.pdf"),
        runtimes=["winmoe", "promoe"],
    )

    assert all("RTX" not in title for title in captured["titles"])
    assert "switch-base-128" in captured["titles"]
    assert "nllb" in captured["titles"]
    assert captured["xtick_counts"] == [2, 1]
    assert captured["xlims"][1][1] < captured["xlims"][0][1]
    assert captured["figure_legends"] == 1
    assert captured["legend_ncols"] == [2]
    assert captured["legend_anchor_y"][0] <= 1.02
    assert captured["axes_legends"] == 0

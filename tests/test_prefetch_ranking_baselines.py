from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiment.scripts.evaluate.evaluate_prefetch_ranking_baselines import (
    DEFAULT_MODEL_SPECS,
    curve_from_scores,
    default_model_specs,
    noisy_or_scores_from_logits,
    rank_major_count_scores_from_logits,
    rank_major_prob_scores_from_logits,
    rank_major_scores_from_logits,
    sample_targets_from_expert_selection,
    token_topk_union_count_scores,
)


def test_token_topk_union_count_scores_counts_unique_token_votes() -> None:
    logits = torch.tensor(
        [[[[9.0, 8.0, 1.0], [7.0, 6.0, 5.0], [0.0, 10.0, 9.0]]]],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    top1 = token_topk_union_count_scores(logits, attention_mask, token_topk=1)
    top2 = token_topk_union_count_scores(logits, attention_mask, token_topk=2)

    assert torch.equal(top1, torch.tensor([[[2.0, 0.0, 0.0]]]))
    assert torch.equal(top2, torch.tensor([[[2.0, 2.0, 0.0]]]))


def test_rank_major_scores_prioritize_all_top1_before_top2() -> None:
    logits = torch.tensor(
        [[[[9.0, 8.0, 1.0], [7.0, 6.0, 5.0]]]],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1]], dtype=torch.long)

    scores = rank_major_scores_from_logits(logits, attention_mask)
    ranked = torch.argsort(scores, dim=-1, descending=True)

    assert ranked.tolist() == [[[0, 1, 2]]]
    assert scores[0, 0, 0] > scores[0, 0, 1] > scores[0, 0, 2]


def test_rank_major_probability_variant_breaks_rank_ties_by_probability_sum() -> None:
    logits = torch.tensor(
        [[[[9.0, 8.0, 1.0], [0.0, 10.0, 1.0], [0.0, 10.0, 1.0]]]],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

    first_seen = rank_major_scores_from_logits(logits, attention_mask)
    prob_ranked = torch.argsort(
        rank_major_prob_scores_from_logits(logits, attention_mask, score_activation='softmax'),
        dim=-1,
        descending=True,
    )

    assert torch.argsort(first_seen, dim=-1, descending=True).tolist() == [[[0, 1, 2]]]
    assert prob_ranked.tolist() == [[[1, 0, 2]]]


def test_rank_major_count_variant_breaks_rank_ties_by_same_rank_vote_count() -> None:
    logits = torch.tensor(
        [[[[9.0, 8.0, 1.0], [0.0, 10.0, 1.0], [0.0, 10.0, 1.0]]]],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)

    count_ranked = torch.argsort(rank_major_count_scores_from_logits(logits, attention_mask), dim=-1, descending=True)

    assert count_ranked.tolist() == [[[1, 0, 2]]]


def test_noisy_or_scores_ignore_padding_tokens() -> None:
    logits = torch.tensor([[[[2.0, 0.0], [0.0, 2.0], [100.0, -100.0]]]], dtype=torch.float32)
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    scores = noisy_or_scores_from_logits(logits, attention_mask, score_activation='softmax')

    probs = torch.softmax(logits[:, :, :2], dim=-1)
    expected = 1.0 - torch.prod(1.0 - probs, dim=2)
    assert torch.allclose(scores, expected)


def test_curve_from_scores_reports_precision_and_recall() -> None:
    scores = torch.tensor([[[0.9, 0.8, 0.1]]], dtype=torch.float32)
    true_set = torch.tensor([[[True, False, True]]])

    rows = curve_from_scores(scores, true_set)

    assert rows[0]['budget'] == 1
    assert rows[0]['precision'] == pytest.approx(1.0)
    assert rows[0]['recall'] == pytest.approx(0.5)
    assert rows[2]['precision'] == pytest.approx(2 / 3)
    assert rows[2]['recall'] == pytest.approx(1.0)


def test_default_model_specs_match_runtime_encoder_predictor_paths() -> None:
    specs = default_model_specs(Path('/repo'))

    assert set(specs) == {'switch-base-128', 'switch-base-256', 'switch-large-128', 'nllb-moe-54b'}
    assert specs['switch-base-128'].ble_path.as_posix().endswith(
        'switch-base-128/sparse-cache-b1-longest-v1/ble/noisyor-from-src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim/encoder_predictor_ble.ts'
    )
    assert specs['switch-base-128'].blte_dir.as_posix().endswith(
        'switch-base-128/sparse-cache-b1-longest-v1/blte/src-simplenn-token-hardce-h384-l1-drop0p5-lr1e4-bs2-seed0-validtrim'
    )
    assert specs['nllb-moe-54b'].ble_path.as_posix().endswith(
        'nllb-moe-54b/sparse-cache-b1-longest-v1/ble/noisyor-from-src-simplenn-token-bce-equal-top2-tokcnt0p004-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim/encoder_predictor_ble.ts'
    )
    assert specs['nllb-moe-54b'].blte_dir.as_posix().endswith(
        'nllb-moe-54b/sparse-cache-b1-longest-v1/blte/src-simplenn-token-bce-equal-top2-tokcnt0p004-h384-l1-drop0p5-lr1e4-bs512-seed0-validtrim'
    )
    assert DEFAULT_MODEL_SPECS == ('switch-base-128', 'switch-base-256', 'switch-large-128', 'nllb-moe-54b')


def test_default_method_order_excludes_train_frequency_and_rank_major_variants() -> None:
    from experiment.scripts.evaluate.evaluate_prefetch_ranking_baselines import DEFAULT_METHOD_ORDER, method_order

    assert DEFAULT_METHOD_ORDER == (
        'ours_noisy_or',
        'token_top1_union_count',
        'token_top2_union_count',
        'rank_major',
        'rank_major_vote_count',
    )
    assert method_order(include_rank_major_variants=False) == DEFAULT_METHOD_ORDER
    assert 'train_frequency' not in DEFAULT_METHOD_ORDER
    assert 'rank_major_prob_sum' not in DEFAULT_METHOD_ORDER
    assert 'rank_major_vote_count' in DEFAULT_METHOD_ORDER


def test_method_order_can_include_rank_major_variants() -> None:
    from experiment.scripts.evaluate.evaluate_prefetch_ranking_baselines import method_order

    assert method_order(include_rank_major_variants=True) == (
        'ours_noisy_or',
        'token_top1_union_count',
        'token_top2_union_count',
        'rank_major',
        'rank_major_vote_count',
        'rank_major_prob_sum',
    )


def test_sample_targets_support_top2_masks() -> None:
    expert_selection = torch.tensor([[[[1, 3], [2, 0], [0, 1]]]], dtype=torch.long)
    expert_selection_mask = torch.tensor([[[[True, True], [True, False], [True, True]]]])
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    true_set, true_count = sample_targets_from_expert_selection(
        expert_selection,
        attention_mask,
        num_experts=4,
        expert_selection_mask=expert_selection_mask,
    )

    assert torch.equal(true_set, torch.tensor([[[False, True, True, True]]]))
    assert torch.equal(true_count, torch.tensor([[3]]))


def test_combined_plot_uses_precision_only_four_panel_layout() -> None:
    from experiment.scripts.evaluate.evaluate_prefetch_ranking_baselines import COMBINED_PLOT_METRICS, combined_subplot_shape

    assert COMBINED_PLOT_METRICS == (('precision', 'Precision@k'),)
    assert combined_subplot_shape(4) == (2, 2)



def test_combined_plot_filename_helpers_separate_paper_and_reference_outputs() -> None:
    from experiment.scripts.evaluate.evaluate_prefetch_ranking_baselines import (
        DEFAULT_METHOD_ORDER,
        reference_method_order,
    )

    assert reference_method_order() == DEFAULT_METHOD_ORDER + (
        'rank_major_prob_sum',
    )


def test_load_existing_curves_reads_requested_model_json(tmp_path: Path) -> None:
    from experiment.scripts.evaluate.evaluate_prefetch_ranking_baselines import ModelSpec, load_existing_curves

    output_dir = tmp_path / 'out'
    model_dir = output_dir / 'switch-base-128'
    model_dir.mkdir(parents=True)
    rows = [{'budget': 1, 'precision': 0.75, 'recall': 0.5, 'mean_overlap': 0.75, 'total_overlap': 3, 'num_sample_layer_items': 4}]
    curves = {
        'ours_noisy_or': rows,
        'token_top1_union_count': rows,
        'token_top2_union_count': rows,
        'rank_major': rows,
        'rank_major_vote_count': rows,
    }
    (model_dir / 'ranking_curves.json').write_text(json.dumps({'model': 'switch-base-128', 'curves': curves}), encoding='utf-8')
    spec = ModelSpec(
        alias='switch-base-128',
        trace_dir=tmp_path / 'trace',
        blte_dir=tmp_path / 'blte',
        ble_path=tmp_path / 'ble.ts',
    )

    loaded = load_existing_curves(output_dir, [spec])

    assert loaded == {'switch-base-128': curves}



def test_load_existing_curves_can_read_alternate_curve_filename(tmp_path: Path) -> None:
    from experiment.scripts.evaluate.evaluate_prefetch_ranking_baselines import ModelSpec, load_existing_curves

    output_dir = tmp_path / 'out'
    model_dir = output_dir / 'switch-base-128'
    model_dir.mkdir(parents=True)
    rows = [{'budget': 1, 'precision': 0.75, 'recall': 0.5, 'mean_overlap': 0.75, 'total_overlap': 3, 'num_sample_layer_items': 4}]
    curves = {
        'ours_noisy_or': rows,
        'token_top1_union_count': rows,
        'token_top2_union_count': rows,
        'rank_major': rows,
        'rank_major_prob_sum': rows,
        'rank_major_vote_count': rows,
    }
    (model_dir / 'ranking_curves_all_methods.json').write_text(
        json.dumps({'model': 'switch-base-128', 'curves': curves}), encoding='utf-8'
    )
    spec = ModelSpec(
        alias='switch-base-128',
        trace_dir=tmp_path / 'trace',
        blte_dir=tmp_path / 'blte',
        ble_path=tmp_path / 'ble.ts',
    )

    loaded = load_existing_curves(
        output_dir,
        [spec],
        include_rank_major_variants=True,
        filename='ranking_curves_all_methods.json',
    )

    assert set(loaded['switch-base-128']) == set(curves)


def test_load_existing_curves_can_read_rank_major_variants_when_requested(tmp_path: Path) -> None:
    from experiment.scripts.evaluate.evaluate_prefetch_ranking_baselines import ModelSpec, load_existing_curves

    output_dir = tmp_path / 'out'
    model_dir = output_dir / 'switch-base-128'
    model_dir.mkdir(parents=True)
    rows = [{'budget': 1, 'precision': 0.75, 'recall': 0.5, 'mean_overlap': 0.75, 'total_overlap': 3, 'num_sample_layer_items': 4}]
    curves = {
        'ours_noisy_or': rows,
        'token_top1_union_count': rows,
        'token_top2_union_count': rows,
        'rank_major': rows,
        'rank_major_prob_sum': rows,
        'rank_major_vote_count': rows,
    }
    (model_dir / 'ranking_curves.json').write_text(json.dumps({'model': 'switch-base-128', 'curves': curves}), encoding='utf-8')
    spec = ModelSpec(
        alias='switch-base-128',
        trace_dir=tmp_path / 'trace',
        blte_dir=tmp_path / 'blte',
        ble_path=tmp_path / 'ble.ts',
    )

    loaded = load_existing_curves(output_dir, [spec], include_rank_major_variants=True)

    assert set(loaded['switch-base-128']) == set(curves)

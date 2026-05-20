from pathlib import Path
import csv
import importlib.util
import sys

SCRIPT = Path('/mnt/huwf5/promoe/performance_baseline/compare/analyze/analyze_encoder_cache.py')
spec = importlib.util.spec_from_file_location('analyze_encoder_cache', SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_new_log_parser_splits_entry_and_expert_use_by_experiment(tmp_path):
    log_on = tmp_path / 'run_on.log'
    log_on.write_text('''ENABLE_DECODER_WARMUP_OVERLAP=True\nSeq 0/2, original_seq_ids=[100], decoding...\n[ts] cache_stage_occupancy context=encoder_layer layer=5 forward_epoch=1 generate_epoch=1 encoder_experts=500 encoder_ready=490 decoder_experts=76 decoder_ready=70\n[ts] cache_layer_entry_demand context=encoder_layer layer=5 forward_epoch=1 generate_epoch=1 needed_experts=4 cache_hits=2 cache_misses=2 cache_slot_hits=2 cache_slot_misses=2 ready_hits=2 ready_misses=2 experts=[1,2,3,4,] not_ready_experts=[3,4,]\n[ts] cache_expert_use_demand context=encoder_layer layer=5 forward_epoch=1 generate_epoch=1 needed_experts=4 cache_hits=3 cache_misses=1 cache_slot_hits=4 cache_slot_misses=0 ready_hits=3 ready_misses=1 experts=[1,2,3,4,] miss_experts=[4,]\nexplicit_input_ms:1.0 explicit_cache_init_ms:10.0 explicit_ttft_ms:20.0 explicit_decode_tpot_excl_first_ms:3.0 (gen_forward_steps:2 new_tokens:3)\n''')
    (tmp_path / 'run_on.summary').write_text('enable_decoder_warmup_overlap=True\n')
    log_off = tmp_path / 'run_off.log'
    log_off.write_text('''ENABLE_DECODER_WARMUP_OVERLAP=False\nSeq 0/2, original_seq_ids=[100], decoding...\n[ts] cache_layer_entry_demand context=encoder_layer layer=5 forward_epoch=1 generate_epoch=1 needed_experts=4 cache_hits=1 cache_misses=3 cache_slot_hits=1 cache_slot_misses=3 ready_hits=1 ready_misses=3 experts=[1,2,3,4,] not_ready_experts=[2,3,4,]\n[ts] cache_expert_use_demand context=encoder_layer layer=5 forward_epoch=1 generate_epoch=1 needed_experts=4 cache_hits=2 cache_misses=2 cache_slot_hits=4 cache_slot_misses=0 ready_hits=2 ready_misses=2 experts=[1,2,3,4,] miss_experts=[3,4,]\nexplicit_input_ms:1.0 explicit_cache_init_ms:10.0 explicit_ttft_ms:30.0 explicit_decode_tpot_excl_first_ms:3.0 (gen_forward_steps:2 new_tokens:3)\n''')
    (tmp_path / 'run_off.summary').write_text('enable_decoder_warmup_overlap=False\n')

    samples = []
    for path in sorted(tmp_path.glob('*.log')):
        samples.extend(mod.parse_log(path))
    assert len(samples) == 2
    on = [s for s in samples if s.experiment == 'overlap_on'][0]
    off = [s for s in samples if s.experiment == 'overlap_off'][0]
    assert on.layers[5].entry.cache_misses == 2
    assert on.layers[5].use.cache_misses == 1
    assert on.layers[5].waited_experts() == 1
    assert on.layers[5].rescued_experts() == 1
    assert off.layers[5].entry.cache_misses == 3
    assert off.layers[5].use.cache_misses == 2

    out = tmp_path / 'out'
    mod.write_outputs(samples, out, max_layers=6)
    sample_rows = list(csv.DictReader((out / 'encoder_sample_summary.csv').open()))
    assert {r['experiment'] for r in sample_rows} == {'overlap_on', 'overlap_off'}
    on_row = [r for r in sample_rows if r['experiment'] == 'overlap_on'][0]
    assert on_row['l5_entry_misses'] == '2'
    assert on_row['l5_waited_experts'] == '1'
    assert on_row['l5_rescued_experts'] == '1'
    compare_rows = list(csv.DictReader((out / 'encoder_sample_experiment_compare.csv').open()))
    assert compare_rows[0]['original_seq_ids'] == '100'
    assert compare_rows[0]['ttft_delta_on_minus_off_ms'] == '-10.000000'

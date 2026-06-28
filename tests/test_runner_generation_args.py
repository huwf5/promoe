from sparse_llm_cache.utils.runner_util import parse_args


def test_runner_defaults_to_greedy_generation_for_benchmark():
  parsed = parse_args([])

  assert parsed["do_sample"] is False
  assert parsed["num_beams"] == 1


def test_runner_parses_generation_strategy_overrides():
  parsed = parse_args(["--do_sample", "True", "--num_beams", "4"])

  assert parsed["do_sample"] is True
  assert parsed["num_beams"] == 4

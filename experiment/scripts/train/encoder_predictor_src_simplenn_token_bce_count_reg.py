#!/usr/bin/env python3
"""Train SRC SimpleNN token predictor with equal top2 BCE and count regularization.

This entry point keeps the original hard-CE/BCE trainer unchanged by supplying
a separate default training setting for NLLB-style top2 cache prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from experiment.scripts.train.encoder_predictor_src_simplenn_token_hard_ce import main as _base_main


DEFAULT_COUNT_REG_ARGS = [
    "--loss-type",
    "multi_label_bce",
    "--no-use-expert-weights-in-loss",
    "--token-cardinality-loss-weight",
    "0.1",
    "--layer-count-loss-weight",
    "0.05",
]


def main(argv: list[str] | None = None):
    user_args = sys.argv[1:] if argv is None else list(argv)
    return _base_main(DEFAULT_COUNT_REG_ARGS + user_args)


if __name__ == "__main__":
    main()

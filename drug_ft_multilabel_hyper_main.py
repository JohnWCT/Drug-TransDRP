#!/usr/bin/env python3
"""Thin entry: TransDRP multilabel fine-tuning."""

import sys
from pathlib import Path

_TRANSDRP_ROOT = Path(__file__).resolve().parent
if str(_TRANSDRP_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRANSDRP_ROOT))

from transdrp_multilabel.config import build_finetune_arg_parser, config_from_finetune_args
from transdrp_multilabel.seed import set_global_seed
from transdrp_multilabel.training.runners import FineTuneRunner

def main() -> None:
    parser = build_finetune_arg_parser()
    args = parser.parse_args()
    config = config_from_finetune_args(args)
    set_global_seed(config.seed)
    FineTuneRunner(config).run()

if __name__ == "__main__":
    main()

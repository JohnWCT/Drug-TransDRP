import pytest
from transdrp_multilabel.smoke.smoke_runner import run_smoke_test

def test_integration_smoke():
    # run_smoke_test runs pre-training and both fine-tuning runs (regression & classification)
    # with 2 epochs each using synthetic CPU-only models.
    # It raises no exceptions on success.
    run_smoke_test()

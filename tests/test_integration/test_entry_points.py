import pytest

def test_import_pretrain_entry():
    import pretrain_multilabel_hyper_main
    assert hasattr(pretrain_multilabel_hyper_main, "main")

def test_import_finetune_entry():
    import drug_ft_multilabel_hyper_main
    assert hasattr(drug_ft_multilabel_hyper_main, "main")

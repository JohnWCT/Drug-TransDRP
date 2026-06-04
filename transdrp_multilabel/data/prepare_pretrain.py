from transdrp_multilabel.contracts import PreparedPretrainData, TransDRPMultilabelConfig
from transdrp_multilabel.data.omics import align_omics_features, read_omics_table

def prepare_pretrain_data(config: TransDRPMultilabelConfig) -> PreparedPretrainData:
    source = read_omics_table(config.source_omics_path, config.source_sample_col, "source")
    target = read_omics_table(config.target_omics_path, config.target_sample_col, "target")
    source_aligned, target_aligned, alignment = align_omics_features(source, target)
    return PreparedPretrainData(
        source_omics=source_aligned,
        target_omics=target_aligned,
        feature_alignment=alignment,
    )

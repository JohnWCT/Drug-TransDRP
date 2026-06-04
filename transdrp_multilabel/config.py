from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal
from transdrp_multilabel.contracts import TransDRPMultilabelConfig

# Hardcoded defaults matching train_params.json in case it is missing
_DEFAULT_JSON: dict[str, Any] = {
    "unlabeled": {
        "batch_size": 128,
        "lr": 0.0001,
        "pretrain_num_epochs": 300,
        "train_num_epochs": 300
    },
    "labeled": {
        "classifier_hidden_dims": [64, 32, 16],
        "batch_size": 64,
        "lr": 0.0001,
        "train_num_epochs": 600,
        "uda_num_epochs": 300
    },
    "encoder_hidden_dims": [512, 256, 64],
    "decoder_hidden_dims": [128, 256, 512],
    "latent_dim": 64,
    "drop": 0.2
}

def load_train_params(transdrp_root: Path | None = None, section: Literal["unlabeled", "labeled"] = "unlabeled") -> dict[str, Any]:
    if transdrp_root is None:
        transdrp_root = Path(__file__).resolve().parents[1]

    train_json = transdrp_root / "train_params.json"
    if train_json.is_file():
        try:
            with train_json.open(encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = _DEFAULT_JSON
    else:
        raw = _DEFAULT_JSON

    params = {k: v for k, v in raw.items() if k not in ("unlabeled", "labeled")}
    params.update(raw.get(section, {}))
    params.setdefault("norm_flag", True)
    params.setdefault("retrain_flag", True)
    params.setdefault("alph", 0.2)
    params.setdefault("beta", 0.3)
    return params

def build_pretrain_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TransDRP multilabel pre-training")
    p.add_argument("--source_omics_path", required=True)
    p.add_argument("--target_omics_path", required=True)
    p.add_argument("--source_sample_col", default="Sample_ID")
    p.add_argument("--target_sample_col", default="tissue_id")
    p.add_argument("--method", default="transdrp_ae")
    p.add_argument("--output_dir", required=True)

    # Overwrite behavior: --overwrite or --no-overwrite
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--epochs", type=int, default=None, help="Override pretrain epochs")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--norm_flag", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--retrain_flag", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default=None)

    return p

def build_finetune_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TransDRP multilabel fine-tuning")
    p.add_argument("--task_type", choices=["classification", "regression"], required=True)
    p.add_argument("--source_omics_path", required=True)
    p.add_argument("--target_omics_path", required=True)
    p.add_argument("--source_response_path", required=True)
    p.add_argument("--target_response_path", required=True)

    p.add_argument("--source_sample_col", default="Sample_ID")
    p.add_argument("--target_sample_col", default="tissue_id")
    p.add_argument("--target_response_sample_col", default="Patient_id")
    p.add_argument("--drug_col", default="drug_name")
    p.add_argument("--source_response_col", default="Label")
    p.add_argument("--target_response_col", default="Label")

    p.add_argument("--pretrain_checkpoint", required=True)
    p.add_argument("--method", default="transdrp_ft")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--metric", default=None, help="Early stopping metric, e.g. macro_auroc or macro_mae")
    p.add_argument("--reg_loss", choices=["mae"], default="mae")
    p.add_argument("--prediction_threshold", type=float, default=0.5)
    p.add_argument(
        "--regression_binary_threshold",
        type=float,
        default=1.0,
        help=(
            "Regression only. Threshold on -log2(AUC): used for (1) drug co-occurrence graph "
            "binarization and (2) target hard pred_label for F1/ACC. Default 1.0 = -log2(0.5). "
            "Not used for source MAE loss or target AUROC/AUPRC (raw pred_score)."
        ),
    )
    p.add_argument(
        "--threshold_label",
        type=float,
        default=0.1,
        help="Minimum normalized co-occurrence to keep a drug-drug edge in the label graph (legacy --thres_label).",
    )
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--source_test_size", type=float, default=0.25, help="Test ratio for evaluation")

    p.add_argument("--epochs", type=int, default=None, help="Override fine-tuning epochs")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=2024)

    # TransDRP contrastive loss / GRL loss coefficients
    p.add_argument("--alph", type=float, default=None, help="Transfer loss coefficient")
    p.add_argument("--beta", type=float, default=None, help="Contrastive loss coefficient")

    # SMILES and Cancer type paths.
    # Cancer type maps are REQUIRED: tissue prototypes / latent analysis must run
    # every fold, so a missing map is a hard error rather than a silent skip.
    p.add_argument("--source_cancer_type_path", required=True)
    p.add_argument("--target_cancer_type_path", required=True)
    p.add_argument("--cancer_type_col", default="Cancer_type")
    p.add_argument("--drug_smiles_path", required=True)

    p.add_argument("--norm_flag", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--retrain_flag", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default=None)

    return p

def _resolve_config_dict(args: argparse.Namespace, mode: Literal["pretrain", "finetune"]) -> dict[str, Any]:
    transdrp_root = Path(__file__).resolve().parents[1]
    section: Literal["unlabeled", "labeled"] = "unlabeled" if mode == "pretrain" else "labeled"
    tp = load_train_params(transdrp_root, section=section)

    # Override defaults with CLI args
    batch_size = int(args.batch_size if args.batch_size is not None else tp["batch_size"])
    lr = float(args.lr if args.lr is not None else tp["lr"])

    # For pretraining epochs
    if mode == "pretrain":
        epochs = int(args.epochs if args.epochs is not None else tp["pretrain_num_epochs"])
    else:
        # Fine-tuning epochs (uda_num_epochs is used in adaptation loop)
        epochs = int(args.epochs if args.epochs is not None else tp["uda_num_epochs"])

    norm_flag = bool(args.norm_flag if args.norm_flag is not None else tp["norm_flag"])
    retrain_flag = bool(args.retrain_flag if args.retrain_flag is not None else tp["retrain_flag"])

    alph = float(args.alph if getattr(args, "alph", None) is not None else tp["alph"])
    beta = float(args.beta if getattr(args, "beta", None) is not None else tp["beta"])

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "source_omics_path": args.source_omics_path,
        "target_omics_path": args.target_omics_path,
        "source_sample_col": args.source_sample_col,
        "target_sample_col": args.target_sample_col,
        "target_response_sample_col": getattr(args, "target_response_sample_col", "Patient_id"),
        "drug_col": getattr(args, "drug_col", "drug_name"),
        "source_response_col": getattr(args, "source_response_col", "Label"),
        "target_response_col": getattr(args, "target_response_col", "Label"),
        "method": args.method,
        "output_dir": args.output_dir,
        "overwrite": args.overwrite,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "seed": args.seed,
        "n_splits": int(getattr(args, "n_splits", 5)),
        "source_test_size": float(getattr(args, "source_test_size", 0.25)),
        "reg_loss": "mae",
        "prediction_threshold": float(getattr(args, "prediction_threshold", 0.5)),
        "regression_binary_threshold": float(getattr(args, "regression_binary_threshold", 1.0)),
        "threshold_label": float(getattr(args, "threshold_label", 0.1)),
        "source_cancer_type_path": getattr(args, "source_cancer_type_path", None),
        "target_cancer_type_path": getattr(args, "target_cancer_type_path", None),
        "cancer_type_col": getattr(args, "cancer_type_col", None),
        "drug_smiles_path": getattr(args, "drug_smiles_path", None),
        "alph": alph,
        "beta": beta,
        "latent_dim": int(tp["latent_dim"]),
        "encoder_hidden_dims": tuple(int(x) for x in tp["encoder_hidden_dims"]),
        "decoder_hidden_dims": tuple(int(x) for x in tp["decoder_hidden_dims"]),
        "classifier_hidden_dims": tuple(int(x) for x in tp.get("classifier_hidden_dims", (64, 32, 16))),
        "drop": float(tp["drop"]),
        "norm_flag": norm_flag,
        "retrain_flag": retrain_flag,
        "device": str(device)
    }

def config_from_pretrain_args(args: argparse.Namespace) -> TransDRPMultilabelConfig:
    d = _resolve_config_dict(args, "pretrain")
    return TransDRPMultilabelConfig(
        task_type="classification",
        source_response_path=None,
        target_response_path=None,
        pretrain_checkpoint=None,
        metric=None,
        **d
    )

def config_from_finetune_args(args: argparse.Namespace) -> TransDRPMultilabelConfig:
    if args.task_type not in ("classification", "regression"):
        raise ValueError(f"invalid task_type: {args.task_type}")
    if not args.pretrain_checkpoint:
        raise ValueError("fine-tune requires --pretrain_checkpoint")
    d = _resolve_config_dict(args, "finetune")

    metric = args.metric
    if metric is None:
        metric = "macro_auroc" if args.task_type == "classification" else "macro_mae"

    return TransDRPMultilabelConfig(
        task_type=args.task_type,
        source_response_path=args.source_response_path,
        target_response_path=args.target_response_path,
        pretrain_checkpoint=args.pretrain_checkpoint,
        metric=metric,
        **d
    )

def config_to_dict(config: TransDRPMultilabelConfig) -> dict[str, Any]:
    d = asdict(config)
    d["encoder_hidden_dims"] = list(config.encoder_hidden_dims)
    d["decoder_hidden_dims"] = list(config.decoder_hidden_dims)
    d["classifier_hidden_dims"] = list(config.classifier_hidden_dims)
    return d

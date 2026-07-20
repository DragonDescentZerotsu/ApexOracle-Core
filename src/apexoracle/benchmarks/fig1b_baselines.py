"""Common-fold Chemprop baselines for the three Fig. 1b target datasets."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit


DEFAULT_CONFIG = Path(
    "configs/antibiotic_classification/fig1b_chemprop_baselines.yaml"
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    supported_protocols = {
        "fig1b_common_outer_folds_chemprop_sensitivity",
        "fig1b_common_outer_folds_chemprop_final_ensembles",
    }
    if config.get("protocol") not in supported_protocols:
        raise ValueError("Unexpected Fig. 1b baseline protocol")
    return config


def build_target_table(
    repo_root: Path, config: dict[str, Any], group: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recover canonical SMILES while preserving the ApexOracle target order."""

    targets = {int(item["group"]): item for item in config["targets"]}
    if group not in targets:
        raise ValueError(f"Unknown target group: {group}")
    target = targets[group]
    token_path = _resolve(repo_root, config["tokenized_records"])
    smiles_path = _resolve(repo_root, target["records"])
    tokenized = pd.read_csv(token_path)
    tokenized = tokenized[tokenized["strain_name"] == target["strain"]].copy()
    tokenized["token_length"] = tokenized["SMILES"].map(
        lambda value: len(ast.literal_eval(value))
    )
    eligible = tokenized[
        tokenized["token_length"] <= int(config["max_token_length"])
    ][["DBAASP_id", "MIC", "token_length"]].reset_index(drop=True)
    canonical = pd.read_csv(smiles_path)[["DBAASP_id", "SMILES", "MIC"]]
    if canonical["DBAASP_id"].duplicated().any():
        raise ValueError(f"Duplicate molecule IDs in {smiles_path}")
    table = eligible.merge(
        canonical,
        on="DBAASP_id",
        how="left",
        validate="one_to_one",
        suffixes=("_tokenized", "_canonical"),
        sort=False,
    )
    if table["SMILES"].isna().any():
        raise ValueError("Canonical SMILES lookup left missing rows")
    if not np.array_equal(
        table["MIC_tokenized"].to_numpy(), table["MIC_canonical"].to_numpy()
    ):
        raise ValueError("Tokenized and canonical labels differ")
    output = table.rename(
        columns={"DBAASP_id": "molecule_id", "SMILES": "smiles"}
    )[["molecule_id", "smiles", "MIC_canonical", "token_length"]]
    output = output.rename(columns={"MIC_canonical": "label"})
    output["label"] = output["label"].astype(int)
    if set(output["label"].unique()) != {0, 1}:
        raise ValueError("Target labels must be binary")
    metadata = {
        "group": group,
        "strain": target["strain"],
        "display_name": target["display_name"],
        "profile": target["profile"],
        "num_source_rows": int(len(tokenized)),
        "num_eligible_rows": int(len(output)),
        "num_positive": int(output["label"].sum()),
        "tokenized_records": str(token_path),
        "tokenized_records_sha256": _sha256(token_path),
        "canonical_smiles_records": str(smiles_path),
        "canonical_smiles_records_sha256": _sha256(smiles_path),
    }
    return output, metadata


def prepare_common_folds(
    repo_root: Path,
    config: dict[str, Any],
    *,
    group: int,
    output_dir: Path,
) -> dict[str, Any]:
    table, metadata = build_target_table(repo_root, config, group)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_assignments = np.full(len(table), -1, dtype=int)
    splitter = KFold(
        n_splits=int(config["outer_folds"]),
        shuffle=True,
        random_state=int(config["outer_seed"]),
    )
    fold_reports = []
    for fold, (outer_train, test) in enumerate(splitter.split(table)):
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=float(config["validation_fraction_of_outer_train"]),
            random_state=int(config["validation_seed"]) + fold,
        )
        train_rel, val_rel = next(
            inner.split(outer_train, table.iloc[outer_train]["label"])
        )
        train = outer_train[train_rel]
        validation = outer_train[val_rel]
        fold_assignments[test] = fold
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for name, indices in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            selected = table.iloc[indices]
            selected[["smiles", "label"]].rename(
                columns={"label": "activity"}
            ).to_csv(fold_dir / f"{name}.csv", index=False)
            selected[["molecule_id", "smiles", "label"]].to_csv(
                fold_dir / f"{name}_manifest.csv", index=False
            )
        fold_reports.append(
            {
                "fold": fold,
                "train": int(len(train)),
                "validation": int(len(validation)),
                "test": int(len(test)),
                "test_positive": int(table.iloc[test]["label"].sum()),
            }
        )
    if np.any(fold_assignments < 0):
        raise RuntimeError("Outer folds did not cover every eligible molecule")
    fold_table = table.copy()
    fold_table["fold"] = fold_assignments
    fold_table.to_csv(output_dir / "folds.csv", index=False)
    report = {
        **metadata,
        "protocol": config["protocol"],
        "outer_folds": int(config["outer_folds"]),
        "outer_seed": int(config["outer_seed"]),
        "validation_fraction_of_outer_train": float(
            config["validation_fraction_of_outer_train"]
        ),
        "validation_seed_base": int(config["validation_seed"]),
        "folds": fold_reports,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _chemprop_executable(bin_dir: Path | None, name: str) -> str:
    executable = str(bin_dir / name) if bin_dir is not None else shutil.which(name)
    if not executable or not Path(executable).exists():
        raise FileNotFoundError(f"Cannot find {name}; use --chemprop-bin-dir")
    return executable


def _feature_arguments(profile: dict[str, Any]) -> list[str]:
    """Return the feature flags for one paper-specific baseline profile."""

    generator = profile.get("features_generator")
    if not generator:
        return []
    arguments = ["--features_generator", str(generator)]
    if bool(profile.get("no_features_scaling", False)):
        arguments.append("--no_features_scaling")
    return arguments


def run_fold(
    config: dict[str, Any],
    *,
    group: int,
    fold: int,
    output_dir: Path,
    chemprop_bin_dir: Path | None,
    gpu: int | None,
    ensemble_size: int | None,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    targets = {int(item["group"]): item for item in config["targets"]}
    target = targets[group]
    profile = config["profiles"][target["profile"]]
    common = config["chemprop"]
    selected_ensemble_size = (
        int(profile["ensemble_size"])
        if ensemble_size is None
        else int(ensemble_size)
    )
    if selected_ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive")
    feature_arguments = _feature_arguments(profile)
    fold_dir = output_dir / f"fold_{fold}"
    model_dir = fold_dir / "checkpoints"
    train = _chemprop_executable(chemprop_bin_dir, "chemprop_train")
    predict = _chemprop_executable(chemprop_bin_dir, "chemprop_predict")
    chemprop_environment = os.environ.copy()
    # Chemprop 1.5 checkpoints contain argparse.Namespace. PyTorch >=2.6 changed
    # torch.load's default to weights_only=True, so trusted, locally generated
    # Chemprop checkpoints need the documented compatibility override.
    chemprop_environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    command = [
        train,
        "--data_path",
        str(fold_dir / "train.csv"),
        "--separate_val_path",
        str(fold_dir / "validation.csv"),
        "--separate_test_path",
        str(fold_dir / "test.csv"),
        "--target_columns",
        "activity",
        "--dataset_type",
        "classification",
        "--save_dir",
        str(model_dir),
        "--ensemble_size",
        str(selected_ensemble_size),
        "--epochs",
        str(common["epochs"]),
        "--batch_size",
        str(common["batch_size"]),
        "--depth",
        str(profile["depth"]),
        "--dropout",
        str(profile["dropout"]),
        "--ffn_num_layers",
        str(profile["ffn_num_layers"]),
        "--hidden_size",
        str(profile["hidden_size"]),
        "--init_lr",
        str(common["init_lr"]),
        "--max_lr",
        str(common["max_lr"]),
        "--final_lr",
        str(common["final_lr"]),
        "--metric",
        str(common["checkpoint_metric"]),
        "--extra_metrics",
        *[str(metric) for metric in common["reported_metrics"]],
        "--seed",
        str(int(config["validation_seed"]) + fold),
        "--pytorch_seed",
        str(int(config["validation_seed"]) + fold),
        "--num_workers",
        "0",
        "--quiet",
    ]
    command.extend(feature_arguments)
    if gpu is not None:
        command.extend(["--gpu", str(gpu)])
    checkpoint_paths = list(model_dir.glob("fold_0/model_*/model.pt"))
    training_returncode = 0
    training_reused = (
        reuse_existing and len(checkpoint_paths) == selected_ensemble_size
    )
    if not training_reused:
        try:
            subprocess.run(command, check=True, env=chemprop_environment)
        except subprocess.CalledProcessError as error:
            # Chemprop 1.x performs a redundant test evaluation after the best
            # checkpoint has been written. Some modern dependency combinations
            # fail in that reporting step; a complete checkpoint grid remains
            # valid for the explicit prediction pass below.
            checkpoint_paths = list(model_dir.glob("fold_0/model_*/model.pt"))
            if len(checkpoint_paths) != selected_ensemble_size:
                raise
            training_returncode = int(error.returncode)
    raw_predictions = fold_dir / "chemprop_predictions.csv"
    predict_command = [
        predict,
        "--test_path",
        str(fold_dir / "test.csv"),
        "--checkpoint_dir",
        str(model_dir),
        "--preds_path",
        str(raw_predictions),
        "--num_workers",
        "0",
    ]
    predict_command.extend(feature_arguments)
    if gpu is not None:
        predict_command.extend(["--gpu", str(gpu)])
    subprocess.run(predict_command, check=True, env=chemprop_environment)
    manifest = pd.read_csv(fold_dir / "test_manifest.csv")
    predicted = pd.read_csv(raw_predictions)
    if "activity" not in predicted or len(predicted) != len(manifest):
        raise RuntimeError("Chemprop prediction output does not match test manifest")
    numeric_predictions = pd.to_numeric(predicted["activity"], errors="coerce")
    invalid = numeric_predictions.isna()
    exclusions_path = fold_dir / "prediction_exclusions.csv"
    excluded = manifest.loc[invalid].copy()
    excluded["reason"] = "chemprop_invalid_smiles"
    excluded.to_csv(exclusions_path, index=False)
    manifest = manifest.loc[~invalid].reset_index(drop=True)
    probabilities = numeric_predictions.loc[~invalid].astype(float).to_numpy()
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Chemprop produced non-finite predictions")
    result = manifest.copy()
    result["prediction"] = probabilities
    result["group"] = group
    result["fold"] = fold
    result["model"] = f"chemprop_{target['profile']}"
    prediction_path = fold_dir / "predictions.csv"
    result.to_csv(prediction_path, index=False)
    metrics = {
        "auroc": float(roc_auc_score(result["label"], result["prediction"])),
        "auprc": float(
            average_precision_score(result["label"], result["prediction"])
        ),
    }
    report = {
        "group": group,
        "fold": fold,
        "profile": target["profile"],
        "ensemble_size": selected_ensemble_size,
        "features_generator": profile.get("features_generator"),
        "num_examples": int(len(result)),
        "num_prediction_exclusions": int(invalid.sum()),
        "metrics": metrics,
        "predictions": str(prediction_path),
        "prediction_exclusions": str(exclusions_path),
        "selection_data": "outer_train_internal_validation_only",
        "training_reused": training_reused,
        "chemprop_training_returncode": training_returncode,
    }
    (fold_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def summarize_group(
    config: dict[str, Any], *, group: int, output_dir: Path
) -> dict[str, Any]:
    expected = pd.read_csv(output_dir / "folds.csv")
    predictions = []
    fold_metrics = []
    for fold in range(int(config["outer_folds"])):
        path = output_dir / f"fold_{fold}" / "predictions.csv"
        frame = pd.read_csv(path)
        if set(frame["fold"].unique()) != {fold}:
            raise ValueError(f"Unexpected fold labels in {path}")
        predictions.append(frame)
        fold_metrics.append(
            {
                "fold": fold,
                "num_examples": int(len(frame)),
                "auroc": float(roc_auc_score(frame["label"], frame["prediction"])),
                "auprc": float(
                    average_precision_score(frame["label"], frame["prediction"])
                ),
            }
        )
    combined = pd.concat(predictions, ignore_index=True)
    if combined["molecule_id"].duplicated().any():
        raise ValueError("OOF predictions contain duplicate molecule IDs")
    exclusion_frames = []
    for fold in range(int(config["outer_folds"])):
        path = output_dir / f"fold_{fold}" / "prediction_exclusions.csv"
        if path.exists():
            exclusion_frames.append(pd.read_csv(path))
    exclusions = (
        pd.concat(exclusion_frames, ignore_index=True)
        if exclusion_frames
        else pd.DataFrame(columns=["molecule_id", "smiles", "label", "reason"])
    )
    if exclusions["molecule_id"].duplicated().any():
        raise ValueError("An excluded molecule appears in more than one test fold")
    expected = expected[~expected["molecule_id"].isin(exclusions["molecule_id"])]
    exclusions.to_csv(output_dir / "prediction_exclusions.csv", index=False)
    expected_contract = expected[["molecule_id", "label", "fold"]].sort_values(
        "molecule_id"
    )
    observed_contract = combined[["molecule_id", "label", "fold"]].sort_values(
        "molecule_id"
    )
    pd.testing.assert_frame_equal(
        expected_contract.reset_index(drop=True),
        observed_contract.reset_index(drop=True),
        check_dtype=False,
    )
    order = {molecule_id: index for index, molecule_id in enumerate(expected.molecule_id)}
    combined["_order"] = combined["molecule_id"].map(order)
    combined = combined.sort_values("_order").drop(columns="_order")
    output_path = output_dir / "oof_predictions.csv"
    combined.to_csv(output_path, index=False)
    report = {
        "group": group,
        "num_examples": int(len(combined)),
        "num_positive": int(combined["label"].sum()),
        "num_prediction_exclusions": int(len(exclusions)),
        "prediction_exclusions": str(output_dir / "prediction_exclusions.csv"),
        "fold_metrics": fold_metrics,
        "fold_mean": {
            metric: float(np.mean([item[metric] for item in fold_metrics]))
            for metric in ("auroc", "auprc")
        },
        "fold_sample_sd": {
            metric: float(np.std([item[metric] for item in fold_metrics], ddof=1))
            for metric in ("auroc", "auprc")
        },
        "pooled_oof_metrics": {
            "auroc": float(roc_auc_score(combined["label"], combined["prediction"])),
            "auprc": float(
                average_precision_score(combined["label"], combined["prediction"])
            ),
        },
        "predictions": str(output_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/fig1b_revision/baselines")
    )
    parser.add_argument("--chemprop-bin-dir", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument(
        "--ensemble-size",
        type=int,
        help="Override the paper-specific ensemble size in the profile.",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    output_root = (
        args.output_root if args.output_root.is_absolute() else root / args.output_root
    )
    output_dir = output_root / f"group_{args.group}"
    manifest = prepare_common_folds(
        root, config, group=args.group, output_dir=output_dir
    )
    if args.prepare_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest
    if args.summarize:
        report = summarize_group(config, group=args.group, output_dir=output_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
        return report
    if args.fold is None:
        raise SystemExit("--fold is required unless --prepare-only is used")
    if args.fold < 0 or args.fold >= int(config["outer_folds"]):
        raise SystemExit("--fold is outside the configured outer-fold range")
    report = run_fold(
        config,
        group=args.group,
        fold=args.fold,
        output_dir=output_dir,
        chemprop_bin_dir=args.chemprop_bin_dir,
        gpu=args.gpu,
        ensemble_size=args.ensemble_size,
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()

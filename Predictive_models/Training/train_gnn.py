"""Only dummy data for demonstrating training is included, so the training results are not reproducible."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader


TRAINING_DIR = Path(__file__).resolve().parent
PREDICTIVE_MODELS_DIR = TRAINING_DIR.parent
MODEL_DIR = PREDICTIVE_MODELS_DIR / "Model_architecture"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from DMPNN_model_architecture import DMPNN_BLI, DMPNN_CLI
from GAT_model_architecture import GAT_BLI, GAT_CLI
from model_architecture import GCN_BLI, GCN_CLI

from make_graph_dataloader import MolToGraph, TripleGraphDataset


MODEL_REGISTRY = {
    "gcn-cli": GCN_CLI,
    "gcn-bli": GCN_BLI,
    "gnn-cli": GCN_CLI,
    "gnn-bli": GCN_BLI,
    "gat-cli": GAT_CLI,
    "gat-bli": GAT_BLI,
    "dmpnn-cli": DMPNN_CLI,
    "dmpnn-bli": DMPNN_BLI,
}

SPLIT_FILES = {
    "train": "United_DB_train.csv",
    "valid": "United_DB_valid.csv",
    "test": "United_DB_test.csv",
}

REQUIRED_COLUMNS = {
    "Smiles",
    "elu1_smiles",
    "elu2_smiles",
    "ratio1",
    "ratio2",
    "Rf",
    "sample_id",
}

GRAPH_COLUMNS = {
    "compound": "Smiles",
    "elu1": "elu1_smiles",
    "elu2": "elu2_smiles",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a GCN, GAT, or DMPNN model from the existing "
            "United_DB train/valid/test CSV files."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_REGISTRY),
        help="Model backbone and eluent-interaction head.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=TRAINING_DIR / "DB" / "Rf",
        help="Directory containing the three United_DB split CSV files.",
    )
    parser.add_argument(
        "--standardization-dir",
        type=Path,
        default=TRAINING_DIR / "standardization",
        help="Directory in which mean.npy and std.npy are written.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Run directory for logs and checkpoints. By default, a timestamped "
            "directory is created under Training/runs/."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--scheduler-step", type=int, default=50)
    parser.add_argument("--scheduler-gamma", type=float, default=0.985)
    parser.add_argument(
        "--patience",
        type=int,
        default=300,
        help="Early-stopping patience. Use 0 to disable early stopping.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last_checkpoint.pt in --output-dir.",
    )
    parser.add_argument(
        "--reuse-graphs",
        action="store_true",
        help=(
            "Reuse graph cache files. By default they are rebuilt to ensure "
            "their labels use the newly calculated training statistics."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 1:
        raise ValueError(
            "--batch-size must be greater than 1 because the models use "
            "batch normalization."
        )
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in the range [0, 1).")
    if args.scheduler_step <= 0:
        raise ValueError("--scheduler-step must be positive.")
    if not 0 < args.scheduler_gamma <= 1:
        raise ValueError("--scheduler-gamma must be in the range (0, 1].")
    if args.patience < 0:
        raise ValueError("--patience cannot be negative.")
    if args.resume and args.output_dir is None:
        raise ValueError("--resume requires an explicit --output-dir.")


def configure_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            TRAINING_DIR / "runs" / f"{args.model}_{timestamp}"
        ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("dtlc_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / "training.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{requested_device}' was requested, but CUDA is "
            "not available."
        )
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(
            f"CUDA device index {device.index} is unavailable; "
            f"{torch.cuda.device_count()} CUDA device(s) were detected."
        )
    return device


def load_split_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    data_dir = data_dir.resolve()
    splits: dict[str, pd.DataFrame] = {}

    for split, filename in SPLIT_FILES.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} split: {path}")
        dataframe = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(dataframe.columns)
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {sorted(missing)}"
            )
        if dataframe.empty:
            raise ValueError(f"{path.name} contains no rows.")
        if dataframe[["Rf", "ratio1", "ratio2", "sample_id"]].isna().any().any():
            raise ValueError(
                f"{path.name} contains missing target, ratio, or sample ID values."
            )
        if dataframe["sample_id"].duplicated().any():
            raise ValueError(f"{path.name} contains duplicate sample_id values.")
        splits[split] = dataframe

    return splits


def fit_standardization(
    train_dataframe: pd.DataFrame,
    standardization_dir: Path,
) -> tuple[float, float]:
    """Fit target statistics using the training split only."""
    values = train_dataframe["Rf"].to_numpy(dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("The training Rf mean/std are not valid.")

    standardization_dir.mkdir(parents=True, exist_ok=True)
    np.save(standardization_dir / "mean.npy", np.array([mean]))
    np.save(standardization_dir / "std.npy", np.array([std]))
    return mean, std


def remove_graph_cache(data_dir: Path) -> None:
    processed_dir = data_dir / "processed"
    for split in SPLIT_FILES:
        for graph_name in GRAPH_COLUMNS:
            cache_path = processed_dir / f"{graph_name}_{split}.pt"
            cache_path.unlink(missing_ok=True)


def build_graph_datasets(
    data_dir: Path,
    standardization_dir: Path,
    *,
    reuse_graphs: bool,
) -> dict[str, TripleGraphDataset]:
    if not reuse_graphs:
        remove_graph_cache(data_dir)

    datasets: dict[str, TripleGraphDataset] = {}
    for split, filename in SPLIT_FILES.items():
        graph_datasets = {}
        for graph_name, smiles_column in GRAPH_COLUMNS.items():
            graph_datasets[graph_name] = MolToGraph(
                smiles=smiles_column,
                root=str(data_dir),
                raw_name=filename,
                preocessed_name=f"{graph_name}_{split}.pt",
                molecular_param=False,
                augment=False,
                standardization_dir=str(standardization_dir),
            )
        datasets[split] = TripleGraphDataset(
            graph_datasets["compound"],
            graph_datasets["elu1"],
            graph_datasets["elu2"],
        )
    return datasets


def build_loaders(
    datasets: dict[str, TripleGraphDataset],
    *,
    batch_size: int,
    num_workers: int,
    use_cuda: bool,
) -> dict[str, DataLoader]:
    train_size = len(datasets["train"])
    if train_size < 2:
        raise ValueError("The training split must contain at least two rows.")
    if train_size % batch_size == 1:
        raise ValueError(
            f"--batch-size {batch_size} leaves a final training batch of one "
            "sample, which is incompatible with BatchNorm. Choose a different "
            "batch size."
        )

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    return {
        "train": DataLoader(datasets["train"], shuffle=True, **common),
        "valid": DataLoader(datasets["valid"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def create_model(model_name: str, dropout: float) -> torch.nn.Module:
    return MODEL_REGISTRY[model_name](dropout=dropout)


def masked_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    mask = torch.isfinite(targets)
    if not torch.any(mask):
        raise ValueError("A batch contains no finite target values.")
    return torch.mean(torch.abs(predictions[mask] - targets[mask]))


def regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    finite = np.isfinite(predictions) & np.isfinite(targets)
    predictions = predictions[finite]
    targets = targets[finite]
    if targets.size == 0:
        raise ValueError("No finite values are available for evaluation.")

    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    denominator = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = (
        float(1 - np.sum(errors**2) / denominator)
        if denominator > 0
        else float("nan")
    )
    return {"mae": mae, "rmse": rmse, "r2": r2}


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    prediction_batches = []
    target_batches = []
    sample_id_batches = []

    with torch.no_grad():
        for compound, eluent1, eluent2 in loader:
            compound = compound.to(device)
            eluent1 = eluent1.to(device)
            eluent2 = eluent2.to(device)
            predictions = model(compound, eluent1, eluent2)
            prediction_batches.append(predictions.detach().view(-1).cpu())
            target_batches.append(compound.y.detach().view(-1).cpu())
            sample_id_batches.append(compound.sample_id.detach().view(-1).cpu())

    standardized_predictions = torch.cat(prediction_batches).numpy()
    standardized_targets = torch.cat(target_batches).numpy()
    sample_ids = torch.cat(sample_id_batches).numpy()
    predictions = standardized_predictions * std + mean
    targets = standardized_targets * std + mean

    prediction_frame = pd.DataFrame(
        {
            "sample_id": sample_ids.astype(np.int64),
            "target_Rf": targets,
            "predicted_Rf": predictions,
        }
    )
    return regression_metrics(predictions, targets), prediction_frame


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(destination)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_history(
    history: list[dict[str, float | int]],
    output_dir: Path,
) -> None:
    destination = output_dir / "history.csv"
    temporary_path = output_dir / "history.csv.tmp"
    pd.DataFrame(history).to_csv(temporary_path, index=False)
    temporary_path.replace(destination)


def train_model(
    *,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    device: torch.device,
    mean: float,
    std: float,
    epochs: int,
    patience: int,
    output_dir: Path,
    resume: bool,
    logger: logging.Logger,
    configuration: dict[str, Any],
) -> Path:
    last_path = output_dir / "last_checkpoint.pt"
    best_path = output_dir / "best_checkpoint.pt"
    start_epoch = 0
    best_valid_mae = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    if resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {last_path}")
        checkpoint = load_checkpoint(last_path, device)
        checkpoint_model = checkpoint.get("configuration", {}).get("model")
        if checkpoint_model and checkpoint_model != configuration["model"]:
            raise ValueError(
                f"Checkpoint model '{checkpoint_model}' does not match "
                f"requested model '{configuration['model']}'."
            )
        checkpoint_configuration = checkpoint.get("configuration", {})
        immutable_options = (
            "dropout",
            "learning_rate",
            "weight_decay",
            "scheduler_step",
            "scheduler_gamma",
        )
        changed_options = [
            option
            for option in immutable_options
            if option in checkpoint_configuration
            and checkpoint_configuration[option] != configuration[option]
        ]
        if changed_options:
            raise ValueError(
                "Resume options differ from the checkpoint: "
                f"{changed_options}. Use the original values."
            )
        checkpoint_mean = float(checkpoint.get("mean", mean))
        checkpoint_std = float(checkpoint.get("std", std))
        if not np.isclose(checkpoint_mean, mean) or not np.isclose(
            checkpoint_std,
            std,
        ):
            raise ValueError(
                "The current training-split mean/std do not match the resume "
                "checkpoint. Restore the original split files before resuming."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_valid_mae = float(checkpoint["best_valid_mae"])
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        history = checkpoint.get("history", [])
        restore_rng_state(checkpoint.get("rng_state"))
        logger.info("Resuming from epoch %d.", start_epoch + 1)

    if start_epoch >= epochs:
        raise ValueError(
            f"The checkpoint already reached epoch {start_epoch}, which is "
            f"not below --epochs {epochs}."
        )

    for epoch in range(start_epoch, epochs):
        model.train()
        train_predictions = []
        train_targets = []

        for compound, eluent1, eluent2 in loaders["train"]:
            compound = compound.to(device)
            eluent1 = eluent1.to(device)
            eluent2 = eluent2.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(compound, eluent1, eluent2)
            loss = masked_mae(predictions, compound.y)
            loss.backward()
            optimizer.step()

            train_predictions.append(predictions.detach().view(-1).cpu())
            train_targets.append(compound.y.detach().view(-1).cpu())

        scheduler.step()

        train_prediction_array = torch.cat(train_predictions).numpy() * std + mean
        train_target_array = torch.cat(train_targets).numpy() * std + mean
        train_metrics = regression_metrics(
            train_prediction_array,
            train_target_array,
        )
        valid_metrics, _ = collect_predictions(
            model,
            loaders["valid"],
            device,
            mean,
            std,
        )

        epoch_record: dict[str, float | int] = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "valid_mae": valid_metrics["mae"],
            "valid_rmse": valid_metrics["rmse"],
            "valid_r2": valid_metrics["r2"],
        }
        history.append(epoch_record)

        improved = valid_metrics["mae"] < best_valid_mae
        if improved:
            best_valid_mae = valid_metrics["mae"]
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "valid_metrics": valid_metrics,
                    "configuration": configuration,
                    "mean": mean,
                    "std": std,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        atomic_torch_save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_valid_mae": best_valid_mae,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "configuration": configuration,
                "mean": mean,
                "std": std,
                "rng_state": capture_rng_state(),
            },
            last_path,
        )
        save_history(history, output_dir)

        logger.info(
            "Epoch %d/%d | train MAE %.5f R2 %.5f | "
            "valid MAE %.5f R2 %.5f | lr %.6g%s",
            epoch + 1,
            epochs,
            train_metrics["mae"],
            train_metrics["r2"],
            valid_metrics["mae"],
            valid_metrics["r2"],
            optimizer.param_groups[0]["lr"],
            " | best" if improved else "",
        )

        if patience > 0 and epochs_without_improvement >= patience:
            logger.info(
                "Early stopping after %d epochs without validation improvement.",
                epochs_without_improvement,
            )
            break

    return best_path


def evaluate_best_model(
    *,
    model: torch.nn.Module,
    best_checkpoint_path: Path,
    loaders: dict[str, DataLoader],
    device: torch.device,
    mean: float,
    std: float,
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, dict[str, float]]:
    checkpoint = load_checkpoint(best_checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics: dict[str, dict[str, float]] = {}

    for split in ("train", "valid", "test"):
        split_metrics, predictions = collect_predictions(
            model,
            loaders[split],
            device,
            mean,
            std,
        )
        metrics[split] = split_metrics
        predictions.to_csv(
            output_dir / f"{split}_predictions.csv",
            index=False,
        )
        logger.info(
            "%s | MAE %.5f | RMSE %.5f | R2 %.5f",
            split,
            split_metrics["mae"],
            split_metrics["rmse"],
            split_metrics["r2"],
        )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = configure_output_dir(args)
    logger = configure_logging(output_dir)
    set_seed(args.seed)

    data_dir = args.data_dir.resolve()
    standardization_dir = args.standardization_dir.resolve()
    device = resolve_device(args.device)

    configuration = {
        "model": args.model,
        "data_dir": str(data_dir),
        "standardization_dir": str(standardization_dir),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "scheduler_step": args.scheduler_step,
        "scheduler_gamma": args.scheduler_gamma,
        "patience": args.patience,
        "device": str(device),
        "seed": args.seed,
        "num_workers": args.num_workers,
        "resume": args.resume,
        "reuse_graphs": args.reuse_graphs,
    }
    with (output_dir / "configuration.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(configuration, file, indent=2)

    logger.info("Loading pre-split CSV files from %s.", data_dir)
    split_frames = load_split_csvs(data_dir)
    for split, dataframe in split_frames.items():
        logger.info("%s rows: %d", split, len(dataframe))

    mean, std = fit_standardization(
        split_frames["train"],
        standardization_dir,
    )
    logger.info(
        "Training-only Rf standardization: mean=%.8f std=%.8f",
        mean,
        std,
    )

    if args.reuse_graphs:
        logger.warning(
            "Reusing graph caches. Ensure they were built from the same CSVs "
            "and standardization values."
        )
    else:
        logger.info("Rebuilding compound and eluent graph caches.")
    graph_datasets = build_graph_datasets(
        data_dir,
        standardization_dir,
        reuse_graphs=args.reuse_graphs,
    )
    loaders = build_loaders(
        graph_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cuda=device.type == "cuda",
    )

    model = create_model(args.model, args.dropout).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(
        optimizer,
        step_size=args.scheduler_step,
        gamma=args.scheduler_gamma,
    )
    logger.info("Training %s on %s.", args.model, device)
    logger.info("Run output: %s", output_dir)

    best_checkpoint_path = train_model(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mean=mean,
        std=std,
        epochs=args.epochs,
        patience=args.patience,
        output_dir=output_dir,
        resume=args.resume,
        logger=logger,
        configuration=configuration,
    )

    logger.info("Evaluating best validation checkpoint.")
    evaluate_best_model(
        model=model,
        best_checkpoint_path=best_checkpoint_path,
        loaders=loaders,
        device=device,
        mean=mean,
        std=std,
        output_dir=output_dir,
        logger=logger,
    )
    logger.info("Training and evaluation completed.")


if __name__ == "__main__":
    main()

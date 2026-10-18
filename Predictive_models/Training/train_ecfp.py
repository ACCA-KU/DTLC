"""Only dummy data for demonstrating training is included, so the training results are not reproducible."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset


TRAINING_DIR = Path(__file__).resolve().parent
FINGERPRINT_RADIUS = 2
FINGERPRINT_SIZE = 2048
DESCRIPTOR_SIZE = 6
MOLECULE_FEATURE_SIZE = FINGERPRINT_SIZE + DESCRIPTOR_SIZE
INTERACTION_SIZE = 2 * MOLECULE_FEATURE_SIZE

MODEL_NAMES = ("ecfp-bli", "ecfp-cli")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an ECFP+descriptor MLP from the existing United_DB "
            "train/valid/test CSV files."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_NAMES,
        help="ECFP model and eluent-interaction strategy.",
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
        help="Directory in which target mean.npy and std.npy are written.",
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
    logger = logging.getLogger("dtlc_ecfp_train")
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
        if dataframe["Smiles"].isna().any():
            raise ValueError(f"{path.name} contains missing compound SMILES.")
        missing_eluent1_with_ratio = (
            dataframe["elu1_smiles"].isna() & (dataframe["ratio1"] != 0)
        )
        missing_eluent2_with_ratio = (
            dataframe["elu2_smiles"].isna() & (dataframe["ratio2"] != 0)
        )
        if missing_eluent1_with_ratio.any() or missing_eluent2_with_ratio.any():
            raise ValueError(
                f"{path.name} assigns a nonzero ratio to a missing eluent."
            )
        if dataframe["sample_id"].duplicated().any():
            raise ValueError(f"{path.name} contains duplicate sample_id values.")
        splits[split] = dataframe.reset_index(drop=True)

    return splits


def fit_target_standardization(
    train_dataframe: pd.DataFrame,
    standardization_dir: Path,
    *,
    save: bool,
) -> tuple[float, float]:
    """Fit Rf target statistics using the training split only."""
    values = train_dataframe["Rf"].to_numpy(dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("The training Rf mean/std are not valid.")

    if save:
        standardization_dir.mkdir(parents=True, exist_ok=True)
        np.save(standardization_dir / "mean.npy", np.array([mean]))
        np.save(standardization_dir / "std.npy", np.array([std]))
    return mean, std


def hash_split_files(data_dir: Path) -> dict[str, str]:
    """Hash each split file so resumed runs cannot silently change data."""
    hashes = {}
    for split, filename in SPLIT_FILES.items():
        digest = hashlib.sha256()
        with (data_dir / filename).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[split] = digest.hexdigest()
    return hashes


def molecule_to_ecfp_descriptor(
    smiles: str | float,
    *,
    generator: Any,
    allow_missing: bool,
    column_name: str,
    row_index: int,
) -> np.ndarray:
    """Convert one molecule to a count ECFP plus six RDKit descriptors."""
    if pd.isna(smiles):
        if allow_missing:
            return np.zeros(MOLECULE_FEATURE_SIZE, dtype=np.float32)
        raise ValueError(
            f"Missing SMILES in column '{column_name}' at row {row_index}."
        )

    smiles_text = str(smiles)
    molecule = Chem.MolFromSmiles(smiles_text)
    if molecule is None:
        raise ValueError(
            f"Invalid SMILES in column '{column_name}' at row {row_index}: "
            f"{smiles_text}"
        )

    fingerprint = generator.GetCountFingerprint(molecule)
    fingerprint_array = np.zeros(FINGERPRINT_SIZE, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    descriptors = np.array(
        [
            Descriptors.ExactMolWt(molecule),
            Descriptors.NumRotatableBonds(molecule),
            Descriptors.NumHDonors(molecule),
            Descriptors.NumHAcceptors(molecule),
            Descriptors.MolLogP(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
        ],
        dtype=np.float32,
    )
    return np.concatenate([fingerprint_array, descriptors])


def make_molecule_representations(
    dataframe: pd.DataFrame,
    column_name: str,
    *,
    allow_missing: bool,
) -> np.ndarray:
    """Create ECFP+descriptor vectors, caching repeated SMILES."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS,
        fpSize=FINGERPRINT_SIZE,
    )
    cache: dict[str, np.ndarray] = {}
    representations = []

    for row_index, smiles in enumerate(dataframe[column_name]):
        if pd.isna(smiles):
            representation = molecule_to_ecfp_descriptor(
                smiles,
                generator=generator,
                allow_missing=allow_missing,
                column_name=column_name,
                row_index=row_index,
            )
        else:
            smiles_text = str(smiles)
            if smiles_text not in cache:
                cache[smiles_text] = molecule_to_ecfp_descriptor(
                    smiles_text,
                    generator=generator,
                    allow_missing=allow_missing,
                    column_name=column_name,
                    row_index=row_index,
                )
            representation = cache[smiles_text]
        representations.append(representation)

    return np.vstack(representations).astype(np.float32, copy=False)


def make_raw_model_features(
    dataframe: pd.DataFrame,
    model_name: str,
) -> tuple[np.ndarray, ...]:
    """Build BLI or CLI interaction vectors."""
    compound = make_molecule_representations(
        dataframe,
        "Smiles",
        allow_missing=False,
    )
    eluent1 = make_molecule_representations(
        dataframe,
        "elu1_smiles",
        allow_missing=True,
    )
    eluent2 = make_molecule_representations(
        dataframe,
        "elu2_smiles",
        allow_missing=True,
    )

    if model_name == "ecfp-bli":
        ratio1 = dataframe["ratio1"].to_numpy(dtype=np.float32).reshape(-1, 1)
        ratio2 = dataframe["ratio2"].to_numpy(dtype=np.float32).reshape(-1, 1)
        mixed_eluent = eluent1 * ratio1 + eluent2 * ratio2
        return (np.hstack([compound, mixed_eluent]),)

    interaction1 = np.hstack([compound, eluent1])
    interaction2 = np.hstack([compound, eluent2])
    if model_name == "ecfp-cli":
        return interaction1, interaction2
    raise ValueError(f"Unsupported model: {model_name}")


class FeatureMinMaxScaler:
    """Small serializable equivalent of sklearn's default MinMaxScaler."""

    def __init__(self) -> None:
        self.data_min: np.ndarray | None = None
        self.data_range: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "FeatureMinMaxScaler":
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("Feature scaler requires a non-empty 2D array.")
        if not np.isfinite(values).all():
            raise ValueError("Feature matrix contains non-finite values.")
        self.data_min = np.min(values, axis=0)
        data_max = np.max(values, axis=0)
        self.data_range = data_max - self.data_min
        self.data_range[self.data_range == 0] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.data_min is None or self.data_range is None:
            raise RuntimeError("Feature scaler has not been fitted.")
        values = np.asarray(values, dtype=np.float32)
        values -= self.data_min
        values /= self.data_range
        return values

def scale_feature_splits(
    raw_features: dict[str, tuple[np.ndarray, ...]],
    output_dir: Path,
    *,
    save_scaler: bool,
) -> tuple[dict[str, tuple[np.ndarray, ...]], str]:
    """Scale each interaction branch exactly as concatenated MinMax scaling."""
    branch_widths = [branch.shape[1] for branch in raw_features["train"]]
    scalers = [
        FeatureMinMaxScaler().fit(branch)
        for branch in raw_features["train"]
    ]
    all_minimums = np.concatenate(
        [scaler.data_min for scaler in scalers if scaler.data_min is not None]
    )
    all_ranges = np.concatenate(
        [
            scaler.data_range
            for scaler in scalers
            if scaler.data_range is not None
        ]
    )
    if save_scaler:
        np.savez(
            output_dir / "feature_scaler.npz",
            data_min=all_minimums,
            data_range=all_ranges,
            branch_widths=np.asarray(branch_widths, dtype=np.int64),
        )

    scaled_features: dict[str, tuple[np.ndarray, ...]] = {}
    for split, branches in raw_features.items():
        scaled_features[split] = tuple(
            scaler.transform(branch)
            for scaler, branch in zip(scalers, branches)
        )

    digest = hashlib.sha256()
    digest.update(all_minimums.tobytes())
    digest.update(all_ranges.tobytes())
    digest.update(np.asarray(branch_widths, dtype=np.int64).tobytes())
    return scaled_features, digest.hexdigest()


def make_interaction_layers(input_size: int, dropout: float) -> nn.ModuleList:
    return nn.ModuleList(
        [
            nn.Linear(input_size, 2048),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(2048),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
        ]
    )


def make_prediction_layers() -> nn.ModuleList:
    return nn.ModuleList(
        [
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
        ]
    )


def apply_layers(values: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
    for layer in layers:
        values = layer(values)
    return values


def initialize_linear_layers(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class BLI_ECFP_MLP(nn.Module):
    """Predict Rf from a compound and a pre-mixed eluent representation."""

    def __init__(self, input_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.interaction_layers = make_interaction_layers(input_size, dropout)
        self.prediction_layers = make_prediction_layers()
        initialize_linear_layers(self)

    def forward(self, compound_mixed_eluent: torch.Tensor) -> torch.Tensor:
        interaction = apply_layers(
            compound_mixed_eluent,
            self.interaction_layers,
        )
        return apply_layers(interaction, self.prediction_layers)


class CLI_ECFP_MLP(nn.Module):
    """Combine two compound-eluent interactions using eluent ratios."""

    def __init__(self, input_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.interaction_layers = make_interaction_layers(input_size, dropout)
        self.prediction_layers = make_prediction_layers()
        initialize_linear_layers(self)

    def forward(
        self,
        interaction1: torch.Tensor,
        interaction2: torch.Tensor,
        ratio1: torch.Tensor,
        ratio2: torch.Tensor,
    ) -> torch.Tensor:
        vector1 = apply_layers(interaction1, self.interaction_layers)
        vector2 = apply_layers(interaction2, self.interaction_layers)
        united = (
            vector1 * ratio1.view(-1, 1).float()
            + vector2 * ratio2.view(-1, 1).float()
        )
        return apply_layers(united, self.prediction_layers)


def create_model(model_name: str, dropout: float) -> nn.Module:
    if model_name == "ecfp-bli":
        return BLI_ECFP_MLP(INTERACTION_SIZE, dropout)
    if model_name == "ecfp-cli":
        return CLI_ECFP_MLP(INTERACTION_SIZE, dropout)
    raise ValueError(f"Unsupported model: {model_name}")


class ECFPDataset(Dataset):
    def __init__(
        self,
        features: Sequence[np.ndarray],
        dataframe: pd.DataFrame,
        mean: float,
        std: float,
    ) -> None:
        self.features = tuple(
            torch.from_numpy(
                np.ascontiguousarray(feature, dtype=np.float32)
            )
            for feature in features
        )
        self.labels = torch.tensor(
            (
                (dataframe["Rf"].to_numpy(dtype=np.float32) - mean)
                / std
            ).reshape(-1, 1),
            dtype=torch.float32,
        )
        self.ratio1 = torch.tensor(
            dataframe["ratio1"].to_numpy(dtype=np.float32).reshape(-1, 1)
        )
        self.ratio2 = torch.tensor(
            dataframe["ratio2"].to_numpy(dtype=np.float32).reshape(-1, 1)
        )
        self.sample_ids = torch.tensor(
            dataframe["sample_id"].to_numpy(dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "features": tuple(feature[index] for feature in self.features),
            "label": self.labels[index],
            "ratio1": self.ratio1[index],
            "ratio2": self.ratio2[index],
            "sample_id": self.sample_ids[index],
        }


def build_loaders(
    scaled_features: dict[str, tuple[np.ndarray, ...]],
    split_frames: dict[str, pd.DataFrame],
    *,
    mean: float,
    std: float,
    batch_size: int,
    num_workers: int,
    use_cuda: bool,
) -> dict[str, DataLoader]:
    datasets = {
        split: ECFPDataset(
            scaled_features[split],
            split_frames[split],
            mean,
            std,
        )
        for split in SPLIT_FILES
    }
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


def forward_batch(
    model: nn.Module,
    model_name: str,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = [feature.to(device) for feature in batch["features"]]
    labels = batch["label"].to(device)
    ratio1 = batch["ratio1"].to(device)
    ratio2 = batch["ratio2"].to(device)

    if model_name == "ecfp-bli":
        predictions = model(features[0])
    elif model_name == "ecfp-cli":
        predictions = model(features[0], features[1], ratio1, ratio2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return predictions, labels, batch["sample_id"]


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
    model: nn.Module,
    model_name: str,
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
        for batch in loader:
            predictions, targets, sample_ids = forward_batch(
                model,
                model_name,
                batch,
                device,
            )
            prediction_batches.append(predictions.detach().view(-1).cpu())
            target_batches.append(targets.detach().view(-1).cpu())
            sample_id_batches.append(sample_ids.detach().view(-1).cpu())

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
    model: nn.Module,
    model_name: str,
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
        checkpoint_configuration = checkpoint.get("configuration", {})
        immutable_options = (
            "model",
            "batch_size",
            "seed",
            "dropout",
            "learning_rate",
            "weight_decay",
            "scheduler_step",
            "scheduler_gamma",
            "feature_scaler_hash",
            "data_hashes",
        )
        changed_options = [
            option
            for option in immutable_options
            if option in checkpoint_configuration
            and checkpoint_configuration[option] != configuration[option]
        ]
        if changed_options:
            raise ValueError(
                "Resume options or features differ from the checkpoint: "
                f"{changed_options}. Use the original data and settings."
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

        for batch in loaders["train"]:
            predictions, targets, _ = forward_batch(
                model,
                model_name,
                batch,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mae(predictions, targets)
            loss.backward()
            optimizer.step()

            train_predictions.append(predictions.detach().view(-1).cpu())
            train_targets.append(targets.detach().view(-1).cpu())

        scheduler.step()

        train_prediction_array = torch.cat(train_predictions).numpy() * std + mean
        train_target_array = torch.cat(train_targets).numpy() * std + mean
        train_metrics = regression_metrics(
            train_prediction_array,
            train_target_array,
        )
        valid_metrics, _ = collect_predictions(
            model,
            model_name,
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
    model: nn.Module,
    model_name: str,
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
            model_name,
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

    logger.info("Loading pre-split CSV files from %s.", data_dir)
    split_frames = load_split_csvs(data_dir)
    for split, dataframe in split_frames.items():
        logger.info("%s rows: %d", split, len(dataframe))

    mean, std = fit_target_standardization(
        split_frames["train"],
        standardization_dir,
        save=not args.resume,
    )
    if not args.resume:
        np.save(output_dir / "mean.npy", np.array([mean]))
        np.save(output_dir / "std.npy", np.array([std]))
    logger.info(
        "Training-only Rf standardization: mean=%.8f std=%.8f",
        mean,
        std,
    )

    logger.info(
        "Generating radius-%d, %d-count ECFP plus %d descriptors.",
        FINGERPRINT_RADIUS,
        FINGERPRINT_SIZE,
        DESCRIPTOR_SIZE,
    )
    raw_features = {
        split: make_raw_model_features(dataframe, args.model)
        for split, dataframe in split_frames.items()
    }
    if args.resume and not (output_dir / "feature_scaler.npz").is_file():
        raise FileNotFoundError(
            f"Resume scaler not found: {output_dir / 'feature_scaler.npz'}"
        )
    scaled_features, feature_scaler_hash = scale_feature_splits(
        raw_features,
        output_dir,
        save_scaler=not args.resume,
    )
    loaders = build_loaders(
        scaled_features,
        split_frames,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cuda=device.type == "cuda",
    )
    del raw_features, scaled_features

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
        "fingerprint_radius": FINGERPRINT_RADIUS,
        "fingerprint_size": FINGERPRINT_SIZE,
        "descriptor_size": DESCRIPTOR_SIZE,
        "feature_scaler_hash": feature_scaler_hash,
        "data_hashes": hash_split_files(data_dir),
    }
    if not args.resume:
        with (output_dir / "configuration.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(configuration, file, indent=2)

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
        model_name=args.model,
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
        model_name=args.model,
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

# Predictive Model Training

The command-line training scripts use the already cleaned and pre-split TLC
datasets. They do not perform data filtering, dataset splitting, exploratory
analysis, or plotting.

## Requirements

- Python 3.10 or later
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy and pandas

## Required data

Place these files in `DB/Rf/`:

- `United_DB_train.csv`
- `United_DB_valid.csv`
- `United_DB_test.csv`

Each file must contain:

`Smiles`, `elu1_smiles`, `elu2_smiles`, `ratio1`, `ratio2`, `Rf`, and
`sample_id`.

Target normalization statistics are calculated from `United_DB_train.csv`
only and saved as `standardization/mean.npy` and
`standardization/std.npy`.

## Run GNN training

Run the command from the `Predictive_models/Training` directory:

```bash
python train_gnn.py --model gcn-bli
```

The default maximum is 10,000 epochs with early stopping after 300 epochs
without validation-MAE improvement.

Available model names:

- `gcn-cli`
- `gcn-bli`
- `gnn-cli` (alias of `gcn-cli`, matching the trained-model folder name)
- `gnn-bli` (alias of `gcn-bli`)
- `gat-cli`
- `gat-bli`
- `dmpnn-cli`
- `dmpnn-bli`

Example with common options:

```bash
python train_gnn.py --model gat-bli --epochs 1000 --batch-size 500 --device cuda:0
```

Use `--device cpu` for CPU training or leave the default `--device auto` to
select CUDA when it is available.

See all options:

```bash
python train_gnn.py --help
```

## Run ECFP training

`train_ecfp.py` contains the complete ECFP conversion, RDKit descriptor,
feature-scaling, neural-network definition, training, and evaluation workflow.

Choose one of the following models:

- `ecfp-bli`
- `ecfp-cli`

Example:

```bash
python train_ecfp.py --model ecfp-cli
```

Example with common options:

```bash
python train_ecfp.py --model ecfp-bli --epochs 1000 --batch-size 500 --device cuda:0
```

See all ECFP options:

```bash
python train_ecfp.py --help
```

The ECFP workflow uses radius-2, 2,048-dimensional count fingerprints and six
RDKit molecular descriptors. Feature scaling and Rf standardization are fitted
using the training split only.

## Outputs

Both scripts create a timestamped directory under `runs/` containing:

- `training.log`: terminal-style epoch log
- `history.csv`: train and validation metrics for every epoch
- `last_checkpoint.pt`: checkpoint used to resume training
- `best_checkpoint.pt`: model with the lowest validation MAE
- `metrics.json`: final train, validation, and test metrics
- `*_predictions.csv`: target and predicted Rf values for each split
- `configuration.json`: command configuration used for the run

ECFP runs also save `feature_scaler.npz` and run-local `mean.npy`/`std.npy`
files.

## Resume a GNN run

Provide the existing run directory and increase `--epochs` if needed:

```bash
python train_gnn.py --model gcn-cli --epochs 2000 --resume --output-dir runs/gcn-cli_YYYYMMDD_HHMMSS
```

Graph caches are rebuilt by default so that labels use the current
training-derived mean and standard deviation. Use `--reuse-graphs` only when
the split CSVs and standardization values have not changed.

Resume an ECFP run in the same way:

```bash
python train_ecfp.py --model ecfp-cli --epochs 2000 --resume --output-dir runs/ecfp-cli_YYYYMMDD_HHMMSS
```

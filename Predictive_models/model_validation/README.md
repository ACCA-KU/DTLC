# Model Validation: Rf Prediction (`rf_prediction.ipynb`)

This notebook uses the trained **BLI-GAT** model (`Trained_models/GAT_BLI`) to
predict TLC retention factors (Rf) for compound / binary-eluent combinations,
and compares the predictions with experimental values.

It reproduces **Figure 5**, **Figure 8**, and **Figure S8** of the paper. To
predict Rf for your own molecules, edit the SMILES, eluent SMILES, or ratios
in the input CSV files.

## Requirements

- Python 3.10 or later
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy, pandas, Matplotlib
- Jupyter

## Directory layout

The notebook resolves all paths relative to its own location, so it **must be
run from `Predictive_models/model_validation/`**.

```
Predictive_models/
├── Model_architecture/
│   └── GAT_model_architecture.py      # GAT_BLI model class
├── Trained_models/
│   └── GAT_BLI/opt_checkpoint.pt      # trained weights used for prediction
├── Training/
│   ├── make_graph_dataloader.py       # MolToGraph, TripleGraphDataset
│   ├── standardization/mean.npy, std.npy   # Rf target (de)normalization
│   └── DB/
│       ├── data_process/eluent_standard.csv  # eluent densities
│       └── Rf/                         # intermediate files and outputs
└── model_validation/
    ├── rf_prediction.ipynb
    ├── pred_list1.csv                 # input for Figure 8 / S8
    └── pred_list2.csv                 # input for Figure 5
```

## Input file format

Each input CSV must contain the following columns:

| Column        | Description                                           |
|---------------|-------------------------------------------------------|
| `Smiles`      | SMILES of the compound                                |
| `elu1_smiles` | SMILES of eluent 1 (e.g. `CCCCCC`, hexane)            |
| `elu2_smiles` | SMILES of eluent 2 (e.g. `CCOC(C)=O`, ethyl acetate)  |
| `ratio1`      | **Volume** fraction of eluent 1                       |
| `ratio2`      | **Volume** fraction of eluent 2                       |
| `Rf`          | *(optional)* experimental Rf, used for plotting       |

Notes:

- Ratios are given as volume ratios (v/v). The notebook converts them to
  mole fractions internally using the densities in
  `Training/DB/data_process/eluent_standard.csv` before passing them to the
  model, because the model was trained on mole-fraction ratios.
- Every eluent SMILES must appear in the `hex_Smiles` column of
  `eluent_standard.csv`; otherwise a `ValueError` lists the missing eluents.
- If `Rf` is omitted, prediction still works, but the experimental curve in
  the plots will be empty.

Provided input files:

| File             | Rows | Content                                                         | Figure        |
|------------------|------|-----------------------------------------------------------------|---------------|
| `pred_list1.csv` | 176  | 16 compounds in Hex/EA, EA volume fraction 0 → 1 in 0.1 steps    | Figure 8, S8  |
| `pred_list2.csv` | 38   | 8 compound/eluent-pair series at integer volume ratios (e.g. 5:1, 3:1, 1:1, 1:2) | Figure 5 |

## Notebook workflow

1. **Setup** – sets up paths, imports `GAT_BLI`, and adds `Training/` to
   `sys.path`.
2. **`vol_2_mol_ratio`** – converts v/v ratios to normalized mole fractions.
3. **`make_graph_predict`** – loads `opt_checkpoint.pt`, builds molecular
   graphs for the compound and both eluents (`MolToGraph`), runs inference,
   and reverses the target standardization with `mean.npy` / `std.npy`.
4. **`predict_vol_ratio_csv(input_csv, output_csv)`** – end-to-end
   prediction. Predictions are clipped to [0, 1] and added as the
   `BLI_GAT_Rf` column.
5. **Plotting**
   - `plot_functional(result)` – one panel per compound, x-axis = EA volume
     fraction (Figure 8 / S8).
   - `plot_solvent(result)` – one panel per (compound, eluent 1, eluent 2)
     series, x-axis = integer volume ratio such as `3:1` (Figure 5).

Figure 5 panel mapping (`plot_solvent` panel numbers):

- Figure 5(a): 1, 2, 3, 4
- Figure 5(b): 5, 6
- Figure 5(c): 7, 8

## Usage

```bash
cd Predictive_models/model_validation
jupyter notebook rf_prediction.ipynb
```

Run all cells in order. To predict with your own data, create a CSV in the
format above and call:

```python
input_csv = os.path.join(DTLC_ROOT, 'Predictive_models/model_validation', 'my_list.csv')
result = predict_vol_ratio_csv(input_csv, output_csv='./DB/Rf/my_list_result.csv')
plot_functional(result)   # or plot_solvent(result)
```

## Output files

Prediction runs inside `Predictive_models/Training/`, so **relative paths
(including `output_csv`) are resolved from `Training/`**, not from
`model_validation/`.

| File                                                       | Description                                      |
|------------------------------------------------------------|--------------------------------------------------|
| `output_csv` (default `Training/DB/Rf/<name>_molfrac_predicted.csv`) | Original input + `BLI_GAT_Rf` prediction column |
| `Training/DB/Rf/<name>_molfrac_input.csv`                  | Model input with ratios converted to mole fractions |
| `Training/DB/Rf/processed/experimental/<name>_molfrac_input_{compound,elu1,elu2}.pt` | Cached graph datasets (regenerated on each run) |

In the notebook as provided:

- `pred_list1.csv` → `Training/pred_result.csv`
- `pred_list2.csv` → `Training/DB/Rf/pred_result2.csv`

## Notes

- A GPU is used automatically if available; otherwise the model runs on CPU.
- The output CSV keeps the original **volume** ratios; only the intermediate
  `*_molfrac_input.csv` contains mole fractions.

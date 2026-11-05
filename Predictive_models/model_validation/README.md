# Model Validation

This notebook `rf_prediction.ipynb` uses the trained **BLI-GAT** model (`Trained_models/GAT_BLI`) to
predict TLC retention factors (Rf) for compound / binary-eluent combinations,
and compares the predictions with experimental values.

It reproduces **Figure 5**, **Figure 8**, and **Figure S8** of the paper. To
predict Rf for your own molecules, edit the SMILES, eluent SMILES, or ratios
in the input CSV files.

## Input file format
| File             | Content                                                         | Figure        |
|------------------|-----------------------------------------------------------------|---------------|
| `pred_list1.csv` | 16 compounds in Hex/EA, EA volume fraction 0 → 1 in 0.1 steps    | Figure 8, S8  |
| `pred_list2.csv` | 8 compounds in Hex/EA or DCM/MeOH series at specific volume ratios | Figure 5 |

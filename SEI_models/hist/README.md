# Figure 7: Functional-group contribution histograms

This folder contains the data and notebook needed to reproduce **Figure 7** of the paper.

Each panel is a histogram of the contribution scores that the segment model assigns to one functional group (fragment). It compares three hexane : ethyl acetate eluent ratios: **10:1**, **4:1** and **1:1**.

---

## Files

| File | Description |
|---|---|
| `histogram.ipynb` | Notebook that draws graphs of Figure 7 |

### Panel ↔ data mapping

| Panel | Title | Fragment SMILES | Data file |
|---|---|---|---|
| a1 | Hydroxyl | `*CO` | `a1_hist_data_CO.csv` |
| a2 | Amino | `*N` | `a2_hist_data_N.csv` |
| a3 | Nitrile | `*C#N` | `a3_hist_data_C#N.csv` |
| a4 | Carboxyl | `*C(=O)O` | `a4_hist_data_C(=O)O.csv` |
| a5 | Aldehyde | `*C=O` | `a5_hist_data_C=O.csv` |
| a6 | Nitro | `*[N+](=O)[O-]` | `a6_hist_data_[N+](=O)[O-].csv` |
| b1 | Benzene | `c1ccccc1` | `b1_hist_data_c1ccccc1.csv` |
| b2 | Methyl | `*C` | `b2_hist_data_C.csv` |
| b3 | Butyl | `*CCCC` | `b3_hist_data_CCCC.csv` |
| b4 | Cyclohexane | `C1CCCCC1` | `b4_hist_data_C1CCCCC1.csv` |
| b5 | Ethyl | `*CC` | `b5_hist_data_CC.csv` |
| b6 | Isopropyl | `*C(C)C` | `b6_hist_data_C(C)C.csv` |

### CSV format

Each CSV has **two header rows**, followed by one contribution value per row:

```
*CO,*CO,*CO                  <- row 1: fragment SMILES
10:1,4:1,1:1                 <- row 2: hexane:EtOAc ratio (used as column names)
-0.8039,-0.7446,-0.4728      <- contribution values
...
```

The three columns can have different lengths. Empty cells are dropped before plotting.

---

## How to reproduce

1. Open `histogram.ipynb` in Jupyter and select a kernel that has `numpy`, `pandas` and `matplotlib` installed.

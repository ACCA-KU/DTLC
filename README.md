## 중요 오류! 수정 必
### !!score 분포 재현 오류!!
SEI_models/contribution_analysis.ipynb 부분 
GROUPS = [
    ('*O', 'Hydroxyl'), ('*N', 'Amino'), ('*C#N', 'Nitrile'),
    ('*C(=O)O', 'Carboxyl'), ('*C=O', 'Aldehyde'), ('*[N+](=O)[O-]', 'Nitro'),
    ('c1ccccc1', 'Benzene'), ('*C', 'Methyl'), ('*CCC', 'n-Propyl'),
    ('C1CCCCC1', 'Cyclohexane'), ('*CC', 'Ethyl'), ('*C(C)C', 'Isopropyl'),
]

수정 !! *O --> *CO


# DTLC: Deep learning for Thin-Layer Chromatography

---

## Installation

> **Note**: This repository uses **Git LFS** for large model files (`.pt`). Make sure Git LFS is installed before cloning to download full model weights.

```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone repository with LFS files
git clone https://github.com/ACCA-KU/DTLC
cd DTLC

# Set up conda environment
conda env create -f environment.yml
```







## DB/LLM_data_Extractor/GPT_API_whole_paper.ipynb
This notebook provides a tool for automatically extracting data from research papers using the OpenAI API (GPT-4.1-mini).

*You will need to create and provide your own OpenAI API key to use this notebook.*

---

## Train_valid_test.ipynb
This notebook contains code for training and validating a model that predicts TLC Rf values.


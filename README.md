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

### 📂 Project Structure

```text
Predictive_models/
├── Training/              # Code for training the TLC Rf prediction model
└── model_validation/      # Code for validating the trained model
```

This notebook contains code for training and validating a model that predicts TLC Rf values.



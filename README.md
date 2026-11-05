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



## 추가 수정 & 결정 사항
[결정사항]

데이터 100개만 남김 (train 80, val 10, test 10)

sei models/ ~~~ train.ipynb 에서 모델 고르도록 모델명 옵션 적어두기



setup 설정 완료 후 read me에 공지
pip install git+https://github.com/~~~~최종 레퍼지토리 이름~~~~.git

통합 셋업 어려우면 predictive, SEI 나누어서

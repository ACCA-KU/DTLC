# DTLC: Deep learning for Thin-Layer Chromatography

---

## Introduction
`DTLC` is ~

## DB/LLM_data_Extractor/GPT_API_whole_paper.ipynb
This notebook provides a tool for automatically extracting data from research papers using the OpenAI API (GPT-4.1-mini).

*You will need to create and provide your own OpenAI API key to use this notebook.*

---

## Train_valid_test.ipynb
This notebook contains code for training and validating a model that predicts TLC Rf values.


## BLI-GAT model

<img width="1590" height="1430" alt="image" src="https://github.com/user-attachments/assets/d0275b9f-b9a4-49a6-bc3d-fa9b3bd6caa6" />
<img width="790" height="1430" alt="image" src="https://github.com/user-attachments/assets/171e4ef1-fbed-4667-8fe5-47291c8b81ad" />



## BLI-GAT-SEI model
<img width="1987" height="1290" alt="image" src="https://github.com/user-attachments/assets/45eb9385-1176-4ef3-b693-ce9171377cf2" />





---
# Model Requirements

## Notes
- Requires an NVIDIA driver supporting CUDA 11.8.


Conda environment for running:

- Python 3.10
- PyTorch 2.1.2 + CUDA 11.8 
- DGL 2.1.0 + CUDA 11.8
- RDKit 2024.9.6

## Setup

```bash
conda env create -f environment.yml
conda activate tlc
```


---

## 추가 수정 & 결정 사항
[결정사항]

데이터 100개만 남김 (train 80, val 10, test 10)

sei models/ ~~~ train.ipynb 에서 모델 고르도록 모델명 옵션 적어두기



setup 설정 완료 후 read me에 공지
pip install git+https://github.com/~~~~최종 레퍼지토리 이름~~~~.git

통합 셋업 어려우면 predictive, SEI 나누어서

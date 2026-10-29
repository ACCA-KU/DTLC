# DTLC: Deep learning for Thin-Layer Chromatography

---

## DB/LLM_data_Extractor/GPT_API_whole_paper.ipynb
This notebook provides a tool for automatically extracting data from research papers using the OpenAI API (GPT-4.1-mini).

*You will need to create and provide your own OpenAI API key to use this notebook.*

---

## Train_valid_test.ipynb
This notebook contains code for training and validating a model that predicts TLC Rf values.





---

## 추가 수정 & 결정 사항
[결정사항]

데이터 100개만 남김 (train 80, val 10, test 10)

sei models/ ~~~ train.ipynb 에서 모델 고르도록 모델명 옵션 적어두기




{아인님 수정 필요}

[predictive models]

~~1. Predictive_models\model_validation : 논문에 예측 모델 결과 뽑은거 재현 해두는 코드~~ 
/Predictive_models/model_validation/model_validation.ipynb
/Predictive_models/model_validation/rf_prediction.ipynb

[SEI models]

~~1. \SEI_models\SSC\src\SSCnet : 여기에 현재 사용하는 group contribution모델이 없는것 같아서 넣어줘햐 할 거같아요~~

~~2. \SEI_models\SSC\src\network_refer.yaml : 이것도 마찬가지...~~

~~3. \SEI_models\Trained_models: SEI 의 ckpt 모델 논문에 쓴것만 가져다 놓기 (여기엔 다 삭제해둠)~~

~~4. SEI_models\_analysis_copy_segmentmodel.ipynb : figure에 있는 데이터 재현하는 코드로 변경해야함~~



setup 설정 완료 후 read me에 공지
pip install git+https://github.com/~~~~최종 레퍼지토리 이름~~~~.git

통합 셋업 어려우면 predictive, SEI 나누어서

---
# DTLC Model environment

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

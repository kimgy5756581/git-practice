# Titanic Survival Prediction

Kaggle Titanic 데이터로 승객 생존 여부를 분류하는 학습 프로젝트입니다. 노트북은 결측값 처리와 범주형 값 변환 후 `RandomForestClassifier`로 예측하고 제출 CSV를 생성합니다.

## Files

- `titanic-random-forest.ipynb`: 전처리, 학습, 예측 과정
- `train.csv`, `test.csv`: Kaggle Titanic 데이터
- `submission*.csv`: 실습 중 생성한 제출 결과

## Run

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install pandas scikit-learn jupyter
jupyter notebook titanic-random-forest.ipynb
```

데이터 출처: [Kaggle Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

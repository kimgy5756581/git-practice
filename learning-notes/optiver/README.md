# Optiver Study Notes

Optiver 금융 시장 데이터셋을 대상으로 한 탐색적 분석과 AutoGluon 기반 예측 실험입니다. 이 자료는 House Prices 프로젝트와 데이터 도메인이 달라 별도 학습 노트로 분리했습니다.

| File | Purpose |
| --- | --- |
| `basic-eda.py` | 메모리 최적화, 결측치·분포·상관관계 탐색 |
| `eda-and-colab-experiments.md` | 그룹별 통계·이상치·시간 구간 EDA와 Colab용 AutoGluon/OpenFE 실험 메모 |

`basic-eda.py`는 Python 스크립트이며, 나머지 실험 메모에는 Colab 전용 설치 셀이 섞여 있어 노트 형식으로 보관했습니다. 실행 전 `PATH`를 자신의 데이터 위치로 바꿔야 하며, 대회 데이터와 생성 모델 파일은 저장소에 포함하지 않습니다.

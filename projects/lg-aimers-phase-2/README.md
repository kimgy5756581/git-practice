# LG Aimers 9th Phase 2 — KBO Pitch Control Prediction

KBO 투수가 의도한 곳에 공을 던졌는지(`control_success`)를 확률로 예측한 LG Aimers 9기 Phase 2 프로젝트입니다.

| Result | Value |
| --- | --- |
| Metric | Brier Skill Score |
| Final score | 1163.1106 |
| Ranking | 59th of 1,090 teams |
| Prediction period | Train: 2019–2024 / Test: 2025 |

## Approach

최종 파이프라인은 트랙맨 이력 매칭, 시계열·선수 기반 피처, LightGBM·CatBoost·Logistic Regression·TabM 앙상블, 그리고 분포 이동 보정으로 구성됩니다. 핵심 교훈은 단순 모델 교체보다 검증 구조와 사후 보정 근거가 성능에 더 큰 영향을 준다는 점입니다.

### Technical decisions

- **Temporal generalization:** 2019–2024 시즌 데이터로 2025 시즌을 예측하는 문제로 보고, 시즌 간 평균·신인 비율 차이를 별도 보정 대상으로 다뤘습니다.
- **Ensemble design:** 트리·선형 모델과 TabM을 결합하고, TabM의 수치·범주형 임베딩을 중심 모델로 사용했습니다.
- **Evidence-driven calibration:** Brier Skill Score의 이차 형태를 이용해 제출 결과 한 번으로 보정 축의 기울기와 최적점을 추정하는 절차를 기록했습니다.
- **Reproducibility boundary:** CUDA 임베딩 역전파의 비결정성을 명시하고, 결정적으로 재현 가능한 산출물과 그렇지 않은 산출물을 구분했습니다.

자세한 실험 근거와 실패 기록은 [METHODOLOGY.md](METHODOLOGY.md)에, 환경·재현성 조건은 [ENVIRONMENT.md](ENVIRONMENT.md)에 정리했습니다. 점수와 순위는 프로젝트 진행 당시의 최종 기록이며, 이 저장소에는 대회 원본 데이터나 모델 가중치를 포함하지 않습니다.

## Repository contents

| File | Purpose |
| --- | --- |
| `train_full.py` | 원본 CSV부터 모델·추론 스크립트·제출 ZIP을 만드는 통합 학습 코드 |
| `verify.py` | 재생성된 산출물과 제출본을 항목별로 대조하는 개발용 검증 도구 |
| `env_dump.py` | Google Colab 환경 정보를 수집하는 도구 |
| `METHODOLOGY.md` | 실험 과정, 리더보드 기반 상수 탐색, 실패와 교훈 |
| `PROJECT_NOTES.md` | 원본 프로젝트 안내와 실행 체크포인트 |
| `METHODOLOGY-summary.md` / `README-original.md` | 초기 버전 문서 보관본 |

## Reproduction

원본 대회 데이터는 포함하지 않습니다. 대회 데이터 이용 약관에 따라 직접 내려받아 아래처럼 배치해야 합니다.

```text
data/
  train.csv
  trackman_history.csv
  test.csv
  sample_submission.csv
```

Google Colab GPU 환경에서 실행하도록 작성됐으며, 학습 시간은 약 155분입니다.

```bash
python env_dump.py
python train_full.py
```

학습 후 `verify.py`는 로컬 생성 산출물과 원본 제출 ZIP을 비교합니다. 이 검증에는 별도로 보관한 `submit_q-0.2.zip`이 필요하며 저장소에는 포함하지 않습니다.

## Reproducibility notes

시드와 결정성 옵션을 설정했지만 TabM의 CUDA 임베딩 역전파 특성상 GPU·드라이버·라이브러리 버전이 다르면 가중치가 비트 단위로 일치하지 않을 수 있습니다. 데이터, 모델, 생성된 ZIP과 출력 폴더는 공개 저장소에서 제외합니다.

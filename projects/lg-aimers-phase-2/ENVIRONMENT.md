# 개발 환경 및 라이브러리 버전

> 아래 `<...>` 부분은 `env_dump.py` 실행 결과로 채워 주세요.

## 실행 환경

| 항목 | 값 |
|---|---|
| 플랫폼 | Google Colab |
| OS | `<Ubuntu 22.04.x LTS>` |
| Python | `<3.11.x>` |
| GPU | `<Tesla T4 / A100-SXM4-40GB>` |
| CUDA | `<12.x>` |

## 라이브러리

| 패키지 | 버전 |
|---|---|
| torch | `<2.x.x+cu121>` |
| numpy | `<...>` |
| pandas | `<...>` |
| scipy | `<...>` |
| scikit-learn | `<...>` |
| lightgbm | 4.7.0 |
| catboost | 1.2.10 |
| tabm | `<...>` |
| rtdl_num_embeddings | `<...>` |

## 파일 구성

```
train_full.py     학습 코드 — rf.pkl / tabm.pt 생성 (약 150분)
verify.py         산출물을 제출 zip 과 대조 검증
script.py         추론 코드 — 리더보드 제출본과 동일
data/train.csv
data/trackman_history.csv
```

## 실행

```bash
python train_full.py     # → out/model/rf.pkl , out/model/tabm.pt
python verify.py         # → 제출본과 항목별 대조
```

## 재현성에 관하여

학습 코드에는 시드 고정과 결정성 플래그를 적용했습니다.

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
```

다만 TabM 은 범주형 임베딩을 사용하며, PyTorch 임베딩 역전파의 CUDA
scatter-add 는 원자적 덧셈 순서가 비결정적입니다. 따라서 **GPU 모델이나
드라이버/라이브러리 버전이 다르면 가중치가 비트 단위로 동일하지 않을 수
있습니다.** 최종 산출물은 리더보드에 제출된 `tabm.pt` 입니다.

LightGBM · CatBoost · LogisticRegression · KMeans · 트랙맨 매칭 ·
전처리 통계는 모두 결정적이며 동일 환경에서 완전히 재현됩니다.

## 리더보드 튜닝 상수에 관하여

후처리 상수(`TARGET`, `a_scale_A`, `a_quad`, `W_TABM`, `KSM`, 수축계수,
`s_unk`, `s_F`, `k_m`, `theta23`)는 Public 리더보드 점수를 이용한
하이퍼파라미터 선택으로 결정했습니다. 방법은 다음과 같습니다.

1. 현재 값을 앵커로 두고 한 값만 바꾼 후보를 제출
2. 두 점수의 차이로 목적함수의 기울기 `b` 를 역산
3. 이차함수의 꼭짓점 `θ* = b / 2C` 를 계산해 최적값 확정

제출 횟수 제한 내에서 수행했으며, 각 상수의 산출 근거는 `train_full.py`
상단 주석에 곡률과 함께 기록해 두었습니다.

## 행 독립성

추론 시 test 행 간 정보 공유가 없습니다.

- 모든 상수는 학습 단계에서 확정되어 `rf.pkl` 에 저장됩니다
- `script.py` 는 저장된 상수를 적용만 하며 `find_shift` 를 재호출하지 않습니다
- `season_form` 은 자기 `id` 로 앵커 dict 를 조회만 합니다
- 그룹 시프트는 해당 행 자신의 피처로 그룹을 판정하며, 이는 범주형 효과를
  모형에 넣는 것과 동일한 구조입니다

지금까지 나온 핵심 용어 전부 정리할게.

---

### 모델 종류

| 용어 | 풀네임 | 한줄 설명 |
|---|---|---|
| MLP | Multi-Layer Perceptron | 가장 기본적인 NN. Linear+ReLU 반복 |
| ResNet | Residual Network | MLP + 지름길(x 더하기) |
| CNN | Convolutional Neural Network | 이미지용 NN |
| RNN | Recurrent Neural Network | 시계열용 NN |
| Transformer | - | Attention으로 피처끼리 서로 참조 |
| GBDT | Gradient Boosted Decision Tree | XGBoost, LightGBM 같은 트리 계열 통칭 |

---

### NN 구성요소

| 용어 | 설명 |
|---|---|
| Linear | 가중치 행렬 곱 = 피처 섞기 = 회전 |
| ReLU | 음수→0, 양수→그대로. 비선형성 추가 |
| PReLU | ReLU인데 음수 기울기 α를 학습 |
| BatchNorm | 각 층 입력을 평균0 분산1로 정규화 |
| Dropout | 노드 랜덤으로 끔. 과적합 방지 |
| Residual | ResNet의 핵심. F(x)+x. 잔차만 학습 |
| Attention | 피처끼리 서로 얼마나 참조할지 가중치 |

---

### 학습 관련

| 용어 | 설명 |
|---|---|
| 가중치 | NN이 학습으로 찾는 숫자들 |
| 역전파 | 오차를 역방향으로 전달해서 가중치 조정 |
| gradient | 가중치를 어느 방향으로 얼마나 조정할지 |
| gradient vanishing | 층 깊어질수록 학습 신호 사라지는 문제 |
| 과적합 | 훈련 데이터만 외워버림. 새 데이터 성능 떨어짐 |
| 학습률 | 가중치 조정 보폭. 처음엔 크게, 나중엔 작게 |
| HPO | HyperParameter Optimization. 하이퍼파라미터 탐색 |
| 하이퍼파라미터 | 층 수, 노드 수, 학습률 등 사람이 정하는 것 |

---

### 임베딩

| 용어 | 설명 |
|---|---|
| Scalar | 숫자 하나. 예: 나이=30 |
| Embedding | 숫자를 벡터로 변환 |
| One-hot | 범주형을 0/1 벡터로 변환. 예: [1,0,0] |
| PLE | Piecewise Linear Encoding. 구간별로 쪼개서 벡터화 |
| Periodic | sin/cos으로 변환. 주기적 패턴 표현 |
| PLR | ReLU∘Linear∘Periodic. 가장 성능 좋았던 임베딩 |

---

### 전처리

| 용어 | 설명 |
|---|---|
| Robust scale | 이상치에 강한 스케일링. 중앙값 기준 |
| Smooth-clip | 극단값을 부드럽게 잘라냄. gradient 안 끊김 |
| Learnable scaling | 스케일링 계수도 학습 |

---

### 기타

| 용어 | 설명 |
|---|---|
| Baseline | 비교 기준 모델. 이것보단 좋아야 함 |
| Backbone | 모델의 핵심 구조. MLP냐 Transformer냐 |
| Vanilla | 아무것도 추가 안 한 기본 모델 |
| Tabular | 표 형태 데이터 |
| Scalar → Vector | 임베딩의 핵심. 표현력 향상 |
| Meta-tuned | 여러 데이터셋에서 미리 찾아놓은 기본값 |
| Multimodal | 이미지+텍스트+표 등 여러 데이터 종류 동시 처리 |
오늘 채팅 전체에서 나온 개념어 전부 정리할게.

---

### 1. 모델 종류

| 용어 | 풀네임 | 설명 |
|---|---|---|
| ML | Machine Learning | 머신러닝. 데이터로 스스로 학습하는 것 전체 |
| DL | Deep Learning | 딥러닝. ML의 부분집합, 층 깊게 쌓는 것 |
| NN | Neural Network | 신경망. 입력→여러 층→출력 |
| MLP | Multi-Layer Perceptron | 가장 기본적인 NN, Linear+ReLU 반복 |
| ResNet | Residual Network | MLP + 지름길(x 더하기) |
| CNN | Convolutional NN | 이미지용 NN |
| RNN | Recurrent NN | 시계열용 NN |
| Transformer | - | Attention으로 관계 학습하는 NN |
| GBDT | Gradient Boosted Decision Tree | XGBoost, LightGBM 등 트리 계열 통칭 |

---

### 2. NN 구성요소

| 용어 | 설명 |
|---|---|
| Linear | 가중치 행렬 곱 = 피처 섞기 = 회전 |
| ReLU | 음수→0, 양수→그대로. 비선형성 + gradient 부분 해결 |
| PReLU | ReLU인데 음수 기울기 α를 학습 (정보 손실 줄임) |
| BatchNorm | 각 층 입력을 평균0 분산1로 정규화. gradient exploding 방지 |
| Dropout | 노드 랜덤으로 끔. 과적합 방지 |
| Residual | ResNet 핵심. F(x)+x. 잔차만 학습 |
| Smooth-clip | 극단값 부드럽게 자름. gradient 안 끊기게 |
| GLU | Gated Linear Unit. 정보 통과량 조절하는 문 |

---

### 3. 학습 관련

| 용어 | 설명 |
|---|---|
| 가중치(Weight/Parameter) | NN이 학습으로 찾는 값. Linear 단계에 있음 |
| bias(b) | y=wx+b의 b. 직선 위치를 자유롭게 (원점 안 지나도 됨) |
| 역전파(Backpropagation) | 오차를 역방향 전달해서 가중치 조정. 피드백 |
| gradient | ∂E/∂w. 가중치를 어느 방향으로 얼마나 바꿀지 (부호+크기 둘 다 중요) |
| chain rule | 역전파 원리. 층마다 편미분 곱해서 전달 |
| gradient vanishing | 층 깊어질수록 gradient→0, 앞층 학습 안 됨 |
| gradient exploding | gradient가 너무 커짐, 학습 불안정 |
| 과적합(Overfitting) | 훈련 데이터만 외움, 새 데이터 성능 떨어짐 |
| 학습률(Learning Rate) | 가중치 조정 보폭. 처음 크게→나중 작게 |
| HPO | HyperParameter Optimization. 하이퍼파라미터 탐색 |
| 하이퍼파라미터 | 층 수, 노드 수, 학습률 등 사람이 정하는 것 |
| pre-training | 정답 없이 데이터 구조 먼저 학습 |
| Contrastive | 비슷한 건 가깝게, 다른 건 멀게 학습 |
| Denoising | 일부러 망가뜨리고 복원하도록 학습 |

---

### 4. 임베딩

| 용어 | 설명 |
|---|---|
| Scalar | 숫자 하나. 예: 나이=30 |
| Vector | 숫자 여러개 (1줄). 예: [0.2, 0.8] |
| Matrix | 숫자 여러개 (여러줄) |
| Embedding | 숫자/카테고리를 벡터로 변환 |
| One-hot | 범주형을 0/1로. [1,0,0]. 어느 행인지 가리키는 포인터 |
| Lookup table | 카테고리별 벡터 저장한 표. 학습됨 |
| PLE | Piecewise Linear Encoding. 구간별로 쪼개서 벡터화 |
| Periodic | sin/cos으로 변환. 주기 패턴 표현 |
| PLR | ReLU∘Linear∘Periodic. 성능 좋았던 임베딩 |

---

### 5. 전처리

| 용어 | 설명 |
|---|---|
| Robust scale | 이상치에 강한 스케일링 (중앙값 기준) |
| Smooth-clip | 극단값 부드럽게 자름 |
| Learnable scaling | 스케일링 계수도 학습 |
| Scaling/정규화 | NN은 필수, 트리는 불필요 (트리는 순서만 봄) |

---

### 6. Attention 관련 (가장 헷갈리는 부분)

| 용어 | 설명 |
|---|---|
| Attention | 요소끼리 얼마나 참조할지 동적 가중치 |
| Q, K, V | Query(찾는 것), Key(가진 정보), Value(실제 내용) |
| Self-Attention | 같은 입력 내부 참조 (피처끼리, 같은 행) |
| Cross-Attention | 다른 입력끼리 (텍스트↔이미지) |
| Multi-Head | 여러 관점에서 동시에 attention |
| CLS token | 전체 정보 모으는 대표 토큰. 최종 예측에 사용 |
| Token | 처리 최소 단위. 테이블에선 피처 하나 |
| Sequence | 토큰들을 순서대로 모은 것 |

**같은 개념, 다른 이름 (논문마다 용어 다름):**

| 개념 | NPT 용어 | SAINT 용어 |
|---|---|---|
| 행끼리 비교 | Datapoint Attention | Intersample Attention |
| 열끼리 비교(같은 행 안) | Attribute Attention | Feature-wise Self-Attention |

---

### 7. 모델별 핵심 (오늘 배운 것들)

| 모델 | 핵심 아이디어 |
|---|---|
| MLP-PLR | 임베딩 개선 (Periodic) |
| RealMLP | 전처리+학습 개선 (제대로 된 MLP) |
| TabR | MLP + 검색(retrieval). 비슷한 샘플 참조 (W_K,W_Y,T 추가) |
| ModernNCA | 거리 기반 soft KNN. 임베딩만으로 (TabR보다 단순) |
| TabM | 앙상블. 가중치 공유로 효율적 |
| TabTransformer | 범주형만 Attention (반쪽) |
| FT-Transformer | 모든 피처 토큰화→Attention (표준 틀) |
| NPT/SAINT | 행+열 Attention 둘 다 |
| T2G-Former | 관계 그래프로 관련 피처만 연결 |
| ExcelFormer | 해로운 정보 흐름 차단 (반투과) |
| AMFormer | 산술연산(곱셈) 내장 (log-exp 트릭) |

---

### 8. 핵심 개념

| 용어 | 설명 |
|---|---|
| Tabular/Tab | 표 형태 데이터 |
| Baseline | 비교 기준 모델 |
| Backbone | 모델 핵심 구조 |
| Vanilla | 아무것도 안 더한 기본 모델 |
| Inductive bias | 모델이 학습 전부터 가진 가정. 데이터와 맞으면 강함 |
| KNN | K-Nearest Neighbors. 가까운 K개 참조 |
| Hard vs Soft | Hard: K개만. Soft: 거리 따라 가중치 (미분 가능, 학습 가능) |
| Parametric | 가중치만 씀 (예측 시 데이터 안 봄) |
| Non-Parametric | 가중치 + 데이터 직접 참조 |
| Target leakage | 정답이 실수로 새어들어감. 컨닝 효과 |
| Multimodal | 이미지+텍스트+표 등 여러 종류 동시 처리 |

---

### 9. 핵심 수식 패턴 (역전파)

```
순전파: a = w × 입력 (출력값, 앞으로 전달)
역전파: 
  ∂E/∂w = ∂E/∂a × 출력값  → 가중치 업데이트용
  ∂E/∂a_prev = ∂E/∂a × 가중치  → 앞층 전달용
  (곱셈 미분하면 상대방이 남음)
```

---

### 10. 오늘의 큰 결론

```
1. 트리(GBDT)는 여전히 강력 (특히 데이터 적을 때)
2. 딥러닝이 만능 아님
3. 데이터 특성에 맞는 inductive bias 고르는 게 핵심
4. "There is still no universal solution"
5. 데이터 부족한 상황 판단력 = 사람의 가장 오래갈 가치
```

필요할 때 이 정리 보면서 복습하면 돼.
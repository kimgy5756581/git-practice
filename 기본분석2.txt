# 1. 드라이브 마운트=>구글 드라이브를 해당 위치에 연결(마운트):드라이브 파일접근가능
from google.colab import drive
drive.mount('/content/drive')

# 2. 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
# 불필요한 경고 메세지 숨김
warnings.filterwarnings('ignore')


# 3. 경로 설정
PATH = '/content/drive/MyDrive/optiver/'

# 4. 메모리 최적화하면서 읽기
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
          # 숫자 타입일 경우만 처리(object=문자열)
            c_min = df[col].min()
            c_max = df[col].max()
            # 해당 컬럼의 최댓값 최솟값 파악해서 범위확인

            col_type = df[col].dtype
            # # 예: int64, float32, object 등

            # str(col_type)
            # # dtype을 문자열로 변환
            # # int64 → "int64"
            # # float32 → "float32"

            # str(col_type)[:3]
            # # 문자열에서 앞 3글자만 자르기
            # # "int64"   → "int"
            # # "float32" → "flo"
            # # "object"  → "obj"
            if str(col_type)[:3] == 'int':

                if c_min > -128 and c_max < 127:
                    df[col] = df[col].astype(np.int8)
                    # -128~127 범위면 int8 (1바이트)로 변환해서 메모리 아끼
                elif c_min > -32768 and c_max < 32767:
                    df[col] = df[col].astype(np.int16)
                elif c_min > -2147483648 and c_max < 2147483647:
                    df[col] = df[col].astype(np.int32)
            else:# 실수형 컬럼이면 (float)
                if c_min > np.finfo(np.float16).min and\
                 c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                    # finfo = float info:float16 타입의 정보 가져오기
                    # \는 "다음 줄도 이 코드의 연속이야" 라는 뜻
    return df

# 5. 샘플만 먼저 읽기
df = pd.read_csv(PATH + 'train.csv', nrows=100000)
# PATH + 'train.csv':경로합치기:'/content/drive/MyDrive/optiver/train.csv'
# 전체 528만행 중 10만행만 읽기
df = reduce_mem_usage(df)
# 읽은 데이터 메모리 최적화
print(f"메모리 사용량: {df.memory_usage().sum() / 1024**2:.1f} MB")
# / 1024**2
# # 바이트 → MB 변환
# # 1024 = 1KB
# # 1024² = 1MB

# :.1f
# # 소수점 1자리까지
print(df.shape)
df.head()

# 1. 기본 정보
print(df.info())
#  각 컬럼의
#   이름, 결측치 개수, 타입
#    한눈에 보여줌
print(df.describe())
# 숫자 컬럼 전부의
#   평균, 표준편차, 최솟값, 최댓값 등
#   통계 요약 보여줌
# 2. 결측치 확인
print(df.isnull().sum())
# 컬럼별 결측치(빈칸) 개수 출력

# 3. target 분포
df['target'].hist(bins=100)
# → 히스토그램 그리기
# → target 값을 100개 구간으로 나눠서
#   각 구간에 몇 개나 있는지 막대로 표시
plt.show()
# 그래프를 화면에 출력

# 4. imbalance → target 관계
df.groupby('imbalance_buy_sell_flag')['target'].mean()
# → -1, 0, +1 세 그룹으로 묶기
# 출력:
# imbalance_buy_sell_flag
# -1    -0.358  ← 매도 우세일 때 target 평균
#  0    -0.343  ← 균형일 때 target 평균
# +1     0.167  ← 매수 우세일 때 target 평균

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt# 히스토그램, 선그래프, 산점도 등
from IPython.display import display
# from IPython.display import display
# display(missing)
# → 코랩에서 예쁜 표로 렌더링
# → print() 대신 display() 쓰면 됨
# → 중간 줄이어도 표로 출력
import warnings
warnings.filterwarnings('ignore')

# ── 1. 기본 구조 ──────────────────────────────────
print("=" * 50)
print("1. 기본 구조")
print("=" * 50)
print(f"행 수: {df.shape[0]:,}") #:,=>천의 자리 쉼표
print(f"컬럼 수: {df.shape[1]}")
print(f"메모리: {df.memory_usage().sum() / 1024**2:.1f} MB") #sum이 없으면 컬럼별 출력

# ── 2. 컬럼 타입 및 결측치 ──────────────────────────
print("\n" + "=" * 50)
print("2. 컬럼 타입 및 결측치")
print("=" * 50)
# pd.DataFrame({})=>딕셔너리로 새 표 만들기:컬럼명은 내가 문자열로 지정하고 행은 df의 컬럼명이 담당
missing = pd.DataFrame({
    '타입': df.dtypes,
    '결측치 수': df.isnull().sum(),
    # sum 안붙이면 컬럼마다 따로따로 나옴
    '결측치 비율(%)': (df.isnull().sum() / len(df) * 100).round(1),
    # f.isnull()
    # 각 칸이 비어있으면 True, 있으면 False
    # True = 1, False = 0으로 계산됨
    # df.size   # → 100000 × 17 = 1700000 이게 셀 개수
    # df.shape  # → (100000, 17) 행/컬럼 둘 다
    # len(df)   # → 100000 행 수만
    '유니크 값 수': df.nunique()
    #컬럼별 고유값(중복없이 어떤값들이 있나) 개수 출력
})

display(missing)

# ── 3. 수치형 컬럼 통계 ──────────────────────────────
print("\n" + "=" * 50)
print("3. 수치형 컬럼 통계")
print("=" * 50)
display(df.describe().T)  # .T = 행열 전치 (보기 편하게)

# ── 4. target 분포 ───────────────────────────────────
print("\n" + "=" * 50)
print("4. target 분포")
print("=" * 50)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
# figsize(14*4)인 도화지를 1행 2열 axes로 쪼갠다는 얘기
# 전체 분포
df['target'].hist(bins=100, ax=axes[0])
# target 히스토그램을 왼쪽(axes[0])에 그리기
# bins=100 = 전체 범위를 100개 구간으로 나눔
axes[0].set_title('target 전체 분포')
axes[0].set_xlabel('target (basis points)')
axes[0].set_ylabel('빈도')

# 이상치 제외 분포 (-30 ~ +30)
df['target'].clip(-30, 30).hist(bins=100, ax=axes[1])
# .clip(-30, 30):
# -30보다 작은 값 → -30으로 고정
# +30보다 큰 값  → +30으로 고정
# → 이상치 제외하고 그래프 그리기
# → 오른쪽(axes[1])에 그리기
axes[1].set_title('target 분포 (-30 ~ +30)')
axes[1].set_xlabel('target (basis points)')
axes[1].set_ylabel('빈도')

plt.tight_layout()
#  두 그래프가 겹치지 않게 자동 간격 조정
plt.show()
# 그래프 출력

# ── 5. imbalance → target 관계 ───────────────────────
print("\n" + "=" * 50)
print("5. imbalance_buy_sell_flag별 target 평균")
print("=" * 50)
display(df.groupby('imbalance_buy_sell_flag')['target'].agg(['mean', 'std', 'count']))
# .groupby('imbalance_buy_sell_flag') 집계함수
# -1, 0, +1 세 그룹으로 묶기=>imbalance_buy_sell_flag가 -1(매도우세),0,1(매 우세)로만 이루어졌으니

# ['target']
# target 컬럼만 보기

# .agg(['mean', 'std', 'count'])
# agg = aggregate (집계)
# 세 가지 동시에 계산:
# mean  = 평균
# std   = 표준편차 (얼마나 퍼져있나)
# count = 개수

# 출력 예시:
#                          mean    std    count
# imbalance_buy_sell_flag
# -1                      -0.36  8.21    33421
#  0                      -0.34  7.98    32156
#  1                       0.17  8.45    34423

# ── 6. seconds_in_bucket별 패턴 ──────────────────────
print("\n" + "=" * 50)
print("6. seconds_in_bucket별 far/near_price 값 존재 개수")
print("=" * 50)
display(df.groupby('seconds_in_bucket')[['far_price', 'near_price']].count().head(10))
# 결과:
#                    far_price  near_price
# seconds_in_bucket
# 0                          0           0
# 10                       245         312
# 20                      1823        1945

# 이렇게 하면 결측치 개수
display(df.groupby('seconds_in_bucket')  # 초별로 묶기 0,10,20...
  [['far_price', 'near_price']]  # 두 컬럼만
  .apply(lambda x: x.isnull().sum())  # 각 그룹에 함수 적용(lambda x:=>1회용 함수),groupby로 묶은 group이 x로 들어옴
  .head(10))  # 처음 10행만
# ── 7. 이상치 확인 ───────────────────────────────────
print("\n" + "=" * 50)
print("7. 이상치 확인 (상위 5개)")
print("=" * 50)
numeric_cols = df.select_dtypes(include=[np.number]).columns
# 숫자형 컬럼만 선택
# np.number = 모든 숫자 타입
# (int8, int16, float32, float64 전부)
# .columns
# 선택된 컬럼들의 이름만 가져오기
# → ['stock_id', 'date_id', 'imbalance_size' ...]
for col in numeric_cols:
    q99 = df[col].quantile(0.99)
    #     quantile = 분위수
    # 0.99 = 상위 1% 경계값
    # → 이 값보다 큰 것이 상위 1%
    q01 = df[col].quantile(0.01)
      # 0.01 = 하위 1% 경계값
  # → 이 값보다 작은 것이 하위 1%

  # 예시: target q01 = -24.1
  # → target이 -24.1 미만이면 하위 1%
    outliers = ((df[col] > q99) | (df[col] < q01)).sum() #이상치 개수
    print(f"{col:35s} 상위1%: {q99:15.4f}  하위1%: {q01:15.4f}  이상치수: {outliers:6,}")
#     {col:35s}
# # 컬럼명을 35칸으로 출력
# # → 줄 맞추려고 (가독성)

# {q99:15.4f}
# # 15칸, 소수점 4자리
# # 예: "      25.3000"

# {q01:15.4f}
# # 15칸, 소수점 4자리
# # 예: "     -24.1000"

# {outliers:6,}
# # 6칸, 천단위 콤마
# # 예: " 2,000"

# ── 8. imbalance_size → target 관계 ──=>매도 매수 불균형 크기에 따른 변동성
print("=" * 50)
print("8. imbalance_size 크기별 target")
print("=" * 50)

# 0 제외하고 분석
df_nonzero = df[df['imbalance_size'] > 0].copy()
# → imbalance_size가 0보다 큰 행만 필터링
# .copy()
# → 원본 df 건드리지 않고
#   별도 복사본 만들기

# 5구간으로 나누기

df_nonzero['imb_size_bin'] = pd.qcut(
    df_nonzero['imbalance_size'],
    5,
    labels=['최소', '소', '중', '대', '최대']
)
# qcut = quantile cut
# →값을 오름차순으로 정리한뒤 데이터를 개수가 균등하게 되도록 구간으로 나눔+레이블칼럼추
# 가=>cut:범위를 균등하게 나누기
print("imbalance_size > 0인 경우만 (0 제외)")
display(df_nonzero.groupby('imb_size_bin')['target'].agg(['mean', 'std', 'count']))
# 출력:
#              mean     std   count
# imb_size_bin
# 최소        -0.307   9.074   14310
# 소          -0.167   8.336   14310
# 중          +0.006   8.038   14309
# 대          -0.216   7.949   14309
# 최대        +0.364   8.225   14310
# flag랑 size 조합
print("\nflag × size 조합:")

display(df_nonzero.groupby(
    ['imbalance_buy_sell_flag', 'imb_size_bin']
)['target'].mean().unstack())

# .unstack()
# 결과를 2차원 표로 펼치기

# unstack() 없으면:
# imbalance_buy_sell_flag  imb_size_bin
# -1                       최소    -0.650
#                          소      -0.224
#                          중      -0.277
# 1                        최소    -0.013
# ...
# → 세로로 길게 나옴

# unstack() 있으면:
# imb_size_bin   최소    소     중     대    최대
# flag
# -1           -0.650  -0.224 -0.277 -0.519 -0.095
# 1            -0.013  -0.122 +0.239 +0.016 +0.686
# → 2차원 표로 펼쳐서 한눈에 비교 가능=>imb_size_bin 부분이 가로로 빠져줌.

# ── 9. seconds_in_bucket별 target 변동성 ──=>시간에 따른 변동성
print("=" * 50)
print("9. seconds_in_bucket별 target 변동성")
print("=" * 50)
display(df.groupby('seconds_in_bucket')['target'].agg(['mean', 'std', 'count']))

# ── 10. target과의 상관관계 ──
print("=" * 50)
print("10. target과의 상관관계")
print("=" * 50)

corr = df.select_dtypes(include=[np.number]).corr()['target'].sort_values()
# df.select_dtypes(include=[np.number])
# # 숫자형 컬럼만 선택
# # row_id(문자열) 제외
# .corr()
# 모든 숫자 컬럼 간 상관관계 계산
# 결과: 컬럼 수 × 컬럼 수 정방 행렬
# 예: 16×16 표
#
#                 stock_id  imbalance_size  target ...
# stock_id          1.000           0.012  -0.013
# imbalance_size    0.012           1.000   0.014
# target           -0.013           0.014   1.000
# ...
# ['target']
# # 그 중 target 열만 뽑기
# # → target과 각 변수의 상관관계만

# .sort_values()
# # 작은 값부터 큰 값 순서로 정렬
# # → 한눈에 어떤 변수가 가장 관련있는지
display(corr)

# 히트맵으로 시각화
import seaborn as sns #히트맵, 박스플롯 등이 matplotlib보다 간단

plt.figure(figsize=(10, 8))# 10×8인치 캔버스 생성
# plt.figure()   → 그래프 1개 그릴 때
# plt.subplots() → 그래프 여러 개 나란히 그릴 때

sns.heatmap(
    df.select_dtypes(include=[np.number]).corr(),
    # 전체 상관관계 행렬
# 16×16 표 전체를 히트맵으로
    annot=True,
    # 각 칸에 숫자 표시
# False면 색상만 보임
# True면 -0.07, +0.03 등 숫자도 표
    fmt='.2f',
    # 숫자 표시 형식
# 소수점 2자리
    cmap='coolwarm',
    # 색상 팔레트
# cool(파랑) = 음수 상관관계
# warm(빨강) = 양수 상관관계
# 중간(흰색) = 0 (관계없음)
    center=0
    # 색상의 중심값을 0으로 설정
)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

!pip install uv -q
!uv pip install autogluon.tabular --system -q
# ── 설치 (처음 한 번만) ──
!pip install uv -q
!uv pip install autogluon.tabular openfe --system -q

# ── 라이브러리 ──
from autogluon.tabular import TabularPredictor
from openfe import OpenFE, transform
import pandas as pd
import numpy as np
from google.colab import drive

drive.mount('/content/drive')
PATH = '/content/drive/MyDrive/optiver/'



# ── 1. 데이터 로드 ──
df = pd.read_csv(PATH + 'train.csv', nrows=100000)
df = df.drop(columns=['row_id', 'time_id'])
df['far_price']  = df['far_price'].fillna(df['reference_price'])
df['near_price'] = df['near_price'].fillna(df['reference_price'])

split_idx = int(len(df) * 0.8)
train_df  = df.iloc[:split_idx].copy()
val_df    = df.iloc[split_idx:].copy()
print(f"train: {len(train_df)}행 / val: {len(val_df)}행")

# ── 2. 수동 feature engineering ──
def add_features(df):
    df = df.copy()
    df['imbalance_momentum'] = df['imbalance_size'] * df['imbalance_buy_sell_flag']
    df['time_to_close']      = 540 - df['seconds_in_bucket']
    df['price_spread']       = df['ask_price'] - df['bid_price']
    df['mid_price']          = (df['ask_price'] + df['bid_price']) / 2
    df['wap_vs_mid']         = df['wap'] - df['mid_price']
    df['imbalance_ratio']    = df['imbalance_size'] / (df['matched_size'] + 1)
    df['log_imbalance']      = np.log1p(df['imbalance_size'])
    df['log_matched']        = np.log1p(df['matched_size'])
    return df

train_df = add_features(train_df)
val_df   = add_features(val_df)
print(f"수동 feature 후 컬럼 수: {train_df.shape[1]}개")

# ── 3. OpenFE 자동 feature engineering ──
train_X = train_df.drop(columns=['target'])
train_y = train_df['target']

ofe = OpenFE()
features = ofe.fit(
    data=train_X,
    label=train_y,
    n_jobs=1,
    n_data_blocks=8,
    min_candidate_features=500,
    stage2_params={'verbose': -1}
)

train_new, val_new = transform(
    train_X,
    val_df.drop(columns=['target']),
    features[:10],
    n_jobs=1
)
train_new['target'] = train_y.values
val_new['target']   = val_df['target'].values

print(f"OpenFE 후 컬럼 수: {train_new.shape[1]}개")
print("\nOpenFE 상위 5개 feature:")
for f in features[:5]:
    print(f"  {f.name}")

# ── 4. AutoGluon 학습 ──
predictor = TabularPredictor(
    label='target',
    eval_metric='mae',
    path=PATH + 'autogluon_openfe_model'
).fit(
    train_data=train_new,
    time_limit=300,
    presets='medium_quality',
    excluded_model_types=['CAT', 'NN_TORCH'],
)

# ── 5. 평가 ──
print("=" * 50)
print("모델별 성능 (MAE, 낮을수록 좋음)")
print("=" * 50)
leaderboard = predictor.leaderboard(val_new, silent=True)
print(leaderboard[['model', 'score_val']].to_string())

print("\n" + "=" * 50)
print("Feature 중요도 상위 15개")
print("=" * 50)
importance = predictor.feature_importance(val_new)
print(importance.head(15).to_string())

# ── AutoGluon 베이스라인 ──
from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

PATH = '/content/drive/MyDrive/optiver/'

# ── 1. 데이터 준비 ──────────────────────────────
# 코랩 무료라 전체 528만행 무리
# 10만행으로 train/val 나누기
df = pd.read_csv(PATH + 'train.csv', nrows=100000)

# 쓸모없는 컬럼 제거
drop_cols = ['row_id', 'time_id']
df = df.drop(columns=drop_cols)

# 결측치 간단 처리
# far/near_price → reference_price로 채우기
df['far_price'] = df['far_price'].fillna(df['reference_price'])
df['near_price'] = df['near_price'].fillna(df['reference_price'])

# 시계열이라 시간 순서 기준으로 split
# 앞 80% train, 뒤 20% val
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
val_df   = df.iloc[split_idx:]

print(f"train: {len(train_df)}행")
print(f"val:   {len(val_df)}행")

# ── 2. 기본 feature engineering ────────────────
def add_features(df):
    df = df.copy()

    # 방향 × 크기 (핵심 변수)
    df['imbalance_momentum'] = df['imbalance_size'] * df['imbalance_buy_sell_flag']

    # 마감 임박도
    df['time_to_close'] = 540 - df['seconds_in_bucket']

    # 호가 차이
    df['price_spread'] = df['ask_price'] - df['bid_price']

    # 호가 중간값
    df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2

    # wap vs 중간값 차이
    df['wap_vs_mid'] = df['wap'] - df['mid_price']

    # 불균형 비율
    df['imbalance_ratio'] = df['imbalance_size'] / (df['matched_size'] + 1)

    # 로그 변환 (치우친 분포 완화)
    df['log_imbalance'] = np.log1p(df['imbalance_size'])
    df['log_matched']   = np.log1p(df['matched_size'])

    return df

train_df = add_features(train_df)
val_df   = add_features(val_df)

print(f"feature 수: {train_df.shape[1]}개")

# ── 3. AutoGluon 학습 ───────────────────────────
predictor = TabularPredictor(
    label='target',        # 예측할 컬럼
    eval_metric='mae',     # 대회 평가지표
    path=PATH + 'autogluon_model'  # 모델 저장 경로
).fit(
    train_data=train_df,
    time_limit=300,        # 5분 제한 (코랩 무료 기준)
    presets='medium_quality',  # 코랩 무료: medium_quality
    # presets 옵션:
    # 'medium_quality' → 빠름, 메모리 적게
    # 'high_quality'   → 느림, 메모리 많이
    # 'best_quality'   → 가장 느림, 코랩 무료 무리
    excluded_model_types=['CAT', 'NN_TORCH'],
    # CAT(CatBoost), 딥러닝 제외
    # → 메모리 아끼려고
)

# ── 4. 평가 ────────────────────────────────────
print("\n" + "=" * 50)
print("검증 성능")
print("=" * 50)
leaderboard = predictor.leaderboard(val_df, silent=True)
print(leaderboard[['model', 'score_val']].to_string())

# ── 5. feature 중요도 ───────────────────────────
print("\n" + "=" * 50)
print("Feature 중요도 (상위 10개)")
print("=" * 50)
importance = predictor.feature_importance(val_df)
print(importance.head(10))
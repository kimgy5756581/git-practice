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

# ── 8. imbalance_size → target 관계 ──
print("=" * 50)
print("8. imbalance_size 크기별 target")
print("=" * 50)

# 0 제외하고 분석
df_nonzero = df[df['imbalance_size'] > 0].copy()

# 5구간으로 나누기
df_nonzero['imb_size_bin'] = pd.qcut(
    df_nonzero['imbalance_size'],
    5,
    labels=['최소', '소', '중', '대', '최대']
)

print("imbalance_size > 0인 경우만 (0 제외)")
display(df_nonzero.groupby('imb_size_bin')['target'].agg(['mean', 'std', 'count']))

# flag랑 size 조합
print("\nflag × size 조합:")
display(df_nonzero.groupby(
    ['imbalance_buy_sell_flag', 'imb_size_bin']
)['target'].mean().unstack())

# ── 9. seconds_in_bucket별 target 변동성 ──
print("=" * 50)
print("9. seconds_in_bucket별 target 변동성")
print("=" * 50)
display(df.groupby('seconds_in_bucket')['target'].agg(['mean', 'std', 'count']))

# ── 10. target과의 상관관계 ──
print("=" * 50)
print("10. target과의 상관관계")
print("=" * 50)

corr = df.select_dtypes(include=[np.number]).corr()['target'].sort_values()
display(corr)

# 히트맵으로 시각화
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(
    df.select_dtypes(include=[np.number]).corr(),
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0
)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
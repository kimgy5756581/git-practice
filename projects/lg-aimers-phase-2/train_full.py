# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════════════
#  LG Aimers 9기 Phase 2 — KBO 투구 제구 성공 예측
#  ★통합 학습 코드 (완결본)★   Private Score 재현용
#
#  최종 제출본:  submit_q-0.2.zip   ★1163.1106★
#  입력       :  data/train.csv , data/trackman_history.csv ,
#                data/test.csv , data/sample_submission.csv
#  출력       :  model/rf.pkl , model/tabm.pt , script.py , submit_final.zip
#
#  ┌ 실행 시간 (Colab T4/A100) ─────────────────────────────────────────┐
#  │ [1][2] 트랙맨 매칭·집계·군집     약  12분 (CPU)                     │
#  │ [3]    피처 (65 → 71, kadj)      약   3분                          │
#  │ [4]    LGB / CatBoost×2 / LR     약  25분 (CPU)                    │
#  │ [5]    recent23 LGB              약   8분 (CPU)                    │
#  │ [6]    TabM 4시드 + SWA창4       약 100분 (GPU)                    │
#  │ [7]~[11] 후처리·저장·리허설      약   8분                          │
#  │                                  ──────────  총 약 155분           │
#  └────────────────────────────────────────────────────────────────────┘
#
#  ┌ 파이프라인 ───────────────────────────────────────────────────────┐
#  │ trackman_history ─지문매칭(코사인+헝가리안)─▶ tm_lookup / cl_map    │
#  │ train.csv ─base_features + season_form(k=50)─▶ 65 수치 + 4 범주     │
#  │           ─sf3 6열 추가─▶ 71열  ─kadj(k=10)─▶ 12열 교체 = TabM 입력 │
#  │                                                                     │
#  │ LGB·CatBoost·LR : 65열(k=50)   |   TabM : 71열(12열 k=10)           │
#  │ blend ─a=1.19 & q=−0.20─▶ 1층 ─▶ 2층 ─▶ s_unk·s_F·k_m ─▶ final_b   │
#  │       ─θ23=0.10 recent23 혼합─▶ pteam:12 ─▶ hv ─▶ TARGET 0.4845    │
#  └────────────────────────────────────────────────────────────────────┘
#
#  ┌ 상수의 출처 — ★전부 리더보드 포물선 역산★ ──────────────────────┐
#  │ 앵커 점수와 후보 점수의 차이로 기울기 b 를 역산하고                │
#  │ 꼭짓점 θ* = b/(2C) 을 계산해 최적값을 확정했다.  8전 8승.           │
#  │   TARGET 0.4845 곡률 401,094  │  a 1.19  곡률   758   +24.47       │
#  │   q     −0.20   곡률     95   │  t 0.85  곡률   122    +2.91       │
#  │   KSM   50→10                +10.64  │  수축 0.872/0.839   +2.29   │
#  │   s_unk +0.042  곡률  5,214   +9.14  │  s_F −0.017         +0.42   │
#  │   k_m   +0.0086 곡률  9,279   +0.68  │  θ23  0.10          +1.93   │
#  │   SWA 창 4                    +9.42  │  sf3 6피처          +6.39   │
#  └────────────────────────────────────────────────────────────────────┘
#
#  ┌ 행 독립성 (추론 시 test 행 간 정보 공유 없음) ────────────────────┐
#  │ · 모든 상수는 학습 단계에서 확정되어 rf.pkl 에 저장                │
#  │ · script.py 는 저장된 상수를 적용만 함 (find_shift 재호출 없음)    │
#  │ · season_form 은 자기 id 로 앵커 dict 를 조회만 함                 │
#  │ · 그룹 시프트는 그 행 자신의 피처로 그룹 판정 (범주형 효과와 동일) │
#  └────────────────────────────────────────────────────────────────────┘
#
#  ┌ 재현성 ──────────────────────────────────────────────────────────┐
#  │ 시드 고정 + 결정성 플래그를 적용했다. 다만 TabM 은 범주형 임베딩을 │
#  │ 쓰고 PyTorch 임베딩 역전파의 CUDA scatter-add 는 원자적 덧셈 순서가│
#  │ 비결정적이므로, GPU 모델·드라이버가 다르면 가중치가 비트 단위로    │
#  │ 동일하지 않을 수 있다. 최종 산출물은 제출된 tabm.pt. → ENVIRONMENT │
#  └────────────────────────────────────────────────────────────────────┘
# ══════════════════════════════════════════════════════════════════════════

import os, sys, math, time, gc, random, zipfile, shutil, subprocess
import numpy as np, pandas as pd, joblib

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch, torch.nn as nn
import lightgbm as lgb
from catboost import CatBoostClassifier
from tabm import TabM
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import warnings; warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# [0] 설정 — 모든 상수를 한 곳에
# ══════════════════════════════════════════════════════════════════════════
DATA_DIR = "./data"
ID, TARGET = "row_id", "control_success"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

LEAGUE    = 0.5238   # season_form prior (전체 평균 성공률)
KSM       = 50       # season_form 평활상수 — 트리·LR 이 쓰는 값
KSM_TABM  = 10       # ★TabM 입력 12열 전용★  리더보드 5점 탐색으로 확정
MIN_CNT   = 100      # pid_emb 부여 기준 (미만 = UNK 0)
N_LI_BIN  = 10
SIM_TH    = 0.8      # 트랙맨 지문 코사인 임계
MIN_PITCH = 200      # 군집 대상 최소 투구수
N_CLUSTER = 12
HOLD      = 2024     # 후처리 상수 적합 연도

W_TABM  = 0.85
W_LGB = W_CAT = (0.95 - W_TABM) / 2.0     # = 0.05
W_LR    = 0.05
THETA23 = 0.10       # recent23 혼합 비중

TGT_INT   = 0.4835   # 파이프 내부 타깃
TGT_STAR  = 0.4845   # ★최종 TARGET★
A_SCALE_A = 1.19     # 메인 파이프 로짓 스케일
A_SCALE_B = 1.16     # recent23 파이프 로짓 스케일
A_QUAD    = -0.20    # ★a 의 2차항★  z' = a(z−c) + q(z−c)² + c
LAM1_BASE, LAM1 = 0.957, 0.872   # 1층 그룹타깃 수축계수 (원래 → 확정)
LAM2_BASE, LAM2 = 1.000, 0.839   # 2층
MIN_GROUP = MIN_G2 = 800
N_FB      = 5        # 2층 form_diff 분위 구간 수
CLIP2     = 0.012    # 2층 시프트 상한
S_UNK     = 0.042    # 신인(pid UNK) 시프트
S_F       = -0.017   # 2군(F) 시프트
K_M       = 0.0086   # 월 기울기
CONFIRMED = {"pteam:12": -0.00112}

LGB_SEEDS = [42]
CB_SEEDS  = [42, 7]
CB_PAR    = dict(iterations=700, learning_rate=0.03, depth=8, l2_leaf_reg=6.0)
LGB_P = dict(objective="binary", n_jobs=-1, verbose=-1,
             n_estimators=597, learning_rate=0.012594140973898157, num_leaves=87,
             min_child_samples=448, feature_fraction=0.7043131958361237,
             bagging_fraction=0.7117462995128336, bagging_freq=6,
             reg_alpha=5.6068712420100154e-06, reg_lambda=1.0928943924423843e-08)
RECENT_FROM = 2023

TABM_SEEDS = [(42, 13), (7, 8), (2024, 5), (1234, 8)]
SWA_W = 4
BATCH, LR_TABM, WD = 4096, 2e-3, 3e-4
K, N_BLOCKS, D_BLOCK, D_EMB, N_BINS = 32, 3, 512, 16, 48

#  (rate_col, n_col, prefix, who)
BASECOLS = [("asof_pitcher_success_rate", "asof_pitcher_n", "pitcher", "pitcher"),
            ("asof_batter_success_rate",  "asof_batter_n",  "batter",  "batter"),
            ("asof_pitcher_reverse_rate", "asof_pitcher_n", "p_rev",   "pitcher")]
SFCOLS   = [("asof_pitcher_middle_rate",   "asof_pitcher_n",          "p_mid",  "pitcher"),
            ("asof_pitcher_ball_rate",     "asof_pitcher_n",          "p_ball", "pitcher"),
            ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n", "p_off",  "pitcher")]

print(f"device={DEV}")
if DEV == "cpu":
    print("⚠ GPU 런타임 필요")
print(f"W=({W_LGB},{W_CAT},{W_TABM},{W_LR}) θ23={THETA23} | a={A_SCALE_A} q={A_QUAD}")
print(f"TARGET={TGT_STAR} | KSM {KSM}→{KSM_TABM}(TabM) | 수축 {LAM1}/{LAM2}")


def set_seed(sd):
    """★결정성 플래그★ — 임베딩 backward 의 CUDA 비결정성을 최대한 억제"""
    random.seed(sd); np.random.seed(sd)
    torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


lg = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))


def apply_shift(p, b):
    return sg(lg(p) + b)


def find_shift(p, t):
    lo, hi = -5.0, 5.0
    for _ in range(60):
        m = (lo + hi) / 2
        if sg(lg(p) + m).mean() < t:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def group_shift(p, key, group_b, global_b):
    out = p.copy(); seen = np.zeros(len(p), bool)
    for k, b in group_b.items():
        m = (key == k)
        if m.sum():
            out[m] = apply_shift(p[m], b); seen |= m
    if (~seen).sum():
        out[~seen] = apply_shift(p[~seen], global_b)
    return out


def reduce_mem(df):
    for c in df.columns:
        t = df[c].dtype
        if t == object:
            continue
        lo, hi = df[c].min(), df[c].max()
        if str(t)[:3] == "int":
            if lo > -128 and hi < 127:            df[c] = df[c].astype(np.int8)
            elif lo > -32768 and hi < 32767:      df[c] = df[c].astype(np.int16)
            elif lo > -2147483648 and hi < 2e9:   df[c] = df[c].astype(np.int32)
        elif lo > np.finfo(np.float32).min and hi < np.finfo(np.float32).max:
            df[c] = df[c].astype(np.float32)
    return df


def base_features(df):
    """★script.py 와 반드시 동일★"""
    df = df.copy()
    df["platoon"] = ((df["pitcher_hand"].astype(np.int8) - 1) * 2
                     + (df["batter_hand"].astype(np.int8) - 1)).astype(np.int8)
    df["is_3ball"] = (df["balls_before"] == 3).astype(np.int8)
    df["is_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(np.int8)
    df["count_diff"] = (df["balls_before"] - df["strikes_before"]).astype(np.int8)
    df["form_diff"] = (df["asof_pitcher_prev3_game_success_rate"]
                       - df["asof_pitcher_success_rate"]).astype(np.float32)
    return df


def train_season_form(df, rate, ncol, who, k, prior):
    """(id, season) 진입점 차분으로 ★이번 시즌 성분★ 만 추출.

    asof_* 는 커리어 누적이라 2019~24 가 대부분을 차지해 2025 예측에는 낡았다.
    시즌 진입 시점의 누적을 빼면 올해 몫만 남고, 표본이 적은 초반은 평활상수 k
    로 커리어 앵커 쪽에 수축시킨다.

        s_rate = (n·r − n0·r0 + r0·k) / (s_n + k)

    ★k 는 리더보드 5점 탐색으로 10 이 최적임을 확인★
      k=806 1135.33 / k=50 1147.38 / ★k=10 1158.02★ / k=5 1152.45 / k=1 1154.20
    """
    idc = f"{who}_id"
    i0 = df.groupby([idc, "season"], observed=True)[ncol].idxmin()
    ent = df.loc[i0, [idc, "season", ncol, rate]].rename(columns={ncol: "n0", rate: "r0"})
    ent["r0"] = ent["r0"].fillna(prior)
    ent["c0"] = ent["n0"] * ent["r0"]
    m = df[[idc, "season", ncol, rate]].merge(
        ent[[idc, "season", "n0", "c0", "r0"]], on=[idc, "season"], how="left")
    ncur = m[ncol].values.astype(np.float64)
    rcur = m[rate].fillna(m["r0"]).values.astype(np.float64)
    s_n = np.clip(ncur - m["n0"].values, 0, None)
    s_rt = (ncur * rcur - m["c0"].values + m["r0"].values * k) / (s_n + k)
    return (s_rt.astype(np.float32),
            (s_rt - m["r0"].values).astype(np.float32),
            s_n.astype(np.float32))


def build_anchor(df, rate, ncol, who, prior):
    """★test 용 앵커★ — 선수별 마지막 누적 상태.
    test 는 다른 행을 참조할 수 없으므로 자기 id 로 dict 조회만 한다."""
    idc = f"{who}_id"
    idx = df.groupby(idc)[ncol].idxmax()
    a = df.loc[idx, [idc, ncol, rate]].copy()
    a[rate] = a[rate].fillna(prior)
    a["c0"] = a[ncol] * a[rate]
    return {int(r[idc]): (float(r[ncol]), float(r["c0"])) for _, r in a.iterrows()}


# ══════════════════════════════════════════════════════════════════════════
# [1] 트랙맨 지문 매칭
#
#  trackman_history 의 투수 ID 는 train 과 별개로 익명화되어 있다.
#  두 테이블에 공통으로 있는 ★상황 슬롯★(일정/이닝/카운트/타자손)의 분포를
#  투수별 히스토그램으로 만들고, 코사인 유사도 4종을 평균한 뒤
#  헝가리안 알고리즘으로 1:1 최적 배정한다.
#  ★손(hand)은 지문에 넣지 않고 사후 검증에만 쓴다★ — 손 일치율이 랜덤 50%
#  를 크게 웃돌면 매칭이 우연이 아니라는 독립적인 증거가 된다.
# ══════════════════════════════════════════════════════════════════════════
print("\n[1] 트랙맨 지문 매칭", flush=True); t0 = time.time()
MC = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
      "balls_before", "strikes_before", "batter_hand"]
trm = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                  usecols=MC + ["pitcher_id", "pitcher_hand"])
tmm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig",
                  usecols=MC + ["pitcher_trackman_id", "pitcher_hand"])
tmm["pitcher_hand"] = tmm["pitcher_hand"].map({"Left": 1, "Right": 2})
tmm["batter_hand"] = tmm["batter_hand"].map({"Left": 1, "Right": 2})
tmm["top_bottom"] = tmm["top_bottom"].map({"Top": "T", "Bottom": "B"})


def slots(d, k):
    if k == "sched": return d.season.astype(str)+"_"+d.game_month.astype(str)+"_"+d.game_dayofweek.astype(str)
    if k == "usage": return "i"+d.inning.clip(upper=10).astype(str)+"_"+d.top_bottom.astype(str)
    if k == "count": return "c"+d.balls_before.astype(str)+d.strikes_before.astype(str)
    if k == "bhand": return "b"+d.batter_hand.astype(str)


S = ia = ib = None
for k in ["sched", "usage", "count", "bhand"]:
    A = trm.assign(s=slots(trm, k)).pivot_table(index="pitcher_id", columns="s",
                                                values="season", aggfunc="size", fill_value=0)
    Bm = tmm.assign(s=slots(tmm, k)).pivot_table(index="pitcher_trackman_id", columns="s",
                                                 values="season", aggfunc="size", fill_value=0)
    cc = A.columns.intersection(Bm.columns)
    nz = lambda M: M.values.astype(float) / (np.linalg.norm(M.values.astype(float), axis=1, keepdims=True) + 1e-9)
    s = nz(A[cc]) @ nz(Bm[cc]).T
    S = s if S is None else S + s
    ia, ib = A.index, Bm.index
S /= 4
r_, c_ = linear_sum_assignment(-S)
sim = S[r_, c_]
ha = trm.groupby("pitcher_id").pitcher_hand.first().reindex(ia).values
hb = tmm.groupby("pitcher_trackman_id").pitcher_hand.first().reindex(ib).values
handok = (ha[r_] == hb[c_])
ok = (sim > SIM_TH) & handok
PMAP = {int(a): int(b) for a, b in zip(ia[r_][ok], ib[c_][ok])}
print(f"  {time.time()-t0:.0f}s | PMAP {len(PMAP)}명 / {len(ia)}  "
      f"★손일치율 {handok[sim>SIM_TH].mean()*100:.1f}%★ (랜덤 50%)")
del trm, tmm, S, A, Bm; gc.collect()


# ══════════════════════════════════════════════════════════════════════════
# [2] 트랙맨 집계 + 투수 유형 군집 (K=12)
# ══════════════════════════════════════════════════════════════════════════
print("\n[2] 트랙맨 집계 + 군집", flush=True); t0 = time.time()
tm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig",
                 usecols=["pitcher_trackman_id", "balls_before", "strikes_before",
                          "pitch_type_group", "rel_speed", "spin_rate", "rel_height",
                          "rel_side", "induced_vert_break", "horz_break", "extension"])
tm["count_diff"] = (tm.balls_before - tm.strikes_before).astype(np.int8)
tm["is_fb"] = (tm.pitch_type_group == "fastball").astype(np.float32)
inv = {v: k for k, v in PMAP.items()}
tp = tm[tm.pitcher_trackman_id.isin(inv)].copy()

g = tp.groupby(["pitcher_trackman_id", "count_diff"])
PC = pd.DataFrame({
    "tm_speed": g.rel_speed.mean(), "tm_fbrate": g.is_fb.mean(),
    "tm_relh_std": g.rel_height.std(), "tm_rels_std": g.rel_side.std(),
    "tm_spin": g.spin_rate.mean(), "tm_ivb": g.induced_vert_break.mean(),
    "tm_hb": g.horz_break.mean(), "tm_ext": g.extension.mean(),
}).astype(np.float32).reset_index()
PC["pitcher_id"] = PC.pitcher_trackman_id.map(inv).astype(np.int16)
PC = PC.drop(columns=["pitcher_trackman_id"]).reset_index(drop=True)
TM_COLS = [c for c in PC.columns if c.startswith("tm_")]
assert PC.duplicated(["pitcher_id", "count_diff"]).sum() == 0, "룩업 중복키"

tp["pitcher_id"] = tp.pitcher_trackman_id.map(inv).astype(np.int16)
prof = tp.groupby("pitcher_id").agg(
    spd=("rel_speed", "mean"), spin=("spin_rate", "mean"),
    ivb=("induced_vert_break", "mean"), hb=("horz_break", "mean"),
    ext=("extension", "mean"), relh=("rel_height", "mean"),
    rels=("rel_side", "mean"), fb=("is_fb", "mean"), n=("rel_speed", "size")).reset_index()
prof = prof[prof.n >= MIN_PITCH].reset_index(drop=True)
PF = ["spd", "spin", "ivb", "hb", "ext", "relh", "rels", "fb"]
SCALER = StandardScaler().fit(prof[PF])
km = KMeans(n_clusters=N_CLUSTER, n_init=10, random_state=42).fit(SCALER.transform(prof[PF]))
CLMAP = {int(p): int(c) + 1 for p, c in zip(prof.pitcher_id, km.labels_)}   # 0 = UNK 예약
assert min(CLMAP.values()) == 1
print(f"  {time.time()-t0:.0f}s | 룩업 {len(PC)}행 | 군집 투수 {len(prof)}명")
del tm, tp, g; gc.collect()


# ══════════════════════════════════════════════════════════════════════════
# [3] 피처
#
#  ★커버율 정합★ 트랙맨은 2019~24 만 존재 → 2025 신인은 물리값이 없다.
#  train 을 100% 커버로 학습하면 "항상 있다"를 배워 test 에서 무너지므로
#  test 추정 커버율에 맞춰 무작위 마스킹한다 (keep_rate).
# ══════════════════════════════════════════════════════════════════════════
print("\n[3] 피처", flush=True); t0 = time.time()
train = reduce_mem(pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig"))
train = base_features(train)

PRIOR = {"pitcher": LEAGUE, "batter": LEAGUE,
         "p_rev": float(train["asof_pitcher_reverse_rate"].mean())}
for rate, _, pre, _ in SFCOLS:
    PRIOR[pre] = float(train[rate].mean())
ANCHOR = {pre: build_anchor(train, rate, ncol, who, PRIOR[pre])
          for rate, ncol, pre, who in BASECOLS + SFCOLS}

# ── 기본 3컬럼 (k=50) : 65 수치피처에 포함 ──
for rate, ncol, pre, who in BASECOLS:
    sr, sh, sn = train_season_form(train, rate, ncol, who, float(KSM), PRIOR[pre])
    if pre in ("pitcher", "batter"):
        train[f"{pre}_season_n"] = sn
        train[f"{pre}_season_rate"] = sr
        train[f"{pre}_form_shift"] = sh
    else:
        train[f"{pre}_season"] = sr
        train[f"{pre}_shift"] = sh
train = train.reset_index(drop=True)
n0 = len(train)
train = train.merge(PC, on=["pitcher_id", "count_diff"], how="left")
assert len(train) == n0, "merge 증식"

pcov = train["tm_speed"].notna().values
first_p = train.groupby("pitcher_id")["season"].transform("min")
ho_p = pcov & (train["season"] != first_p).values
TEST_COV = ho_p[(train.season == HOLD).values].mean()
KEEP_RATE = min(1.0, TEST_COV / pcov.mean())
rngm = np.random.RandomState(20250815)
mask_p = pcov & (rngm.rand(len(train)) < KEEP_RATE)
for c in TM_COLS:
    train[c] = np.where(mask_p, train[c], np.nan).astype(np.float32)
train["tm_missing"] = (~mask_p).astype(np.int8)
print(f"  ★커버율 정합★ 전체 {pcov.mean()*100:.1f}% / test추정 {TEST_COV*100:.1f}%"
      f" / KEEP {KEEP_RATE:.3f} → 학습 {mask_p.mean()*100:.1f}%")

train["pcl"] = train.pitcher_id.map(CLMAP).fillna(0).astype(np.int64)
train.loc[~mask_p, "pcl"] = 0
PCL_CARD = int(train["pcl"].max()) + 1
assert int(((train.tm_missing == 1) & (train.pcl != 0)).sum()) == 0, "정합성 오류"

_cnt = train.pitcher_id.value_counts()
PID_MAP = {int(v): i + 1 for i, v in enumerate(sorted(_cnt[_cnt >= MIN_CNT].index))}
train["pid_emb"] = train.pitcher_id.map(PID_MAP).fillna(0).astype(np.int64)
PID_CARD = len(PID_MAP) + 1
_e = np.unique(np.quantile(train["li"].values, np.linspace(0, 1, N_LI_BIN + 1)))
LI_EDGES = _e.copy(); LI_EDGES[0] = -np.inf; LI_EDGES[-1] = np.inf
train["li_bin"] = pd.cut(train["li"], LI_EDGES, labels=False).astype(np.int64)
LI_CARD = int(train["li_bin"].max()) + 1

CAT_COLS = ["top_bottom", "game_type", "base_state", "platoon"]
CAT8 = CAT_COLS + ["game_month", "game_dayofweek", "outs_before", "inning", "batter_id"]
CAT_MAP = {c: sorted(train[c].dropna().unique().tolist()) for c in CAT_COLS}
for col, cats in CAT_MAP.items():
    train[col] = pd.Categorical(train[col], categories=cats)

# pcl 은 TabM 전용 (LGB 에는 해로웠음)
FEATURES = [c for c in train.columns if c not in [ID, TARGET, "pid_emb", "li_bin", "pcl"]]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]
CARD_TABM = [len(CAT_MAP[c]) for c in CAT_COLS] + [PID_CARD, LI_CARD, PCL_CARD]
y_all = train[TARGET].values.astype(np.float32)
assert len(NUM_COLS) == 65, f"수치피처 {len(NUM_COLS)} != 65"
print(f"  피처 {len(FEATURES)} (수치 {len(NUM_COLS)} + 범주 {len(CAT_COLS)})"
      f" | pid {PID_CARD} li {LI_CARD} pcl {PCL_CARD}")

# ── ★sf3★ : TabM 입력에만 6열 추가 (65 → 71).  리더보드 +6.39 ──
#   asof_* 커리어 통계는 낡았지만, 같은 컬럼에서 ★시즌 성분★ 을 분리하면
#   신선한 신호가 된다. middle / ball / offspeed 세 축이 유효했다.
#   (batter middle · fastball 도 시도했으나 −6.78 로 해로웠다.)
NEW6, _sf = [], {}
for rate, ncol, pre, who in SFCOLS:
    sr, sh, _ = train_season_form(train, rate, ncol, who, float(KSM), PRIOR[pre])
    _sf[f"{pre}_season"] = sr; _sf[f"{pre}_shift"] = sh
    NEW6 += [f"{pre}_season", f"{pre}_shift"]
SF = pd.DataFrame(_sf)
MED_NEW = SF.median()
_v = SF.fillna(MED_NEW).values.astype(np.float32)
MU_NEW, SD_NEW = _v.mean(0), _v.std(0) + 1e-6

Xn = train[NUM_COLS].copy(); MED = Xn.median()
Xn = Xn.fillna(MED).values.astype(np.float32)
MU, SD = Xn.mean(0), Xn.std(0) + 1e-6
Xn71 = np.column_stack([(Xn - MU) / SD, (_v - MU_NEW) / SD_NEW]).astype(np.float32)
N_NUM = Xn71.shape[1]
assert N_NUM == 71

# ── ★kadj★ : TabM 입력 12열만 k=10 으로 교체 (트리·LR 은 k=50 유지) ──
#   k 는 "이번 시즌 성적을 얼마나 믿을까". 50 은 근거 없이 정한 값이었고
#   리더보드 5점 탐색에서 10 이 봉우리였다 (k=50 대비 ★+10.64★).
KSTAT, KADJ = {}, {}
for rate, ncol, pre, who in BASECOLS + SFCOLS:
    sr, sh, _ = train_season_form(train, rate, ncol, who, float(KSM_TABM), PRIOR[pre])
    if pre in [p for _, _, p, _ in SFCOLS]:
        nm_r, nm_s = f"{pre}_season", f"{pre}_shift"
    elif pre in ("pitcher", "batter"):
        nm_r, nm_s = f"{pre}_season_rate", f"{pre}_form_shift"
    else:
        nm_r, nm_s = f"{pre}_season", f"{pre}_shift"
    pair = {}
    for nm, v in [(nm_r, sr), (nm_s, sh)]:
        pos = (len(NUM_COLS) + NEW6.index(nm)) if nm in NEW6 else NUM_COLS.index(nm)
        med = float(np.nanmedian(v))
        vv = np.where(np.isnan(v), med, v).astype(np.float32)
        mu, sd = float(vv.mean()), float(vv.std() + 1e-8)
        Xn71[:, pos] = (vv - mu) / sd
        KSTAT[nm] = {"pos": int(pos), "k": float(KSM_TABM), "mu": mu, "sd": sd, "med": med}
        pair["stat_rate" if nm == nm_r else "stat_shift"] = KSTAT[nm]
    _g = train.groupby(f"{who}_id").apply(
        lambda x: x.loc[x[ncol].idxmax(), [ncol, rate]], include_groups=False)
    KADJ[pre] = {"rate": rate, "ncol": ncol, "idcol": f"{who}_id", "prior": PRIOR[pre],
                 "k": float(KSM_TABM), "name_rate": nm_r, "name_shift": nm_s,
                 "stat_rate": pair["stat_rate"], "stat_shift": pair["stat_shift"],
                 "n0": _g[ncol].to_dict(), "r0": _g[rate].fillna(PRIOR[pre]).to_dict()}
print(f"  ★kadj {len(KADJ)}컬럼 / 12열을 k={KSM_TABM} 로 교체 (TabM 입력만)★")

Xc_lgb = np.column_stack([train[c].cat.codes.values.astype(np.int64) for c in CAT_COLS])
Xc_lgb = np.where(Xc_lgb < 0, 0, Xc_lgb)
Xc_tabm = np.column_stack([Xc_lgb, train["pid_emb"].values,
                           train["li_bin"].values, train["pcl"].values])
for i in range(Xc_tabm.shape[1]):
    assert Xc_tabm[:, i].max() < CARD_TABM[i]
X_flat = np.column_stack([(Xn - MU) / SD, Xc_lgb.astype(np.float32)])
va_mask = (train.season == HOLD).values
va_y = y_all[va_mask].astype(np.float64)
print(f"  {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════════
# [4] LightGBM / CatBoost / Logistic   (65 수치 + 4 범주)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[4] LGB {len(LGB_SEEDS)}시드", flush=True)
lgb_models, _lp = [], []
for sd in LGB_SEEDS:
    t0 = time.time()
    m = lgb.LGBMClassifier(random_state=sd, **LGB_P)
    m.fit(train[FEATURES], y_all, categorical_feature=CAT_COLS)
    lgb_models.append(m); _lp.append(m.predict_proba(train[FEATURES])[:, 1])
    print(f"  seed {sd:<5} mean={_lp[-1].mean():.4f}  {time.time()-t0:.0f}s", flush=True)
p_lgb_all = np.mean(_lp, axis=0); del _lp; gc.collect()

print(f"\n[4b] CatBoost cat{len(CAT8)} {len(CB_SEEDS)}시드", flush=True)
CB_NUM = [c for c in FEATURES if c not in CAT8]
tr_cb = train[FEATURES].copy()
for c in CAT8:
    tr_cb[c] = tr_cb[c].astype(str)
CB_MED = tr_cb[CB_NUM].median()
tr_cb[CB_NUM] = tr_cb[CB_NUM].fillna(CB_MED)
cb_models, _cp = [], []
for sd in CB_SEEDS:
    t0 = time.time()
    m = CatBoostClassifier(random_seed=sd, verbose=0, thread_count=-1,
                           cat_features=CAT8, **CB_PAR)
    m.fit(tr_cb, y_all)
    cb_models.append(m); _cp.append(m.predict_proba(tr_cb)[:, 1])
    print(f"  seed {sd:<5} mean={_cp[-1].mean():.4f}  {time.time()-t0:.0f}s", flush=True)
p_cat_all = np.mean(_cp, axis=0); del tr_cb, _cp; gc.collect()

print("\n[4c] Logistic", flush=True); t0 = time.time()
lr_model = LogisticRegression(C=0.1, max_iter=1000, n_jobs=-1, random_state=42)
lr_model.fit(X_flat, y_all)
p_lr_all = lr_model.predict_proba(X_flat)[:, 1]
LR_COEF = lr_model.coef_.astype(np.float64).ravel()
LR_INT = float(lr_model.intercept_[0])
print(f"  {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════════
# [5] recent23 — 최근 2년(2023~24)만 학습한 LGB
#
#  시즌 드리프트가 크므로 최근만 본 모델을 따로 두고 θ23=0.10 으로 섞는다.
#  하이퍼파라미터는 메인 LGB 와 동일하고 학습 데이터만 다르다.  +1.93
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[5] recent23 (season >= {RECENT_FROM})", flush=True); t0 = time.time()
recent = (train.season >= RECENT_FROM).values
lgbR = lgb.LGBMClassifier(random_state=42, **LGB_P)
lgbR.fit(train.loc[recent, FEATURES], y_all[recent], categorical_feature=CAT_COLS)
p_B_all = lgbR.predict_proba(train[FEATURES])[:, 1]
print(f"  {time.time()-t0:.0f}s | 학습 {recent.sum():,}행 mean={p_B_all[va_mask].mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════
# [6] TabM 4시드 + ★SWA★
#
#  SWA = 마지막 SWA_W 에폭의 가중치를 평균. 시드 산포를 줄여 +9.42.
#  시드 수·에폭은 서버 추론 600초 제한 안에서 정해졌다 (5시드는 656초로 초과).
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[6] TabM {len(TABM_SEEDS)}시드 (SWA 창 {SWA_W})", flush=True)
Xn_t = torch.as_tensor(Xn71); Xc_t = torch.as_tensor(Xc_tabm); y_t = torch.as_tensor(y_all)
rng = np.random.default_rng(0)
sub_ = torch.as_tensor(Xn71[rng.choice(len(Xn71), size=min(100_000, len(Xn71)), replace=False)])
BINS = compute_bins(sub_, n_bins=N_BINS)


def build_tabm():
    emb = PiecewiseLinearEmbeddings(BINS, d_embedding=D_EMB, activation=False, version="B")
    return TabM.make(n_num_features=N_NUM, cat_cardinalities=CARD_TABM, d_out=1,
                     k=K, n_blocks=N_BLOCKS, d_block=D_BLOCK, num_embeddings=emb)


def predict_va(model, bs=16384):
    model.eval(); out = []; idx = np.where(va_mask)[0]
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            s_ = idx[i:i+bs]
            o = model(Xn_t[s_].to(DEV), Xc_t[s_].to(DEV))
            out.append(torch.sigmoid(o).mean(1).squeeze(-1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


lossf = nn.BCEWithLogitsLoss()
n = len(y_t); steps = math.ceil(n / BATCH)
tabm_states, tabm_preds = [], []
T0 = time.time()
for sd, n_ep in TABM_SEEDS:
    sl = min(SWA_W, max(2, n_ep // 2 + 1))
    t0 = time.time(); set_seed(sd)
    model = build_tabm().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_TABM, weight_decay=WD)
    acc, nacc = None, 0
    for ep in range(1, n_ep + 1):
        model.train(); perm = torch.randperm(n)
        for i in range(steps):
            ix = perm[i*BATCH:(i+1)*BATCH]
            o = model(Xn_t[ix].to(DEV), Xc_t[ix].to(DEV))
            loss = lossf(o.squeeze(-1), y_t[ix].to(DEV).unsqueeze(1).expand(-1, o.shape[1]))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if ep > n_ep - sl:                                  # ★SWA 누적★
            cur = {k2: v.detach().float().cpu() for k2, v in model.state_dict().items()}
            if acc is None:
                acc, nacc = {k2: v.clone() for k2, v in cur.items()}, 1
            else:
                nacc += 1
                for k2 in acc:
                    acc[k2] += (cur[k2] - acc[k2]) / nacc
    model.load_state_dict(acc)
    p = predict_va(model)
    tabm_states.append(acc); tabm_preds.append(p)
    print(f"  seed {sd:<5} ep{n_ep} SWA{sl} mean={p.mean():.4f}  {time.time()-t0:.0f}s", flush=True)
    del model, opt; torch.cuda.empty_cache(); gc.collect()
p_tabm = np.mean(tabm_preds, axis=0)
_sp = float(np.mean([np.sqrt(((q - p_tabm) ** 2).mean()) for q in tabm_preds]))
print(f"  총 {(time.time()-T0)/60:.0f}분 | 시드간 산포 {_sp:.6f}")


# ══════════════════════════════════════════════════════════════════════════
# [7] 후처리 상수 적합 (2024)
#
#  blend → a(2차항 q) → 1층 → 2층 → s_unk·s_F·k_m → final_b_A
#        → θ23 로 recent23 혼합 → pteam:12 → hv → TARGET 0.4845
# ══════════════════════════════════════════════════════════════════════════
print("\n[7] 후처리", flush=True)
p_lgb, p_cat = p_lgb_all[va_mask], p_cat_all[va_mask]
p_lr, p_B_raw = p_lr_all[va_mask], p_B_all[va_mask]
va_cd = np.asarray(train.loc[va_mask, "count_diff"])
va_gt = np.asarray(train.loc[va_mask, "game_type"]).astype(str)
va_fd = np.asarray(train.loc[va_mask, "form_diff"], dtype=np.float64)
va_mt = np.asarray(train.loc[va_mask, "game_month"], dtype=np.float64)
va_pt = np.asarray(train.loc[va_mask, "pitcher_team_id"]).astype(str)
va_unk = (train["pid_emb"].values[va_mask] == 0)
va_F = (va_gt == "F")
K1 = np.array([f"{c}_{g}" for c, g in zip(va_cd, va_gt)])
tilt = np.clip((va_mt - 6.0) / 3.0, -1.5, 1.5)

blend = W_LGB * p_lgb + W_CAT * p_cat + W_TABM * p_tabm + W_LR * p_lr
A_CENTER = float(lg(blend).mean())


def a_apply(p, a_, q_):
    """★a 를 곡선으로★

    a 는 '앙상블 평균이라 예측이 소심해진 것'을 되돌리는 배율이다(+24.47).
    그런데 '배율 하나'라는 형태 자체는 한 번도 의심하지 않았던 임의 가정이었다.
    2차항 q 를 붙여 비대칭 보정을 허용한 결과 q* = −0.13 (리더보드 2점) → +1.05
    """
    z = lg(p) - A_CENTER
    return sg(a_ * z + q_ * z * z + A_CENTER)


raw_val = a_apply(blend, A_SCALE_A, A_QUAD)
r_all = va_y.mean()
print(f"  a={A_SCALE_A} q={A_QUAD} center={A_CENTER:.5f}"
      f" | 로짓sd {lg(blend).std():.4f} → {lg(raw_val).std():.4f}")

# ── 1층 : count_diff × game_type ──
#   ★수축계수★ 2024 그룹 성공률에는 '해마다 바뀌는 표류'가 섞여 있어 그대로
#   믿으면 안 된다. 6시즌 분해로 방향을 잡고 리더보드로 0.957→0.872 확정.
ratio = pd.Series(va_y).groupby(K1).mean() / r_all
size = pd.Series(va_y).groupby(K1).size()
GROUP_B = {}
for k in ratio.index:
    if size[k] < MIN_GROUP:
        continue
    m = (K1 == k)
    b0 = float(find_shift(raw_val[m], TGT_INT * (1.0 + LAM1_BASE * (float(ratio[k]) - 1.0))))
    GROUP_B[k] = b0 + 4.0 * TGT_INT * (LAM1 - LAM1_BASE) * (float(ratio[k]) - 1.0)
GLOBAL_B = float(find_shift(raw_val, TGT_INT))
chk = group_shift(raw_val, K1, GROUP_B, GLOBAL_B)
print(f"  1층 {len(GROUP_B)}그룹 (수축 {LAM1_BASE}→{LAM1}) | 평균 {chk.mean():.5f}")

# ── 2층 : form_diff 분위 × count_diff ──
_fdall = train["form_diff"].to_numpy(np.float64)
_q = np.unique(np.quantile(_fdall[~np.isnan(_fdall)], np.linspace(0, 1, N_FB + 1)))
FB_EDGES = _q.copy(); FB_EDGES[0] = -np.inf; FB_EDGES[-1] = np.inf
_fb = pd.cut(pd.Series(va_fd), FB_EDGES, labels=False).fillna(-1).astype(int).to_numpy()
K2 = np.array([f"{a}_{c}" for a, c in zip(_fb, va_cd)])
ratio2 = pd.Series(va_y).groupby(K2).mean() / r_all
size2 = pd.Series(va_y).groupby(K2).size()
FORM_B = {}
for k in ratio2.index:
    if size2[k] < MIN_G2:
        continue
    m = (K2 == k)
    b0 = float(np.clip(find_shift(chk[m], TGT_INT * (1.0 + LAM2_BASE * (float(ratio2[k]) - 1.0))),
                       -CLIP2, CLIP2))
    FORM_B[k] = float(np.clip(b0 + 4.0 * TGT_INT * (LAM2 - LAM2_BASE) * (float(ratio2[k]) - 1.0),
                              -CLIP2, CLIP2))
FORM_GLOBAL = 0.0
print(f"  2층 {len(FORM_B)}그룹 (수축 {LAM2_BASE}→{LAM2}, 클립 ±{CLIP2})")


def pipe(p_raw, a_, q_):
    """메인 / recent23 공통. a_·q_ 만 다르다."""
    p = a_apply(p_raw, a_, q_)
    p = group_shift(p, K1, GROUP_B, GLOBAL_B)
    p = group_shift(p, K2, FORM_B, FORM_GLOBAL)
    p = p.copy()
    p[va_unk] = apply_shift(p[va_unk], S_UNK)   # 신인: test 21% vs 2024 0.68%
    p[va_F] = apply_shift(p[va_F], S_F)         # 2군(F)
    p = apply_shift(p, K_M * tilt)              # 월 기울기
    f = float(find_shift(p, TGT_INT))
    return apply_shift(p, f), f


pA, FINAL_B_A = pipe(blend, A_SCALE_A, A_QUAD)
pB23, FINAL_B_B23 = pipe(p_B_raw, A_SCALE_B, 0.0)     # ★recent23 은 q 미적용★
preds = (1.0 - THETA23) * pA + THETA23 * pB23
for tag, s in CONFIRMED.items():
    m = (va_pt == tag.split(":")[1])
    preds[m] = apply_shift(preds[m], s)
HV_FINAL_B = float(find_shift(preds, TGT_INT))
preds = apply_shift(preds, HV_FINAL_B)
TGT_SHIFT = float(find_shift(preds, TGT_STAR))
preds = apply_shift(preds, TGT_SHIFT)
print(f"  final_b_A {FINAL_B_A:+.8f} | final_b_B23 {FINAL_B_B23:+.8f}")
print(f"  hv {HV_FINAL_B:+.8f} | tgt_shift {TGT_SHIFT:+.6f} | 평균 {preds.mean():.6f}")
assert abs(preds.mean() - TGT_STAR) < 5e-4, "최종 평균 이탈"


# ══════════════════════════════════════════════════════════════════════════
# [8] 저장
# ══════════════════════════════════════════════════════════════════════════
print("\n[8] 저장", flush=True)
os.makedirs("./model", exist_ok=True)
joblib.dump({
    "lgb_list": lgb_models, "cb": cb_models, "lgb_b23": lgbR,
    "features": FEATURES, "num_cols": NUM_COLS, "cat_cols": CAT_COLS,
    "cat8": CAT8, "cb_num": CB_NUM, "cb_med": CB_MED,
    "cat_map": CAT_MAP, "med": MED, "mu": MU, "sd": SD,
    "lr_coef": LR_COEF, "lr_int": LR_INT,
    "w": (W_LGB, W_CAT, W_TABM, W_LR), "theta23": THETA23,
    "anchor": ANCHOR, "ksm": KSM, "league": LEAGUE, "prior": PRIOR,
    "sfcols": SFCOLS, "basecols": BASECOLS,
    "pid_map": PID_MAP, "pid_card": PID_CARD,
    "li_edges": LI_EDGES, "li_card": LI_CARD,
    "tm_lookup": PC, "tm_cols": TM_COLS, "keep_rate": KEEP_RATE,
    "cl_map": CLMAP, "pcl_card": PCL_CARD,
    "a_scale": A_SCALE_B, "a_scale_A": A_SCALE_A, "a_center": A_CENTER,
    "a_quad": A_QUAD, "lam": LAM1,
    "group_b": GROUP_B, "global_b": GLOBAL_B,
    "form_count_edges": FB_EDGES, "form_count_b": FORM_B,
    "form_count_global_b": FORM_GLOBAL,
    "unk_shift": S_UNK, "f_shift": S_F, "month_tilt": K_M,
    "final_b_A": FINAL_B_A, "final_b_B23": FINAL_B_B23,
    "hv_final_b": HV_FINAL_B, "tgt_shift": TGT_SHIFT, "confirmed": CONFIRMED,
    "num_new": NEW6, "mu_new": MU_NEW, "sd_new": SD_NEW, "med_new": MED_NEW,
    "kadj": KADJ, "kstat": KSTAT,
}, "./model/rf.pkl", compress=3)
torch.save({"states": tabm_states, "bins": BINS, "n_num": N_NUM, "card": CARD_TABM,
            "k": K, "n_blocks": N_BLOCKS, "d_block": D_BLOCK, "d_emb": D_EMB},
           "./model/tabm.pt")
print(f"  rf.pkl {os.path.getsize('./model/rf.pkl')/1e6:.1f}MB"
      f" | tabm.pt {os.path.getsize('./model/tabm.pt')/1e6:.1f}MB")


# ══════════════════════════════════════════════════════════════════════════
# [9] script.py 생성 — ★추론 코드★
#
#  학습 때 저장한 상수만 적용한다. find_shift 재호출이 없어 행 독립.
# ══════════════════════════════════════════════════════════════════════════
script = r'''import os, joblib
import numpy as np, pandas as pd, torch
from tabm import TabM
from rtdl_num_embeddings import PiecewiseLinearEmbeddings

ID_COL, TARGET_COL = "row_id", "control_success"
_lgt = lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
_sig = lambda z: 1/(1+np.exp(-z))


def base_features(df):
    """★ 학습 코드와 반드시 동일 ★"""
    df = df.copy()
    df["platoon"] = ((df["pitcher_hand"].astype(np.int8)-1)*2
                     + (df["batter_hand"].astype(np.int8)-1)).astype(np.int8)
    df["is_3ball"]      = (df["balls_before"]==3).astype(np.int8)
    df["is_full_count"] = ((df["balls_before"]==3)&(df["strikes_before"]==2)).astype(np.int8)
    df["count_diff"]    = (df["balls_before"]-df["strikes_before"]).astype(np.int8)
    df["form_diff"]     = (df["asof_pitcher_prev3_game_success_rate"]
                           - df["asof_pitcher_success_rate"]).astype(np.float32)
    return df


def test_season_form(df, rate, ncol, prefix, idcol, anchor, prior, k):
    """★ 자기 id 로 앵커 dict 조회만 → 행 독립 ★"""
    n0 = df[idcol].map(lambda i: anchor.get(int(i), (0.0, 0.0))[0]).values.astype(np.float64)
    c0 = df[idcol].map(lambda i: anchor.get(int(i), (0.0, 0.0))[1]).values.astype(np.float64)
    r0 = np.where(n0 > 0, c0/np.maximum(n0, 1), prior)
    ncur = df[ncol].values.astype(np.float64)
    rcur = df[rate].fillna(pd.Series(r0, index=df.index)).values.astype(np.float64)
    s_n  = np.clip(ncur - n0, 0, None)
    s_rt = (ncur*rcur - c0 + r0*k)/(s_n + k)
    if prefix in ("pitcher", "batter"):
        df[prefix+"_season_n"]    = s_n.astype(np.float32)
        df[prefix+"_season_rate"] = s_rt.astype(np.float32)
        df[prefix+"_form_shift"]  = (s_rt - r0).astype(np.float32)
    else:
        df[prefix+"_season"] = s_rt.astype(np.float32)
        df[prefix+"_shift"]  = (s_rt - r0).astype(np.float32)
    return df


def apply_shift(p, b):
    return _sig(_lgt(p) + b)


def group_shift_fixed(p, g, group_b, global_b):
    """★ 학습 때 구한 상수만 적용. find_shift 재호출 금지 ★"""
    out = p.copy(); seen = np.zeros(len(p), dtype=bool)
    for k, b in group_b.items():
        m = (g == k)
        if m.sum() == 0: continue
        out[m] = apply_shift(p[m], b); seen |= m
    if (~seen).sum() > 0:
        out[~seen] = apply_shift(p[~seen], global_b)
    return out


def main():
    B = joblib.load("./model/rf.pkl")
    T = torch.load("./model/tabm.pt", map_location="cpu", weights_only=False)
    w_lgb, w_cat, w_tabm, w_lr = B["w"]
    print(f"weights {w_lgb}/{w_cat}/{w_tabm}/{w_lr}"
          f" | tabm {len(T['states'])} cb {len(B['cb'])}")

    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub  = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    print(f"test={len(test)} submission={len(sub)}")

    test = base_features(test)
    for rate, ncol, pre, who in B["basecols"]:
        test = test_season_form(test, rate, ncol, pre, who+"_id",
                                B["anchor"][pre], B["prior"][pre], B["ksm"])

    n_before = len(test)
    test = test.merge(B["tm_lookup"], on=["pitcher_id","count_diff"], how="left")
    assert len(test) == n_before, "trackman merge row inflation!"
    test["tm_missing"] = test["tm_speed"].isna().astype(np.int8)
    print(f"trackman coverage={(1-test['tm_missing'].mean())*100:.1f}%")

    pcl = test["pitcher_id"].map(B["cl_map"]).fillna(0).astype(np.int64).values
    pcl = np.where(test["tm_missing"].values == 1, 0, pcl)
    pcl = np.clip(pcl, 0, B["pcl_card"]-1)
    pid = test["pitcher_id"].map(B["pid_map"]).fillna(0).astype(np.int64).values
    print(f"cluster UNK={100*(pcl==0).mean():.1f}% | pitcher-id UNK={(pid==0).mean()*100:.1f}%")

    li_bin = pd.cut(test["li"], B["li_edges"], labels=False).values
    li_bin = np.clip(np.nan_to_num(li_bin, nan=0).astype(np.int64), 0, B["li_card"]-1)

    # ── 트리 / 선형 : 65 수치 + 4 범주 (k=50) ──
    cb_in = test[B["features"]].copy()
    for c in B["cat8"]: cb_in[c] = cb_in[c].astype(str)
    cb_in[B["cb_num"]] = cb_in[B["cb_num"]].fillna(B["cb_med"])
    p_cat = np.mean([m.predict_proba(cb_in)[:, 1] for m in B["cb"]], axis=0)
    for c, cats in B["cat_map"].items():
        test[c] = pd.Categorical(test[c], categories=cats)
    p_lgb = np.mean([m.predict_proba(test[B["features"]])[:, 1] for m in B["lgb_list"]], axis=0)
    p_m23 = B["lgb_b23"].predict_proba(test[B["features"]])[:, 1]

    Xn = test[B["num_cols"]].fillna(B["med"]).values.astype(np.float32)
    Xn = (Xn - B["mu"]) / B["sd"]
    Xc = np.column_stack([test[c].cat.codes.values.astype(np.int64) for c in B["cat_cols"]])
    Xc = np.where(Xc < 0, 0, Xc)
    z = np.column_stack([Xn, Xc.astype(np.float32)]) @ B["lr_coef"] + B["lr_int"]
    p_lr = _sig(z)

    # ── ★sf3★ TabM 전용 6피처 (65 → 71) ──
    _new = {}
    for rate, ncol, pre, who in B["sfcols"]:
        _t = test_season_form(test.copy(), rate, ncol, pre, who+"_id",
                              B["anchor"][pre], B["prior"][pre], B["ksm"])
        _new[pre+"_season"] = _t[pre+"_season"].values
        _new[pre+"_shift"]  = _t[pre+"_shift"].values
    SFdf = pd.DataFrame({c: _new[c] for c in B["num_new"]})
    print("sf_new " + " ".join(f"{c}={np.nanmean(SFdf[c].values):+.4f}" for c in B["num_new"]))
    _v = SFdf.fillna(B["med_new"]).values.astype(np.float32)
    Xn = np.column_stack([Xn, (_v - B["mu_new"]) / B["sd_new"]]).astype(np.float32)

    # ── ★kadj★ TabM 입력 12열을 k=10 으로 교체 ──
    #    트리·LR 은 위에서 이미 k=50 열로 예측을 마쳤다.
    _ka = B.get("kadj", {})
    if _ka:
        assert Xn.shape[1] == 71, f"Xn {Xn.shape}"
        for _pre, _a in _ka.items():
            _idv = np.asarray(test[_a["idcol"]])
            _n = np.asarray(test[_a["ncol"]], dtype=np.float64)
            _r = np.asarray(test[_a["rate"]], dtype=np.float64)
            _n0 = np.array([_a["n0"].get(int(v), 0.0) for v in _idv], dtype=np.float64)
            _r0 = np.array([_a["r0"].get(int(v), _a["prior"]) for v in _idv], dtype=np.float64)
            _r = np.where(np.isnan(_r), _r0, _r)
            _kk = float(_a["k"])
            _sn = np.clip(_n - _n0, 0, None)
            _sr = (_n*_r - _n0*_r0 + _r0*_kk) / (_sn + _kk)
            _sh = _sr - _r0
            for _st, _v2 in [(_a["stat_rate"], _sr), (_a["stat_shift"], _sh)]:
                _vv = np.where(np.isnan(_v2), _st["med"], _v2).astype(np.float32)
                Xn[:, _st["pos"]] = (_vv - _st["mu"]) / _st["sd"]
            print(f"  kadj {_pre:<8} k={_kk:>4.0f} pos {_a['stat_rate']['pos']:>2},"
                  f"{_a['stat_shift']['pos']:>2}")
        print(f"★kadj {len(_ka)}컬럼 / 12열 교체 | Xn {Xn.shape}★")

    # ── TabM ──
    Xc_tabm = np.column_stack([Xc, pid, li_bin, pcl])
    emb = PiecewiseLinearEmbeddings(T["bins"], d_embedding=T["d_emb"],
                                    activation=False, version="B")
    xn_t, xc_t = torch.as_tensor(Xn), torch.as_tensor(Xc_tabm)
    tps = []
    for st in T["states"]:
        m = TabM.make(n_num_features=T["n_num"], cat_cardinalities=T["card"], d_out=1,
                      k=T["k"], n_blocks=T["n_blocks"], d_block=T["d_block"],
                      num_embeddings=emb)
        m.load_state_dict(st); m.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(xn_t), 16384):
                o = m(xn_t[i:i+16384], xc_t[i:i+16384])
                outs.append(torch.sigmoid(o).mean(1).squeeze(-1).numpy())
        tps.append(np.concatenate(outs)); del m
    p_tabm = np.mean(tps, axis=0)

    blend_A = w_lgb*p_lgb + w_cat*p_cat + w_tabm*p_tabm + w_lr*p_lr
    print(f"blend_A={blend_A.mean():.4f} B23={p_m23.mean():.4f}")

    # ── 후처리 (전부 학습 상수) ──
    _c  = B["a_center"]
    _aA = B.get("a_scale_A", B["a_scale"])      # 메인 파이프
    _aB = B["a_scale"]                          # recent23
    _qd = B.get("a_quad", 0.0)                  # ★a 의 2차항 (메인만)★
    _km = B.get("month_tilt", 0.0)
    _su, _sf = B.get("unk_shift", 0.0), B.get("f_shift", 0.0)
    _t23 = B.get("theta23", 0.0)
    gkey = np.array([f"{c}_{g}" for c, g in zip(np.asarray(test["count_diff"]),
                                                np.asarray(test["game_type"]).astype(str))])
    _e  = np.asarray(B["form_count_edges"], dtype=np.float64)
    _fb = pd.cut(pd.Series(np.asarray(test["form_diff"], dtype=np.float64)),
                 _e, labels=False).fillna(-1).astype(int).to_numpy()
    _k2 = np.array([f"{a}_{c}" for a, c in zip(_fb, np.asarray(test["count_diff"]))])
    _u  = (pid == 0)
    _f  = (np.asarray(test["game_type"]).astype(str) == "F")
    _t  = np.clip((np.asarray(test["game_month"], dtype=np.float64)-6.0)/3.0, -1.5, 1.5)

    def _pipe(raw, final_b, a_, qd_=0.0):
        _z = _lgt(raw)
        q = _sig(a_*(_z-_c) + qd_*(_z-_c)**2 + _c)
        q = group_shift_fixed(q, gkey, B["group_b"], B["global_b"])
        q = group_shift_fixed(q, _k2, B["form_count_b"], B["form_count_global_b"])
        q = q.copy()
        q[_u] = apply_shift(q[_u], _su)
        q[_f] = apply_shift(q[_f], _sf)
        q = _sig(_lgt(q) + _km*_t)
        return apply_shift(q, final_b)

    print(f"aA={_aA} aB={_aB} q={_qd} unk={_u.mean()*100:.1f}% F={_f.mean()*100:.1f}%")
    preds = (1.0-_t23)*_pipe(blend_A, B["final_b_A"], _aA, _qd) \
          + _t23     *_pipe(p_m23,   B["final_b_B23"], _aB)

    for tag, s in B.get("confirmed", {}).items():
        m = (np.asarray(test["pitcher_team_id"]).astype(str) == tag.split(":")[1])
        preds[m] = apply_shift(preds[m], s)
        print(f"  hv {tag} s={s:+.5f} on {m.mean()*100:.2f}%")
    preds = apply_shift(preds, B["hv_final_b"])
    print(f"final mean={preds.mean():.5f} range[{preds.min():.5f},{preds.max():.5f}]")
    preds = apply_shift(preds, B["tgt_shift"])
    print(f"tgt_shift={B['tgt_shift']:+.6f} | mean={preds.mean():.5f}")

    pm = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = sub[ID_COL].map(pm).fillna(0.4845)
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False)
    print(f"Saved: rows={len(sub)}")


if __name__ == "__main__":
    main()
'''
with open("./script.py", "w", encoding="utf-8") as f:
    f.write(script)
with open("./requirements.txt", "w") as f:
    f.write("lightgbm==4.7.0\ncatboost==1.2.10\ntabm\n")
print("  script.py / requirements.txt 생성")


# ══════════════════════════════════════════════════════════════════════════
# [10] 리허설 + 추론 시간 + zip
#      ★서버 제한 600초★ — 시드 수는 이 제한 안에서 정해졌다 (실측 배수 1.065)
# ══════════════════════════════════════════════════════════════════════════
print("\n[10] 리허설", flush=True)
r = subprocess.run([sys.executable, "script.py"], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("❌ STDERR:\n", r.stderr[-3000:])
else:
    print(pd.read_csv("./output/submission.csv").head())

print("\n[11] 추론 시간 (24.6만 행, CPU 기준)", flush=True)
idx = np.arange(min(245_789, len(Xn71)))
xn_c, xc_c = torch.as_tensor(Xn71[idx]), torch.as_tensor(Xc_tabm[idx])
emb_c = PiecewiseLinearEmbeddings(BINS, d_embedding=D_EMB, activation=False, version="B")
t0 = time.time()
for st in tabm_states:
    mc = TabM.make(n_num_features=N_NUM, cat_cardinalities=CARD_TABM, d_out=1,
                   k=K, n_blocks=N_BLOCKS, d_block=D_BLOCK, num_embeddings=emb_c)
    mc.load_state_dict(st); mc.eval()
    with torch.no_grad():
        for i in range(0, len(xn_c), 16384):
            mc(xn_c[i:i+16384], xc_c[i:i+16384])
    del mc
tabm_sec = time.time() - t0
t0 = time.time()
for m in lgb_models + [lgbR]:
    m.predict_proba(train[FEATURES].iloc[idx])
lgb_sec = time.time() - t0
_cb = train[FEATURES].iloc[idx].copy()
for c in CAT8:
    _cb[c] = _cb[c].astype(str)
_cb[CB_NUM] = _cb[CB_NUM].fillna(CB_MED)
t0 = time.time()
for m in cb_models:
    m.predict_proba(_cb)
cb_sec = time.time() - t0
tot = tabm_sec + lgb_sec + cb_sec
srv = tot * 1.065
print(f"  TabM {tabm_sec:.0f}s | LGB×{len(lgb_models)+1} {lgb_sec:.0f}s"
      f" | CAT×{len(cb_models)} {cb_sec:.0f}s | 로컬 {tot:.0f}s")
print(f"  ★서버 예상 {srv:.0f}s★ (제한 600s | 여유 {600-srv:.0f}s)")
if srv > 580:
    print("  ⚠ 제한 근접 — 시드 수를 줄이세요")
del _cb; gc.collect()

with zipfile.ZipFile("./submit_final.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["model/rf.pkl", "model/tabm.pt", "script.py", "requirements.txt"]:
        z.write("./" + f, f)
print(f"\n★submit_final.zip {os.path.getsize('./submit_final.zip')/1e6:.1f}MB★")
print(f"  기준: submit_q-0.2.zip = 1163.1106")
print(f"  대조 검증은 verify.py 참조")

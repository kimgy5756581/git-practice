# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════════════
#  verify.py — train_full.py 산출물을 ★제출본 submit_q-0.2.zip★ 과 대조
#
#  목적:  재구성 코드가 실제 제출물을 만들어내는지 항목별로 확인한다.
#  사용:  python verify.py   (model/rf.pkl 과 submit_q-0.2.zip 이 있어야 함)
#
#  ┌ 판정 기준 ───────────────────────────────────────────────────────┐
#  │  ✅ 결정적 산출물 — 완전 일치해야 함                                │
#  │     tm_lookup · cl_map · pid_map · li_edges · keep_rate            │
#  │     num_cols · cat_map · mu/sd/med · form_count_edges              │
#  │  ✅ 트리 계열 — 시드 고정이라 사실상 일치                            │
#  │     lr_coef · lr_int                                                │
#  │  ⚠ TabM 가중치 — CUDA scatter-add 비결정성으로 미세차 가능           │
#  │     그로 인해 group_b / final_b / tgt_shift 도 미세하게 달라질 수 있음│
#  └──────────────────────────────────────────────────────────────────┘
# ══════════════════════════════════════════════════════════════════════════
import os, zipfile, shutil, joblib, numpy as np, torch

NEW = "./model/rf.pkl"
REF_ZIP = "./submit_q-0.2.zip"
assert os.path.exists(NEW), f"{NEW} 없음 — train_full.py 를 먼저 실행"
assert os.path.exists(REF_ZIP), f"{REF_ZIP} 없음"

shutil.rmtree("./_ref", ignore_errors=True)
with zipfile.ZipFile(REF_ZIP) as z:
    z.extractall("./_ref")
A = joblib.load(NEW)                      # 재구성본
B = joblib.load("./_ref/model/rf.pkl")    # 제출본
TA = torch.load("./model/tabm.pt", map_location="cpu", weights_only=False)
TB = torch.load("./_ref/model/tabm.pt", map_location="cpu", weights_only=False)

OK = FAIL = 0


def rep(name, ok, detail=""):
    global OK, FAIL
    mark = "✅" if ok else "❌"
    print(f"  {mark} {name:<26}{detail}")
    if ok: OK += 1
    else:  FAIL += 1


print("═" * 62)
print("① 결정적 산출물 (완전 일치해야 함)")
print("═" * 62)
rep("num_cols", A["num_cols"] == B["num_cols"], f"{len(A['num_cols'])}열")
rep("cat_cols", A["cat_cols"] == B["cat_cols"])
rep("cat_map", A["cat_map"] == B["cat_map"])
rep("keep_rate", abs(A["keep_rate"] - B["keep_rate"]) < 1e-12,
    f"{A['keep_rate']:.6f} vs {B['keep_rate']:.6f}")
rep("pid_map", A["pid_map"] == B["pid_map"], f"{len(A['pid_map'])}명")
rep("cl_map", A["cl_map"] == B["cl_map"], f"{len(A['cl_map'])}명")
rep("li_edges", np.allclose(A["li_edges"], B["li_edges"]))
rep("form_count_edges", np.allclose(A["form_count_edges"], B["form_count_edges"]))
for k in ["mu", "sd"]:
    rep(k, np.allclose(np.asarray(A[k]), np.asarray(B[k]), atol=1e-6),
        f"최대차 {np.abs(np.asarray(A[k])-np.asarray(B[k])).max():.2e}")
rep("med", np.allclose(A["med"].values, B["med"].values, atol=1e-6))

# 트랙맨 룩업
pa = A["tm_lookup"].sort_values(["pitcher_id", "count_diff"]).reset_index(drop=True)
pb = B["tm_lookup"].sort_values(["pitcher_id", "count_diff"]).reset_index(drop=True)
same_shape = pa.shape == pb.shape
d = np.nanmax(np.abs(pa[A["tm_cols"]].values - pb[B["tm_cols"]].values)) if same_shape else float("nan")
rep("tm_lookup", same_shape and d < 1e-4, f"{pa.shape} 최대차 {d:.2e}")

print("\n" + "═" * 62)
print("② 트리 / 선형 (시드 고정 → 사실상 일치)")
print("═" * 62)
rep("lr_coef", np.allclose(A["lr_coef"], B["lr_coef"], atol=1e-5),
    f"최대차 {np.abs(A['lr_coef']-B['lr_coef']).max():.2e}")
rep("lr_int", abs(A["lr_int"] - B["lr_int"]) < 1e-5,
    f"{A['lr_int']:.8f} vs {B['lr_int']:.8f}")
rep("CatBoost 개수", len(A["cb"]) == len(B["cb"]), f"{len(A['cb'])} vs {len(B['cb'])}")
_la = A.get("lgb_list", [A.get("lgb")])
_lb = B.get("lgb_list", [B.get("lgb")])
rep("LGB 개수", len(_la) == len(_lb), f"{len(_la)} vs {len(_lb)}")

print("\n" + "═" * 62)
print("③ TabM 구조 / kadj")
print("═" * 62)
rep("n_num", TA["n_num"] == TB["n_num"], f"{TA['n_num']} vs {TB['n_num']}")
rep("card", list(TA["card"]) == list(TB["card"]))
rep("시드 수", len(TA["states"]) == len(TB["states"]),
    f"{len(TA['states'])} vs {len(TB['states'])}")
ka, kb = A.get("kadj", {}), B.get("kadj", {})
rep("kadj 컬럼 수", len(ka) == len(kb), f"{len(ka)} vs {len(kb)}")
rep("kadj k 값", {v["k"] for v in ka.values()} == {v["k"] for v in kb.values()},
    f"{sorted({v['k'] for v in ka.values()})}")
if ka and kb:
    dp = all(ka[p]["stat_rate"]["pos"] == kb[p]["stat_rate"]["pos"] for p in ka)
    rep("kadj 열 위치", dp)

print("\n" + "═" * 62)
print("④ 후처리 상수  (TabM 비결정성 영향 있음)")
print("═" * 62)
for k in ["a_scale_A", "a_scale", "a_center", "a_quad", "theta23",
          "unk_shift", "f_shift", "month_tilt",
          "final_b_A", "final_b_B23", "hv_final_b", "tgt_shift"]:
    if k not in A or k not in B:
        rep(k, False, "키 없음"); continue
    va, vb = float(A[k]), float(B[k])
    rep(k, abs(va - vb) < 1e-4, f"{va:+.8f} vs {vb:+.8f}  차 {abs(va-vb):.2e}")
rep("w", np.allclose(A["w"], B["w"]), f"{tuple(round(x,4) for x in A['w'])}")

# 그룹 시프트
for key in ["group_b", "form_count_b"]:
    ga, gb = A[key], B[key]
    same_keys = set(ga) == set(gb)
    dm = max(abs(ga[k] - gb[k]) for k in ga) if same_keys and ga else float("nan")
    rep(key, same_keys and dm < 1e-4,
        f"{len(ga)}그룹 최대차 {dm:.2e}" if same_keys else f"키 불일치 {len(ga)} vs {len(gb)}")

print("\n" + "═" * 62)
print("⑤ TabM 가중치 (참고 — 완전 일치는 기대하지 않음)")
print("═" * 62)
for i, (sa, sb) in enumerate(zip(TA["states"], TB["states"])):
    ks = [k for k in sa if sa[k].is_floating_point()]
    d = max(float((sa[k] - sb[k]).abs().max()) for k in ks)
    r = max(float((sa[k] - sb[k]).abs().max() / (sb[k].abs().max() + 1e-12)) for k in ks)
    print(f"  시드 {i}  최대차 {d:.3e}  상대 {r:.3e}"
          + ("   ★비트 일치★" if d == 0 else ""))

print("\n" + "═" * 62)
print(f"결과  ✅ {OK}  /  ❌ {FAIL}")
print("═" * 62)
if FAIL == 0:
    print("★모든 항목 일치 — 재구성 성공★")
else:
    print("불일치 항목을 확인하세요. TabM 비결정성이 원인이면 ④ 만 미세하게 어긋납니다.")
    print("①②③ 이 어긋나면 전처리/시드/데이터 경로를 점검해야 합니다.")

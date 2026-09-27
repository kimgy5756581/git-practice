# -*- coding: utf-8 -*-
"""ENVIRONMENT.md 를 채우기 위한 환경 정보 수집.  Colab 에서 그대로 실행."""
import sys, subprocess, platform


def sh(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


print("=" * 56)
print("실행 환경")
print("=" * 56)
print(f"  플랫폼    Google Colab")
print(f"  OS        {sh(chr(39).join([]) or 'grep PRETTY_NAME /etc/os-release').split('=')[-1].strip(chr(34))}")
print(f"  커널      {platform.release()}")
print(f"  Python    {sys.version.split()[0]}")
print(f"  GPU       {sh('nvidia-smi --query-gpu=name --format=csv,noheader') or '(없음)'}")
print(f"  드라이버  {sh('nvidia-smi --query-gpu=driver_version --format=csv,noheader')}")
print(f"  CUDA      {sh('nvcc --version | tail -1')}")

print("\n" + "=" * 56)
print("라이브러리")
print("=" * 56)
for name in ["torch", "numpy", "pandas", "scipy", "sklearn",
             "lightgbm", "catboost", "tabm", "rtdl_num_embeddings"]:
    try:
        m = __import__(name)
        v = getattr(m, "__version__", None)
        if v is None:
            v = sh(f"pip show {name} | grep ^Version") or "?"
        print(f"  {name:<22}{v}")
    except Exception as e:
        print(f"  {name:<22}(import 실패: {type(e).__name__})")

try:
    import torch
    print(f"\n  torch.version.cuda      {torch.version.cuda}")
    print(f"  torch.cuda.is_available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cuDNN                   {torch.backends.cudnn.version()}")
except Exception:
    pass

print("\n→ 위 값을 ENVIRONMENT.md 의 <...> 자리에 채워 넣으세요.")

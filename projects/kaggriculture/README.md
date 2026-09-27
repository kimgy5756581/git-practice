# Kaggriculture Strategy League

Kaggriculture 에이전트 전략을 같은 조건에서 비교하기 위한 로컬 리그 도구입니다. 단순 파라미터 변형이 아니라 서로 다른 의사결정 구조를 가진 전략을 별도 프로세스에서 실행해, 좌석을 교대하며 결과를 기록합니다.

## Included strategies

| ID | Core idea |
| --- | --- |
| `master` | Replay and repair baseline |
| `conditional-memory` | Opponent-state pattern matching and market-order adjustment |
| `two-coins` | Shop-dependent routes with a protected selling window |
| `v48` | Public-state route router and clone detection |
| `pure-architecture` | State-based production and economy rules |
| `agroboss` | Greedy live-board job scheduling |

`two-coins` is a multi-file bundle. Its bundled `LICENSE.txt` and `NOTICE.txt` are retained. Attribution notices embedded in strategy source files are also preserved.

## Setup

Python 3.10+ and the pinned engine are required.

```powershell
python -m pip install -r requirements.txt
```

```text
agents/          # Strategies included in the league
agent_host.py    # Isolated strategy process
worker.py        # One game runner
league.py        # Scheduling, snapshots, and result aggregation
manifest.json    # Strategy registry and seed sets
```

## Run

```powershell
# Unit tests for scheduling, pairing, and safe bundle extraction
python -m unittest -v test_league.py

# List registered strategies
.\run.ps1 --list

# Smoke round robin: one seed, both seats for every pairing
.\run.ps1 --seed-set smoke

# Compare a new standalone submission against the default pool
.\run.ps1 --candidate C:\path\to\main.py --seed-set development
```

Each game runs in its own engine process, and each agent runs in a separate process. This prevents module-level state from leaking between games. Results are written to `runs/<timestamp>/` and ignored by Git.

## Reading results correctly

- A seed played in both seats is one scenario, not two independent samples.
- Keep final cash, margin, status, error output, and code hash together.
- Local league results select research candidates; they do **not** prove Kaggle leaderboard rating or certify a submission package.
- Candidate code must still be checked separately against Kaggle's submission interface and time limits.

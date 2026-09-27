# Solo Adventure Maze

`Java`와 CSV 맵 파일로 만든 텍스트 기반 미로 탐험 게임입니다. 방을 이동하며 무기와 물약을 얻고, 트롤에게서 열쇠를 획득해 마스터 문으로 탈출하는 것이 목표입니다.

## Run

JDK 17 이상에서 프로젝트 루트에서 실행합니다.

```bash
javac -encoding UTF-8 -d out src/itm/comlang/*.java
java -cp out itm.comlang.Main
```

## Controls

| Key | Action |
| --- | --- |
| `u` / `d` / `l` / `r` | Move up / down / left / right |
| `a` | Attack an adjacent monster |
| `q` | Quit |

## Map symbols

`S/W/X`는 무기, `G/O/T`는 고블린·오크·트롤, `m/B`는 물약, `d:filename`은 일반 문, `D`는 마스터 문, `*`는 열쇠를 뜻합니다.

게임을 시작하면 원본 CSV 맵을 복사한 `run_<timestamp>` 폴더가 생성됩니다. 이 실행 기록은 Git에서 제외됩니다.

package itm.comlang;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

/**
 * 게임 전체 흐름을 담당하는 엔진.
 *  - 시작 방을 불러오고 영웅을 배치한다.
 *  - 메인 루프에서 명령을 받아 이동/전투/문 이동/아이템 처리/저장을 수행한다.
 *
 * 게임 진행 중 일어나는 파일 오류는 이 클래스 안에서 잡아서 안내만 하고
 * 계속 진행하므로, 루프가 중간에 죽지 않는다(robustness).
 */
public class Game {

    // 인접 방향: 상, 하, 좌, 우 (행 변화, 열 변화)
    private static final int[][] DIRS = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    private Hero hero;
    private Room current;       // 현재 방
    private String runDir;      // 이번 실행의 저장 폴더
    private RoomLoader loader;
    private Scanner scanner;
    private Random random;
    private boolean running;    // 루프가 도는 동안 true
    private boolean won;        // 마스터 문으로 탈출하면 true

    public Game(String runDir) {
        this.runDir = runDir;
        this.loader = new RoomLoader();
        this.hero = new Hero();
        this.scanner = new Scanner(System.in);
        this.random = new Random();
        this.running = true;
        this.won = false;
    }

    /**
     * 게임을 시작한다. 시작 방(start.csv)을 불러오고 영웅을 배치한 뒤 메인 루프를 돈다.
     * start.csv 자체를 못 읽는 경우는 치명적이므로 예외를 위로 던진다(Main이 처리).
     */
    public void run() throws IOException, MapFormatException {
        // 시작 방은 항상 start.csv 이다 (유일하게 이름을 가정할 수 있는 파일).
        current = loader.load(runDir, "start.csv");
        placeHeroAtStart();

        while (running) {
            current.render(hero);
            showAdjacentMonsters();   // 인접 몬스터가 있으면 HP와 함께 안내(액션 메뉴)
            System.out.print("명령을 입력하세요 (u/d/l/r 이동, a 공격, q 종료): ");

            if (!scanner.hasNextLine()) {
                break;                // 입력이 끝나면 종료
            }
            String input = scanner.nextLine().trim();

            handleCommand(input);

            if (hero.isDead()) {
                System.out.println("\n영웅이 쓰러졌습니다... 게임 오버.");
                running = false;
            }
        }

        if (won) {
            System.out.println("\n축하합니다! 마스터 문을 열고 탈출했습니다. 승리!");
        }
    }

    // ===================== 영웅 배치 =====================

    /** 게임 시작 시 영웅 위치를 정한다 (스펙의 세 가지 규칙). */
    private void placeHeroAtStart() {
        // 1) start.csv 안에 @ 가 있었으면 그 위치
        if (current.getHeroStartRow() >= 0) {
            hero.setPosition(current.getHeroStartRow(), current.getHeroStartCol());
            return;
        }
        // 2) (1,1)이 비어 있으면 그 자리
        if (current.inBounds(1, 1) && current.objectAt(1, 1) instanceof EmptyCell) {
            hero.setPosition(1, 1);
            return;
        }
        // 3) 그 외에는 빈 칸 중 무작위
        placeHeroAtRandomEmpty(current);
    }

    /** 주어진 방의 빈 칸 중 한 곳을 무작위로 골라 영웅을 놓는다. */
    private void placeHeroAtRandomEmpty(Room room) {
        ArrayList<int[]> empties = new ArrayList<>();
        for (int r = 0; r < room.getRows(); r++) {
            for (int c = 0; c < room.getCols(); c++) {
                if (room.objectAt(r, c) instanceof EmptyCell) {
                    empties.add(new int[]{r, c});
                }
            }
        }
        if (empties.isEmpty()) {
            hero.setPosition(0, 0);   // 비상시(빈 칸이 전혀 없으면) 좌상단
        } else {
            int[] pick = empties.get(random.nextInt(empties.size()));
            hero.setPosition(pick[0], pick[1]);
        }
    }

    // ===================== 명령 처리 =====================

    private void handleCommand(String input) {
        if (input.equals("q")) {
            running = false;
            return;
        }
        if (input.equals("a")) {
            attackAdjacentMonster();
            return;
        }
        if (input.equals("u") || input.equals("d") || input.equals("l") || input.equals("r")) {
            move(input);
            return;
        }
        // 그 외의 입력은 무시하고 안내만 한다 (프로그램이 죽지 않는다).
        System.out.println("알 수 없는 명령입니다: \"" + input + "\". (u/d/l/r, a, q 중 하나를 입력하세요)");
    }

    // ===================== 이동 =====================

    private void move(String dir) {
        int nr = hero.getRow();
        int nc = hero.getCol();
        if (dir.equals("u")) nr--;
        else if (dir.equals("d")) nr++;
        else if (dir.equals("l")) nc--;
        else if (dir.equals("r")) nc++;

        if (!current.inBounds(nr, nc)) {
            System.out.println("그쪽은 벽입니다. 이동할 수 없습니다.");
            return;
        }

        Drawable target = current.objectAt(nr, nc);

        if (target instanceof EmptyCell) {
            hero.setPosition(nr, nc);                     // 빈 칸: 그냥 이동
        } else if (target instanceof Key) {
            hero.giveKey();                               // 열쇠: 줍고 이동
            current.setObjectAt(nr, nc, new EmptyCell());
            hero.setPosition(nr, nc);
            System.out.println("열쇠를 주웠습니다!");
        } else if (target instanceof Weapon) {
            handleWeaponCell((Weapon) target, nr, nc);    // 무기: 줍기/교체
        } else if (target instanceof Potion) {
            handlePotionCell((Potion) target, nr, nc);    // 물약: 자동 회복
        } else if (target instanceof Monster) {
            System.out.println("몬스터가 막고 있습니다. 'a'로 공격하세요.");  // 못 지나감
        } else if (target instanceof MasterDoor) {
            handleMasterDoor();                           // 마스터 문
        } else if (target instanceof RegularDoor) {
            goThroughDoor((RegularDoor) target);          // 일반 문: 방 이동
        }
    }

    /** 무기 칸으로 이동했을 때: 맨손이면 자동 장착, 무장 상태면 교체 여부를 묻는다. */
    private void handleWeaponCell(Weapon weaponOnFloor, int nr, int nc) {
        if (!hero.isArmed()) {
            hero.equip(weaponOnFloor);
            current.setObjectAt(nr, nc, new EmptyCell());   // 주운 무기는 바닥에서 사라짐
            hero.setPosition(nr, nc);
            System.out.println(weaponOnFloor.getName() + "을(를) 주웠습니다.");
            return;
        }
        System.out.print("이미 " + hero.getWeapon().getName() + "을(를) 들고 있습니다. "
                + weaponOnFloor.getName() + "(으)로 교체할까요? (y/n): ");
        String answer = scanner.hasNextLine() ? scanner.nextLine().trim() : "n";
        if (answer.equals("y")) {
            Weapon old = hero.getWeapon();
            hero.equip(weaponOnFloor);
            current.setObjectAt(nr, nc, old);   // 기존 무기를 그 칸에 내려놓는다
            System.out.println(weaponOnFloor.getName() + "(으)로 교체했습니다. "
                    + old.getName() + "은(는) 바닥에 두었습니다.");
        } else {
            System.out.println("교체하지 않았습니다. (새 무기는 바닥에 그대로 남습니다)");
            // 격자를 바꾸지 않으므로 새 무기는 그 칸에 그대로 남는다.
        }
        hero.setPosition(nr, nc);   // 어느 쪽이든 영웅은 그 칸으로 이동
    }

    /** 물약 칸으로 이동했을 때: 자동으로 마신다. 단, 풀피이면 마시지 않고 남겨둔다. */
    private void handlePotionCell(Potion potion, int nr, int nc) {
        if (hero.isFullHp()) {
            System.out.println("체력이 가득 차 있어 물약을 남겨둡니다.");
            // 격자를 바꾸지 않으므로 물약은 그 칸에 그대로 남는다.
        } else {
            potion.applyTo(hero);
            current.setObjectAt(nr, nc, new EmptyCell());   // 마신 물약은 사라짐
            System.out.println("물약을 마셔 체력을 회복했습니다. (현재 HP: " + hero.getHp() + ")");
        }
        hero.setPosition(nr, nc);
    }

    /** 마스터 문으로 이동했을 때: 열쇠가 있으면 승리, 없으면 안내. */
    private void handleMasterDoor() {
        if (hero.hasKey()) {
            won = true;
            running = false;
        } else {
            System.out.println("마스터 문은 잠겨 있습니다. 열쇠가 필요합니다.");
        }
    }

    // ===================== 방 이동(문) =====================

    /** 일반 문으로 이동: 현재 방을 저장하고 다음 방을 불러와 영웅을 재배치한다. */
    private void goThroughDoor(RegularDoor door) {
        String targetFile = door.getTargetFile();
        String fromFile = current.getFileName();

        // 1) 현재 방 상태를 저장 (아이템/몬스터 HP/교체 등 반영). 실패해도 게임은 계속.
        try {
            current.save(runDir);
        } catch (IOException e) {
            System.out.println("현재 방 상태 저장에 실패했습니다: " + e.getMessage());
        }

        // 2) 다음 방을 불러온다. 실패하면 현재 방에 그대로 머문다 (프로그램은 죽지 않음).
        Room next;
        try {
            next = loader.load(runDir, targetFile);
        } catch (Exception e) {
            System.out.println("문이 가리키는 방을 열 수 없습니다: " + targetFile
                    + " (" + e.getMessage() + ")");
            return;
        }

        // 3) 새 방에서 '내가 떠나온 방'을 가리키는 문 옆 칸에 영웅을 놓는다.
        placeHeroByReturnDoor(next, fromFile);

        current = next;
        System.out.println(targetFile + " 방으로 이동했습니다.");
    }

    /** 새 방에서, 떠나온 방(fromFile)을 가리키는 문을 찾아 그 안쪽 칸에 영웅을 배치한다. */
    private void placeHeroByReturnDoor(Room room, String fromFile) {
        ArrayList<int[]> doors = room.findRegularDoors();
        int[] returnDoor = null;
        for (int[] pos : doors) {
            RegularDoor d = (RegularDoor) room.objectAt(pos[0], pos[1]);
            if (d.getTargetFile().equals(fromFile)) {
                returnDoor = pos;
                break;
            }
        }

        if (returnDoor != null) {
            // 문은 테두리에 있으므로, 한 칸 안쪽 위치를 계산한다.
            int[] inside = innerNeighbor(room, returnDoor[0], returnDoor[1]);
            if (inside != null && room.objectAt(inside[0], inside[1]) instanceof EmptyCell) {
                hero.setPosition(inside[0], inside[1]);
                return;
            }
        }
        // 돌아갈 문을 못 찾거나 안쪽이 막혀 있으면 빈 칸 아무 곳에나 배치
        placeHeroAtRandomEmpty(room);
    }

    /** 테두리에 있는 문 위치 (r,c)의 '방 안쪽으로 한 칸' 위치를 돌려준다. */
    private int[] innerNeighbor(Room room, int r, int c) {
        if (r == 0)                    return new int[]{r + 1, c};   // 위쪽 테두리 -> 아래로
        if (r == room.getRows() - 1)   return new int[]{r - 1, c};   // 아래쪽 테두리 -> 위로
        if (c == 0)                    return new int[]{r, c + 1};   // 왼쪽 테두리 -> 오른쪽으로
        if (c == room.getCols() - 1)   return new int[]{r, c - 1};   // 오른쪽 테두리 -> 왼쪽으로
        return null;                                                 // (테두리가 아니면 없음)
    }

    // ===================== 전투 =====================

    /** 현재 영웅과 인접한 몬스터 칸들의 위치 목록 */
    private ArrayList<int[]> adjacentMonsterCells() {
        ArrayList<int[]> result = new ArrayList<>();
        for (int[] d : DIRS) {
            int r = hero.getRow() + d[0];
            int c = hero.getCol() + d[1];
            if (current.inBounds(r, c) && current.objectAt(r, c) instanceof Monster) {
                result.add(new int[]{r, c});
            }
        }
        return result;
    }

    /** 인접 몬스터가 있으면 종류와 현재 HP를 보여 준다 (전투 액션 메뉴). */
    private void showAdjacentMonsters() {
        ArrayList<int[]> cells = adjacentMonsterCells();
        if (cells.isEmpty()) {
            return;
        }
        StringBuilder sb = new StringBuilder("인접한 몬스터 -> ");
        for (int i = 0; i < cells.size(); i++) {
            Monster m = (Monster) current.objectAt(cells.get(i)[0], cells.get(i)[1]);
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(m.getClass().getSimpleName()).append(" (HP ").append(m.getHp()).append(")");
        }
        sb.append("  [a: 공격 / 이동키: 회피]");
        System.out.println(sb.toString());
    }

    /** 인접 몬스터를 공격한다. 영웅과 몬스터가 동시에 피해를 주고받는다. */
    private void attackAdjacentMonster() {
        if (!hero.isArmed()) {
            System.out.println("무기가 없어 공격할 수 없습니다.");
            return;
        }
        ArrayList<int[]> cells = adjacentMonsterCells();
        if (cells.isEmpty()) {
            System.out.println("공격할 몬스터가 인접해 있지 않습니다.");
            return;
        }
        // 인접 몬스터가 여럿이면 상,하,좌,우 순서로 첫 번째를 공격한다.
        int mr = cells.get(0)[0];
        int mc = cells.get(0)[1];
        Monster monster = (Monster) current.objectAt(mr, mc);

        // 동시 피해: 몬스터는 내 무기 공격력만큼, 나는 몬스터 공격력만큼.
        monster.takeDamage(hero.getWeapon().getDamage());
        hero.takeDamage(monster.getDamage());

        System.out.println(monster.getClass().getSimpleName() + " 공격! "
                + "(몬스터 HP: " + monster.getHp() + " / 내 HP: " + hero.getHp() + ")");

        if (monster.isDead()) {
            System.out.println(monster.getClass().getSimpleName() + "을(를) 처치했습니다!");
            if (monster.dropsKey()) {
                current.setObjectAt(mr, mc, new Key());   // 트롤은 그 자리에 열쇠를 떨어뜨림
                System.out.println("몬스터가 열쇠를 떨어뜨렸습니다! ('*'를 주우세요)");
            } else {
                current.setObjectAt(mr, mc, new EmptyCell());
            }
        }
    }
}

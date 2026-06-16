package itm.comlang;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

/**
 * 하나의 '방'. 2차원 격자(grid)에 Drawable 객체들을 담는다.
 * 격자에 영웅(@)은 들어 있지 않다 (영웅 위치는 Hero가 따로 관리).
 */
public class Room {

    private Drawable[][] grid;   // 방의 칸들. 모든 칸은 Drawable 타입(다형성).
    private int rows;
    private int cols;
    private String fileName;     // 이 방의 CSV 파일명 (예: "room2.csv")

    // start.csv 안에 '@'가 있으면 그 위치를 기억한다. 없으면 -1.
    private int heroStartRow = -1;
    private int heroStartCol = -1;

    public Room(Drawable[][] grid, int rows, int cols, String fileName) {
        this.grid = grid;
        this.rows = rows;
        this.cols = cols;
        this.fileName = fileName;
    }

    public int getRows() { return rows; }
    public int getCols() { return cols; }
    public String getFileName() { return fileName; }

    public void setHeroStart(int r, int c) {
        this.heroStartRow = r;
        this.heroStartCol = c;
    }
    public int getHeroStartRow() { return heroStartRow; }
    public int getHeroStartCol() { return heroStartCol; }

    /** (r, c)가 격자 범위 안인지 확인 */
    public boolean inBounds(int r, int c) {
        return r >= 0 && r < rows && c >= 0 && c < cols;
    }

    public Drawable objectAt(int r, int c) {
        return grid[r][c];
    }

    public void setObjectAt(int r, int c, Drawable obj) {
        grid[r][c] = obj;
    }

    /**
     * 방을 화면에 그린다.
     * 맨 위에 영웅 상태 줄(HP/무기/열쇠)을 출력하고, 벽(+,-,|)으로 둘러싼 격자를 그린다.
     * 영웅이 있는 칸에는 그 칸의 원래 글자 대신 '@'를 겹쳐 그린다.
     */
    public void render(Hero hero) {
        // 1) 상태 줄 (방을 그릴 때마다 맨 위에 표시)
        String weaponText = hero.isArmed()
                ? hero.getWeapon().getName() + " (" + hero.getWeapon().getDamage() + ")"
                : "None";
        String keyText = hero.hasKey() ? "Yes" : "No";
        System.out.println("HP: " + hero.getHp() + "/" + Hero.MAX_HP
                + " | Weapon: " + weaponText + " | Key: " + keyText);

        // 2) 위쪽 벽
        System.out.println(buildWall());

        // 3) 각 행을 좌우 벽(|)과 함께 출력
        for (int r = 0; r < rows; r++) {
            StringBuilder line = new StringBuilder();
            line.append('|');
            for (int c = 0; c < cols; c++) {
                if (hero.getRow() == r && hero.getCol() == c) {
                    line.append('@');                  // 영웅 위치
                } else {
                    line.append(grid[r][c].getSymbol()); // 칸의 종류와 무관하게 글자만 받음(다형성)
                }
            }
            line.append('|');
            System.out.println(line.toString());
        }

        // 4) 아래쪽 벽
        System.out.println(buildWall());
    }

    /** "+----+" 모양의 가로 벽 한 줄을 만든다. */
    private String buildWall() {
        StringBuilder wall = new StringBuilder();
        wall.append('+');
        for (int c = 0; c < cols; c++) {
            wall.append('-');
        }
        wall.append('+');
        return wall.toString();
    }

    /**
     * 이 방 안의 모든 일반 문(RegularDoor)의 위치를 찾아 목록으로 돌려준다.
     * (다른 방에서 들어올 때 '어느 문으로 돌아와야 하는지' 찾는 데 사용)
     */
    public ArrayList<int[]> findRegularDoors() {
        ArrayList<int[]> doors = new ArrayList<>();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (grid[r][c] instanceof RegularDoor) {
                    doors.add(new int[]{r, c});
                }
            }
        }
        return doors;
    }

    /**
     * 현재 방 상태를 CSV로 저장한다.
     * 첫 줄에 "행,열"을 적고, 각 칸은 toCsv()로 직렬화하여 콤마로 잇는다.
     * 각 칸이 자기 자신을 어떻게 저장할지 스스로 알기 때문에(다형성),
     * 몬스터의 현재 HP, 아이템 줍기/교체, 처치된 몬스터(빈칸) 등이 자동으로 반영된다.
     */
    public void save(String runDir) throws IOException {
        ArrayList<String> lines = new ArrayList<>();
        lines.add(rows + "," + cols);                 // 헤더
        for (int r = 0; r < rows; r++) {
            StringBuilder sb = new StringBuilder();
            for (int c = 0; c < cols; c++) {
                if (c > 0) {
                    sb.append(',');
                }
                sb.append(grid[r][c].toCsv());
            }
            lines.add(sb.toString());
        }
        Path path = Paths.get(runDir, fileName);
        Files.write(path, lines);                     // 복사본 폴더(runDir)에만 저장
    }
}

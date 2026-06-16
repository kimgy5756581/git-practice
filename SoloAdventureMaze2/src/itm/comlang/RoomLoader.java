package itm.comlang;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * CSV 파일을 읽어 Room 객체로 만드는 책임을 가진 클래스.
 * 첫 줄은 "행,열" 헤더이고, 그 다음 줄들이 각 칸의 내용이다.
 *
 * 칸 글자 -> 객체 변환 규칙:
 *   (빈칸)      -> EmptyCell
 *   S/W/X       -> Stick / WeakSword / StrongSword
 *   G/O/T       -> Goblin / Orc / Troll   (":HP"가 붙으면 그 HP, 없으면 기본 HP)
 *   m/B         -> MinorFlask / BigFlask
 *   *           -> Key
 *   d:파일명     -> RegularDoor
 *   D           -> MasterDoor
 *   @           -> 영웅 시작 위치(빈칸으로 두고 위치만 기억)
 */
public class RoomLoader {

    /**
     * runDir 폴더 안의 fileName CSV를 읽어 Room으로 만든다.
     * @throws IOException 파일을 읽을 수 없을 때 (없는 파일 등)
     * @throws MapFormatException 파일 형식이 잘못되었을 때
     */
    public Room load(String runDir, String fileName) throws IOException, MapFormatException {
        Path path = Paths.get(runDir, fileName);
        // readAllLines는 줄끝 문자(\n, \r\n)를 자동으로 떼어 준다.
        List<String> lines = Files.readAllLines(path);

        if (lines.isEmpty()) {
            throw new MapFormatException(fileName + " : 파일이 비어 있습니다.");
        }

        // ---- 첫 줄: "행,열" 헤더 파싱 ----
        String[] header = lines.get(0).trim().split(",");
        if (header.length < 2) {
            throw new MapFormatException(fileName + " : 첫 줄에 '행,열' 정보가 없습니다.");
        }
        int rows;
        int cols;
        try {
            rows = Integer.parseInt(header[0].trim());
            cols = Integer.parseInt(header[1].trim());
        } catch (NumberFormatException e) {
            throw new MapFormatException(fileName + " : 행/열 숫자를 읽을 수 없습니다.");
        }
        if (rows <= 0 || cols <= 0) {
            throw new MapFormatException(fileName + " : 행/열은 1 이상이어야 합니다.");
        }

        // ---- 격자 채우기 ----
        Drawable[][] grid = new Drawable[rows][cols];
        int heroRow = -1;
        int heroCol = -1;

        for (int r = 0; r < rows; r++) {
            // 헤더 다음 줄부터가 데이터. 줄이 모자라면 빈 줄로 취급한다.
            String line = (r + 1 < lines.size()) ? lines.get(r + 1) : "";
            // 콤마로 나눈다. -1 옵션은 끝쪽 빈 칸도 버리지 않고 보존한다.
            String[] tokens = line.split(",", -1);

            if (tokens.length > cols) {
                throw new MapFormatException(
                        fileName + " : " + (r + 1) + "번째 데이터 줄의 칸 수(" + tokens.length
                                + ")가 열 수(" + cols + ")보다 많습니다.");
            }

            for (int c = 0; c < cols; c++) {
                String token = (c < tokens.length) ? tokens[c].trim() : "";
                if (token.equals("@")) {
                    grid[r][c] = new EmptyCell();   // 영웅 자리는 빈칸으로 두고
                    heroRow = r;                    // 위치만 따로 기억한다
                    heroCol = c;
                } else {
                    grid[r][c] = createCell(token, fileName);
                }
            }
        }

        Room room = new Room(grid, rows, cols, fileName);
        room.setHeroStart(heroRow, heroCol);
        return room;
    }

    /** 칸 글자 하나를 알맞은 Drawable 객체로 바꾼다. */
    private Drawable createCell(String token, String fileName) throws MapFormatException {
        if (token.isEmpty()) {
            return new EmptyCell();
        }
        // ----- 문 -----
        if (token.equals("D")) {
            return new MasterDoor();
        }
        if (token.startsWith("d:")) {
            String target = token.substring(2).trim();
            return new RegularDoor(target);
        }
        // ----- 무기 -----
        if (token.equals("S")) return new Stick();
        if (token.equals("W")) return new WeakSword();
        if (token.equals("X")) return new StrongSword();
        // ----- 물약 -----
        if (token.equals("m")) return new MinorFlask();
        if (token.equals("B")) return new BigFlask();
        // ----- 열쇠 -----
        if (token.equals("*")) return new Key();
        // ----- 몬스터 ("G", "G:3", "O:8", "T" 등) -----
        char first = token.charAt(0);
        if (first == 'G' || first == 'O' || first == 'T') {
            int hp = parseMonsterHp(token, first, fileName);
            if (first == 'G') return new Goblin(hp);
            if (first == 'O') return new Orc(hp);
            return new Troll(hp);
        }
        // 그 외에는 알 수 없는 기호 -> 형식 오류로 처리
        throw new MapFormatException(fileName + " : 알 수 없는 칸 기호 '" + token + "'");
    }

    /** 몬스터 토큰에서 HP를 읽는다. ":HP"가 없으면 종류별 기본 HP를 돌려준다. */
    private int parseMonsterHp(String token, char type, String fileName) throws MapFormatException {
        int defaultHp;
        if (type == 'G') defaultHp = Goblin.DEFAULT_HP;
        else if (type == 'O') defaultHp = Orc.DEFAULT_HP;
        else defaultHp = Troll.DEFAULT_HP;

        int colon = token.indexOf(':');
        if (colon < 0) {
            return defaultHp;                 // "G" 처럼 HP가 생략된 경우
        }
        String hpPart = token.substring(colon + 1).trim();
        if (hpPart.isEmpty()) {
            return defaultHp;
        }
        try {
            return Integer.parseInt(hpPart);
        } catch (NumberFormatException e) {
            throw new MapFormatException(fileName + " : 몬스터 HP를 읽을 수 없습니다 ('" + token + "')");
        }
    }
}

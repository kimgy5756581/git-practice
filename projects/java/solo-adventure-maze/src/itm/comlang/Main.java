package itm.comlang;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

/**
 * 프로그램의 시작점.
 * 1) 이번 실행용 저장 폴더를 만들고, 2) 원본 CSV를 그 폴더로 복사한 뒤, 3) 게임을 실행한다.
 * 파일 관련 오류가 나면 메시지를 출력하고 깔끔하게 종료한다 (비정상 종료하지 않는다).
 */
public class Main {

    public static void main(String[] args) {
        // 한글 출력이 환경(OS/콘솔)에 상관없이 깨지지 않도록 출력 인코딩을 UTF-8로 고정한다.
        System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));

        SaveManager saveManager = new SaveManager();
        try {
            String runDir = saveManager.createRunDir();   // 예: run_1717300000000
            saveManager.copyOriginals(runDir);            // 원본 .csv -> 복사본 폴더

            Game game = new Game(runDir);
            game.run();
        } catch (IOException e) {
            System.out.println("파일 입출력 오류가 발생했습니다: " + e.getMessage());
        } catch (MapFormatException e) {
            System.out.println("지도 파일 형식 오류: " + e.getMessage());
        }
    }
}

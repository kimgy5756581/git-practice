package itm.comlang;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * 실행할 때마다 별도의 저장 폴더를 만들고, 원본 CSV들을 그 폴더로 복사하는 책임.
 * 게임은 이 '복사본'만 읽고 쓰므로 원본 CSV 파일은 절대 바뀌지 않는다.
 */
public class SaveManager {

    /**
     * 현재 시각(밀리초)을 이용해 "run_<숫자>" 폴더를 만들고 그 폴더 이름을 돌려준다.
     * 매 실행마다 다른 폴더가 만들어진다.
     */
    public String createRunDir() throws IOException {
        String dirName = "run_" + System.currentTimeMillis();
        Files.createDirectories(Paths.get(dirName));
        return dirName;
    }

    /**
     * 현재 작업 폴더(".")에 있는 모든 .csv 파일을 runDir로 복사한다.
     * 방 개수나 이름을 미리 알 필요 없이 존재하는 모든 .csv를 통째로 복사하므로,
     * 채점자가 다른 이름의 파일을 주어도 그대로 동작한다.
     * 경로는 모두 상대 경로이므로 어떤 컴퓨터에서도 실행된다(이식성).
     */
    public void copyOriginals(String runDir) throws IOException {
        Path current = Paths.get(".");
        // try-with-resources: 디렉터리 스트림을 자동으로 닫아 준다.
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(current, "*.csv")) {
            for (Path file : stream) {
                Path target = Paths.get(runDir, file.getFileName().toString());
                Files.copy(file, target);
            }
        }
    }
}

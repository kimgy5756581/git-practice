package itm.comlang;

/**
 * 일반 문. CSV 칸에 "d:파일명.csv" 형태로 저장되며, 그 파일명이 연결된 다음 방이다.
 * 파일명은 실행 중에 칸에서 읽어 사용한다 (방 이름을 코드에 하드코딩하지 않는다).
 */
public class RegularDoor extends Door {

    private String targetFile;   // 이 문이 연결하는 방의 CSV 파일명

    public RegularDoor(String targetFile) {
        this.targetFile = targetFile;
    }

    public String getTargetFile() {
        return targetFile;
    }

    /** 일반 문은 언제든 통과할 수 있다. */
    @Override
    public boolean canPass(Hero hero) {
        return true;
    }

    @Override
    public char getSymbol() {
        return 'd';
    }

    @Override
    public String toCsv() {
        return "d:" + targetFile;
    }
}

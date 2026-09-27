package itm.comlang;

/**
 * 아무것도 없는 빈 칸. 영웅이 자유롭게 이동할 수 있는 칸이다.
 * null 대신 이 객체를 격자에 넣어두면, getSymbol() 호출 시
 * NullPointerException이 발생하지 않는다 (Null Object 패턴).
 */
public class EmptyCell implements Drawable {

    @Override
    public char getSymbol() {
        return ' ';   // 빈칸은 공백으로 표시
    }

    @Override
    public String toCsv() {
        return "";    // CSV에서는 빈 문자열 (콤마와 콤마 사이가 비어 있음)
    }
}

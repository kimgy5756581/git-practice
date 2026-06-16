package itm.comlang;

/**
 * 열쇠. 처음에는 어떤 방에도 존재하지 않으며, Troll을 처치하면 그 자리에 떨어진다.
 * 영웅이 열쇠 칸으로 이동하면 자동으로 줍는다.
 */
public class Key implements Drawable {

    @Override
    public char getSymbol() {
        return '*';
    }

    @Override
    public String toCsv() {
        return "*";
    }
}

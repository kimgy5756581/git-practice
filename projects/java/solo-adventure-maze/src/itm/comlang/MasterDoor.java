package itm.comlang;

/**
 * 마스터 문. 열쇠가 있어야 통과(탈출)할 수 있고, 통과하면 게임 승리.
 */
public class MasterDoor extends Door {

    /** 영웅이 열쇠를 가지고 있어야만 통과할 수 있다. */
    @Override
    public boolean canPass(Hero hero) {
        return hero.hasKey();
    }

    @Override
    public char getSymbol() {
        return 'D';
    }

    @Override
    public String toCsv() {
        return "D";
    }
}

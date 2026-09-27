package itm.comlang;

/**
 * 트롤: 기본 HP 15, 공격력 4, 문자 'T'.
 * 게임 전체에 단 하나뿐이며, 처치하면 열쇠를 떨어뜨리는 유일한 몬스터다.
 */
public class Troll extends Monster {
    public static final int DEFAULT_HP = 15;

    public Troll(int hp) {
        super(hp, 4, 'T');
    }

    /** 트롤만 열쇠를 떨어뜨린다. */
    @Override
    public boolean dropsKey() {
        return true;
    }
}

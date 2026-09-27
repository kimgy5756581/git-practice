package itm.comlang;

/** 고블린: 기본 HP 3, 공격력 1, 문자 'G' */
public class Goblin extends Monster {
    public static final int DEFAULT_HP = 3;   // CSV에 HP가 없을 때 쓰는 기본값

    public Goblin(int hp) {
        super(hp, 1, 'G');
    }
}

package itm.comlang;

/** 오크: 기본 HP 8, 공격력 3, 문자 'O' */
public class Orc extends Monster {
    public static final int DEFAULT_HP = 8;

    public Orc(int hp) {
        super(hp, 3, 'O');
    }
}

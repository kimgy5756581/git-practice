package itm.comlang;

/**
 * 몬스터의 공통 부모(추상 클래스).
 * 세 몬스터(Goblin, Orc, Troll)는 모두 'HP'와 '공격력'을 가지며,
 * 피해를 받는 동작/죽었는지 판단하는 동작이 동일하다. 값만 다르므로 상속으로 묶는다.
 */
public abstract class Monster implements Drawable {

    private int hp;        // 현재 체력
    private int damage;    // 공격력
    private char symbol;   // 화면 문자 (G, O, T)

    public Monster(int hp, int damage, char symbol) {
        this.hp = hp;
        this.damage = damage;
        this.symbol = symbol;
    }

    /** 피해를 받아 HP를 깎는다. (0 밑으로는 내려가지 않게 한다) */
    public void takeDamage(int amount) {
        hp = hp - amount;
        if (hp < 0) {
            hp = 0;
        }
    }

    /** HP가 0 이하이면 죽은 것이다. */
    public boolean isDead() {
        return hp <= 0;
    }

    public int getHp() {
        return hp;
    }

    public int getDamage() {
        return damage;
    }

    /**
     * 이 몬스터가 열쇠를 떨어뜨리는지 여부.
     * 기본값은 false이고, Troll만 true로 재정의(override)한다.
     */
    public boolean dropsKey() {
        return false;
    }

    @Override
    public char getSymbol() {
        return symbol;
    }

    /** 저장할 때는 '현재 HP'를 함께 적는다. 예: 오크가 5 남았으면 "O:5" */
    @Override
    public String toCsv() {
        return symbol + ":" + hp;
    }
}

package itm.comlang;

/**
 * 회복 물약의 공통 부모(추상 클래스).
 * 두 물약(MinorFlask, BigFlask)은 회복량만 다르고
 * "영웅을 회복시킨다"는 동작이 같으므로 상속으로 묶는다.
 */
public abstract class Potion implements Drawable {

    private int heal;     // 회복량
    private char symbol;  // 화면 문자 (m, B)

    public Potion(int heal, char symbol) {
        this.heal = heal;
        this.symbol = symbol;
    }

    /** 영웅의 HP를 회복량만큼 올린다 (최대치 제한은 Hero.heal()이 처리). */
    public void applyTo(Hero hero) {
        hero.heal(heal);
    }

    public int getHeal() {
        return heal;
    }

    @Override
    public char getSymbol() {
        return symbol;
    }

    @Override
    public String toCsv() {
        return String.valueOf(symbol);
    }
}

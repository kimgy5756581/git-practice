package itm.comlang;

/**
 * 무기의 공통 부모(추상 클래스).
 * 세 가지 무기(Stick, WeakSword, StrongSword)는 모두 '이름'과 '공격력'을 가지며,
 * 공격력을 알려주는 동작이 완전히 같고 값만 다르다. 따라서 상속으로 묶는다.
 * (타입 필드 + if문 대신 상속을 쓰면, 무기를 추가할 때 클래스 하나만 만들면 된다.)
 *
 * Drawable을 구현하므로 무기도 격자 칸에 놓일 수 있다.
 */
public abstract class Weapon implements Drawable {

    private String name;    // 무기 이름 (예: "Weak Sword")
    private int damage;     // 공격력
    private char symbol;    // CSV/화면에서 쓰는 문자 (S, W, X)

    /** 자식 클래스가 생성자에서 이름/공격력/문자를 넘겨준다. */
    public Weapon(String name, int damage, char symbol) {
        this.name = name;
        this.damage = damage;
        this.symbol = symbol;
    }

    public int getDamage() {
        return damage;
    }

    public String getName() {
        return name;
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

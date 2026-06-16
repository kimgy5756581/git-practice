package itm.comlang;

/**
 * 영웅(플레이어).
 * 영웅은 격자(grid) 안에 객체로 넣지 않고, 위치(row, col)를 따로 들고 다닌다.
 * 이렇게 하면 영웅이 물약/무기 칸 '위에 서 있는' 상황을 자연스럽게 표현할 수 있다.
 * (그 칸 아래의 원래 객체는 격자에 그대로 남아 있고, 화면에는 @만 겹쳐 그린다.)
 *
 * 모든 필드는 private이며, 메서드를 통해서만 접근한다 (캡슐화).
 */
public class Hero {

    public static final int MAX_HP = 25;   // 영웅의 최대 체력

    private int hp;
    private Weapon weapon;     // 현재 장착한 무기 (없으면 null = 맨손)
    private boolean hasKey;
    private int row;
    private int col;

    public Hero() {
        this.hp = MAX_HP;
        this.weapon = null;
        this.hasKey = false;
    }

    // ---------- 위치 ----------
    public int getRow() { return row; }
    public int getCol() { return col; }

    public void setPosition(int row, int col) {
        this.row = row;
        this.col = col;
    }

    // ---------- 체력 ----------
    public int getHp() { return hp; }

    public void takeDamage(int amount) {
        hp = hp - amount;
        if (hp < 0) {
            hp = 0;
        }
    }

    public void heal(int amount) {
        hp = hp + amount;
        if (hp > MAX_HP) {
            hp = MAX_HP;   // 최대치를 넘지 않는다
        }
    }

    public boolean isDead() {
        return hp <= 0;
    }

    public boolean isFullHp() {
        return hp >= MAX_HP;
    }

    // ---------- 무기 ----------
    public boolean isArmed() {
        return weapon != null;
    }

    public Weapon getWeapon() {
        return weapon;
    }

    public void equip(Weapon weapon) {
        this.weapon = weapon;
    }

    // ---------- 열쇠 ----------
    public boolean hasKey() {
        return hasKey;
    }

    public void giveKey() {
        this.hasKey = true;
    }
}

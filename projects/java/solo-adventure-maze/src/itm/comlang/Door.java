package itm.comlang;

/**
 * 문의 공통 부모(추상 클래스).
 * 일반 문(RegularDoor)과 마스터 문(MasterDoor)은
 * "영웅이 통과할 수 있는가?"를 판단하는 방식이 서로 다르다.
 * 그래서 canPass()를 추상 메서드로 두고 각 자식이 다르게 구현한다.
 */
public abstract class Door implements Drawable {

    /** 영웅이 이 문을 통과할 수 있는지 여부 */
    public abstract boolean canPass(Hero hero);
}

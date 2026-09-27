package itm.comlang;

/**
 * CSV 지도 파일이 잘못된 형식일 때 던지는 예외.
 * (헤더가 없거나, 칸 수가 맞지 않거나, 알 수 없는 기호가 있는 경우 등)
 * 이 예외는 Main에서 잡아서 깔끔한 메시지를 출력하고 종료하는 데 쓰인다.
 */
public class MapFormatException extends Exception {
    public MapFormatException(String message) {
        super(message);
    }
}

import binance.BinanceUtil;
import bot.Session;

public class SessionMain {
  public static void main(String[] args) throws Exception {
    String baseUrl = BinanceUtil.testnetBaseUrl;
    String apiKey = "jWna3EA6Ija8rDcW0En5Rv8KSFadFixbpTpHXzl3975ly6W9UUvAZTfiutDKuMS0";
    String secretKey = "uNB9tbzwOqDC9psp06XD6Vni3fxr9vyY495mrFh7X4v8m70CX08t9r3k37o9xpaq";
    Session session = new Session(apiKey, secretKey, "BTCUSDT", baseUrl,
        100, Long.MAX_VALUE, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
    System.out.println("Starting...");
    session.start();
    session.join();
  }
}
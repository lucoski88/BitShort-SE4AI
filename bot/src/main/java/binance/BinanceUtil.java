package binance;

public class BinanceUtil {
  public static final String mainnetBaseUrl = "https://api.binance.com";
  public static final String testnetBaseUrl = "https://testnet.binance.vision";
  public static final double minTradeAmount = 5.0;
  
  public static final String orderStatusNew = "NEW";
  public static final String orderStatusPartiallyFilled = "PARTIALLY_FILLED";
  public static final String orderStatusFilled = "FILLED";
  public static final String orderStatusCanceled = "CANCELED";
  public static final String orderStatusExpired = "EXPIRED";
  public static final String orderStatusRejected = "REJECTED";
  
  private BinanceUtil() {}
}
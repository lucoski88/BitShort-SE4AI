package binance.types;

import org.json.JSONObject;

public class BinanceOrder {
  private final String symbol;
  private final long orderId;
  private final int orderListId;
  private final String clientOrderId;
  private final double price;
  private final double origQty;
  private final double executedQty;
  private final double cummulativeQuoteQty;
  private final String status;
  private final String timeInForce;
  private final String type;
  private final String side;
  private final double stopPrice;
  private final double icebergQty;
  private final long time;
  private final long updateTime;
  private final boolean isWorking;
  private final long workingTime;
  private final double origQuoteOrderQty;
  private final String selfTradePreventionMode;

  public BinanceOrder(String symbol, long orderId, int orderListId, String clientOrderId,
                      double price, double origQty, double executedQty, double cummulativeQuoteQty,
                      String status, String timeInForce, String type, String side,
                      double stopPrice, double icebergQty, long time, long updateTime,
                      boolean isWorking, long workingTime, double origQuoteOrderQty,
                      String selfTradePreventionMode) {
    this.symbol = symbol;
    this.orderId = orderId;
    this.orderListId = orderListId;
    this.clientOrderId = clientOrderId;
    this.price = price;
    this.origQty = origQty;
    this.executedQty = executedQty;
    this.cummulativeQuoteQty = cummulativeQuoteQty;
    this.status = status;
    this.timeInForce = timeInForce;
    this.type = type;
    this.side = side;
    this.stopPrice = stopPrice;
    this.icebergQty = icebergQty;
    this.time = time;
    this.updateTime = updateTime;
    this.isWorking = isWorking;
    this.workingTime = workingTime;
    this.origQuoteOrderQty = origQuoteOrderQty;
    this.selfTradePreventionMode = selfTradePreventionMode;
  }

  public static BinanceOrder parse(JSONObject jsonObj) {
    return new BinanceOrder(jsonObj.getString("symbol"), jsonObj.getLong("orderId"),
        jsonObj.getInt("orderListId"), jsonObj.getString("clientOrderId"),
        jsonObj.getDouble("price"), jsonObj.getDouble("origQty"),
        jsonObj.getDouble("executedQty"), jsonObj.getDouble("cummulativeQuoteQty"),
        jsonObj.getString("status"), jsonObj.getString("timeInForce"),
        jsonObj.getString("type"), jsonObj.getString("side"),
        jsonObj.getDouble("stopPrice"), jsonObj.getDouble("icebergQty"),
        jsonObj.getLong("time"), jsonObj.getLong("updateTime"),
        jsonObj.getBoolean("isWorking"), jsonObj.getLong("workingTime"),
        jsonObj.getDouble("origQuoteOrderQty"), jsonObj.getString("selfTradePreventionMode"));
  }

  public String getSymbol() {
    return symbol;
  }

  public long getOrderId() {
    return orderId;
  }

  public int getOrderListId() {
    return orderListId;
  }

  public String getClientOrderId() {
    return clientOrderId;
  }

  public double getPrice() {
    return price;
  }

  public double getOrigQty() {
    return origQty;
  }

  public double getExecutedQty() {
    return executedQty;
  }

  public double getCummulativeQuoteQty() {
    return cummulativeQuoteQty;
  }

  public String getStatus() {
    return status;
  }

  public String getTimeInForce() {
    return timeInForce;
  }

  public String getType() {
    return type;
  }

  public String getSide() {
    return side;
  }

  public double getStopPrice() {
    return stopPrice;
  }

  public double getIcebergQty() {
    return icebergQty;
  }

  public long getTime() {
    return time;
  }

  public long getUpdateTime() {
    return updateTime;
  }

  public boolean isWorking() {
    return isWorking;
  }

  public long getWorkingTime() {
    return workingTime;
  }

  public double getOrigQuoteOrderQty() {
    return origQuoteOrderQty;
  }

  public String getSelfTradePreventionMode() {
    return selfTradePreventionMode;
  }

  @Override
  public String toString() {
    return "BinanceOrder{" +
        "symbol='" + symbol + '\'' +
        ", orderId=" + orderId +
        ", orderListId=" + orderListId +
        ", clientOrderId='" + clientOrderId + '\'' +
        ", price=" + price +
        ", origQty=" + origQty +
        ", executedQty=" + executedQty +
        ", cummulativeQuoteQty=" + cummulativeQuoteQty +
        ", status='" + status + '\'' +
        ", timeInForce='" + timeInForce + '\'' +
        ", type='" + type + '\'' +
        ", side='" + side + '\'' +
        ", stopPrice=" + stopPrice +
        ", icebergQty=" + icebergQty +
        ", time=" + time +
        ", updateTime=" + updateTime +
        ", isWorking=" + isWorking +
        ", workingTime=" + workingTime +
        ", origQuoteOrderQty=" + origQuoteOrderQty +
        ", selfTradePreventionMode='" + selfTradePreventionMode + '\'' +
        '}';
  }
}
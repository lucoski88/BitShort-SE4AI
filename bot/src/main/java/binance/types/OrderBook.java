package binance.types;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class OrderBook {
  private final long lastUpdateId;
  private final List<OrderBookLine> bids;
  private final List<OrderBookLine> asks;

  public OrderBook(long lastUpdateId, List<OrderBookLine> bids, List<OrderBookLine> asks) {
    this.lastUpdateId = lastUpdateId;
    this.bids = bids;
    this.asks = asks;
  }

  public static OrderBook parse(JSONObject json) {
    long id = json.getLong("lastUpdateId");
    JSONArray jsonBids = json.getJSONArray("bids");
    JSONArray jsonAsks = json.getJSONArray("asks");

    List<OrderBookLine> b = new ArrayList<>(jsonBids.length());
    List<OrderBookLine> a = new ArrayList<>(jsonAsks.length());

    for (int i = 0; i < jsonBids.length(); i++) {
      JSONArray tmp = jsonBids.getJSONArray(i);
      b.add(new OrderBookLine(tmp.getDouble(0), tmp.getDouble(1)));
    }

    for (int i = 0; i < jsonAsks.length(); i++) {
      JSONArray tmp = jsonAsks.getJSONArray(i);
      a.add(new OrderBookLine(tmp.getDouble(0), tmp.getDouble(1)));
    }

    return new OrderBook(id, b, a);
  }

  public long getLastUpdateId() {
    return lastUpdateId;
  }

  public List<OrderBookLine> getBids() {
    return bids;
  }

  public List<OrderBookLine> getAsks() {
    return asks;
  }

  public double getMinSellPrice() {
    return asks.get(0).getPrice();
  }

  public double getMinSellQuantity() {
    return asks.get(0).getQty();
  }

  public double getMinSellQuoteQuantity() {
    return asks.get(0).getQuoteQuantity();
  }

  public double getMaxBuyPrice() {
    return bids.get(0).getPrice();
  }

  public double getMaxBuyQuantity() {
    return bids.get(0).getQty();
  }

  public double getMaxBuyQuoteQuantity() {
    return bids.get(0).getQuoteQuantity();
  }

  @Override
  public String toString() {
    return "OrderBook{" +
        "lastUpdateId=" + lastUpdateId +
        ", bids=" + bids +
        ", asks=" + asks +
        '}';
  }
}
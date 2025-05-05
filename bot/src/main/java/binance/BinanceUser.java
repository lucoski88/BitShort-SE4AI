package binance;

import binance.types.BinanceOrder;
import com.binance.connector.client.SpotClient;
import com.binance.connector.client.impl.SpotClientImpl;
import com.binance.connector.client.impl.spot.Trade;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class BinanceUser {
  private final SpotClient client;
  private final Trade trade;

  public BinanceUser(String apiKey, String secretKey, String baseUrl) {
    client = new SpotClientImpl(apiKey, secretKey, baseUrl);
    trade = client.createTrade();
  }

  public BinanceUser(SpotClient client) {
    this.client = client;
    trade = client.createTrade();
  }

  public double getFreeBalance(String asset) throws RuntimeException {
    Map<String, Object> params = new LinkedHashMap<>();

    params.put("timestamp", System.currentTimeMillis());
    String response = trade.account(params);

    JSONObject jsonObj = new JSONObject(response);
    JSONArray jsonArray = jsonObj.getJSONArray("balances");

    for (int i = 0; i < jsonArray.length(); i++) {
      JSONObject obj = jsonArray.getJSONObject(i);
      if (obj.getString("asset").equals(asset)) {
        return obj.getDouble("free");
      }
    }

    throw new RuntimeException("Asset not found");
  }

  public long buy(String symbol, double quantity) {
    Map<String, Object> params = new LinkedHashMap<>();
    params.put("symbol", symbol);
    params.put("side", "BUY");
    params.put("type", "MARKET");
    params.put("quoteOrderQty", quantity);
    params.put("timestamp", System.currentTimeMillis());

    String response = trade.newOrder(params);
    JSONObject jsonObject = new JSONObject(response);

    return jsonObject.getLong("orderId");
  }

  public long sell(String symbol, double quantity) {
    Map<String, Object> params = new LinkedHashMap<>();
    params.put("symbol", symbol);
    params.put("side", "SELL");
    params.put("type", "MARKET");
    params.put("quantity", quantity);
    params.put("timestamp", System.currentTimeMillis());

    String response = trade.newOrder(params);
    JSONObject jsonObject = new JSONObject(response);

    return jsonObject.getLong("orderId");
  }

  public long placeSellOrder(String symbol, double quantity, double price) {
    Map<String, Object> params = new LinkedHashMap<>();
    params.put("symbol", symbol);
    params.put("side", "SELL");
    params.put("type", "LIMIT");
    params.put("timeInForce", "GTC");
    params.put("quantity", quantity);
    params.put("price", price);
    params.put("timestamp", System.currentTimeMillis());

    String response = trade.newOrder(params);
    JSONObject jsonObject = new JSONObject(response);

    return jsonObject.getLong("orderId");
  }

  public BinanceOrder getOrderById(String symbol, long orderId) {
    Map<String, Object> params = new LinkedHashMap<>();
    params.put("symbol", symbol);
    params.put("orderId", orderId);
    params.put("timestamp", System.currentTimeMillis());

    String response = trade.getOrder(params);

    return BinanceOrder.parse(new JSONObject(response));
  }

  public List<BinanceOrder> getAllOrders(String symbol) {
    Map<String, Object> params = new LinkedHashMap<>();
    params.put("symbol", symbol);
    params.put("timestamp", System.currentTimeMillis());

    String response = trade.getOrders(params);

    List<BinanceOrder> list = new ArrayList<>();
    JSONArray jsonArray = new JSONArray(response);
    for (int i = 0; i < jsonArray.length(); i++) {
      list.add(BinanceOrder.parse(jsonArray.getJSONObject(i)));
    }

    return list;
  }

  public void cancelOrder(String symbol, long orderId) {
    Map<String, Object> params = new LinkedHashMap<>();
    params.put("symbol", symbol);
    params.put("orderId", orderId);
    params.put("timestamp", System.currentTimeMillis());

    String response = trade.cancelOrder(params);
  }
}
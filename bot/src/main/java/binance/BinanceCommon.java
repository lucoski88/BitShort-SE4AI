package binance;

import binance.types.KLine;
import binance.types.OrderBook;
import com.binance.connector.client.SpotClient;
import com.binance.connector.client.impl.SpotClientImpl;
import com.binance.connector.client.impl.spot.Market;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class BinanceCommon {
  private final Market market;

  public BinanceCommon(String baseUrl) {
    market = new SpotClientImpl(baseUrl).createMarket();
  }

  public BinanceCommon(SpotClient client) {
    market = client.createMarket();
  }

  public List<KLine> getKLines(String symbol, String interval, int limit) {
    Map<String, Object> params = new LinkedHashMap<>();
    params.put("symbol", symbol);
    params.put("interval", interval);
    params.put("limit", limit);

    List<KLine> list = new ArrayList<>();

    String response = market.klines(params);
    JSONArray jsonArray = new JSONArray(response);
    for (int i = 0; i < jsonArray.length(); i++) {
      list.add(KLine.parse(jsonArray.getJSONArray(i)));
    }

    return list;
  }

  public double getCurrentPrice(String symbol) {
    Map<String, Object> params = new LinkedHashMap<>();
    params.put("symbol", symbol);

    String response = market.tickerSymbol(params);

    JSONObject jsonObject = new JSONObject(response);
    return jsonObject.getDouble("price");
  }

  public OrderBook getOrderBook(String symbol, int limit) {
    Map<String, Object> params = new LinkedHashMap<>();
    params.put("symbol", symbol);
    params.put("limit", limit);

    String response = market.depth(params);

    return OrderBook.parse(new JSONObject(response));
  }
}
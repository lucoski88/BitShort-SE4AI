package bot.server;

import binance.BinanceUtil;
import bot.Session;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class HttpHandlerImpl implements HttpHandler {
    private final Map<String, Session> sessions = new HashMap<>();

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        String method = exchange.getRequestMethod();
        switch (method) {
            case "POST":
                doPost(exchange);
                break;
            default:
                exchange.sendResponseHeaders(405, -1);
                break;
        }
    }
    
    private void doPost(HttpExchange exchange) throws IOException {
        URIParser uriParser = new URIParser(exchange.getRequestURI().toString());
        ParameterParser paramsParser = new ParameterParser(new String(exchange.getRequestBody().readAllBytes(),
                StandardCharsets.UTF_8));
        List<String> pathList = uriParser.getPath();
        Map<String, Object> paramsMap = paramsParser.getParameters();
        
        Map<String, String> keyParams = getKeyParameters(paramsMap);

        if (keyParams.isEmpty() || !keyParams.containsKey("apiKey")) {
            exchange.sendResponseHeaders(401, -1);
            return;
        }

        String response = null;
        if (pathList.size() > 1) {
            switch (pathList.get(1)) {
                case "startSession":
                    if (keyParams.size() < 2) {
                        exchange.sendResponseHeaders(401, -1);
                        return;
                    }
                    response = startSession(paramsMap, keyParams);
                    break;
                case "stopSession":
                    response = stopSession(paramsMap, keyParams);
                    break;
                case "logSession":
                    response = logSession(paramsMap, keyParams);
                    break;
                default:
                    break;
            }
        }

        if (response != null) {
            exchange.sendResponseHeaders(200, response.length());
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            OutputStream out = exchange.getResponseBody();
            out.write(response.getBytes(StandardCharsets.UTF_8));
            out.flush();
            out.close();
            return;
        }

        exchange.sendResponseHeaders(400, -1);
    }
    
    private String startSession(Map<String, Object> params, Map<String, String> keyParams) {
        if (params.containsKey("symbol")) {
            String symbol = (String) params.get("symbol");

            if (sessions.containsKey(keyParams.get("apiKey"))) {
                return new JSONObject().put("response", "Already on session").toString();
            }

            if (params.containsKey("amount")) {
                double amount = Double.parseDouble("" + params.get("amount"));

                Long stopTime;
                if (params.containsKey("stopTime")) {
                    stopTime = (Long) params.get("stopTime");
                } else {
                    stopTime = Long.MAX_VALUE;
                }

                Double stopEarn;
                if (params.containsKey("stopEarn")) {
                    stopEarn = Double.parseDouble("" + params.get("stopEarn"));
                } else {
                    stopEarn = Double.POSITIVE_INFINITY;
                }

                Double stopLoss;
                if (params.containsKey("stopLoss")) {
                    stopLoss = Double.parseDouble("" + params.get("stopLoss"));
                } else {
                    stopLoss = Double.POSITIVE_INFINITY;
                }

                Session session = null;

                try {
                    session = new Session((String) keyParams.get("apiKey"), (String) keyParams.get("secretKey"), symbol, BinanceUtil.testnetBaseUrl, amount, stopTime, stopEarn, stopLoss);
                } catch (Exception e) {
                    return new JSONObject().put("response", e.getMessage()).toString();
                }

                session.start();
                sessions.put(keyParams.get("apiKey"), session);
                return new JSONObject().put("response", "Started").toString();

            } else {
                return new JSONObject().put("response", "Missing amount").toString();
            }
        } else {
            return new JSONObject().put("response", "Missing symbol").toString();
        }
    }
    
    private String stopSession(Map<String, Object> params, Map<String, String> keyParams) {
        if (params.containsKey("symbol")) {
            String symbol = (String) params.get("symbol");
            if (sessions.containsKey(keyParams.get("apiKey"))) {
                Session session = sessions.get(keyParams.get("apiKey"));
                if (session.getSymbol().equals(symbol)) {
                    session.interrupt();
                    return new JSONObject().put("response", "Stopped").toString();
                } else {
                    return new JSONObject().put("response", "No session with this symbol").toString();
                }
            } else {
                return new JSONObject().put("response", "No session running for this api key").toString();
            }
        } else {
            return new JSONObject().put("response", "Missing symbol").toString();
        }
    }

    private String logSession(Map<String, Object> params, Map<String, String> keyParams) {
        if (params.containsKey("symbol")) {
            String symbol = (String) params.get("symbol");
            if (sessions.containsKey(keyParams.get("apiKey"))) {
                Session session = sessions.get(keyParams.get("apiKey"));
                if (session.getSymbol().equals(symbol)) {
                    Queue<String> logs = session.getLogs();
                    JSONArray jsonArr = new JSONArray();
                    synchronized (logs) {
                        while (!logs.isEmpty()) {
                            jsonArr.put(logs.poll());
                        }
                    }
                    return new JSONObject().put("response", jsonArr).toString();
                } else {
                    return new JSONObject().put("response", "No session with this symbol").toString();
                }
            } else {
                return new JSONObject().put("response", "No session running for this api key").toString();
            }
        } else {
            return new JSONObject().put("response", "Missing symbol").toString();
        }
    }
    
    private Map<String, String> getKeyParameters(Map<String, Object> parameters) {
        Map<String, String> keyParameters = new LinkedHashMap<>();

        if (parameters.containsKey("apiKey")) {
            keyParameters.put("apiKey", (String)parameters.get("apiKey"));
            parameters.remove("apiKey");
        }

        if (parameters.containsKey("secretKey")) {
            keyParameters.put("secretKey", (String)parameters.get("secretKey"));
            parameters.remove("secretKey");
        }
        
        return keyParameters;
    }
}

package modelserver;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.json.JSONObject;

import java.io.IOException;
import java.io.OutputStream;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Scanner;

public class HttpHandlerImplML implements HttpHandler {
    private LinearRegressionModel model;
    
    public HttpHandlerImplML(LinearRegressionModel model) {
        this.model = model;
    }
    
    @Override
    public void handle(HttpExchange httpExchange) throws IOException {
        System.out.println("Handling");
        String method = httpExchange.getRequestMethod();
        switch (method) {
            case "POST":
                doPost(httpExchange);
                break;
            default:
                httpExchange.sendResponseHeaders(405, -1);
                break;
        }
    }
    
    private void doPost(HttpExchange httpExchange) throws IOException {
        System.out.println("Doing post");
        String parameters = new String(new Scanner(httpExchange.getRequestBody()).nextLine()
                .getBytes(StandardCharsets.UTF_8));
        ParameterParser parser = new ParameterParser(parameters);
        Map<String, Object> parametersMap = parser.getParameters();
        System.out.println(parametersMap);
        if (parametersMap.size() == 5) {
            if(parametersMap.containsKey("open") && parametersMap.containsKey("high") &&
                parametersMap.containsKey("low") && parametersMap.containsKey("close") &&
                    parametersMap.containsKey("actualOpen")) {
                /*double[] params = new double[]{(double) parametersMap.get("open"),
                    (double) parametersMap.get("low"),
                    (double) parametersMap.get("actualOpen")};*/
                double[] params = new double[]{(double) parametersMap.get("actualOpen"),
                        (double) parametersMap.get("open"),
                        (double) parametersMap.get("high"),
                        (double) parametersMap.get("low")};
                double prediction = model.predict(new DenseVector(params));
                double truncatedPrediction = BigDecimal.valueOf(prediction)
                                .setScale(2, RoundingMode.DOWN).doubleValue();
                System.out.println(prediction);
                JSONObject json = new JSONObject();
                json.put("prediction", truncatedPrediction);
                httpExchange.sendResponseHeaders(200, json.toString().length());
                httpExchange.getResponseHeaders().set("Content-Type", "application/json");
                OutputStream out = httpExchange.getResponseBody();
                out.write(json.toString().getBytes());
                out.flush();
                out.close();
                return;
            }
        }
        
        httpExchange.sendResponseHeaders(400, -1);
    }
}

import com.sun.net.httpserver.HttpServer;
import modelserver.HttpHandlerImplML;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.SparkSession;

import java.net.InetSocketAddress;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

public class Main {
    public static void main(String[] args) throws Exception {
        SparkSession session = SparkSession
                .builder()
                .appName("ModelServer")
                .master("local")
                .getOrCreate();
        
        LinearRegressionModel model = LinearRegressionModel.load("high_predictor_large_dataset");
        
        InetSocketAddress addr = new InetSocketAddress(6666);
        HttpServer server = HttpServer.create(addr, 128);
        server.createContext("/ml", new HttpHandlerImplML(model));
        ThreadPoolExecutor threadPoolExecutor = (ThreadPoolExecutor) Executors.newFixedThreadPool(10);
        server.setExecutor(threadPoolExecutor);
        server.start();
        System.out.println("Model server listening on port 6666");
    }
}
package bot;

import binance.BinanceUtil;
import binance.types.KLine;
import org.json.JSONObject;

import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class DataProducer extends Thread {
    private static final Map<String, DataProducer> instancesMapMain = new ConcurrentHashMap<>();
    private static final Map<String, DataProducer> instancesMapTest = new ConcurrentHashMap<>();
    private final DataObtainer dataObtainer;
    private final String symbol;
    private final String baseUrl;
    private double modelProcessedData;
    
    private DataProducer(String symbol, String baseUrl) {
        this.symbol = symbol;
        this.baseUrl = baseUrl;
        dataObtainer = new DataObtainer(symbol, baseUrl);
    }
    
    public static DataProducer getInstance(String symbol, String baseUrl) throws RuntimeException {
        //Verificare cosa succede con symbol invalido
        if (baseUrl.equals(BinanceUtil.mainnetBaseUrl)) {
            if (!instancesMapMain.containsKey(symbol)) {
                synchronized (DataProducer.class) {
                    if (!instancesMapMain.containsKey(symbol)) {
                        DataProducer dp = new DataProducer(symbol, baseUrl);
                        instancesMapMain.put(symbol, dp);
                        dp.start();
                        System.out.println("Nuova istanza creata");
                    } else {
                        System.out.println("Ritorno istanza gia esistente dal sync");
                    }
                }
            } else {
                System.out.println("Ritorno istanza gia esistente");
            }
            return instancesMapMain.get(symbol);
        }
        if (baseUrl.equals(BinanceUtil.testnetBaseUrl)) {
            if (!instancesMapTest.containsKey(symbol)) {
                synchronized (DataProducer.class) {
                    if (!instancesMapTest.containsKey(symbol)) {
                        DataProducer dp = new DataProducer(symbol, baseUrl);
                        instancesMapTest.put(symbol, dp);
                        dp.start();
                    }
                }
            }
            return instancesMapTest.get(symbol);
        }
        
        throw new RuntimeException("Invalid baseUrl");
    }
    
    public double getModelProcessedData() {
        return modelProcessedData;
    }
    
    public String getSymbol() {
        return symbol;
    }
    
    public String getBaseUrl() {
        return baseUrl;
    }
    
    @Override
    public void run() {
        dataObtainer.start();
        while (true) {
            synchronized (dataObtainer) {
                try {
                    dataObtainer.wait();
                } catch (InterruptedException e) {
                    break;
                }
            }
            List<KLine> obtainedData = dataObtainer.getData();
            KLine prevMin5 = obtainedData.get(0);
            KLine prevMin4 = obtainedData.get(1);
            KLine prevMin3 = obtainedData.get(2);
            KLine prevMin2 = obtainedData.get(3);
            KLine prevMin1 = obtainedData.get(4);
            KLine actualMin = obtainedData.get(5);
            
            String request = "close5=" + prevMin5.getClose() +
                    "&" + "close4=" + prevMin4.getClose() +
                    "&" + "close3=" + prevMin3.getClose() +
                    "&" + "close2=" + prevMin2.getClose() +
                    "&" + "close1=" + prevMin1.getClose();
            URL url = null;
            HttpURLConnection conn = null;
                    try{
                        url = new URL("http://localhost:6666/ml");
                        conn = (HttpURLConnection) url.openConnection();
                        conn.setRequestMethod("POST");
                        conn.setDoOutput(true);
                        conn.setDoInput(true);
                        conn.getOutputStream().write(request.getBytes(StandardCharsets.UTF_8));
                        JSONObject json = new JSONObject(new String(conn.getInputStream().readAllBytes(), StandardCharsets.UTF_8));
                        modelProcessedData = json.getDouble("prediction");
                    } catch (Exception e) {
                        System.out.println("[DATA PRODUCER] Couldn't connect to model server");
                        continue;
                    }
            System.out.println("[DATA PRODUCER] Data produced");
            synchronized (this) {
                this.notifyAll();
            }
        }
    }
}

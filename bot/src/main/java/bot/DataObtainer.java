/*
    This class obtains data for the model from Binance every minute.
    Notifies the owner of this object to make it use data as soon
    as they are ready.
 */

package bot;

import binance.BinanceCommon;
import binance.types.KLine;
import com.binance.connector.client.exceptions.BinanceConnectorException;

import java.util.List;
import java.util.Map;
import java.util.Timer;
import java.util.TimerTask;

public class DataObtainer {
    private BinanceCommon common;
    private Timer timer;
    private Map<String, Object> parameters;
    private List<KLine> retrievedData;
    private final String symbol;
    private final DataObtainer thisDataObtainer = this;
    
    public DataObtainer(String symbol, String baseUrl) {
        common = new BinanceCommon(baseUrl);
        timer = new Timer();
        this.symbol = symbol;
    }
    
    public void start() {
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                try {
                    do {
                        retrievedData = common.getKLines(symbol, "1m", 6);
                    } while (retrievedData.get(5).getTimestamp() != calculateStartMinuteTimestamp());
                    synchronized (thisDataObtainer) {
                        thisDataObtainer.notify();
                    }
                } catch (BinanceConnectorException bce) {
                    System.out.println("[DATA OBTAINER] Couldn't get data");
                    //bce.printStackTrace();
                }
            }
        }, calculateDelay(), 60000);
    }
    
    public List<KLine> getData() {
        return retrievedData;
    }
    
    private long calculateDelay() {
        long now = System.currentTimeMillis();
        return ((long)Math.ceil(now / 60000) * 60000 + 60000) - now;
    }
    
    private long calculateStartMinuteTimestamp() {
        long now = System.currentTimeMillis();
        return ((long)Math.ceil(now / 60000) * 60000);
    }
}

package bot;

import binance.BinanceCommon;
import binance.BinanceUser;
import binance.BinanceUtil;
import binance.types.BinanceOrder;
import binance.types.OrderBook;
import binance.types.OrderBookLine;
import com.binance.connector.client.SpotClient;
import com.binance.connector.client.impl.SpotClientImpl;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class Session extends Thread {
    private final String apiKey;
    private final DataProducer dataProducer;
    private final BinanceCommon common;
    private final BinanceUser user;
    private final String symbol;
    private final double usableAmount;
    private final Long stopTime;
    private final Double stopEarn;
    private final Double stopLoss;
    private double earned;
    private final Queue<String> logs;
    
    public Session(String apiKey, String secretKey, String symbol, String baseUrl, double amount, Long stopTime,

                   Double stopEarn, Double stopLoss) throws Exception {
        if (BinanceUtil.minTradeAmount > amount) {
            throw new Exception("Need higher amount");
        }
        //Session min duration is 2000 seconds
        if (System.currentTimeMillis() + 120000 > stopTime) {
            throw new Exception("Need longer stopTime");
        }
        this.apiKey = apiKey;
        this.symbol = symbol;
        dataProducer = DataProducer.getInstance(symbol, baseUrl);
        SpotClient client = new SpotClientImpl(apiKey,secretKey, baseUrl);
        common = new BinanceCommon(client);
        user = new BinanceUser(client);
        usableAmount = amount;

        if (user.getFreeBalance("USDT") < usableAmount) {
            throw new Exception("Insufficient balance");
        }

        this.stopTime = stopTime;
        this.stopEarn = stopEarn;
        this.stopLoss = stopLoss;
        earned = 0;

        logs = new LinkedList<>();
    }

    @Override
    public void run() {
        double amount = earned <= 0 ? usableAmount : usableAmount + earned;
        double residualFromBuy = 0;
        Long prevOrderId = null;

        double lastBuyPrice = 0;
        
        double partiallyFilledAmount = 0;

        synchronized (logs) {
            System.out.println("[SESSION] Started\n[SESSION] Balance: " + user.getFreeBalance("USDT") + "USDT");
            logs.offer("[SESSION] Started\n[SESSION] Balance: " + user.getFreeBalance("USDT") + "USDT");
        }
            while (true) {
                synchronized (logs) {

                    //color
                    if (earned > 0) {
                        System.out.println("\u001B[32m");
                    }
                    if (earned < 0) {
                        System.out.println("\u001B[31m");
                    }

                    System.out.println("[SESSION] Max amount: " + usableAmount + "\t|\tamount: " + amount + "\t|\tresidual: " +
                        residualFromBuy + "\t|\tearning: " + earned);

                    //reset color
                    System.out.println("\u001B[0m");

                    System.out.println("[SESSION] Waiting for prediction...");

                    logs.offer("[SESSION] Max amount: " + usableAmount + "\t|\tamount: " + amount + "\t|\tresidual: " +
                        residualFromBuy + "\t|\tearning: " + earned + "\n[SESSION] Waiting for prediction...");
                }
                /*
                 * wait for prediction from dataProducer
                 */
                synchronized (dataProducer) {
                    try {
                        dataProducer.wait();
                    } catch (InterruptedException e) {
                        synchronized (logs) {
                            System.out.println("[SESSION] Session stopped");
                            logs.offer("[SESSION] Session stopped");
                        }
                        break;
                    }
                }
                synchronized (logs) {
                    double prediction = dataProducer.getModelProcessedData();
                    System.out.println("[SESSION] Notified, back to work");
                    logs.offer("[SESSION] Notified, back to work");
                    try {
                        if (prevOrderId != null) {
                            //check open sell order from last iteration
                            BinanceOrder prevOrder = user.getOrderById(symbol, prevOrderId);
                            String prevOrderStatus = prevOrder.getStatus();

                            if (prevOrderStatus.equals(BinanceUtil.orderStatusPartiallyFilled) ||
                                prevOrderStatus.equals(BinanceUtil.orderStatusNew)) {
                                System.out.println("[SESSION] Sell order status " + prevOrderStatus);
                                logs.offer("[SESSION] Sell order status " + prevOrderStatus);

                                BigDecimal orderBookQtySum = BigDecimal.valueOf(0.0);

                                OrderBook orderBook = common.getOrderBook(symbol, 100);
                                while (orderBook.getBids().isEmpty()) {
                                    orderBook = common.getOrderBook(symbol, 100);
                                }
                                List<OrderBookLine> bids = orderBook.getBids();
                                int i = 0;
                                for (; i < bids.size() && bids.get(i).getPrice() > lastBuyPrice; i++) {
                                    orderBookQtySum.add(BigDecimal.valueOf(bids.get(i).getQty()));
                                }

                                if (i >= bids.size()) {
                                    i = bids.size() - 1;
                                }

                                BigDecimal tmpPartiallyFilled = BigDecimal.valueOf(prevOrder.getOrigQty())
                                    .subtract(BigDecimal.valueOf(prevOrder.getExecutedQty()));

                                double curr = bids.get(i).getPrice();

                                if (tmpPartiallyFilled.multiply(BigDecimal.valueOf(bids.get(0).getPrice())).doubleValue() < BinanceUtil.minTradeAmount) {
                                    System.out.println("[SESSION] Amount to sell lower than MIN_NOTIONAL, order won't be cancelled, waiting to be filled...");
                                    logs.offer("[SESSION] Amount to sell lower than MIN_NOTIONAL, order won't be cancelled, waiting to be filled...");
                                }

                                try {
                                    if (orderBookQtySum.doubleValue() >= tmpPartiallyFilled.doubleValue()) {

                                        System.out.println("[SESSION] Sell order price too high," +
                                            " but current price (" + curr + ") is higher than last bought price (" + lastBuyPrice + ")");
                                        logs.offer("[SESSION] Sell order price too high," +
                                            " but current price (" + curr + ") is higher than last bought price (" + lastBuyPrice + ")");

                                        user.cancelOrder(symbol, prevOrderId);

                                        prevOrder = user.getOrderById(symbol, prevOrderId);

                                        partiallyFilledAmount = BigDecimal.valueOf(partiallyFilledAmount)
                                            .add(BigDecimal.valueOf(prevOrder.getCummulativeQuoteQty()))
                                            .doubleValue();

                                        System.out.println("[SESSION] Order cancelled, placing new sell order at current price...");
                                        logs.offer("[SESSION] Order cancelled, placing new sell order at current price...");

                                        double toSell = BigDecimal.valueOf(prevOrder.getOrigQty())
                                            .subtract(BigDecimal.valueOf(prevOrder.getExecutedQty())).doubleValue();
                                        prevOrderId = user.sell(symbol, toSell);

                                        prevOrder = user.getOrderById(symbol, prevOrderId);
                                        curr = BigDecimal.valueOf(prevOrder.getCummulativeQuoteQty())
                                            .divide(BigDecimal.valueOf(prevOrder.getExecutedQty()), 2, RoundingMode.UP)
                                            .doubleValue();
                                        System.out.println("[SESSION] Sold at " + curr);
                                        logs.offer("[SESSION] Sold at " + curr);
                                    } else {
                                        System.out.println("[SESSION] Current price (" + curr + ") is lower than last bought price (" + lastBuyPrice + ")");
                                        logs.offer("[SESSION] Current price (" + curr + ") is lower than last bought price (" + lastBuyPrice + ")");
                                        if (prediction > lastBuyPrice) {
                                            System.out.println("[SESSION] Prediction (" + prediction +
                                                ") is more than last buy price (" + lastBuyPrice + ")");
                                            logs.offer("[SESSION] Prediction (" + prediction +
                                                ") is more than last buy price (" + lastBuyPrice + ")");

                                            if (prediction > prevOrder.getPrice()) {
                                                System.out.println("[SESSION] Leaving previous sell order because prediction is higher (if prediction is correct our old sell order will be filled for sure)");
                                                logs.offer("[SESSION] Leaving previous sell order because prediction is higher (if prediction is correct our old sell order will be filled for sure)");
                                                continue;
                                            }

                                            user.cancelOrder(symbol, prevOrderId);

                                            prevOrder = user.getOrderById(symbol, prevOrderId);

                                            partiallyFilledAmount = BigDecimal.valueOf(partiallyFilledAmount)
                                                .add(BigDecimal.valueOf(prevOrder.getCummulativeQuoteQty()))
                                                .doubleValue();

                                            System.out.println("[SESSION] Order cancelled, placing new sell order at current prediction (" +
                                                prediction + ")");
                                            logs.offer("[SESSION] Order cancelled, placing new sell order at current prediction (" +
                                                prediction + ")");

                                            double toSell = BigDecimal.valueOf(prevOrder.getOrigQty())
                                                .subtract(BigDecimal.valueOf(prevOrder.getExecutedQty())).doubleValue();
                                            prevOrderId = user.placeSellOrder(symbol, toSell, prediction);

                                            prevOrder = user.getOrderById(symbol, prevOrderId);
                                            System.out.println("[SESSION] New sell order placed, going for the next minute...");
                                            logs.offer("[SESSION] New sell order placed, going for the next minute...");
                                            continue;
                                        } else {
                                            user.cancelOrder(symbol, prevOrderId);

                                            prevOrder = user.getOrderById(symbol, prevOrderId);

                                            partiallyFilledAmount = BigDecimal.valueOf(partiallyFilledAmount)
                                                .add(BigDecimal.valueOf(prevOrder.getCummulativeQuoteQty()))
                                                .doubleValue();

                                            double toSell = BigDecimal.valueOf(prevOrder.getOrigQty())
                                                .subtract(BigDecimal.valueOf(prevOrder.getExecutedQty())).doubleValue();
                                            prevOrderId = user.sell(symbol, toSell);
                                            prevOrder = user.getOrderById(symbol, prevOrderId);

                                            curr = BigDecimal.valueOf(prevOrder.getCummulativeQuoteQty())
                                                .divide(BigDecimal.valueOf(prevOrder.getExecutedQty()), 2, RoundingMode.UP)
                                                .doubleValue();

                                            System.out.println("[SESSION] Can't save this sell order, sold at current price (" +
                                                curr + ") :(");
                                            logs.offer("[SESSION] Can't save this sell order, sold at current price (" +
                                                curr + ") :(");
                                        }
                                    }
                                } catch (Exception e) {
                                    e.printStackTrace();
                                    if (e.getMessage().contains("-2011")) {
                                        partiallyFilledAmount = 0;
                                        continue;
                                    }

                                    if (e.getMessage().contains("timeout")) {
                                        System.out.println("[SESSION] Connection error, session stopped");
                                    }

                                    return;
                                }
                            }

                            BigDecimal prevCummulative = BigDecimal.valueOf(prevOrder.getCummulativeQuoteQty())
                                .add(BigDecimal.valueOf(partiallyFilledAmount));
                            BigDecimal tmpAmount = BigDecimal.valueOf(amount);

                            BigDecimal bigEarned = BigDecimal.valueOf(earned);

                            earned = bigEarned.add(prevCummulative.subtract(tmpAmount)).doubleValue();
                            amount = prevCummulative.add(BigDecimal.valueOf(residualFromBuy)).doubleValue();
                            if (amount > usableAmount) {
                                amount = usableAmount;
                            }
                            //residual reset
                            residualFromBuy = 0;
                            partiallyFilledAmount = 0;

                            System.out.println("[SESSION] Sell order filled, earning " + earned + ", balance: " +
                                user.getFreeBalance("USDT") + "USDT");
                            logs.offer("[SESSION] Sell order filled, earning " + earned + ", balance: " +
                                user.getFreeBalance("USDT") + "USDT");
                        }

                        if (checkStopTime()) {
                            if (checkAmount(amount)) {
                                if (checkLoss()) {
                                    if (checkEarned()) {
                                        double currentPrice;
                                        OrderBook orderBook = common.getOrderBook(symbol, 100);
                                        while (orderBook.getAsks().isEmpty()) {
                                            orderBook = common.getOrderBook(symbol, 100);
                                        }

                                        //System.out.println("[SESSION] Prediction: " + prediction + "\tcurrent price: " + currentPrice);

                                        if (prediction > (currentPrice = orderBook.getMinSellPrice())) {

                                            BigDecimal orderBookQuoteQtySum = BigDecimal.valueOf(0.0);

                                            List<OrderBookLine> asks = orderBook.getAsks();
                                            boolean hasBought = false;
                                            int i = 0;
                                            for (; i < asks.size() && asks.get(i).getPrice() < prediction; i++) {
                                                orderBookQuoteQtySum = orderBookQuoteQtySum.add(BigDecimal
                                                    .valueOf(asks.get(i).getQuoteQuantity()));

                                                if (amount <= orderBookQuoteQtySum.doubleValue()) {
                                                    prevOrderId = user.buy(symbol, amount);
                                                    hasBought = true;
                                                    i++;
                                                    break;
                                                }
                                            }

                                            if (!hasBought && orderBookQuoteQtySum.doubleValue() >= BinanceUtil.minTradeAmount) {
                                                prevOrderId = user.buy(symbol, orderBookQuoteQtySum.doubleValue());
                                                hasBought = true;
                                            }

                                            if (!hasBought) {
                                                System.out.println("[SESSION] Prediction: " + prediction + ", current price: " + currentPrice + "\n[SESSION] Couldn't buy, no sell orders satisfied prediction or min trade amount, going for the next minute");
                                                logs.offer("[SESSION] Prediction: " + prediction + ", current price: " + currentPrice + "\n[SESSION] Couldn't buy, no sell orders satisfied prediction or min trade amount, going for the next minute");
                                                prevOrderId = null;
                                                continue;
                                            }

                                            currentPrice = asks.get(i - 1).getPrice();

                                            BinanceOrder buyOrder = user.getOrderById(symbol, prevOrderId);

                                            lastBuyPrice = BigDecimal.valueOf(buyOrder.getCummulativeQuoteQty())
                                                .divide(BigDecimal.valueOf(buyOrder.getExecutedQty()), 2, RoundingMode.UP).doubleValue();

                                            System.out.println("[SESSION] Prediction (" + prediction + ") is more than current price (" +
                                                currentPrice + "), bought at an average price of " + lastBuyPrice + " (filled with " + i + " orders)");
                                            logs.offer("[SESSION] Prediction (" + prediction + ") is more than current price (" +
                                                currentPrice + "), bought at an average price of " + lastBuyPrice + " (filled with " + i + " orders)");

                                            lastBuyPrice = currentPrice;

                                            BigDecimal cummulativeFromBuy = BigDecimal.valueOf(buyOrder.getCummulativeQuoteQty());
                                            BigDecimal tmpResidual = BigDecimal.valueOf(amount).subtract(cummulativeFromBuy);

                                            residualFromBuy = tmpResidual.doubleValue();

                                            amount = cummulativeFromBuy.doubleValue();

                                            System.out.println("[SESSION] Successfully bought " + buyOrder.getExecutedQty() + " (" +
                                                amount + "USDT)");
                                            logs.offer("[SESSION] Successfully bought " + buyOrder.getExecutedQty() + " (" +
                                                amount + "USDT)");

                                            prevOrderId = user.placeSellOrder(symbol, buyOrder.getExecutedQty(),
                                                dataProducer.getModelProcessedData());

                                            BinanceOrder sellOrder = user.getOrderById(symbol, prevOrderId);
                                            System.out.println("[SESSION] Placed sell order at " + sellOrder.getPrice());
                                            logs.offer("[SESSION] Placed sell order at " + sellOrder.getPrice());

                                        } else {
                                            prevOrderId = null;
                                            System.out.println("[SESSION] Prediction (" + prediction + ") is less than current price (" + currentPrice + "), going for the next minute...");
                                            logs.offer("[SESSION] Prediction (" + prediction + ") is less than current price (" + currentPrice + "), going for the next minute...");
                                        }
                                    } else {
                                        System.out.println("Stop earn reached");
                                        logs.offer("Stop earn reached");
                                        break;
                                    }
                                } else {
                                    System.out.println("Stop loss reached");
                                    logs.offer("Stop loss reached");
                                    break;
                                }
                            } else {
                                System.out.println("Insufficient amount or balance");
                                logs.offer("Insufficient amount or balance");
                                break;
                            }
                        } else {
                            System.out.println("Stop time reached");
                            logs.offer("Stop time reached");
                            break;
                        }
                    } catch (Exception e) {
                        //Fare la stessa cosa nell'else del checkBalance()
                        //Dovrà ritornare qualcosa tipo "Insufficient amount or balance"
                        if (e.getMessage().contains("timeout")) {
                            System.out.println("[SESSION] Connection error, session stopped");
                        }
                        e.printStackTrace();
                        return;
                    }
                }
            }


        try {
            if (prevOrderId != null) {
                BinanceOrder prevOrder = user.getOrderById(symbol, prevOrderId);

                if (prevOrder.getStatus().equals(BinanceUtil.orderStatusNew)
                    || prevOrder.getStatus().equals(BinanceUtil.orderStatusPartiallyFilled)) {

                    user.cancelOrder(symbol, prevOrderId);

                    prevOrder = user.getOrderById(symbol, prevOrderId);

                    partiallyFilledAmount = BigDecimal.valueOf(partiallyFilledAmount)
                        .add(BigDecimal.valueOf(prevOrder.getCummulativeQuoteQty()))
                        .doubleValue();

                    double toSell = BigDecimal.valueOf(prevOrder.getOrigQty())
                        .subtract(BigDecimal.valueOf(prevOrder.getExecutedQty())).doubleValue();
                    prevOrderId = user.sell(symbol, toSell);

                    BigDecimal prevCummulative = BigDecimal.valueOf(prevOrder.getCummulativeQuoteQty())
                        .add(BigDecimal.valueOf(partiallyFilledAmount));
                    BigDecimal tmpAmount = BigDecimal.valueOf(amount);

                    BigDecimal bigEarned = BigDecimal.valueOf(earned);

                    earned = bigEarned.add(prevCummulative.subtract(tmpAmount)).doubleValue();
                    synchronized (logs) {
                        System.out.println("[SESSION] Session stopped, closed pending orders, earning " + earned + ", balance: " +
                            user.getFreeBalance("USDT") + "USDT");
                        logs.offer("[SESSION] Session stopped, closed pending orders, earning " + earned + ", balance: " +
                            user.getFreeBalance("USDT") + "USDT");
                    }
                }
            }
        } catch (Exception e) {
            if (e.getMessage().contains("-2011")) {
                return;
            }

            if (e.getMessage().contains("timeout")) {
                System.out.println("[SESSION] Connection error, session stopped");
            }
        }
    }
    
    private boolean checkStopTime() {
        return stopTime == null || System.currentTimeMillis() < stopTime;
    }
    
    private boolean checkAmount(double amount) {
        return amount >= BinanceUtil.minTradeAmount && amount < user.getFreeBalance("USDT");
    }

    public boolean checkLoss() {
        return earned > (stopLoss * -1);
    }
    
    private boolean checkEarned() {
        return earned < stopEarn;
    }

    public String getSymbol() {
        return symbol;
    }

    public Queue<String> getLogs() {
        return logs;
    }
}

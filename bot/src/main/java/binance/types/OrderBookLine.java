package binance.types;

import java.math.BigDecimal;

public class OrderBookLine {
  private final double price;
  private final double qty;

  public OrderBookLine(double price, double qty) {
    this.price = price;
    this.qty = qty;
  }

  public double getPrice() {
    return price;
  }

  public double getQty() {
    return qty;
  }

  public double getQuoteQuantity() {
    BigDecimal price = BigDecimal.valueOf(this.price);
    BigDecimal qty = BigDecimal.valueOf(this.qty);

    return price.multiply(qty).doubleValue();
  }

  @Override
  public String toString() {
    return "OrderBookLine{" +
        "price=" + price +
        ", qty=" + qty +
        '}';
  }
}
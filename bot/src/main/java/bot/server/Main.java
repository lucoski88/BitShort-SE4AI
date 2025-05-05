package bot.server;

import com.sun.net.httpserver.HttpServer;

import java.net.InetSocketAddress;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

public class Main {
    private static final int PORT = 7777;
    private static final int BACK_LOG = 128;

    public static void main(String[] args) throws Exception {
        InetSocketAddress addr = new InetSocketAddress(PORT);
        HttpServer server = HttpServer.create(addr, BACK_LOG);
        server.createContext("/bot", new HttpHandlerImpl());
        ThreadPoolExecutor threadPoolExecutor = (ThreadPoolExecutor) Executors.newFixedThreadPool(10);
        server.setExecutor(threadPoolExecutor);
        server.start();

        System.out.println("Bot listening on port " + PORT);
    }
}
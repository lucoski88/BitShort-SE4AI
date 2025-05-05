package bot.server;

import java.util.*;

public class URIParser {
  private String uri;
  private List<String> path;
  private Map<String, Object> params;

  public URIParser(String uri) {
    this.uri = uri;

    path = new ArrayList<>();
    params = new LinkedHashMap<>();

    if (uri.contains("?") && uri.indexOf("?") < uri.length() - 1) {
      String pathStr = uri.substring(1, uri.indexOf("?"));
      String paramsStr = uri.substring(uri.indexOf("?") + 1);

      path.addAll(Arrays.asList(pathStr.split("/")));

      ParameterParser parameterParser = new ParameterParser(paramsStr);
      params = parameterParser.getParameters();

    } else if (uri.contains("?")) {
      String pathStr = uri.substring(1, uri.indexOf("?"));
      path.addAll(Arrays.asList(pathStr.split("/")));
    } else {
      String pathStr = uri.substring(1);
      path.addAll(Arrays.asList(pathStr.split("/")));
    }
  }

  public List<String> getPath() {
    return path;
  }

  public Map<String, Object> getParameters() {
    return params;
  }
}
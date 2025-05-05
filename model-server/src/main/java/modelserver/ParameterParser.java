package modelserver;

import java.util.LinkedHashMap;
import java.util.Map;

public class ParameterParser {
  private String paramsStr;
  private Map<String, Object> params;

  public ParameterParser(String paramsStr) {
    this.paramsStr = paramsStr;
    this.params = new LinkedHashMap<>();

    String[] paramsArr = paramsStr.split("&");
    for (String str : paramsArr) {
      String[] tmp = str.split("=");
      if (tmp.length == 2) {
        try {
          params.put(tmp[0], Integer.parseInt(tmp[1]));
        } catch (NumberFormatException e) {
          try {
            params.put(tmp[0], Long.parseLong(tmp[1]));
          } catch (NumberFormatException e1) {
            try {
              params.put(tmp[0], Double.parseDouble(tmp[1]));
            } catch (NumberFormatException e2) {
              params.put(tmp[0], tmp[1]);
            }
          }
        }
      }
    }
  }

  public Map<String, Object> getParameters() {
    return params;
  }
}
public class SimplePrettyPrinter {

    public static String prettyPrintJson(String json) {
        StringBuilder prettyJson = new StringBuilder();
        int indent = 0;
        boolean inQuotes = false;

        for (char c : json.toCharArray()) {
            switch (c) {
                case '"':
                    prettyJson.append(c);
                    if (c == '"') {
                        inQuotes = !inQuotes;
                    }
                    break;
                case '{':
                case '[':
                    prettyJson.append(c);
                    if (!inQuotes) {
                        prettyJson.append('\n');
                        indent++;
                        appendIndent(prettyJson, indent);
                    }
                    break;
                case '}':
                case ']':
                    if (!inQuotes) {
                        prettyJson.append('\n');
                        indent--;
                        appendIndent(prettyJson, indent);
                    }
                    prettyJson.append(c);
                    break;
                case ',':
                    prettyJson.append(c);
                    if (!inQuotes) {
                        prettyJson.append('\n');
                        appendIndent(prettyJson, indent);
                    }
                    break;
                case ':':
                    prettyJson.append(c);
                    if (!inQuotes) {
                        prettyJson.append(' ');
                    }
                    break;
                default:
                    prettyJson.append(c);
            }
        }
        return prettyJson.toString();
    }

    private static void appendIndent(StringBuilder sb, int indent) {
        for (int i = 0; i < indent; i++) {
            sb.append("  "); // 两个空格
        }
    }

    public static void main(String[] args) {
        String json = "{\"name\":\"John\",\"age\":null,\"city\":\"New York\"}";
        System.out.println(prettyPrintJson(json));
    }
}

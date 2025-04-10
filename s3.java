import software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider;
import software.amazon.awssdk.core.ResponseInputStream;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.*;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class S3SelectExample {
    public static void main(String[] args) {
        String bucket = "my-bucket";
        String key = "sample-data.csv";

        // 初始化 S3 客户端（使用默认配置文件）
        S3Client s3 = S3Client.builder()
                .credentialsProvider(ProfileCredentialsProvider.create())
                .build();

        // 构建 SelectObjectContent 请求
        SelectObjectContentRequest selectRequest = SelectObjectContentRequest.builder()
                .bucket(bucket)
                .key(key)
                .expression("SELECT * FROM S3Object s WHERE s.age > 30")
                .expressionType(ExpressionType.SQL)
                .inputSerialization(InputSerialization.builder()
                        .csv(CSVInput.builder().fileHeaderInfo(FileHeaderInfo.USE).build())
                        .compressionType(CompressionType.NONE)
                        .build())
                .outputSerialization(OutputSerialization.builder()
                        .csv(CSVOutput.builder().build())
                        .build())
                .build();

        try (ResponseInputStream<SelectObjectContentResponse> result = s3.selectObjectContent(selectRequest);
             BufferedReader reader = new BufferedReader(new InputStreamReader(result))) {

            String line;
            System.out.println("Query Results:");
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        s3.close();
    }
}

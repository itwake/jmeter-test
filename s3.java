import software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider;
import software.amazon.awssdk.services.s3.S3AsyncClient;
import software.amazon.awssdk.services.s3.model.*;

import java.util.concurrent.CompletableFuture;

public class S3SelectAsyncExample {
    public static void main(String[] args) {
        String bucket = "your-bucket-name";
        String key = "your-object-key";

        S3AsyncClient s3AsyncClient = S3AsyncClient.builder()
                .credentialsProvider(ProfileCredentialsProvider.create())
                .build();

        SelectObjectContentRequest request = SelectObjectContentRequest.builder()
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

        CompletableFuture<SelectObjectContentResponse> future = s3AsyncClient.selectObjectContent(request, new SelectObjectContentResponseHandler() {
            @Override
            public void responseReceived(SelectObjectContentResponse response) {
                // 处理响应元数据
            }

            @Override
            public void onEventStream(SdkPublisher<SelectObjectContentEventStream> publisher) {
                publisher.subscribe(event -> {
                    if (event instanceof RecordsEvent) {
                        RecordsEvent recordsEvent = (RecordsEvent) event;
                        // 处理返回的记录
                        System.out.println(new String(recordsEvent.payload().asByteArray()));
                    }
                });
            }

            @Override
            public void exceptionOccurred(Throwable throwable) {
                throwable.printStackTrace();
            }

            @Override
            public void complete() {
                System.out.println("SelectObjectContent completed");
            }
        });

        future.join(); // 等待操作完成
        s3AsyncClient.close();
    }
}

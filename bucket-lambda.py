import os
import boto3
import mimetypes
import base64

s3 = boto3.client('s3')
BUCKET = os.environ['BUCKET']  # e.g. "my-secure-bucket"

def lambda_handler(event, context):
    # HTTP API v2 path 在 event['rawPath']
    # REST API v1 path 在 event['pathParameters']['proxy']
    path = event.get('rawPath') or event.get('pathParameters', {}).get('proxy', '')
    key = path.lstrip('/') or 'index.html'  # 默认回退到 index.html

    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        body_bytes = obj['Body'].read()

        # 动态推断 Content-Type
        content_type, _ = mimetypes.guess_type(key)
        if not content_type:
            content_type = 'application/octet-stream'

        # 判断是否需要 Base64
        is_binary = not content_type.startswith('text/') and 'application/javascript' not in content_type

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': content_type,
                'Cache-Control': 'max-age=3600, public'
            },
            'isBase64Encoded': is_binary,
            'body': base64.b64encode(body_bytes).decode('utf-8') if is_binary else body_bytes.decode('utf-8')
        }

    except s3.exceptions.NoSuchKey:
        return {'statusCode': 404, 'body': 'Not Found'}
    except Exception as e:
        return {
            'statusCode': 500,
            'body': f'Error: {str(e)}'
        }

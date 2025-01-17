import json
import boto3
from botocore.exceptions import ClientError

def lambda_handler(event, context):
    ssm_client = boto3.client('ssm')
    parameter_name = '/my/parameter/key'  # 替换为您的参数名称

    try:
        # 尝试获取参数
        response = ssm_client.get_parameter(
            Name=parameter_name,
            WithDecryption=False  # 如果参数是加密的，设置为 True
        )
        current_value = int(response['Parameter']['Value'])
        new_value = current_value + 1
        print(f"参数存在，当前值: {current_value}，更新为: {new_value}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ParameterNotFound':
            # 参数不存在，初始化为 1
            new_value = 1
            print(f"参数不存在，创建并设置为: {new_value}")
        else:
            # 其他错误，重新抛出
            raise e

    try:
        # 设置参数值
        ssm_client.put_parameter(
            Name=parameter_name,
            Value=str(new_value),
            Type='String',  # 根据需要选择 'String', 'StringList' 或 'SecureString'
            Overwrite=True  # 如果参数存在则覆盖
        )
        return {
            'statusCode': 200,
            'body': json.dumps({
                'parameter': parameter_name,
                'value': new_value
            })
        }
    except ClientError as e:
        print(f"设置参数时出错: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Failed to set parameter'})
        }

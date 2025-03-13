import boto3
import base64
from cryptography.fernet import Fernet

# 初始化
kms_client = boto3.client('kms', region_name='ap-southeast-1')  # 替换成你的区域
key_id = 'arn:aws:kms:your-region:your-account-id:key/your-key-id'  # 你的KMS Key ARN

# 生成Data Key
response = kms_client.generate_data_key(KeyId=key_id, KeySpec='AES_256')

plaintext_data_key = response['Plaintext']          # 真实加密用的Key
encrypted_data_key = response['CiphertextBlob']     # 被KMS加密的Key

# 明文Data Key转换成Fernet Key（Fernet是AES加密的一个高级封装）
fernet_key = base64.urlsafe_b64encode(plaintext_data_key)

# 准备Fernet加密器
cipher = Fernet(fernet_key)

# 要加密的内容
plaintext = "MySuperSecretPass123!".encode('utf-8')

# 加密数据
encrypted_data = cipher.encrypt(plaintext)

# 打印结果
print("加密后的数据（Base64）：", base64.b64encode(encrypted_data).decode())
print("加密后的Data Key（Base64）：", base64.b64encode(encrypted_data_key).decode())

# ===> 存起来：encrypted_data 和 encrypted_data_key


# 假设你从存储里取回这两个
stored_encrypted_data_key = base64.b64decode("上面生成的加密Data Key的Base64")
stored_encrypted_data = base64.b64decode("上面生成的加密数据的Base64")

# 解密Data Key
response = kms_client.decrypt(CiphertextBlob=stored_encrypted_data_key)

plaintext_data_key = response['Plaintext']  # 还原明文Data Key

# 明文Data Key转换成Fernet Key
fernet_key = base64.urlsafe_b64encode(plaintext_data_key)

# 准备Fernet解密器
cipher = Fernet(fernet_key)

# 解密数据
decrypted_data = cipher.decrypt(stored_encrypted_data)

print("解密后的原文：", decrypted_data.decode())

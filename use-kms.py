import boto3
import base64

# 初始化KMS客户端
kms_client = boto3.client("kms", region_name="ap-southeast-1")  # 改成你的AWS区域

# 你的KMS密钥ID或ARN
key_id = "arn:aws:kms:your-region:your-account-id:key/your-key-id"

# 要加密的字符串
plaintext = "MySuperSecretPass123!"

# KMS加密
response = kms_client.encrypt(
    KeyId=key_id,
    Plaintext=plaintext.encode("utf-8")  # KMS要求输入是字节类型
)

# 获取加密后的数据
ciphertext_blob = response["CiphertextBlob"]

# 转换成Base64（方便存储或传输）
ciphertext_base64 = base64.b64encode(ciphertext_blob).decode("utf-8")

print("加密后的字符串（Base64）：")
print(ciphertext_base64)
# Base64解回二进制
ciphertext_blob = base64.b64decode(ciphertext_base64)

# KMS解密
response = kms_client.decrypt(CiphertextBlob=ciphertext_blob)

# 取出明文并转换成字符串
decrypted_plaintext = response["Plaintext"].decode("utf-8")

print("解密后的字符串：", decrypted_plaintext)

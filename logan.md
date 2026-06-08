下面是一份**可以直接开始测试**的安装与使用说明，分为两条路径：

1. **本地 Windows 测试**：推荐 Windows + WSL2 + Docker Desktop，适合开发、功能验证、full-stack smoke。
2. **AWS EKS / Kubernetes 测试**：适合验证接近云上部署的路径，但仍按“可落地试用”而不是完整生产级别来处理。

这份说明基于当前分支：

```text
repo: https://github.com/itwake/llm-powered-log-analytic
branch: codex/logan-platform-foundation
commit: ab86b966c5a2ce5202d7afb0eae96f1a09c2e293
```

当前项目目标是 LogAn-style 的 case-based incident log diagnosis platform：创建 case、上传日志、运行分析，并查看 Data Summary、Temporal View、Tabular Logs、Causal Graph、Causal Summary 五个视图；这也符合 LogAn 论文中“Drain 模板化 + representative log lines + golden signals / fault categories / entities + 五视图报告”的设计。 当前 README 也说明项目已经包含 FastAPI 后端、SQLAlchemy 存储、Temporal worker、MinIO/S3 上传、ClickHouse/OpenSearch 可选分析 sink、Next.js workbench、ECharts Temporal View 和 Cytoscape Causal Graph。([GitHub][1])

---

# 一、本地 Windows 安装与使用

## 1. 推荐机器配置

本地测试建议：

```text
Windows 11 或 Windows 10 2004+
CPU: 4 核以上
内存: 最低 16GB，推荐 32GB
磁盘: 至少 30GB 空闲
网络: 可访问 GitHub、Docker Hub / mirror.gcr.io、npm、PyPI
```

Docker full-stack smoke 会启动 PostgreSQL、MinIO、ClickHouse、OpenSearch、Temporal、API、worker 等服务；项目文档也提醒 OpenSearch 虽然被限制为 512MB heap，但本地仍要给 Docker 留出数 GB 空闲内存。([GitHub][2])

---

## 2. 安装 WSL2

用 **管理员 PowerShell** 执行：

```powershell
wsl --install -d Ubuntu-24.04
```

安装完成后重启机器。第一次打开 Ubuntu 时，设置 Linux 用户名和密码。

检查 WSL 版本：

```powershell
wsl -l -v
```

如果 Ubuntu 不是 WSL2，执行：

```powershell
wsl --set-version Ubuntu-24.04 2
```

Microsoft 官方文档说明，Windows 10 2004+ 或 Windows 11 可以用 `wsl --install` 一条命令安装 WSL，并默认安装 Ubuntu；WSL2 是后续 Linux 开发和 Docker Desktop 集成的推荐路径。([Microsoft Learn][3])

---

## 3. 安装 Docker Desktop

先确认 Windows 有 winget：

```powershell
winget --version
```

如果 winget 不存在，先从 Microsoft Store 更新或安装 **App Installer**。Microsoft 文档说明 WinGet 是 Windows Package Manager 的命令行工具，可用于搜索、安装和管理 Windows 应用。([Microsoft Learn][4])

安装 Docker Desktop：

```powershell
winget install -e --id Docker.DockerDesktop
```

安装后打开 Docker Desktop，确认：

```text
Settings → General → Use the WSL 2 based engine
Settings → Resources → WSL Integration → Enable integration with Ubuntu-24.04
```

Docker 官方文档说明 Docker Desktop on Windows 支持 WSL2 backend，并建议在 WSL2 distribution 中启用 Docker integration，之后可以在 WSL terminal 里直接使用 `docker` 命令。([Docker Documentation][5]) ([Docker Documentation][6])

进入 Ubuntu，验证：

```bash
docker version
docker compose version
```

看到 client/server 信息就说明 Docker 可用。

---

## 4. 在 WSL Ubuntu 中安装基础工具

打开 Ubuntu 终端，执行：

```bash
sudo apt update
sudo apt install -y \
  git curl unzip zip make jq ca-certificates gnupg \
  build-essential \
  python3 python3-venv python3-pip python-is-python3
```

检查 Python 版本。项目要求 Python 3.11+，当前 `pyproject.toml` 也明确要求 `requires-python >=3.11`。([GitHub][7])

```bash
python --version
```

如果显示 `Python 3.12.x` 或 `3.11.x`，可以继续。

安装 Node.js 24 LTS。Node.js 当前下载页显示 24.x 是 LTS；README 要求 Node 20+，因此 Node 24 LTS 可用。([Node.js][8]) ([GitHub][1])

```bash
curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
sudo apt-get install -y nodejs
node -v
npm -v
```

启用 pnpm：

```bash
corepack enable
corepack prepare pnpm@10.13.1 --activate
pnpm -v
```

项目 README 中也使用 `corepack prepare pnpm@10.13.1 --activate`。([GitHub][1])

---

## 5. 克隆当前分支和 commit

建议把代码放在 WSL 文件系统里，不要放在 `/mnt/c/...`，否则 Node、Docker bind mount 和文件监听会慢很多。

```bash
mkdir -p ~/work
cd ~/work

git clone -b codex/logan-platform-foundation https://github.com/itwake/llm-powered-log-analytic.git
cd llm-powered-log-analytic

git checkout ab86b966c5a2ce5202d7afb0eae96f1a09c2e293
git rev-parse HEAD
```

最后一行应输出：

```text
ab86b966c5a2ce5202d7afb0eae96f1a09c2e293
```

---

## 6. 安装 Python 和 Web 依赖

```bash
cd ~/work/llm-powered-log-analytic

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e . pytest pytest-asyncio ruff

pnpm install
```

---

## 7. 先跑轻量测试

```bash
source .venv/bin/activate

python -m pytest -q
ruff check apps tests scripts
```

你前面已经跑出过：

```text
133 passed, 3 skipped
ruff check passed
```

本地重新跑时，真实 Copilot staging 测试在没有 token 时会 skip，这是正确行为。

---

## 8. 跑离线 benchmark

```bash
source .venv/bin/activate

python -m logan_workers.evaluation.run \
  --benchmark benchmarks/logan/checkout_incident \
  --out .logan/evaluation/report.json \
  --markdown .logan/evaluation/report.md
```

项目 operations 文档说明这个 benchmark 使用 synthetic checkout incident fixture，并用 MockCopilotAnnotationGateway，因此不需要 Docker、Temporal、GitHub Copilot credentials 或外部网络。([GitHub][2])

查看结果：

```bash
cat .logan/evaluation/report.md
```

重点看：

```text
review_load_reduction
golden_signal_macro_f1
fault_category_micro_f1
root_cause_hit_at_k
causal_summary_quality
```

你之前的 scale quick 已经得到 review-load reduction `0.944444`，说明当前 mock pipeline 下的压缩目标达到了可测试水平。

---

## 9. 跑 scale quick

```bash
source .venv/bin/activate
make PYTHON=.venv/bin/python scale-benchmark
```

operations 文档说明 scale benchmark 会生成混合格式日志，包括 `.log`、`.jsonl`、`.log.gz`、`.zip`、跨服务依赖失败、重试、资源饱和、gateway failure 和 multiline stack trace，并输出 wall time、peak RSS、template 数、model call 数、review-load reduction 等指标。([GitHub][2])

---

## 10. 启动本地轻量模式 Web + API

这个模式不需要 Docker、PostgreSQL、MinIO、Temporal、ClickHouse、OpenSearch，也不需要真实 Copilot token。它适合先看 UI 和五视图。

打开第一个 Ubuntu 终端：

```bash
cd ~/work/llm-powered-log-analytic
source .venv/bin/activate

export LOGAN_STORE_BACKEND=memory
export LOGAN_OBJECT_STORE_BACKEND=local
export LOGAN_LOCAL_OBJECT_STORE_DIR=.logan/object-store
export LOGAN_LLM_PROVIDER=mock
export LOGAN_CORS_ALLOWED_ORIGINS=http://localhost:3000
export LOGAN_RATE_LIMIT_ENABLED=false
export LOGAN_METRICS_ENABLED=true

python -m uvicorn app.main:app --app-dir apps/api --host 127.0.0.1 --port 8000
```

打开第二个 Ubuntu 终端：

```bash
cd ~/work/llm-powered-log-analytic

export NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
corepack pnpm --filter @logan/web dev --hostname 127.0.0.1 --port 3000
```

打开浏览器：

```text
http://localhost:3000
```

本地 API 默认使用 in-memory metadata，上传文件默认写到 `.logan/object-store`；README 也说明本地 browser 使用 `http://localhost:3000`，Web 使用 `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`，这样 cookie 和 CORS 能匹配。([GitHub][1])

### 使用步骤

进入 Web 后按这个路径测试：

```text
1. Register / Login
2. Create Case
3. Upload log / zip / jsonl / txt
4. Start analysis
5. 查看 Data Summary
6. 点击 Temporal View 的时间窗口
7. 查看 Tabular Logs
8. 查看 Causal Graph
9. 查看 Causal Summary
10. 提交 feedback 或 export summary
```

没有自己的日志时，可以先打包项目里的测试日志：

```bash
cd ~/work/llm-powered-log-analytic
zip -r checkout_incident.zip tests/fixtures/logs/checkout_incident
```

然后在 UI 上传 `checkout_incident.zip`。

---

## 11. 跑浏览器 E2E

首次安装 Playwright 浏览器依赖：

```bash
corepack pnpm exec playwright install --with-deps chromium
```

运行 E2E：

```bash
corepack pnpm e2e
```

operations 文档说明 E2E 会启动 API 和 Next.js workbench，使用 memory store、local object store 和 mock LLM，并验证 Data Summary、Temporal View、Tabular Logs、Causal Graph、Causal Summary。([GitHub][2])

你之前已经跑出：

```text
corepack pnpm e2e: 2 passed
```

---

## 12. 跑本地 full-stack Docker smoke

这个模式更接近实际部署，会启动：

```text
PostgreSQL
MinIO
ClickHouse
OpenSearch
Temporal
FastAPI API
Temporal worker
smoke runner
```

执行：

```bash
cd ~/work/llm-powered-log-analytic
source .venv/bin/activate

docker compose config
make full-stack-smoke
```

完成后清理：

```bash
make full-stack-down
```

operations 文档说明 `make full-stack-smoke` 会启动 PostgreSQL、MinIO、ClickHouse、OpenSearch、Temporal、API 和 worker，并验证 API health、注册/登录、case 创建、MinIO 上传、Temporal worker 分析、五个 report endpoint、ClickHouse/OpenSearch 写入和外部查询审计。([GitHub][2])

默认端口：

```text
Web:        http://localhost:3000
API:        http://localhost:8000
PostgreSQL: localhost:5432
MinIO:      localhost:9000 / 9001
ClickHouse: localhost:8123
OpenSearch: localhost:9200
Temporal:   localhost:7233
```

这些端口也在 operations 文档中列出。([GitHub][2])

---

## 13. 真实 GitHub Copilot Plugin 模型测试

没有 token 时，所有真实 Copilot staging smoke 都应该 skip。不要伪造成通过。

项目 README 说明默认 LLM provider 是 `github_copilot`，默认模型是 `gpt-5.4`；本地/CI 可用 `LOGAN_LLM_PROVIDER=mock` 做确定性分析，真实 Copilot staging smoke 需要显式提供 `LOGAN_GITHUB_COPILOT_TOKEN` 或 `LOGAN_GITHUB_SOURCE_TOKEN`。([GitHub][1])

拿到 token 后，在 WSL 中执行：

```bash
cd ~/work/llm-powered-log-analytic
source .venv/bin/activate

export LOGAN_GITHUB_COPILOT_TOKEN="你的 token"
make copilot-staging-smoke
```

或：

```bash
export LOGAN_GITHUB_SOURCE_TOKEN="你的 source OAuth / PAT token"
make copilot-staging-smoke
```

API 文档说明 backend model gateway 的凭证解析顺序是：stored Copilot plugin token、stored GitHub source OAuth 换取的 Copilot token、`LOGAN_GITHUB_COPILOT_TOKEN`、`LOGAN_GITHUB_SOURCE_TOKEN`。它不会把 token 返回给前端。([GitHub][9])

---

# 二、AWS EKS / Kubernetes 安装与使用

下面是**云上测试版**说明。它的目标是验证：

```text
ECR 镜像
EKS workload
PostgreSQL
Temporal
worker pipeline
S3 上传
API/Web
五视图报告
```

为了减少公网、证书、DNS 和 Ingress 变量，第一轮建议使用：

```text
kubectl port-forward 访问 Web/API
AWS S3 作为对象存储
in-cluster PostgreSQL / Temporal / ClickHouse / OpenSearch
mock LLM provider
```

也就是说，第一轮先不做公网 ALB/Ingress。现有 repo 里有 `infra/k8s/ingress.yaml`，但它使用占位 host `logan.example.com`。([GitHub][10]) 对测试来说，port-forward 最少坑。

---

## 1. AWS 费用提醒

EKS 会产生费用。测试时至少会用到：

```text
EKS control plane
EC2 worker nodes
EBS volumes
ECR storage
S3 bucket
Load balancer，如果后续启用 Ingress / LoadBalancer
NAT Gateway，如果 eksctl 创建了带 NAT 的 VPC
```

测试完成后一定执行清理命令，见本节最后。

---

## 2. 在本机安装 AWS/EKS 工具

建议继续在前面装好的 **WSL Ubuntu** 中执行。

安装 AWS CLI v2：

```bash
cd /tmp
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
sudo ./aws/install

aws --version
```

AWS 官方文档说明 Linux x86_64 可以通过下载 `awscli-exe-linux-x86_64.zip`、解压并执行 `sudo ./aws/install` 安装 AWS CLI v2；Windows 也可通过 MSI 安装。([AWS 文档][11])

配置 AWS 凭证：

```bash
aws configure
```

建议使用 IAM 用户或 IAM role，不要用 root account。AWS CLI 入门文档也建议不要使用 root credentials，而应使用最小权限身份。([AWS 文档][12])

检查身份：

```bash
aws sts get-caller-identity
```

安装 `kubectl`：

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/kubectl

kubectl version --client
```

AWS EKS 文档说明 `kubectl` 是管理 Kubernetes cluster resources 的主要工具，并提醒 kubectl 版本应与 EKS control plane 版本相差不超过一个 minor version。([AWS 文档][13])

安装 `eksctl`：

```bash
ARCH=amd64
PLATFORM=$(uname -s)_$ARCH

curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_${PLATFORM}.tar.gz"
curl -sL "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_checksums.txt" \
  | grep "$PLATFORM" \
  | sha256sum --check

tar -xzf "eksctl_${PLATFORM}.tar.gz" -C /tmp
sudo install -m 0755 /tmp/eksctl /usr/local/bin
rm "eksctl_${PLATFORM}.tar.gz" /tmp/eksctl

eksctl version
```

eksctl 官方安装文档建议从官方 GitHub releases 安装，并给出了 Unix 安装命令。([Eksctl][14])

Helm 不是第一轮必须项，但后续装 ALB Controller / 监控组件会用到：

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm version
```

AWS EKS Helm 文档说明 Helm 是 Kubernetes package manager，可以用于在 EKS 上安装和管理 charts。([AWS 文档][15])

---

## 3. 创建 EKS cluster

设置变量：

```bash
export AWS_REGION=us-east-1
export CLUSTER_NAME=logan-pilot
```

创建 `cluster.yaml`：

```bash
cat > cluster.yaml <<'YAML'
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: logan-pilot
  region: us-east-1

iam:
  withOIDC: true

managedNodeGroups:
  - name: logan-ng
    instanceType: m6i.xlarge
    desiredCapacity: 3
    minSize: 2
    maxSize: 4
    volumeSize: 100
YAML
```

如果你的 region 不是 `us-east-1`，同步修改文件里的 region。

创建 cluster：

```bash
eksctl create cluster -f cluster.yaml
```

AWS EKS 文档说明 `eksctl` 可以创建 Amazon EKS cluster 和 nodes，并自动创建多个原本需要手工创建的资源。([AWS 文档][16])

配置 kubeconfig：

```bash
aws eks update-kubeconfig --region "$AWS_REGION" --name "$CLUSTER_NAME"
kubectl get nodes
```

AWS EKS 文档说明可用 `aws eks update-kubeconfig --region region-code --name my-cluster` 创建或更新 kubeconfig。([AWS 文档][17])

确认 storage class：

```bash
kubectl get storageclass
```

如果没有默认 StorageClass，或者 PVC 一直 Pending，需要安装 EBS CSI driver。AWS 文档建议通过 Amazon EKS add-on 安装 EBS CSI driver；EKS add-on 名称是 `aws-ebs-csi-driver`。([AWS 文档][18]) ([AWS 文档][19])

---

## 4. 创建 ECR repository 并推送镜像

设置变量：

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
export IMAGE_TAG="ab86b966"
```

创建 ECR repositories：

```bash
aws ecr create-repository --repository-name logan-api --region "$AWS_REGION" \
  --image-scanning-configuration scanOnPush=true || true

aws ecr create-repository --repository-name logan-web --region "$AWS_REGION" \
  --image-scanning-configuration scanOnPush=true || true

aws ecr create-repository --repository-name logan-worker --region "$AWS_REGION" \
  --image-scanning-configuration scanOnPush=true || true
```

AWS ECR 文档说明可以用 `aws ecr create-repository` 创建 repository。([AWS 文档][20])

登录 ECR：

```bash
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR"
```

AWS ECR 文档说明 Docker 登录 ECR 应使用 `aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <registry>`。([AWS 文档][21])

构建并推送镜像：

```bash
cd ~/work/llm-powered-log-analytic

docker build -f infra/docker/api.Dockerfile \
  -t "$ECR/logan-api:$IMAGE_TAG" .

docker build -f infra/docker/web.Dockerfile \
  -t "$ECR/logan-web:$IMAGE_TAG" .

docker build -f infra/docker/worker.Dockerfile \
  -t "$ECR/logan-worker:$IMAGE_TAG" .

docker push "$ECR/logan-api:$IMAGE_TAG"
docker push "$ECR/logan-web:$IMAGE_TAG"
docker push "$ECR/logan-worker:$IMAGE_TAG"
```

当前 Dockerfiles 已经是更接近部署的形态：API 和 worker 使用非 editable install，Web 先 build Next.js，再用 `next start` 服务，并包含 `/healthz` healthcheck。([GitHub][22]) ([GitHub][23]) ([GitHub][24])

---

## 5. 创建 S3 bucket 作为日志对象存储

这一点很重要：EKS 第一轮测试建议用 AWS S3，不建议用 in-cluster MinIO。原因是浏览器直传 presigned URL 时，浏览器能访问 AWS S3，但通常不能直接访问 cluster 内部的 `http://minio:9000`。

设置 bucket 名：

```bash
export S3_BUCKET="logan-pilot-${AWS_ACCOUNT_ID}-${AWS_REGION}"
```

创建 bucket：

```bash
if [ "$AWS_REGION" = "us-east-1" ]; then
  aws s3api create-bucket --bucket "$S3_BUCKET" --region "$AWS_REGION" || true
else
  aws s3api create-bucket \
    --bucket "$S3_BUCKET" \
    --region "$AWS_REGION" \
    --create-bucket-configuration LocationConstraint="$AWS_REGION" || true
fi
```

给浏览器上传设置 CORS。第一轮用 port-forward，所以 origin 是 `http://localhost:3000`：

```bash
cat > s3-cors.json <<'JSON'
{
  "CORSRules": [
    {
      "AllowedOrigins": ["http://localhost:3000"],
      "AllowedMethods": ["GET", "PUT", "POST", "HEAD"],
      "AllowedHeaders": ["*"],
      "ExposeHeaders": ["ETag"],
      "MaxAgeSeconds": 3000
    }
  ]
}
JSON

aws s3api put-bucket-cors \
  --bucket "$S3_BUCKET" \
  --cors-configuration file://s3-cors.json
```

---

## 6. 创建 Kubernetes namespace、ConfigMap、Secret

创建 namespace：

```bash
kubectl create namespace logan --dry-run=client -o yaml | kubectl apply -f -
```

生成测试用 secrets：

```bash
export APP_SECRET=$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')
export ENC_SECRET=$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')
```

为了让 in-cluster PostgreSQL 的密码和 API 的 `LOGAN_DATABASE_URL` 匹配，下面用同一个 `APP_SECRET` 作为 Postgres 密码。测试可以这样做；生产不建议把 app secret 和 DB password 复用。

创建 secret。这里假设你当前 shell 里的 `AWS_ACCESS_KEY_ID` 和 `AWS_SECRET_ACCESS_KEY` 是一个只允许访问该 S3 bucket 的测试用 IAM access key：

```bash
kubectl -n logan create secret generic logan-secrets \
  --from-literal=LOGAN_SECRET_KEY="$APP_SECRET" \
  --from-literal=LOGAN_CREDENTIAL_ENCRYPTION_KEY="$ENC_SECRET" \
  --from-literal=LOGAN_DATABASE_URL="postgresql+psycopg://logan:${APP_SECRET}@postgres:5432/logan" \
  --from-literal=LOGAN_REDIS_URL="redis://redis:6379/0" \
  --from-literal=LOGAN_S3_ACCESS_KEY="${AWS_ACCESS_KEY_ID}" \
  --from-literal=LOGAN_S3_SECRET_KEY="${AWS_SECRET_ACCESS_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

没有 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` 时，先不要继续。后续更稳的做法是 IRSA / Pod Identity，而不是静态 key；但第一轮测试用受限 key 更简单。

创建 ConfigMap。第一轮使用 mock LLM，确保没有 Copilot token 也能跑通平台：

```bash
kubectl -n logan create configmap logan-config \
  --from-literal=LOGAN_ENV=production \
  --from-literal=LOGAN_LLM_PROVIDER=mock \
  --from-literal=LOGAN_COPILOT_MODEL=gpt-5.4 \
  --from-literal=LOGAN_COPILOT_REASONING_EFFORT=high \
  --from-literal=LOGAN_ANALYSIS_ORCHESTRATOR=temporal \
  --from-literal=LOGAN_TEMPORAL_ADDRESS=temporal:7233 \
  --from-literal=LOGAN_TEMPORAL_NAMESPACE=default \
  --from-literal=LOGAN_TEMPORAL_TASK_QUEUE=logan-analysis \
  --from-literal=LOGAN_TEMPORAL_ACTIVITY_START_TO_CLOSE_SECONDS=3600 \
  --from-literal=LOGAN_TEMPORAL_ACTIVITY_MAX_ATTEMPTS=3 \
  --from-literal=LOGAN_OBJECT_STORE_BACKEND=s3 \
  --from-literal=LOGAN_S3_BUCKET="$S3_BUCKET" \
  --from-literal=LOGAN_ANALYSIS_INPUT_TMP_DIR=.logan/analysis-inputs \
  --from-literal=AWS_REGION="$AWS_REGION" \
  --from-literal=AWS_DEFAULT_REGION="$AWS_REGION" \
  --from-literal=LOGAN_STEP_ARTIFACTS_ENABLED=true \
  --from-literal=LOGAN_RATE_LIMIT_ENABLED=false \
  --from-literal=LOGAN_RATE_LIMIT_REQUESTS_PER_MINUTE=120 \
  --from-literal=LOGAN_METRICS_ENABLED=true \
  --from-literal=LOGAN_METRICS_PATH=/metrics \
  --from-literal=LOGAN_API_WORKERS=2 \
  --from-literal=LOGAN_CORS_ALLOWED_ORIGINS=http://localhost:3000 \
  --from-literal=NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 \
  --from-literal=LOGAN_ANALYTICS_SINKS_ENABLED=true \
  --from-literal=LOGAN_CLICKHOUSE_URL=http://clickhouse:8123 \
  --from-literal=LOGAN_OPENSEARCH_URL=http://opensearch:9200 \
  --from-literal=LOGAN_EXTERNAL_ANALYTICS_QUERIES_ENABLED=true \
  --from-literal=LOGAN_EXTERNAL_ANALYTICS_QUERY_TIMEOUT_SECONDS=10 \
  --from-literal=LOGAN_RAW_LOG_RETENTION_DAYS=30 \
  --from-literal=LOGAN_REPORT_RETENTION_DAYS=365 \
  --from-literal=LOGAN_AUDIT_RETENTION_DAYS=730 \
  --dry-run=client -o yaml | kubectl apply -f -
```

项目自带的 `infra/k8s/configmap.yaml` 已经包含 `LOGAN_ANALYSIS_ORCHESTRATOR=temporal`、`LOGAN_TEMPORAL_ADDRESS=temporal:7233`、`LOGAN_API_WORKERS=2`、metrics、CORS 等基础配置。([GitHub][25]) 上面这份 ConfigMap 增加了 AWS S3、ClickHouse/OpenSearch sink 和本地 port-forward CORS 的测试配置。

---

## 7. 部署基础服务

先部署 PostgreSQL、Redis、ClickHouse、OpenSearch、Temporal。第一轮使用 AWS S3，所以不部署 MinIO：

```bash
kubectl apply -f infra/k8s/postgres.yaml
kubectl apply -f infra/k8s/redis.yaml
kubectl apply -f infra/k8s/clickhouse.yaml
kubectl apply -f infra/k8s/opensearch.yaml
kubectl apply -f infra/k8s/temporal.yaml
```

repo 里的 Kubernetes manifests 包含 PostgreSQL PVC、MinIO PVC、ClickHouse、OpenSearch、Temporal、Redis、API、Web、worker、migration job 等文件。([GitHub][26]) PostgreSQL manifest 默认创建 20Gi PVC，MinIO manifest 默认创建 50Gi PVC；本说明第一轮用 S3，因此可以暂时不 apply MinIO。([GitHub][27]) ([GitHub][28])

等待基础服务：

```bash
kubectl -n logan rollout status deployment/postgres --timeout=300s
kubectl -n logan rollout status deployment/redis --timeout=300s
kubectl -n logan rollout status deployment/clickhouse --timeout=300s
kubectl -n logan rollout status deployment/opensearch --timeout=300s
kubectl -n logan rollout status deployment/temporal --timeout=300s
```

检查：

```bash
kubectl -n logan get pods
kubectl -n logan get pvc
```

如果 PVC Pending，先处理 StorageClass / EBS CSI。

---

## 8. 运行数据库 migration

migration job manifest 默认镜像是 `ghcr.io/itwake/logan-api:latest`，需要替换为刚推到 ECR 的 image。生成临时文件：

```bash
mkdir -p .logan/k8s

cp infra/k8s/migration-job.yaml .logan/k8s/migration-job.yaml

sed -i "s#ghcr.io/itwake/logan-api:latest#$ECR/logan-api:$IMAGE_TAG#g" \
  .logan/k8s/migration-job.yaml
```

执行：

```bash
kubectl -n logan delete job logan-migration --ignore-not-found
kubectl apply -f .logan/k8s/migration-job.yaml
kubectl -n logan wait --for=condition=complete job/logan-migration --timeout=300s
```

查看日志：

```bash
kubectl -n logan logs job/logan-migration
```

当前 migration job 已经改为调用 `python scripts/run_migrations.py`，而不是 Alembic。([GitHub][29])

---

## 9. 部署 API、worker、Web

先 apply manifests：

```bash
kubectl apply -f infra/k8s/api-deployment.yaml
kubectl apply -f infra/k8s/worker-deployment.yaml
kubectl apply -f infra/k8s/web-deployment.yaml
```

替换镜像为 ECR：

```bash
kubectl -n logan set image deployment/logan-api api="$ECR/logan-api:$IMAGE_TAG"
kubectl -n logan set image deployment/logan-worker worker="$ECR/logan-worker:$IMAGE_TAG"
kubectl -n logan set image deployment/logan-web web="$ECR/logan-web:$IMAGE_TAG"
```

等待 rollout：

```bash
kubectl -n logan rollout status deployment/logan-api --timeout=300s
kubectl -n logan rollout status deployment/logan-worker --timeout=300s
kubectl -n logan rollout status deployment/logan-web --timeout=300s
```

API deployment 使用 `/healthz` readiness/liveness probe，Web deployment 也使用 `/healthz` probe，worker 运行 `python -m logan_workers.temporal_worker`。([GitHub][30]) ([GitHub][31]) ([GitHub][32])

检查：

```bash
kubectl -n logan get pods -o wide
kubectl -n logan get svc
```

---

## 10. 通过 port-forward 访问 EKS 上的系统

开三个终端。

终端 1：

```bash
kubectl -n logan port-forward svc/logan-api 8000:8000
```

终端 2：

```bash
kubectl -n logan port-forward svc/logan-web 3000:3000
```

终端 3：

```bash
curl http://localhost:8000/healthz
```

浏览器打开：

```text
http://localhost:3000
```

测试步骤：

```text
1. Register / Login
2. Create Case
3. Upload zip/log/jsonl
4. Start analysis
5. 等待 analysis run 完成
6. 查看五视图
7. Export Causal Summary
8. Submit feedback
```

建议第一轮上传：

```bash
cd ~/work/llm-powered-log-analytic
zip -r checkout_incident.zip tests/fixtures/logs/checkout_incident
```

然后在 UI 上传 `checkout_incident.zip`。

---

## 11. EKS 上切换到真实 Copilot Plugin 模型

第一轮建议先用 `LOGAN_LLM_PROVIDER=mock` 验证平台链路。确认 EKS 上传、分析、Temporal worker、五视图都可用后，再切真实模型。

准备 token：

```bash
export LOGAN_GITHUB_COPILOT_TOKEN="你的 token"
```

把 token 加到 secret：

```bash
kubectl -n logan create secret generic logan-secrets \
  --from-literal=LOGAN_SECRET_KEY="$APP_SECRET" \
  --from-literal=LOGAN_CREDENTIAL_ENCRYPTION_KEY="$ENC_SECRET" \
  --from-literal=LOGAN_DATABASE_URL="postgresql+psycopg://logan:${APP_SECRET}@postgres:5432/logan" \
  --from-literal=LOGAN_REDIS_URL="redis://redis:6379/0" \
  --from-literal=LOGAN_S3_ACCESS_KEY="${AWS_ACCESS_KEY_ID}" \
  --from-literal=LOGAN_S3_SECRET_KEY="${AWS_SECRET_ACCESS_KEY}" \
  --from-literal=LOGAN_GITHUB_COPILOT_TOKEN="$LOGAN_GITHUB_COPILOT_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -
```

把 provider 改成 `github_copilot`：

```bash
kubectl -n logan create configmap logan-config \
  --from-literal=LOGAN_ENV=production \
  --from-literal=LOGAN_LLM_PROVIDER=github_copilot \
  --from-literal=LOGAN_COPILOT_MODEL=gpt-5.4 \
  --from-literal=LOGAN_COPILOT_REASONING_EFFORT=high \
  --from-literal=LOGAN_ANALYSIS_ORCHESTRATOR=temporal \
  --from-literal=LOGAN_TEMPORAL_ADDRESS=temporal:7233 \
  --from-literal=LOGAN_TEMPORAL_NAMESPACE=default \
  --from-literal=LOGAN_TEMPORAL_TASK_QUEUE=logan-analysis \
  --from-literal=LOGAN_TEMPORAL_ACTIVITY_START_TO_CLOSE_SECONDS=3600 \
  --from-literal=LOGAN_TEMPORAL_ACTIVITY_MAX_ATTEMPTS=3 \
  --from-literal=LOGAN_OBJECT_STORE_BACKEND=s3 \
  --from-literal=LOGAN_S3_BUCKET="$S3_BUCKET" \
  --from-literal=LOGAN_ANALYSIS_INPUT_TMP_DIR=.logan/analysis-inputs \
  --from-literal=AWS_REGION="$AWS_REGION" \
  --from-literal=AWS_DEFAULT_REGION="$AWS_REGION" \
  --from-literal=LOGAN_STEP_ARTIFACTS_ENABLED=true \
  --from-literal=LOGAN_RATE_LIMIT_ENABLED=false \
  --from-literal=LOGAN_METRICS_ENABLED=true \
  --from-literal=LOGAN_METRICS_PATH=/metrics \
  --from-literal=LOGAN_API_WORKERS=2 \
  --from-literal=LOGAN_CORS_ALLOWED_ORIGINS=http://localhost:3000 \
  --from-literal=NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 \
  --from-literal=LOGAN_ANALYTICS_SINKS_ENABLED=true \
  --from-literal=LOGAN_CLICKHOUSE_URL=http://clickhouse:8123 \
  --from-literal=LOGAN_OPENSEARCH_URL=http://opensearch:9200 \
  --from-literal=LOGAN_EXTERNAL_ANALYTICS_QUERIES_ENABLED=true \
  --from-literal=LOGAN_EXTERNAL_ANALYTICS_QUERY_TIMEOUT_SECONDS=10 \
  --from-literal=LOGAN_RAW_LOG_RETENTION_DAYS=30 \
  --from-literal=LOGAN_REPORT_RETENTION_DAYS=365 \
  --from-literal=LOGAN_AUDIT_RETENTION_DAYS=730 \
  --dry-run=client -o yaml | kubectl apply -f -
```

重启 API 和 worker：

```bash
kubectl -n logan rollout restart deployment/logan-api
kubectl -n logan rollout restart deployment/logan-worker

kubectl -n logan rollout status deployment/logan-api --timeout=300s
kubectl -n logan rollout status deployment/logan-worker --timeout=300s
```

然后重新上传一个小日志 case 测真实模型。不要一开始拿大日志压真实模型。

---

## 12. EKS 清理

删除应用：

```bash
kubectl delete namespace logan --ignore-not-found
```

删除 cluster：

```bash
eksctl delete cluster --name "$CLUSTER_NAME" --region "$AWS_REGION"
```

删除 ECR repositories：

```bash
aws ecr delete-repository --repository-name logan-api --region "$AWS_REGION" --force || true
aws ecr delete-repository --repository-name logan-web --region "$AWS_REGION" --force || true
aws ecr delete-repository --repository-name logan-worker --region "$AWS_REGION" --force || true
```

删除 S3 bucket：

```bash
aws s3 rm "s3://$S3_BUCKET" --recursive
aws s3api delete-bucket --bucket "$S3_BUCKET" --region "$AWS_REGION"
```

确认没有遗留 EBS volumes：

```bash
aws ec2 describe-volumes \
  --region "$AWS_REGION" \
  --filters Name=status,Values=available \
  --query "Volumes[*].{VolumeId:VolumeId,Size:Size,Tags:Tags}" \
  --output table
```

---

# 三、测试时必须关注的要点

## 1. 先 mock，后真实 Copilot

第一轮测试目标是验证平台链路，不是模型效果：

```text
mock: 验证系统可用性、五视图、Temporal、S3、报告、导出
real Copilot: 验证真实 golden signal / fault category / entities / summary 质量
```

没有 token 时 staging smoke skip 是正确行为。README 明确说明 full-stack smoke 不需要 Copilot credentials，而真实 Copilot staging smoke 只有显式设置 token 才会跑。([GitHub][1])

---

## 2. EKS 第一轮不要急着上 Ingress

先用：

```bash
kubectl port-forward svc/logan-web 3000:3000
kubectl port-forward svc/logan-api 8000:8000
```

原因：

```text
减少 ALB Controller / DNS / ACM / CORS / NEXT_PUBLIC_API_BASE_URL 变量
方便快速判断系统是否真正可用
```

后续做公网入口时，再处理：

```text
AWS Load Balancer Controller 或 EKS Auto Mode ALB
ACM certificate
Route53 DNS
https://logan.example.com
LOGAN_CORS_ALLOWED_ORIGINS
Web build-time/runtime API base URL
```

AWS EKS Auto Mode 文档说明 Ingress 可以触发创建 ALB，但前提是 Auto Mode cluster、IngressClassParams、IngressClass 和相关 subnet tags 等都正确配置。([AWS 文档][33])

---

## 3. 上传路径要验证清楚

本地轻量模式：

```text
local object store
memory store
local analysis
```

本地 full-stack smoke：

```text
MinIO
Temporal
PostgreSQL
ClickHouse
OpenSearch
mock LLM
```

EKS 推荐第一轮：

```text
AWS S3
Temporal
PostgreSQL
ClickHouse
OpenSearch
mock LLM
```

不要在 EKS 第一轮用 in-cluster MinIO 做浏览器直传，除非你已经处理好 MinIO 的公网/port-forward endpoint 和 CORS。否则浏览器拿到的 presigned URL 很可能是 cluster 内部地址，无法上传。

---

## 4. Review-load reduction 是第一指标

每次测试记录：

```text
raw log lines
template count
representative sample count
model call count
offending template count
review-load reduction
analysis wall time
summary 是否可用
causal graph 是否有证据
```

LogAn 论文中 Data Summary 通过只展示被分类为 error、availability、saturation、latency、traffic 的 unique offending log lines，平均减少约 90% 需要审查的数据。 当前项目 benchmark 和 scale benchmark 也已经把 review-load reduction、golden-signal F1、fault-category F1、entity F1、root-cause hit@k、causal-edge recall、summary rubric 作为评估指标。([GitHub][2])

---

## 5. 因果图只能当候选证据

UI 和报告里都应保持：

```text
candidate cause
needs validation
confidence
evidence
lag
support windows
```

不要把 Causal Graph 当成“确定根因”。README 也明确说明 PGEM-style transition scores 和 Granger-style lagged-linear scores 只是候选因果证据，用于帮助排序和验证，不证明 root cause truth。([GitHub][1])

---

# 四、常见问题排查

| 问题                          | 处理方式                                                                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `docker: command not found` | Docker Desktop 没启动，或 WSL integration 没开。打开 Docker Desktop → Settings → Resources → WSL Integration。                                   |
| `python` 版本低于 3.11          | 换 Ubuntu 24.04 WSL，或安装 Python 3.11+。项目要求 Python 3.11+。                                                                                |
| `pnpm: command not found`   | 执行 `corepack enable && corepack prepare pnpm@10.13.1 --activate`。                                                                     |
| Web 调 API 报 CORS            | 本地设置 `LOGAN_CORS_ALLOWED_ORIGINS=http://localhost:3000`。README 说明 credentialed browser requests 需要配置允许的 browser origins。([GitHub][1]) |
| EKS Pod Pending             | 先查 `kubectl get pvc -n logan`，多数是 StorageClass / EBS CSI 问题。                                                                          |
| EKS API 连接 PostgreSQL 失败    | 确认 `LOGAN_DATABASE_URL` 密码和 Postgres `POSTGRES_PASSWORD` 一致。上面的说明用 `APP_SECRET` 统一两者。                                                 |
| S3 上传失败                     | 检查 bucket CORS、S3 access key 权限、`LOGAN_OBJECT_STORE_BACKEND=s3`、`LOGAN_S3_BUCKET`、region。                                             |
| Copilot staging skip        | 没有 `LOGAN_GITHUB_COPILOT_TOKEN` 或 `LOGAN_GITHUB_SOURCE_TOKEN`，这是正确行为。                                                                 |
| OpenSearch 起不来              | 节点资源不足或安全/内存设置问题；第一轮可以先关闭 external analytics sink，只验证 PostgreSQL + report fallback。                                                   |
| EKS 网页能开但上传失败               | 浏览器直传 S3 需要 bucket CORS 允许 `http://localhost:3000`，并暴露 `ETag`。                                                                        |
| EKS Ingress 没地址             | 第一轮不要用 Ingress；先 port-forward。公网入口后续再接 ALB Controller / EKS Auto Mode。                                                                |

---

# 五、建议的实际测试顺序

按这个顺序最稳：

```text
1. 本地 WSL 轻量模式
   python tests → benchmark → web/api mock

2. 本地 full-stack Docker smoke
   PostgreSQL + MinIO + Temporal + ClickHouse + OpenSearch + worker

3. 本地真实 Copilot staging smoke
   只在拿到 token 后执行

4. EKS mock 模式
   ECR images + EKS + S3 + Temporal + worker + 五视图

5. EKS 真实 Copilot 小日志 case
   先 1 个小 zip，再 5–10 个历史 incident

6. 工程师评价
   Data Summary 是否减少阅读量
   Temporal View 是否帮助定位窗口
   Causal Graph 是否有用
   Causal Summary 是否可贴到 ticket
```

当前最关键的验收不是“能不能跑起来”，而是：

```text
工程师是否真的少看日志
候选 offending lines 是否相关
因果链是否可验证
summary 是否能用于交接
真实 Copilot 输出是否稳定
```

只要这几项在 5–10 个真实历史 case 上通过，就可以进入内部受控 pilot。

[1]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/README.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/docs/operations.md "raw.githubusercontent.com"
[3]: https://learn.microsoft.com/en-us/windows/wsl/install "Install WSL | Microsoft Learn"
[4]: https://learn.microsoft.com/en-us/windows/package-manager/winget/ "Use WinGet to install and manage applications | Microsoft Learn"
[5]: https://docs.docker.com/desktop/setup/install/windows-install/ "Install Docker Desktop on Windows | Docker Docs"
[6]: https://docs.docker.com/desktop/features/wsl/ "Docker Desktop WSL 2 backend on Windows | Docker Docs"
[7]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/pyproject.toml "raw.githubusercontent.com"
[8]: https://nodejs.org/en/download "Node.js — Download Node.js®"
[9]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/docs/api.md "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/ingress.yaml "raw.githubusercontent.com"
[11]: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html "Installing or updating to the latest version of the AWS CLI - AWS Command Line Interface"
[12]: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html?utm_source=chatgpt.com "Getting started with the AWS CLI"
[13]: https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html "Set up kubectl and eksctl - Amazon EKS"
[14]: https://eksctl.io/installation/ "Installation options for Eksctl - Eksctl User Guide"
[15]: https://docs.aws.amazon.com/eks/latest/userguide/helm.html "Deploy applications with Helm on Amazon EKS - Amazon EKS"
[16]: https://docs.aws.amazon.com/eks/latest/userguide/getting-started-eksctl.html "Get started with Amazon EKS – eksctl - Amazon EKS"
[17]: https://docs.aws.amazon.com/eks/latest/userguide/create-kubeconfig.html "Connect kubectl to an EKS cluster by creating a kubeconfig file - Amazon EKS"
[18]: https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html?utm_source=chatgpt.com "Use Kubernetes volume storage with Amazon EBS"
[19]: https://docs.aws.amazon.com/eks/latest/userguide/workloads-add-ons-available-eks.html?utm_source=chatgpt.com "AWS add-ons - Amazon EKS"
[20]: https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html?utm_source=chatgpt.com "Creating an Amazon ECR private repository to store images"
[21]: https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html?utm_source=chatgpt.com "push a Docker image to an Amazon ECR repository"
[22]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/docker/api.Dockerfile "raw.githubusercontent.com"
[23]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/docker/web.Dockerfile "raw.githubusercontent.com"
[24]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/docker/worker.Dockerfile "raw.githubusercontent.com"
[25]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/configmap.yaml "raw.githubusercontent.com"
[26]: https://github.com/itwake/llm-powered-log-analytic/tree/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s "llm-powered-log-analytic/infra/k8s at ab86b966c5a2ce5202d7afb0eae96f1a09c2e293 · itwake/llm-powered-log-analytic · GitHub"
[27]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/postgres.yaml "raw.githubusercontent.com"
[28]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/minio.yaml "raw.githubusercontent.com"
[29]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/migration-job.yaml "raw.githubusercontent.com"
[30]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/api-deployment.yaml "raw.githubusercontent.com"
[31]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/web-deployment.yaml "raw.githubusercontent.com"
[32]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/worker-deployment.yaml "raw.githubusercontent.com"
[33]: https://docs.aws.amazon.com/eks/latest/userguide/auto-configure-alb.html?utm_source=chatgpt.com "Create an IngressClass to configure an Application Load ..."

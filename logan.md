可以。下面这份可以直接作为测试安装说明使用。我按当前远端分支 `codex/logan-platform-foundation`、commit `ab86b966c5a2ce5202d7afb0eae96f1a09c2e293` 来写。

当前项目已经具备：FastAPI 后端、Next.js 工作台、PostgreSQL/SQLite 持久化、MinIO/S3 上传、Temporal worker、ClickHouse/OpenSearch 可选分析 sink、五个报告视图、Copilot-backed chat、RBAC、审计、retention、Docker 和 K8s 部署脚手架。README 也明确说明本项目的五个核心视图是 Data Summary、Temporal View、Tabular Logs、Causal Graph 和 Causal Summary。([GitHub][1])

---

# 一、测试前必须知道的要点

## 1. 先用 mock 模型跑通平台，再跑真实 Copilot

本地和 EKS 初次测试建议先设置：

```bash
LOGAN_LLM_PROVIDER=mock
```

这样可以验证上传、分析、Temporal、worker、报告、导出、UI，不需要 GitHub Copilot token。仓库文档说明 full-stack smoke 默认使用 mock LLM，不需要 Copilot credentials。([GitHub][1])

等平台链路稳定后，再切到：

```bash
LOGAN_LLM_PROVIDER=github_copilot
```

真实 Copilot 方式有两种：

1. 用户在 UI 里走 GitHub device-code 授权；
2. 后端环境提供 `LOGAN_GITHUB_COPILOT_TOKEN` 或 `LOGAN_GITHUB_SOURCE_TOKEN`。

API 文档里说明了 `/api/copilot/auth/start`、`/api/copilot/auth/check`、`/api/copilot/auth/credential` 三个授权接口，并且说明后端模型网关会按“已缓存 Copilot plugin token → GitHub source OAuth 交换 → 环境变量 token”的顺序解析凭证。([GitHub][2])

## 2. 因果图只能当候选证据

Causal Graph 的边是 `candidate_cause`，不是确定 RCA。security 文档也要求摘要使用 `candidate cause`、`likely`、`evidence suggests`、`needs validation` 这类谨慎语言。([GitHub][3])

## 3. Windows 本地建议用 WSL2 + Docker Desktop

Docker 官方建议 Windows Docker Desktop 使用 WSL 2 backend，并且安装后在 Docker Desktop 设置里启用 “Use WSL 2 based engine”。([Docker Documentation][4])

## 4. EKS 首测建议先用 port-forward，不急着配公网 Ingress

EKS 上先通过：

```bash
kubectl port-forward svc/logan-web 3000:3000 -n logan
kubectl port-forward svc/logan-api 8000:8000 -n logan
```

在本地浏览器访问 `http://localhost:3000`。这样可以避开 ALB、DNS、TLS、CORS、Next.js build-time env 的复杂性。

## 5. 当前 K8s manifest 需要小改才能顺利测试

当前 K8s scaffold 已有 namespace、config、secrets、API/Web/Worker deployment、migration job、Postgres、Redis、MinIO、Temporal、ClickHouse、OpenSearch、Ingress、HPA 等文件。([GitHub][5])

但测试时要特别注意三点：

1. `postgres.yaml` 当前把 PostgreSQL 密码绑定到了 `LOGAN_SECRET_KEY`，所以 `LOGAN_DATABASE_URL` 里的数据库密码必须和 `LOGAN_SECRET_KEY` 一致，或者你需要改 manifest 使用单独的 `LOGAN_DB_PASSWORD`。([GitHub][6])
2. K8s configmap 示例默认没有把对象存储切到 MinIO/S3；在 K8s 多 pod 场景里不要用 API pod 本地文件存储，应该设置 `LOGAN_OBJECT_STORE_BACKEND=minio` 或 `s3`。
3. Web 镜像的 Dockerfile 是 build Next.js 后用 `next start` 运行。([GitHub][7]) 如果要通过公网域名访问，`NEXT_PUBLIC_API_BASE_URL` 很可能需要在 Web build 阶段就设置好；首测用 port-forward 可以规避这个问题。

---

# 二、本地 Windows 安装和使用

## 2.1 推荐环境

建议使用：

```text
Windows 11
Docker Desktop + WSL2 backend
PowerShell 7 或 Windows Terminal
Git
Python 3.11 或 3.12
Node.js 20+
corepack / pnpm 10.13.1
```

项目 README 写明 Python 3.11+ 是必需的，Node 20+ 和 pnpm 推荐用于 Web workspace。([GitHub][1]) 项目的 `pyproject.toml` 也声明 `requires-python = ">=3.11"`。([GitHub][8])

---

## 2.2 拉取代码

PowerShell：

```powershell
git clone -b codex/logan-platform-foundation https://github.com/itwake/llm-powered-log-analytic.git
cd llm-powered-log-analytic
git checkout ab86b966c5a2ce5202d7afb0eae96f1a09c2e293
```

确认：

```powershell
git status
git log -1 --oneline
```

应该看到：

```text
ab86b96 Harden deployment readiness probes
```

---

## 2.3 本地轻量模式：不依赖 Docker，适合快速看 UI 和跑 mock 分析

这个模式使用：

```text
API: FastAPI
Web: Next.js dev server
Store: memory
Object store: local
LLM: mock
```

安装 Python 依赖：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e . pytest pytest-asyncio ruff
```

安装 Web 依赖：

```powershell
corepack enable
corepack prepare pnpm@10.13.1 --activate
pnpm install
```

启动 API：

```powershell
$env:LOGAN_ENV="development"
$env:LOGAN_LLM_PROVIDER="mock"
$env:LOGAN_STORE_BACKEND="memory"
$env:LOGAN_OBJECT_STORE_BACKEND="local"
$env:LOGAN_LOCAL_OBJECT_STORE_DIR=".logan/object-store"
$env:LOGAN_CORS_ALLOWED_ORIGINS="http://localhost:3000"
$env:LOGAN_METRICS_ENABLED="true"

python -m uvicorn app.main:app --reload --app-dir apps/api --host 127.0.0.1 --port 8000
```

另开一个 PowerShell，启动 Web：

```powershell
cd C:\Work\llm-powered-log-analytic
$env:NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"
corepack pnpm --filter @logan/web dev --hostname 127.0.0.1 --port 3000
```

打开：

```text
http://localhost:3000
```

健康检查：

```powershell
curl.exe http://localhost:8000/healthz
curl.exe http://localhost:8000/metrics
```

README 说明本地 API 默认使用 in-memory metadata，除非设置了 `LOGAN_DATABASE_URL`；上传内容默认写入 `.logan/object-store`。([GitHub][1])

---

## 2.4 本地验证命令

运行单元测试：

```powershell
python -m pytest -q
```

运行 deterministic benchmark：

```powershell
python -m logan_workers.evaluation.run `
  --benchmark benchmarks/logan/checkout_incident `
  --out .logan/evaluation/report.json `
  --markdown .logan/evaluation/report.md
```

operations 文档说明 benchmark 使用 `MockCopilotAnnotationGateway`，不需要 Docker、Temporal、Copilot credentials 或外部网络，并会输出 review-load reduction、golden signal F1、fault category F1、entity F1、root-cause hit@k、causal-edge recall 和 summary quality 等指标。([GitHub][9])

安装 Playwright 浏览器：

```powershell
corepack pnpm exec playwright install chromium
```

运行 E2E：

```powershell
corepack pnpm e2e
```

README 说明 Playwright E2E 会启动 FastAPI `127.0.0.1:8000` 和 Next.js `127.0.0.1:3000`，并默认使用 memory store、local object store 和 mock LLM。([GitHub][1])

---

## 2.5 本地 full-stack Docker 模式

这个模式更接近落地环境，会启动：

```text
PostgreSQL
Redis
MinIO
ClickHouse
OpenSearch
Temporal
FastAPI API
Temporal worker
Next.js Web
```

docker-compose 文档说明 full-stack smoke 会启动 PostgreSQL、MinIO、ClickHouse、OpenSearch、Temporal、API 和 worker，并验证 MinIO 上传、Temporal materialization、五个 report endpoint、ClickHouse/OpenSearch 写入等路径。([GitHub][9])

确认 Docker Desktop 已启动，并给 Docker 至少：

```text
CPU: 4 cores+
Memory: 8 GB+，建议 12–16 GB
Disk: 20 GB+
```

复制环境文件：

```powershell
copy .env.example .env
```

检查 compose 配置：

```powershell
docker compose config
```

启动完整栈：

```powershell
docker compose up -d --build postgres redis minio minio-init clickhouse opensearch temporal api worker web
```

查看状态：

```powershell
docker compose ps
```

看 API health：

```powershell
curl.exe http://localhost:8000/healthz
```

看 Web：

```text
http://localhost:3000
```

运行 smoke：

```powershell
docker compose --profile smoke run --rm smoke
```

停止并清理：

```powershell
docker compose down -v
```

如果你在 WSL2 或 Git Bash 中有 `make`，可以直接用项目命令：

```bash
make full-stack-smoke
make full-stack-down
```

项目文档也列出了这个方式。([GitHub][9])

---

## 2.6 本地 UI 使用流程

进入 `http://localhost:3000` 后，按这个路径测试：

1. 注册一个用户。
2. 登录。
3. 创建 Case，例如：

   ```text
   Title: Checkout incident test
   Product: demo-checkout
   Service: payment-service
   Environment: local
   ```
4. 上传日志文件。可以先用仓库里的测试日志：

   ```text
   tests/fixtures/logs/checkout_incident
   ```
5. Start Analysis。
6. 等 run 状态变成 `succeeded`。
7. 依次检查：

   ```text
   Data Summary
   Temporal View
   Tabular Logs
   Causal Graph
   Causal Summary
   ```
8. 导出 summary。
9. 提交 feedback。

API 文档说明 case 和 analysis API 包括创建 case、上传文件、complete upload、启动 analysis-run、获取 summary/temporal/logs/causal graph/causal summary/export/feedback 等接口。([GitHub][2])

---

## 2.7 本地切换真实 Copilot

轻量模式或 Docker 模式都可以切真实 Copilot。

### 方式 A：UI device-code 授权

启动 API 时：

```powershell
$env:LOGAN_LLM_PROVIDER="github_copilot"
```

登录 Web 后，使用 Copilot 授权入口。后端会通过 `/api/copilot/auth/start` 发起 GitHub device-code flow，通过 `/api/copilot/auth/check` 轮询授权结果。API 文档说明授权成功后不会把 source token、plugin token 或加密字节返回前端。([GitHub][2])

### 方式 B：环境变量 token

PowerShell：

```powershell
$env:LOGAN_LLM_PROVIDER="github_copilot"
$env:LOGAN_GITHUB_COPILOT_TOKEN="<your-copilot-plugin-token>"
```

或者：

```powershell
$env:LOGAN_GITHUB_SOURCE_TOKEN="<your-github-source-oauth-or-pat-token>"
```

真实 staging smoke：

```powershell
$env:LOGAN_RUN_COPILOT_STAGING_SMOKE="true"
$env:LOGAN_GITHUB_COPILOT_TOKEN="<token>"
make copilot-staging-smoke
```

如果 Windows 没有 `make`，建议在 WSL2 里跑这一条。

---

# 三、AWS EKS / Kubernetes 安装和使用

下面给的是“可开始测试”的 EKS 部署方式，不是完整生产化方案。它优先目标是：

```text
EKS 上跑通 API + Web + Worker + PostgreSQL + MinIO + Temporal
先用 mock LLM 验证平台
通过 port-forward 访问
再切真实 Copilot
最后再考虑 ALB/Ingress、RDS、S3、OpenSearch managed service
```

AWS 官方 EKS 快速开始文档说明，使用 `eksctl` 创建集群前需要安装并配置 AWS CLI、kubectl 和 eksctl。([AWS 文档][10]) AWS CLI 的 `aws eks update-kubeconfig` 会创建或更新 kubeconfig，用于让 `kubectl` 连接 EKS cluster。([AWS 文档][11])

---

## 3.1 EKS 前置条件

本地或 AWS CloudShell 需要：

```text
AWS CLI v2
kubectl
eksctl
Docker
ECR push 权限
EKS 创建权限
```

Windows 用户建议在 WSL2 Ubuntu 里跑 EKS 命令。kubectl 官方文档也支持 Windows 上用 winget、Chocolatey 或 Scoop 安装。([Kubernetes][12])

设置变量：

```bash
export AWS_REGION=ap-southeast-1
export CLUSTER_NAME=logan-pilot
export TAG=ab86b966
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REGISTRY=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
```

---

## 3.2 创建 EKS 集群

快速创建一个测试集群：

```bash
eksctl create cluster \
  --name ${CLUSTER_NAME} \
  --region ${AWS_REGION} \
  --managed \
  --nodes 3 \
  --node-type m6i.xlarge \
  --nodes-min 2 \
  --nodes-max 5
```

连接集群：

```bash
aws eks update-kubeconfig \
  --region ${AWS_REGION} \
  --name ${CLUSTER_NAME}
```

确认：

```bash
kubectl get nodes
kubectl get ns
```

如果后续 PVC 一直 Pending，检查 StorageClass：

```bash
kubectl get storageclass
```

如果没有可用 StorageClass，需要安装/启用 EBS CSI driver。AWS 文档建议通过 EKS add-on 安装 Amazon EBS CSI driver。([AWS 文档][13])

---

## 3.3 构建并推送镜像到 ECR

当前 K8s deployment 默认镜像是：

```text
ghcr.io/itwake/logan-api:latest
ghcr.io/itwake/logan-web:latest
ghcr.io/itwake/logan-worker:latest
```

API、Web、Worker deployment manifest 里都使用了这些镜像。([GitHub][14]) ([GitHub][15]) ([GitHub][16])

为了测试稳定，建议构建自己的 ECR 镜像。

创建 ECR repo：

```bash
for repo in logan-api logan-worker logan-web; do
  aws ecr describe-repositories --repository-names ${repo} --region ${AWS_REGION} >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name ${repo} --region ${AWS_REGION}
done
```

登录 ECR：

```bash
aws ecr get-login-password --region ${AWS_REGION} \
  | docker login --username AWS --password-stdin ${ECR_REGISTRY}
```

AWS ECR 文档说明推送镜像前需要用 `aws ecr get-login-password` 给 Docker registry 登录，认证 token 有效期为 12 小时。([AWS 文档][17])

构建镜像：

```bash
docker build -t ${ECR_REGISTRY}/logan-api:${TAG} \
  -f infra/docker/api.Dockerfile .

docker build -t ${ECR_REGISTRY}/logan-worker:${TAG} \
  -f infra/docker/worker.Dockerfile .

docker build -t ${ECR_REGISTRY}/logan-web:${TAG} \
  -f infra/docker/web.Dockerfile .
```

推送：

```bash
docker push ${ECR_REGISTRY}/logan-api:${TAG}
docker push ${ECR_REGISTRY}/logan-worker:${TAG}
docker push ${ECR_REGISTRY}/logan-web:${TAG}
```

API Dockerfile 使用 Python 3.12 slim、非 editable 安装，并通过 `/healthz` 做 healthcheck。([GitHub][18]) Worker Dockerfile 启动 `python -m logan_workers.temporal_worker`。([GitHub][19]) Web Dockerfile build Next.js app 后用 `next start` 运行，并暴露 `/healthz`。([GitHub][7])

---

## 3.4 准备 K8s 配置

创建 namespace：

```bash
kubectl apply -f infra/k8s/namespace.yaml
```

当前 namespace manifest 创建的是 `logan` namespace。([GitHub][20])

生成测试密钥：

```bash
export LOGAN_APP_AND_DB_SECRET=$(openssl rand -hex 32)
export LOGAN_CREDENTIAL_KEY=$(openssl rand -hex 32)
export MINIO_SECRET=$(openssl rand -hex 32)
```

注意：因为当前 `postgres.yaml` 用 `LOGAN_SECRET_KEY` 作为 PostgreSQL 密码，所以这里让 `LOGAN_DATABASE_URL` 的密码和 `LOGAN_SECRET_KEY` 一致。这个只是测试简化方式；后续建议把 DB password 独立成 `LOGAN_DB_PASSWORD`。

创建 secret：

```bash
kubectl create secret generic logan-secrets \
  -n logan \
  --from-literal=LOGAN_SECRET_KEY="${LOGAN_APP_AND_DB_SECRET}" \
  --from-literal=LOGAN_CREDENTIAL_ENCRYPTION_KEY="${LOGAN_CREDENTIAL_KEY}" \
  --from-literal=LOGAN_DATABASE_URL="postgresql+psycopg://logan:${LOGAN_APP_AND_DB_SECRET}@postgres:5432/logan" \
  --from-literal=LOGAN_REDIS_URL="redis://redis:6379/0" \
  --from-literal=LOGAN_S3_ACCESS_KEY="logan" \
  --from-literal=LOGAN_S3_SECRET_KEY="${MINIO_SECRET}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

安全文档说明生产环境下如果 `LOGAN_SECRET_KEY` 或 `LOGAN_CREDENTIAL_ENCRYPTION_KEY` 是默认值或短于 32 字符，API 会拒绝启动。([GitHub][3])

创建 configmap。首测建议使用 mock LLM、MinIO、Temporal，并关闭 ClickHouse/OpenSearch 外部查询：

```bash
cat > /tmp/logan-config.yaml <<'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: logan-config
  namespace: logan
data:
  LOGAN_ENV: "production"

  LOGAN_LLM_PROVIDER: "mock"
  LOGAN_COPILOT_MODEL: "gpt-5.4"
  LOGAN_COPILOT_REASONING_EFFORT: "high"
  LOGAN_COPILOT_OAUTH_CLIENT_ID: "Iv1.b507a08c87ecfe98"

  LOGAN_STORE_BACKEND: "auto"

  LOGAN_ANALYSIS_ORCHESTRATOR: "temporal"
  LOGAN_TEMPORAL_ADDRESS: "temporal:7233"
  LOGAN_TEMPORAL_NAMESPACE: "default"
  LOGAN_TEMPORAL_TASK_QUEUE: "logan-analysis"
  LOGAN_TEMPORAL_ACTIVITY_START_TO_CLOSE_SECONDS: "3600"
  LOGAN_TEMPORAL_ACTIVITY_MAX_ATTEMPTS: "3"

  LOGAN_OBJECT_STORE_BACKEND: "minio"
  LOGAN_S3_ENDPOINT: "http://minio:9000"
  LOGAN_S3_BUCKET: "logan"
  LOGAN_S3_REGION: "us-east-1"
  LOGAN_S3_FORCE_PATH_STYLE: "true"
  LOGAN_S3_PRESIGN_EXPIRES_SECONDS: "900"
  LOGAN_S3_MULTIPART_THRESHOLD_BYTES: "104857600"
  LOGAN_S3_MULTIPART_PART_SIZE_BYTES: "67108864"
  LOGAN_S3_MULTIPART_MAX_PARTS: "10000"
  LOGAN_ANALYSIS_INPUT_TMP_DIR: "/tmp/logan-analysis-inputs"

  LOGAN_STEP_ARTIFACTS_ENABLED: "true"
  LOGAN_STEP_ARTIFACT_FAILURE_MODE: "warn"

  LOGAN_ANALYTICS_SINKS_ENABLED: "false"
  LOGAN_EXTERNAL_ANALYTICS_QUERIES_ENABLED: "false"
  LOGAN_ANALYTICS_SINK_FAILURE_MODE: "warn"

  LOGAN_RAW_LOG_RETENTION_DAYS: "30"
  LOGAN_REPORT_RETENTION_DAYS: "365"
  LOGAN_AUDIT_RETENTION_DAYS: "730"

  LOGAN_RATE_LIMIT_ENABLED: "false"
  LOGAN_RATE_LIMIT_REQUESTS_PER_MINUTE: "120"

  LOGAN_API_WORKERS: "2"
  LOGAN_METRICS_ENABLED: "true"
  LOGAN_METRICS_PATH: "/metrics"

  LOGAN_CORS_ALLOWED_ORIGINS: "http://localhost:3000"
  NEXT_PUBLIC_API_BASE_URL: "http://localhost:8000"

  LOGAN_OTEL_ENABLED: "false"
  LOGAN_OTEL_SERVICE_NAME: "logan-api"
  LOGAN_OTEL_EXPORTER_OTLP_ENDPOINT: ""
EOF

kubectl apply -f /tmp/logan-config.yaml
```

repo 自带的 configmap 示例使用 production、Temporal、GitHub Copilot、`LOGAN_API_WORKERS=2`、`LOGAN_CORS_ALLOWED_ORIGINS=https://logan.example.com` 和 `NEXT_PUBLIC_API_BASE_URL=https://logan.example.com`。 ([GitHub][21]) 我这里为了首测改成了 localhost port-forward 和 mock LLM。

---

## 3.5 部署依赖服务

部署 PostgreSQL、Redis、MinIO、Temporal：

```bash
kubectl apply -f infra/k8s/postgres.yaml
kubectl apply -f infra/k8s/redis.yaml
kubectl apply -f infra/k8s/minio.yaml
kubectl apply -f infra/k8s/temporal.yaml
```

PostgreSQL manifest 包含 20Gi PVC、Deployment 和 Service。([GitHub][6]) MinIO manifest 包含 50Gi PVC、Deployment 和 Service。([GitHub][22]) Temporal manifest 使用 `temporalio/auto-setup` 并连接 PostgreSQL。([GitHub][23])

等待：

```bash
kubectl rollout status deployment/postgres -n logan --timeout=180s
kubectl rollout status deployment/redis -n logan --timeout=180s
kubectl rollout status deployment/minio -n logan --timeout=180s
kubectl rollout status deployment/temporal -n logan --timeout=240s
```

创建 MinIO bucket。当前 K8s 目录没有类似 docker-compose 里的 `minio-init`，所以需要补一个一次性 Job：

```bash
cat > /tmp/minio-init.yaml <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: minio-init
  namespace: logan
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: mc
          image: quay.io/minio/mc:latest
          envFrom:
            - secretRef:
                name: logan-secrets
          command:
            - /bin/sh
            - -ec
            - |
              mc alias set local http://minio:9000 "$LOGAN_S3_ACCESS_KEY" "$LOGAN_S3_SECRET_KEY"
              mc mb --ignore-existing local/logan
EOF

kubectl apply -f /tmp/minio-init.yaml
kubectl wait --for=condition=complete job/minio-init -n logan --timeout=120s
kubectl logs job/minio-init -n logan
```

---

## 3.6 打补丁使用 ECR 镜像

复制一份 manifest：

```bash
rm -rf /tmp/logan-k8s
cp -r infra/k8s /tmp/logan-k8s
```

替换镜像：

```bash
perl -pi -e "s#ghcr.io/itwake/logan-api:latest#$ENV{ECR_REGISTRY}/logan-api:$ENV{TAG}#g" \
  /tmp/logan-k8s/api-deployment.yaml \
  /tmp/logan-k8s/migration-job.yaml

perl -pi -e "s#ghcr.io/itwake/logan-worker:latest#$ENV{ECR_REGISTRY}/logan-worker:$ENV{TAG}#g" \
  /tmp/logan-k8s/worker-deployment.yaml

perl -pi -e "s#ghcr.io/itwake/logan-web:latest#$ENV{ECR_REGISTRY}/logan-web:$ENV{TAG}#g" \
  /tmp/logan-k8s/web-deployment.yaml
```

检查：

```bash
grep -R "${ECR_REGISTRY}" /tmp/logan-k8s
```

---

## 3.7 跑数据库迁移

当前 migration job 使用：

```text
python scripts/run_migrations.py
```

manifest 已经不再调用 Alembic。([GitHub][24])

执行：

```bash
kubectl apply -f /tmp/logan-k8s/migration-job.yaml
kubectl wait --for=condition=complete job/logan-migration -n logan --timeout=180s
kubectl logs job/logan-migration -n logan
```

如果失败，先看：

```bash
kubectl describe job logan-migration -n logan
kubectl logs job/logan-migration -n logan
```

常见原因是：

```text
LOGAN_DATABASE_URL 密码和 postgres 的 POSTGRES_PASSWORD 不一致
Postgres 还没 ready
PVC Pending
```

---

## 3.8 部署 API、Worker、Web

```bash
kubectl apply -f /tmp/logan-k8s/api-deployment.yaml
kubectl apply -f /tmp/logan-k8s/worker-deployment.yaml
kubectl apply -f /tmp/logan-k8s/web-deployment.yaml
```

等待：

```bash
kubectl rollout status deployment/logan-api -n logan --timeout=240s
kubectl rollout status deployment/logan-worker -n logan --timeout=240s
kubectl rollout status deployment/logan-web -n logan --timeout=240s
```

API 和 Web deployment 都配置了 `/healthz` readiness/liveness probes。([GitHub][14]) ([GitHub][15])

检查 pod：

```bash
kubectl get pods -n logan -o wide
kubectl get svc -n logan
```

---

## 3.9 用 port-forward 访问

开第一个终端：

```bash
kubectl port-forward svc/logan-api 8000:8000 -n logan
```

开第二个终端：

```bash
kubectl port-forward svc/logan-web 3000:3000 -n logan
```

打开：

```text
http://localhost:3000
```

健康检查：

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/metrics
```

---

## 3.10 EKS 上的 UI 使用流程

和本地一样：

1. 注册用户。
2. 登录。
3. 创建 Case。
4. 上传日志。
5. Start Analysis。
6. 等 run 变成 `succeeded`。
7. 查看五个视图：

   ```text
   Data Summary
   Temporal View
   Tabular Logs
   Causal Graph
   Causal Summary
   ```
8. 导出 summary。
9. 提交 feedback。

查看 worker 日志：

```bash
kubectl logs deployment/logan-worker -n logan -f
```

查看 API 日志：

```bash
kubectl logs deployment/logan-api -n logan -f
```

查看事件：

```bash
kubectl get events -n logan --sort-by=.metadata.creationTimestamp
```

---

## 3.11 切换 EKS 到真实 Copilot

确认平台 mock 跑通后，更新 configmap：

```bash
kubectl patch configmap logan-config -n logan --type merge -p '
{
  "data": {
    "LOGAN_LLM_PROVIDER": "github_copilot"
  }
}'
```

如果使用环境 token，创建/更新 secret：

```bash
kubectl patch secret logan-secrets -n logan --type merge -p "$(cat <<EOF
{
  "stringData": {
    "LOGAN_GITHUB_COPILOT_TOKEN": "<your-copilot-plugin-token>"
  }
}
EOF
)"
```

重启 API 和 worker：

```bash
kubectl rollout restart deployment/logan-api -n logan
kubectl rollout restart deployment/logan-worker -n logan

kubectl rollout status deployment/logan-api -n logan --timeout=180s
kubectl rollout status deployment/logan-worker -n logan --timeout=180s
```

更安全的方式是不用环境 token，而是让测试用户通过 UI 做 Copilot device-code 授权。security 文档说明 GitHub source OAuth 和 Copilot plugin token 会加密保存，并且不会返回前端。([GitHub][3])

---

## 3.12 可选：通过 Ingress / ALB 暴露

首测不建议直接做公网。等 port-forward 跑通后再加。

当前 ingress manifest 路由：

```text
/api -> logan-api:8000
/    -> logan-web:3000
host -> logan.example.com
```

repo 的 ingress.yaml 已经定义了这个路径路由。([GitHub][25])

EKS 上如果要用 AWS ALB，需要安装 AWS Load Balancer Controller。AWS 文档建议新手用 Helm 安装 AWS Load Balancer Controller。([AWS 文档][26])

Ingress 需要补类似 annotation：

```yaml
metadata:
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
spec:
  ingressClassName: alb
```

然后：

```bash
kubectl apply -f /tmp/logan-k8s/ingress.yaml
kubectl get ingress -n logan
```

拿到 ALB 地址后，在 DNS 里把你的域名 CNAME 到 ALB。

重要：如果用公网域名访问，Web 镜像需要用正确的 API base URL 构建。当前前端代码读取 `process.env.NEXT_PUBLIC_API_BASE_URL`，默认是 `http://localhost:8000`。 ([GitHub][27]) 对公网部署，我建议把前端改为相对路径 `/api`，或者修改 Web Dockerfile 支持 build arg：

```Dockerfile
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
RUN pnpm --filter @logan/web build
```

然后 build：

```bash
docker build \
  --build-arg NEXT_PUBLIC_API_BASE_URL=https://logan.example.com \
  -t ${ECR_REGISTRY}/logan-web:${TAG} \
  -f infra/docker/web.Dockerfile .
```

同时把 configmap 里的 CORS 改掉：

```bash
kubectl patch configmap logan-config -n logan --type merge -p '
{
  "data": {
    "LOGAN_CORS_ALLOWED_ORIGINS": "https://logan.example.com",
    "NEXT_PUBLIC_API_BASE_URL": "https://logan.example.com"
  }
}'
```

---

# 四、建议的测试顺序

按这个顺序推进最稳：

## 第 1 轮：Windows 轻量模式

目标：确认代码和 UI 能跑。

```text
python -m pytest -q
benchmark evaluation
API + Web dev server
上传小日志
mock analysis succeeded
五个视图可打开
```

## 第 2 轮：Windows Docker full-stack

目标：确认接近落地的服务链路。

```text
docker compose up
PostgreSQL
MinIO
Temporal
API
Worker
Web
full-stack smoke pass
```

full-stack smoke 会验证 API health、注册/登录、case 创建、MinIO 上传、Temporal worker materialization、五个 report endpoint、ClickHouse/OpenSearch 写入等路径。([GitHub][9])

## 第 3 轮：EKS mock

目标：确认 K8s 部署链路。

```text
EKS cluster ready
ECR images pulled
PostgreSQL PVC bound
MinIO bucket exists
Temporal reachable
migration complete
API/Web/Worker rollout success
port-forward UI 可用
mock analysis succeeded
```

## 第 4 轮：真实 Copilot staging

目标：确认模型路径。

```text
LOGAN_LLM_PROVIDER=github_copilot
UI device-code auth 或 LOGAN_GITHUB_COPILOT_TOKEN
小日志 case 分析成功
Causal Summary 质量可接受
无 raw log 泄露到 audit/exports
```

## 第 5 轮：真实历史 incident 回放

目标：确认实际价值。

至少跑 5–10 个历史 case，记录：

```text
raw log lines
templates
offending templates
model call count
processing time
review-load reduction
工程师是否认为 Data Summary 有用
工程师是否认为 Causal Graph 有用
summary 是否能贴到 ticket/customer update
root cause candidate 是否进 top 3
```

---

# 五、常见问题排查

## 1. API 起不来

看日志：

```bash
kubectl logs deployment/logan-api -n logan
```

重点检查：

```text
LOGAN_SECRET_KEY 是否少于 32 字符
LOGAN_CREDENTIAL_ENCRYPTION_KEY 是否少于 32 字符
LOGAN_DATABASE_URL 是否能连 Postgres
CORS origin 是否正确
```

security 文档说明 production 模式下短 key 或默认 key 会导致 API 拒绝启动。([GitHub][3])

## 2. migration job 失败

看：

```bash
kubectl logs job/logan-migration -n logan
```

最常见是数据库密码不一致。当前 `postgres.yaml` 用 `LOGAN_SECRET_KEY` 作为 `POSTGRES_PASSWORD`。([GitHub][6]) 所以 `LOGAN_DATABASE_URL` 必须匹配。

## 3. 上传失败

检查对象存储配置：

```bash
kubectl logs deployment/logan-api -n logan | grep -i s3
kubectl get svc minio -n logan
kubectl logs job/minio-init -n logan
```

K8s 多 pod 下不要用 local object store；要用 MinIO 或 S3。API 文档说明 S3/MinIO 上传会返回 presigned PUT URL，并在 complete 时通过 `head_object` 验证对象存在和大小。([GitHub][2])

## 4. analysis 一直 pending

看 worker：

```bash
kubectl logs deployment/logan-worker -n logan -f
```

看 Temporal：

```bash
kubectl logs deployment/temporal -n logan
```

确认 configmap：

```bash
kubectl get configmap logan-config -n logan -o yaml | grep LOGAN_TEMPORAL
```

## 5. Web 能打开，但请求 API 失败

本地 port-forward 模式应使用：

```text
LOGAN_CORS_ALLOWED_ORIGINS=http://localhost:3000
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

如果是公网域名，必须同步处理：

```text
Web build-time NEXT_PUBLIC_API_BASE_URL
API runtime LOGAN_CORS_ALLOWED_ORIGINS
Ingress host/path
cookie credentials
```

README 说明浏览器 API calls 使用 `credentials: "include"`，所以 CORS 必须允许带 cookie 的前端 origin。([GitHub][1])

## 6. Causal Graph 看起来“不准”

这是预期内风险。当前系统输出的是 candidate causal evidence，不是确定 root cause。README 明确说明 PGEM-style transition score 和 Granger-style lagged-linear score 只是候选因果证据，用于排序和验证，不证明根因。([GitHub][1])

---

# 六、什么时候算“可以开始测试成功”

本地 Windows 算成功：

```text
pytest 通过
benchmark 通过
web build/typecheck 通过
API/Web 能启动
能注册/登录
能创建 case
能上传日志
mock analysis succeeded
五个视图都有数据
export 可生成
feedback 可提交
```

EKS 算成功：

```text
kubectl get pods -n logan 全部 Running/Completed
migration job Completed
API /healthz OK
Web /healthz OK
MinIO bucket exists
Temporal worker 有处理日志
port-forward 后 UI 可用
上传真实小日志成功
analysis run succeeded
五个报告 endpoint 非空
```

下一步真正决定能不能扩大试用的是：**真实 Copilot staging smoke + 5–10 个真实历史 incident 回放**。当前安装测试先用 mock，把平台部署、数据流和 UI 工作流跑稳。

[1]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/README.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/docs/api.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/docs/security.md "raw.githubusercontent.com"
[4]: https://docs.docker.com/desktop/features/wsl/?utm_source=chatgpt.com "Docker Desktop WSL 2 backend on Windows"
[5]: https://github.com/itwake/llm-powered-log-analytic/tree/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s "llm-powered-log-analytic/infra/k8s at ab86b966c5a2ce5202d7afb0eae96f1a09c2e293 · itwake/llm-powered-log-analytic · GitHub"
[6]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/postgres.yaml "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/docker/web.Dockerfile "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/pyproject.toml "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/docs/operations.md "raw.githubusercontent.com"
[10]: https://docs.aws.amazon.com/eks/latest/userguide/getting-started-eksctl.html?utm_source=chatgpt.com "Get started with Amazon EKS – eksctl"
[11]: https://docs.aws.amazon.com/eks/latest/userguide/create-kubeconfig.html?utm_source=chatgpt.com "Connect kubectl to an EKS cluster by creating a kubeconfig file"
[12]: https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/?utm_source=chatgpt.com "Install and Set Up kubectl on Windows"
[13]: https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html?utm_source=chatgpt.com "Use Kubernetes volume storage with Amazon EBS"
[14]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/api-deployment.yaml "raw.githubusercontent.com"
[15]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/web-deployment.yaml "raw.githubusercontent.com"
[16]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/worker-deployment.yaml "raw.githubusercontent.com"
[17]: https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html?utm_source=chatgpt.com "push a Docker image to an Amazon ECR repository"
[18]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/docker/api.Dockerfile "raw.githubusercontent.com"
[19]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/docker/worker.Dockerfile "raw.githubusercontent.com"
[20]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/namespace.yaml "raw.githubusercontent.com"
[21]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/configmap.yaml "raw.githubusercontent.com"
[22]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/minio.yaml "raw.githubusercontent.com"
[23]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/temporal.yaml "raw.githubusercontent.com"
[24]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/migration-job.yaml "raw.githubusercontent.com"
[25]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/infra/k8s/ingress.yaml "raw.githubusercontent.com"
[26]: https://docs.aws.amazon.com/eks/latest/userguide/aws-load-balancer-controller.html?utm_source=chatgpt.com "Route internet traffic with AWS Load Balancer Controller"
[27]: https://raw.githubusercontent.com/itwake/llm-powered-log-analytic/ab86b966c5a2ce5202d7afb0eae96f1a09c2e293/apps/web/src/lib/api.ts "raw.githubusercontent.com"

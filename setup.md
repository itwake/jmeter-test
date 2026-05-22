下面给你一个**通用落地方案**：把 VS Code Copilot 的“全局能力”拆成四层：

1. **全局指令层**：VS Code user-level `*.instructions.md`，真正跨所有 workspace 生效。
2. **全局 AGENTS 层**：放一份你自己的 `AGENTS.md`，由全局 instructions 显式引用。
3. **工具层**：安装 `jira` / `confluence` CLI 到 PATH，让 Copilot Agent 能通过终端调用。
4. **记忆层**：`MEMORY.md + memory/daily/YYYY-MM-DD.md`，用 Markdown 文件持久化。

VS Code 官方文档里，user-level instruction 文件可以放在 `~/.copilot/instructions`，而 `AGENTS.md` 主要是 workspace/root/subfolder 级别；所以“全局 AGENTS.md”的可靠做法是：**全局 `.instructions.md` 自动加载，再引用你的全局 `AGENTS.md` 和 `MEMORY.md`**。VS Code 也支持 Markdown link 引用额外指令文件，并可通过 Diagnostics 检查哪些 instructions 被加载。([Visual Studio Code][1])

---

## 0. 推荐目录结构

### macOS

```text
~/.copilot/
  AGENTS.md
  MEMORY.md
  memory/
    daily/
      2026-05-21.md
    projects/
    people/
    archive/
  instructions/
    global-agent.instructions.md

~/.config/atlassian/
  config.json

~/.local/bin/
  jira
  confluence
```

### Windows

```text
%USERPROFILE%\.copilot\
  AGENTS.md
  MEMORY.md
  memory\
    daily\
      2026-05-21.md
    projects\
    people\
    archive\
  instructions\
    global-agent.instructions.md

%APPDATA%\atlassian\
  config.json

%USERPROFILE%\.local\bin\
  jira.exe
  confluence.exe
```

`MEMORY.md + daily notes` 这种结构的核心思想是：长期、稳定、可复用的事实放 `MEMORY.md`；当天上下文、观察、临时任务记录放 daily note；需要时再把 daily 里的高价值内容提炼进 `MEMORY.md`。这个设计和 OpenClaw 的 Markdown memory 模型一致：`MEMORY.md` 用于长期记忆，`memory/YYYY-MM-DD.md` 用于 daily notes，而且模型只会“记住”写到磁盘上的内容。([GitHub][2])

---

# 1. macOS 配置

## 1.1 创建目录

```bash
mkdir -p ~/.copilot/instructions
mkdir -p ~/.copilot/memory/daily
mkdir -p ~/.copilot/memory/projects
mkdir -p ~/.copilot/memory/people
mkdir -p ~/.copilot/memory/archive

mkdir -p ~/.config/atlassian
mkdir -p ~/.local/bin
```

## 1.2 安装 Jira / Confluence CLI

你给的工具仓库当前包含 `jira` 和 `confluence` 两个 CLI，目标是给 humans、shell scripts、LLM/agent workflows 提供稳定 JSON 输出；README 也要求把 `jira` 和 `confluence` 放到 PATH 后用 `version --json` 验证。([GitHub][3])

如果已经有 release artifact：

```bash
# 下载与你系统匹配的 darwin-arm64 或 darwin-amd64 压缩包后：
tar -xzf <archive>.tar.gz
cp jira confluence ~/.local/bin/
chmod +x ~/.local/bin/jira ~/.local/bin/confluence
```

如果暂时没有 release artifact，就从源码构建。仓库 README 提供了 `scripts/build.sh`，build 输出在 `dist/<goos>-<goarch>/` 下；我查看 release 页面时还没有发布包，所以这里给源码构建作为兜底。([GitHub][3])

```bash
mkdir -p ~/src
cd ~/src
git clone <JIRA_CONFLUENCE_TOOLS_REPO_URL> jira-confluence-tools
cd jira-confluence-tools

bash scripts/build.sh --snapshot

ARCH="$(go env GOARCH)"
cp "dist/darwin-${ARCH}/jira" ~/.local/bin/
cp "dist/darwin-${ARCH}/confluence" ~/.local/bin/
chmod +x ~/.local/bin/jira ~/.local/bin/confluence
```

把 `~/.local/bin` 加入 PATH：

```bash
grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.zshrc || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc

source ~/.zshrc
```

验证：

```bash
jira version --json
confluence version --json
```

---

## 1.3 配置凭证

工具支持 `ATLASSIAN_CONFIG` 环境变量覆盖配置路径，macOS/Linux 默认路径是 `~/.config/atlassian/config.json`；支持的认证模式包括 username/password、username/API key、bearer token/PAT。([GitHub][3])

创建：

```bash
cat > ~/.config/atlassian/config.json <<'JSON'
{
  "version": 1,
  "jira": {
    "default_instance": "main",
    "instances": [
      {
        "name": "main",
        "base_url": "https://jira.example.com",
        "rest_path": "/rest/api/2",
        "auth": {
          "type": "basic_api_key",
          "username": "your-email-or-username",
          "api_key": "YOUR_JIRA_API_KEY_OR_PAT"
        },
        "default_project": "PROJ",
        "verify_ssl": true
      }
    ]
  },
  "confluence": {
    "default_instance": "docs",
    "instances": [
      {
        "name": "docs",
        "base_url": "https://confluence.example.com",
        "rest_path": "/rest/api",
        "auth": {
          "type": "bearer_token",
          "token": "YOUR_CONFLUENCE_PAT"
        },
        "default_space": "ENG",
        "verify_ssl": true
      }
    ]
  }
}
JSON

chmod 700 ~/.config/atlassian
chmod 600 ~/.config/atlassian/config.json
```

测试：

```bash
jira auth test --instance main --json
confluence auth test --instance docs --json
```

凭证不要写进 `AGENTS.md`、`MEMORY.md`、daily note 或项目仓库。这个工具的安全文档也强调不要打印 `password`、`api_key`、`token`，配置文件在支持的平台使用 `0600` 权限，且 write/destructive 命令应配合 `--dry-run` / `--yes`。([GitHub][4])

---

## 1.4 创建全局 instruction 文件

```bash
cat > ~/.copilot/instructions/global-agent.instructions.md <<'MD'
---
applyTo: "**"
---

# Global Agent Instructions

Use the global agent guide at [AGENTS](../AGENTS.md) as the default operating guide.

When continuity, prior decisions, preferences, Jira, Confluence, or cross-session context matters, read [MEMORY](../MEMORY.md).

Use `../memory/daily/YYYY-MM-DD.md` for daily running notes. Determine today's date from the system if needed.

Never store credentials, API tokens, passwords, cookies, private keys, or session tokens in AGENTS.md, MEMORY.md, daily notes, prompts, logs, or repository files.

For Jira and Confluence work:
- Use the installed CLI commands `jira` and `confluence`.
- Always request machine-readable output with `--json`.
- First inspect available commands with `jira commands --json` or `confluence commands --json`.
- Before constructing a non-trivial command, inspect the command schema, for example `jira schema issue.create --json`.
- Use `--dry-run` before write operations.
- Use `--yes` only after the user has explicitly confirmed a destructive operation.
- Parse the JSON envelope and branch on `ok`, `data`, `error.code`, and `error.hint`.
MD
```

---

## 1.5 创建全局 AGENTS.md

````bash
cat > ~/.copilot/AGENTS.md <<'MD'
# Global AGENTS.md

## Scope

These rules apply across all local projects unless a workspace-specific instruction file is more specific.

## Tool usage

### Jira

Use the `jira` CLI for Jira tasks.

Default workflow:
1. Run `jira commands --json` when command availability is unclear.
2. Run `jira schema <command> --json` before constructing create, update, transition, assign, search, or other non-trivial commands.
3. Use `--json` on every command.
4. Prefer full issue URLs when the user gives a URL.
5. Add `--instance <name>` only when routing is ambiguous or the user explicitly asks for a specific instance.
6. For write operations, run with `--dry-run` first.
7. For destructive operations, require explicit user confirmation, then use `--yes`.

### Confluence

Use the `confluence` CLI for Confluence tasks.

Default workflow:
1. Run `confluence commands --json` when command availability is unclear.
2. Run `confluence schema <command> --json` before constructing create, update, move, upload, restore, delete, or other non-trivial commands.
3. Use `--json` on every command.
4. Prefer full page URLs when the user gives a URL.
5. Add `--instance <name>` only when routing is ambiguous or the user explicitly asks for a specific instance.
6. For write operations, run with `--dry-run` first.
7. For destructive operations, require explicit user confirmation, then use `--yes`.

## Credentials

Credentials are not memory.

Never store secrets in:
- AGENTS.md
- MEMORY.md
- daily notes
- prompts
- repository files
- generated documentation
- terminal transcripts intentionally saved for later use

Credentials should live only in the configured local credential/config mechanism.

If an authentication command fails, report the failure category without printing secrets.

## Memory

Long-term memory:
- `~/.copilot/MEMORY.md`

Daily notes:
- `~/.copilot/memory/daily/YYYY-MM-DD.md`

Rules:
1. At the start of a task that depends on prior context, read MEMORY.md.
2. Also inspect today's daily note, and yesterday's note if relevant.
3. Store temporary observations in the daily note.
4. Promote only durable, reusable facts to MEMORY.md.
5. Ask before storing personal preferences, sensitive project facts, or anything that could surprise the user.
6. Never store secrets or raw confidential data.
7. Keep memory concise and evidence-based.

Suggested daily note format:

```md
# YYYY-MM-DD

## Active context

## Decisions

## Observations

## Follow-ups

## Candidates to promote to MEMORY.md
````

Suggested long-term memory format:

```md
# MEMORY.md

## Stable preferences

## Durable decisions

## Tooling defaults

## Project-independent lessons

## Open loops
```

MD

````

---

## 1.6 创建 MEMORY.md 和当天 daily note

```bash
cat > ~/.copilot/MEMORY.md <<'MD'
# MEMORY.md

Long-term, curated memory only.

## Stable preferences

## Durable decisions

## Tooling defaults

- Jira and Confluence access is handled through local CLI tools.
- Credentials must never be written into memory files.

## Project-independent lessons

## Open loops
MD

TODAY="$(date +%F)"
cat > "$HOME/.copilot/memory/daily/${TODAY}.md" <<MD
# ${TODAY}

## Active context

## Decisions

## Observations

## Follow-ups

## Candidates to promote to MEMORY.md
MD
````

---

# 2. Windows 配置

下面用 PowerShell。

## 2.1 创建目录

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.copilot\instructions" | Out-Null
New-Item -ItemType Directory -Force "$env:USERPROFILE\.copilot\memory\daily" | Out-Null
New-Item -ItemType Directory -Force "$env:USERPROFILE\.copilot\memory\projects" | Out-Null
New-Item -ItemType Directory -Force "$env:USERPROFILE\.copilot\memory\people" | Out-Null
New-Item -ItemType Directory -Force "$env:USERPROFILE\.copilot\memory\archive" | Out-Null

New-Item -ItemType Directory -Force "$env:APPDATA\atlassian" | Out-Null
New-Item -ItemType Directory -Force "$env:USERPROFILE\.local\bin" | Out-Null
```

## 2.2 安装 Jira / Confluence CLI

如果已有 Windows release artifact：

```powershell
# 解压 windows-amd64 或 windows-arm64 包后：
Copy-Item .\jira.exe "$env:USERPROFILE\.local\bin\jira.exe" -Force
Copy-Item .\confluence.exe "$env:USERPROFILE\.local\bin\confluence.exe" -Force
```

如果没有 release artifact，就源码构建：

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\src" | Out-Null
Set-Location "$env:USERPROFILE\src"

git clone <JIRA_CONFLUENCE_TOOLS_REPO_URL> jira-confluence-tools
Set-Location "$env:USERPROFILE\src\jira-confluence-tools"

.\scripts\build.ps1 --snapshot

$arch = go env GOARCH
Copy-Item ".\dist\windows-$arch\jira.exe" "$env:USERPROFILE\.local\bin\jira.exe" -Force
Copy-Item ".\dist\windows-$arch\confluence.exe" "$env:USERPROFILE\.local\bin\confluence.exe" -Force
```

加入用户 PATH：

```powershell
$bin = "$env:USERPROFILE\.local\bin"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($userPath -notlike "*$bin*") {
  [Environment]::SetEnvironmentVariable("Path", "$bin;$userPath", "User")
}

Write-Host "Restart VS Code or the terminal after updating PATH."
```

重启 VS Code 后验证：

```powershell
jira version --json
confluence version --json
```

---

## 2.3 配置凭证

Windows 默认配置路径是 `%APPDATA%\atlassian\config.json`。([GitHub][3])

```powershell
$config = "$env:APPDATA\atlassian\config.json"

@'
{
  "version": 1,
  "jira": {
    "default_instance": "main",
    "instances": [
      {
        "name": "main",
        "base_url": "https://jira.example.com",
        "rest_path": "/rest/api/2",
        "auth": {
          "type": "basic_api_key",
          "username": "your-email-or-username",
          "api_key": "YOUR_JIRA_API_KEY_OR_PAT"
        },
        "default_project": "PROJ",
        "verify_ssl": true
      }
    ]
  },
  "confluence": {
    "default_instance": "docs",
    "instances": [
      {
        "name": "docs",
        "base_url": "https://confluence.example.com",
        "rest_path": "/rest/api",
        "auth": {
          "type": "bearer_token",
          "token": "YOUR_CONFLUENCE_PAT"
        },
        "default_space": "ENG",
        "verify_ssl": true
      }
    ]
  }
}
'@ | Set-Content -Path $config -Encoding UTF8
```

限制 ACL 到当前用户：

```powershell
$user = "$env:USERDOMAIN\$env:USERNAME"

icacls $config /inheritance:r
icacls $config /grant:r "${user}:(R,W)"
```

测试：

```powershell
jira auth test --instance main --json
confluence auth test --instance docs --json
```

---

## 2.4 创建全局 instruction 文件

```powershell
$instruction = "$env:USERPROFILE\.copilot\instructions\global-agent.instructions.md"

@'
---
applyTo: "**"
---

# Global Agent Instructions

Use the global agent guide at [AGENTS](../AGENTS.md) as the default operating guide.

When continuity, prior decisions, preferences, Jira, Confluence, or cross-session context matters, read [MEMORY](../MEMORY.md).

Use `../memory/daily/YYYY-MM-DD.md` for daily running notes. Determine today's date from the system if needed.

Never store credentials, API tokens, passwords, cookies, private keys, or session tokens in AGENTS.md, MEMORY.md, daily notes, prompts, logs, or repository files.

For Jira and Confluence work:
- Use the installed CLI commands `jira` and `confluence`.
- Always request machine-readable output with `--json`.
- First inspect available commands with `jira commands --json` or `confluence commands --json`.
- Before constructing a non-trivial command, inspect the command schema, for example `jira schema issue.create --json`.
- Use `--dry-run` before write operations.
- Use `--yes` only after the user has explicitly confirmed a destructive operation.
- Parse the JSON envelope and branch on `ok`, `data`, `error.code`, and `error.hint`.
'@ | Set-Content -Path $instruction -Encoding UTF8
```

---

## 2.5 创建全局 AGENTS.md

```powershell
$agents = "$env:USERPROFILE\.copilot\AGENTS.md"

@'
# Global AGENTS.md

## Scope

These rules apply across all local projects unless a workspace-specific instruction file is more specific.

## Tool usage

### Jira

Use the `jira` CLI for Jira tasks.

Default workflow:
1. Run `jira commands --json` when command availability is unclear.
2. Run `jira schema <command> --json` before constructing create, update, transition, assign, search, or other non-trivial commands.
3. Use `--json` on every command.
4. Prefer full issue URLs when the user gives a URL.
5. Add `--instance <name>` only when routing is ambiguous or the user explicitly asks for a specific instance.
6. For write operations, run with `--dry-run` first.
7. For destructive operations, require explicit user confirmation, then use `--yes`.

### Confluence

Use the `confluence` CLI for Confluence tasks.

Default workflow:
1. Run `confluence commands --json` when command availability is unclear.
2. Run `confluence schema <command> --json` before constructing create, update, move, upload, restore, delete, or other non-trivial commands.
3. Use `--json` on every command.
4. Prefer full page URLs when the user gives a URL.
5. Add `--instance <name>` only when routing is ambiguous or the user explicitly asks for a specific instance.
6. For write operations, run with `--dry-run` first.
7. For destructive operations, require explicit user confirmation, then use `--yes`.

## Credentials

Credentials are not memory.

Never store secrets in:
- AGENTS.md
- MEMORY.md
- daily notes
- prompts
- repository files
- generated documentation
- terminal transcripts intentionally saved for later use

Credentials should live only in the configured local credential/config mechanism.

If an authentication command fails, report the failure category without printing secrets.

## Memory

Long-term memory:
- `%USERPROFILE%\.copilot\MEMORY.md`

Daily notes:
- `%USERPROFILE%\.copilot\memory\daily\YYYY-MM-DD.md`

Rules:
1. At the start of a task that depends on prior context, read MEMORY.md.
2. Also inspect today's daily note, and yesterday's note if relevant.
3. Store temporary observations in the daily note.
4. Promote only durable, reusable facts to MEMORY.md.
5. Ask before storing personal preferences, sensitive project facts, or anything that could surprise the user.
6. Never store secrets or raw confidential data.
7. Keep memory concise and evidence-based.
'@ | Set-Content -Path $agents -Encoding UTF8
```

---

## 2.6 创建 MEMORY.md 和当天 daily note

```powershell
$memory = "$env:USERPROFILE\.copilot\MEMORY.md"

@'
# MEMORY.md

Long-term, curated memory only.

## Stable preferences

## Durable decisions

## Tooling defaults

- Jira and Confluence access is handled through local CLI tools.
- Credentials must never be written into memory files.

## Project-independent lessons

## Open loops
'@ | Set-Content -Path $memory -Encoding UTF8

$today = Get-Date -Format "yyyy-MM-dd"
$daily = "$env:USERPROFILE\.copilot\memory\daily\$today.md"

@"
# $today

## Active context

## Decisions

## Observations

## Follow-ups

## Candidates to promote to MEMORY.md
"@ | Set-Content -Path $daily -Encoding UTF8
```

---

# 3. VS Code Settings

打开 VS Code `settings.json`：

macOS：

```text
~/Library/Application Support/Code/User/settings.json
```

Windows：

```text
%APPDATA%\Code\User\settings.json
```

建议加入：

```json
{
  "chat.instructionsFilesLocations": {
    "~/.copilot/instructions": true,
    ".github/instructions": true,
    ".claude/rules": true
  },
  "chat.includeReferencedInstructions": true,
  "chat.useAgentsMdFile": true,
  "chat.useNestedAgentsMdFiles": true
}
```

说明：

* `~/.copilot/instructions` 是 user-level instructions 的推荐位置之一。([Visual Studio Code][1])
* `chat.includeReferencedInstructions` 用来确保 Markdown link 引用的 instructions 能进入上下文；如果发现没生效，可以在 Copilot Chat 的 Diagnostics 里检查加载了哪些文件。([Visual Studio Code][1])
* `chat.useAgentsMdFile` 开启 VS Code 对 workspace 内 `AGENTS.md` 的支持；`chat.useNestedAgentsMdFiles` 支持子目录 `AGENTS.md`。([Visual Studio Code][1])

---

# 4. Copilot Agent 使用规则

之后你可以这样对 Copilot 说：

```text
请先读取我的全局 AGENTS 和 MEMORY。然后检查 Jira 和 Confluence CLI 是否可用，不要执行任何写操作。
```

它应该优先执行：

```bash
jira version --json
confluence version --json
jira commands --json
confluence commands --json
```

做 Jira/Confluence 写操作时，应该先：

```bash
jira schema issue.create --json
confluence schema page.create --json
```

然后用 `--dry-run` 预演。工具文档也明确建议：agent 先用 `commands --json` 检查命令，再用 `schema <command> --json` 检查参数；始终用 `--json`，失败时看 `error.code` 和 `error.hint`，写操作先 `--dry-run`，破坏性操作才用 `--yes`。([GitHub][3])

---

# 5. 凭证“记住”的正确边界

不要让 Copilot “记住凭证”。更安全的边界是：

```text
Copilot 记住：有哪些工具、怎么调用、配置文件在哪里、不能泄露秘密。
CLI 工具记住：凭证本身，通过本地 config 或系统凭证存储。
Memory 记住：偏好、流程、长期决策，不记 token。
```

最小可落地方案是把 token 存在用户目录下的 `config.json`，并限制权限。更安全的方案是用系统 Keychain / Credential Manager / 1Password CLI 之类的方式保存 token，再由启动脚本生成或刷新本地 config；但无论哪种，`AGENTS.md` 和 `MEMORY.md` 都只写“凭证在哪里由工具读取”，不要写真实值。

---

# 6. 最终效果

完成后，你会得到：

```text
全局 Copilot 行为：
  ~/.copilot/instructions/global-agent.instructions.md

全局 agent 操作手册：
  ~/.copilot/AGENTS.md

全局长期记忆：
  ~/.copilot/MEMORY.md

每日上下文：
  ~/.copilot/memory/daily/YYYY-MM-DD.md

工具凭证：
  macOS:   ~/.config/atlassian/config.json
  Windows: %APPDATA%\atlassian\config.json

工具命令：
  jira --json ...
  confluence --json ...
```

这套方案的重点是：**全局 instructions 负责自动注入，AGENTS.md 负责操作规范，MEMORY.md/daily 负责跨会话记忆，Jira/Confluence 凭证由 CLI 配置持久化，不进入 AI 记忆。**

[1]: https://code.visualstudio.com/docs/copilot/customization/custom-instructions "Use custom instructions in VS Code"
[2]: https://github.com/openclaw/openclaw/blob/main/docs/concepts/memory.md "openclaw/docs/concepts/memory.md at main · openclaw/openclaw · GitHub"
[3]: https://github.com/dvnuo/engineering-flow-platform-tools/blob/master/README.md "engineering-flow-platform-tools/README.md at master · dvnuo/engineering-flow-platform-tools · GitHub"
[4]: https://github.com/dvnuo/engineering-flow-platform-tools/blob/master/docs/SECURITY.md "engineering-flow-platform-tools/docs/SECURITY.md at master · dvnuo/engineering-flow-platform-tools · GitHub"

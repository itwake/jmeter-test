对，现在应该收缩成 **“极简常驻指令 + 按需工具 skill + 一个长期记忆文件”**。

VS Code 的 always-on instructions / workspace `AGENTS.md` 会被自动加入 chat context；官方也建议 instructions 保持简短、聚焦。相比之下，VS Code Agent Skills 是 task-specific / on-demand 的，安装很多 skill 也不会都占用上下文，只有相关 skill 和被引用的资源会被加载。([Visual Studio Code][1])

所以我建议你把之前那套大目录 router 改成下面这个极简版本。

---

# 1. 新结构

## macOS

```text
~/.copilot/
  instructions/
    global-min.instructions.md

  skills/
    atlassian-tools/
      SKILL.md
    long-memory/
      SKILL.md

  MEMORY.md
```

## Windows

```text
%USERPROFILE%\.copilot\
  instructions\
    global-min.instructions.md

  skills\
    atlassian-tools\
      SKILL.md
    long-memory\
      SKILL.md

  MEMORY.md
```

不再需要默认加载：

```text
AGENTS.md
memory/README.md
memory/daily/
memory/projects/
memory/people/
memory/decisions/
memory/tools/
memory/prompts/
```

这些不是不能有，而是**不要作为全局常驻上下文**。你的目标是“工具使用 + long memory”，所以只保留：

```text
global-min.instructions.md  # 常驻，极短
atlassian-tools/SKILL.md    # Jira / Confluence 时才加载
long-memory/SKILL.md        # 需要记忆时才加载
MEMORY.md                   # 长期记忆本体，按需读取
```

---

# 2. 极简常驻 instructions

放到：

```text
~/.copilot/instructions/global-min.instructions.md
```

Windows：

```text
%USERPROFILE%\.copilot\instructions\global-min.instructions.md
```

内容建议控制在 30 行以内：

```md
---
applyTo: "**"
---

# Global Minimal Instructions

Keep context lean.

Do not load global memory or tool guides unless the current task needs them.

Use Jira/Confluence tools only when the task mentions Jira, Confluence, issue keys, tickets, pages, spaces, CQL, JQL, or Atlassian URLs.

For Jira/Confluence tasks, use the `atlassian-tools` skill if available. Use local CLI commands `jira` and `confluence` with `--json`.

Use long memory only when the task mentions prior context, preferences, decisions, "remember", "continue", "last time", "上次", "之前", "以后都这样", or when the user explicitly asks to save or recall memory.

For memory tasks, use the `long-memory` skill if available. The long memory file is `~/.copilot/MEMORY.md` on macOS/Linux and `%USERPROFILE%\.copilot\MEMORY.md` on Windows.

Never store credentials, tokens, passwords, cookies, private keys, PATs, session IDs, or authorization headers in memory or instructions.

Low-risk memory writes are allowed when the user explicitly says to remember something. Ask before storing sensitive, personal, surprising, inferred, or credential-related information.
```

重点：这里**不要写 Markdown link**，例如不要写：

```md
[MEMORY](../MEMORY.md)
```

因为 Markdown 引用可能让 VS Code 把被引用文件也纳入上下文。官方文档说明 instructions 可以通过 Markdown links 引用上下文文件，并且 `chat.includeReferencedInstructions` 会影响 referenced instructions 的加载。([Visual Studio Code][1])

用普通路径字符串即可：

```text
~/.copilot/MEMORY.md
```

这样常驻上下文只负责“路由”，不把内容提前塞进去。

---

# 3. Jira / Confluence 按需 skill

放到：

```text
~/.copilot/skills/atlassian-tools/SKILL.md
```

Windows：

```text
%USERPROFILE%\.copilot\skills\atlassian-tools\SKILL.md
```

内容：

````md
---
name: atlassian-tools
description: Use this skill only for Jira or Confluence tasks, including issue keys, tickets, JQL, Confluence pages, spaces, CQL, Atlassian URLs, or Jira/Confluence auth validation.
---

# Atlassian Tools Skill

Use local CLI tools:

- `jira`
- `confluence`

Always request machine-readable output:

```bash
jira ... --json
confluence ... --json
````

## Discovery

When command availability is unclear:

```bash
jira commands --json
confluence commands --json
```

Before non-trivial commands, inspect schema:

```bash
jira schema <command> --json
confluence schema <command> --json
```

Examples:

```bash
jira schema issue.create --json
confluence schema page.create --json
```

## Safety

For write operations, use `--dry-run` first.

For destructive operations, require explicit user confirmation and then use `--yes`.

On errors, inspect:

* `error.code`
* `error.hint`

Do not retry blindly.

## Config and credentials

Credentials are read by the CLI config, not by memory.

Config locations:

macOS/Linux:

```text
~/.config/atlassian/config.json
```

Windows:

```text
%APPDATA%\atlassian\config.json
```

Environment override:

```text
ATLASSIAN_CONFIG
```

Never print or store credential values.

## Auth validation

Jira:

```bash
jira auth test --instance <instance> --json
```

Confluence:

```bash
confluence auth test --instance <instance> --json
```

## Routing

If the user gives a full Jira or Confluence URL, prefer the URL.

Use `--instance <name>` only when routing is ambiguous or the user asks for a specific instance.

## Memory interaction

Do not write Jira/Confluence content into long memory unless the user explicitly asks to remember a durable preference, workflow, default instance, default project, default space, or recurring convention.

Never store raw issue/page payloads in memory.

````

你给的工具仓库本身也强调这几个点：`jira` / `confluence` 是面向 humans、shell scripts 和 LLM/agent workflows 的稳定 JSON CLI；配置路径支持 `ATLASSIAN_CONFIG`、macOS/Linux 默认 `~/.config/atlassian/config.json`、Windows 默认 `%APPDATA%\atlassian\config.json`；agent 应先用 `commands --json` 和 `schema <command> --json`，始终用 `--json`，写操作先 `--dry-run`，破坏性操作用 `--yes`。:contentReference[oaicite:2]{index=2}

---

# 4. Long memory 按需 skill

放到：

```text
~/.copilot/skills/long-memory/SKILL.md
````

Windows：

```text
%USERPROFILE%\.copilot\skills\long-memory\SKILL.md
```

内容：

````md
---
name: long-memory
description: Use this skill only when the task involves long-term memory, prior context, user preferences, durable decisions, reusable workflows, "remember", "continue", "last time", "上次", "之前", or "以后都这样".
---

# Long Memory Skill

Long memory is stored in one file:

macOS/Linux:

```text
~/.copilot/MEMORY.md
````

Windows:

```text
%USERPROFILE%\.copilot\MEMORY.md
```

## When to read memory

Read long memory only when the task involves:

* prior context
* remembered preferences
* durable decisions
* recurring workflows
* "continue"
* "last time"
* "remember"
* "上次"
* "之前"
* "以后都这样"
* explicit memory recall

Do not read memory for ordinary one-shot coding or Q&A tasks.

## When to write memory

Write memory when the user explicitly says:

* 记住这个
* 以后都这样
* save this to memory
* remember this
* make this the default
* 以后默认

Low-risk writes may be done without asking.

Ask first before writing:

* inferred preferences
* sensitive personal information
* people/team notes
* private business context
* surprising facts
* credential-related information
* anything copied from private systems in raw form

## Never write

Never write:

* API tokens
* PATs
* passwords
* cookies
* private keys
* session IDs
* authorization headers
* raw credential config files
* full Jira payloads
* full Confluence page payloads
* large terminal logs
* unconfirmed assumptions as facts

## How to write

Keep entries concise.

Use this format:

```md
- YYYY-MM-DD: <durable fact or preference>
```

Prefer updating an existing section over creating many new sections.

If memory conflicts with the current user instruction, follow the current instruction and note that memory may be stale.

````

---

# 5. MEMORY.md 极简模板

放到：

```text
~/.copilot/MEMORY.md
````

Windows：

```text
%USERPROFILE%\.copilot\MEMORY.md
```

内容：

```md
# MEMORY.md

Long-term curated memory only.

Do not store credentials, tokens, passwords, cookies, private keys, PATs, session IDs, authorization headers, raw Jira payloads, raw Confluence payloads, or large logs.

## User preferences

## Durable decisions

## Reusable workflows

## Tooling defaults

### Jira / Confluence

- Use local CLI tools `jira` and `confluence`.
- Always use `--json`.
- Credentials are handled by local CLI config, not by memory.

## Open loops
```

这个文件可以逐渐增长，但不要默认加载。只有 long-memory skill 触发时才读。

---

# 6. VS Code settings 建议

`settings.json` 可以这样收敛：

```json
{
  "chat.instructionsFilesLocations": {
    "~/.copilot/instructions": true
  },
  "chat.useAgentsMdFile": false,
  "chat.useNestedAgentsMdFiles": false
}
```

如果你还想使用 workspace 里的项目级 `AGENTS.md`，就不要关：

```json
{
  "chat.useAgentsMdFile": true
}
```

但从“减少 token”的角度看，全局方案里不建议依赖 `AGENTS.md`，因为 VS Code 会自动检测 workspace root 的 `AGENTS.md` 并应用到 chat request。([Visual Studio Code][1])

Skills 默认可以放在 user profile 的 `~/.copilot/skills/`，VS Code 文档列出的 personal skills 位置包括 `~/.copilot/skills/`、`~/.claude/skills/`、`~/.agents/skills/`。([Visual Studio Code][2])

---

# 7. 最终触发逻辑

现在上下文加载会变成这样：

## 普通代码任务

只加载：

```text
global-min.instructions.md
```

不读：

```text
MEMORY.md
atlassian-tools/SKILL.md
long-memory/SKILL.md
```

## Jira / Confluence 任务

加载：

```text
global-min.instructions.md
atlassian-tools/SKILL.md
```

必要时使用：

```bash
jira commands --json
jira schema <command> --json
confluence commands --json
confluence schema <command> --json
```

不自动读 `MEMORY.md`，除非任务还涉及“默认项目、上次约定、长期偏好”。

## 记忆任务

加载：

```text
global-min.instructions.md
long-memory/SKILL.md
MEMORY.md
```

只在这些场景读写：

```text
记住这个
以后都这样
上次怎么做的
继续之前的
我的默认偏好是什么
```

---

# 8. 我建议你删除或降级的内容

之前这些内容不适合作为常驻 system prompt：

```text
完整 directory router
daily note 规则
projects/people/decisions/tools/prompts/archive 目录说明
长篇 Jira/Confluence 操作说明
长篇 credential policy
```

把它们改成：

```text
常驻：只保留路由规则
按需：Jira/Confluence 进入 atlassian-tools skill
按需：记忆进入 long-memory skill
长期：只维护 MEMORY.md
```

最终原则就是：

```text
常驻 prompt 只回答两个问题：
1. 什么时候用 Jira/Confluence 工具？
2. 什么时候读写 long memory？

其他细节全部按需加载。
```

[1]: https://code.visualstudio.com/docs/copilot/customization/custom-instructions "Use custom instructions in VS Code"
[2]: https://code.visualstudio.com/docs/copilot/customization/agent-skills "Use Agent Skills in VS Code"

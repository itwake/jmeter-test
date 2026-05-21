对，应该把它改成 **“目录 + 读写路由规则”**，而不是只放目录。
核心原则是：

```text
AGENTS.md 负责告诉 agent：每个目录什么时候读、什么时候写、禁止写什么。
MEMORY.md 负责长期稳定事实。
memory/daily/ 负责当天流水和临时上下文。
memory/projects/、people/、tools/ 等目录负责按主题沉淀。
凭证永远不进入 memory，只让工具从本地 config / credential store 读取。
```

VS Code 的 user-level instructions 可以跨 workspace 生效，`*.instructions.md` 可以放在 `~/.copilot/instructions`；VS Code 也支持 `AGENTS.md`、Markdown 引用和 instruction diagnostics，所以这里的全局方案本质是：**用全局 instructions 自动引入全局 AGENTS，再让 AGENTS 定义 memory router**。([Visual Studio Code][1])

下面是我建议你直接落地的版本。

---

# 1. 推荐目录结构

## macOS

```text
~/.copilot/
  AGENTS.md
  MEMORY.md

  memory/
    README.md
    daily/
      2026-05-21.md
    projects/
    people/
    decisions/
    tools/
      jira.md
      confluence.md
      credentials.md
    prompts/
    archive/

  instructions/
    global-agent.instructions.md
```

## Windows

```text
%USERPROFILE%\.copilot\
  AGENTS.md
  MEMORY.md

  memory\
    README.md
    daily\
      2026-05-21.md
    projects\
    people\
    decisions\
    tools\
      jira.md
      confluence.md
      credentials.md
    prompts\
    archive\

  instructions\
    global-agent.instructions.md
```

我建议新增一个 `memory/README.md`，专门写目录使用规则。
`AGENTS.md` 里只放路由规则摘要，`memory/README.md` 里放更详细的目录契约。

---

# 2. 每个目录的读写职责

| 路径                           | 作用            | 什么时候读                                 | 什么时候写                      |
| ---------------------------- | ------------- | ------------------------------------- | -------------------------- |
| `AGENTS.md`                  | 全局行为规则        | 每次任务开始前都应视为最高优先级的本地规则                 | 很少写；只有你要改 agent 行为时才更新     |
| `MEMORY.md`                  | 长期稳定记忆        | 涉及偏好、长期决策、跨项目上下文、历史选择时读取              | 只写高信噪、长期有效、可复用的信息          |
| `memory/daily/YYYY-MM-DD.md` | 当天运行日志        | 每次复杂任务开始读今天；必要时读昨天                    | 任务进展、临时观察、未确认信息、候选记忆先写这里   |
| `memory/projects/`           | 项目级长期上下文      | 涉及某个项目、仓库、系统、产品时读取对应文件                | 项目目标、架构事实、惯例、长期 TODO       |
| `memory/people/`             | 人和团队上下文       | 涉及某个人、团队、汇报关系、协作习惯时读取                 | 只写非敏感、工作相关、用户允许记录的信息       |
| `memory/decisions/`          | 已确认决策         | 涉及“为什么之前这样做”时读取                       | 只写已经明确确认的决策、取舍、日期、影响范围     |
| `memory/tools/`              | 工具使用手册        | 涉及 Jira、Confluence、GitHub、浏览器、CLI 时读取 | 工具命令习惯、实例名称、非敏感配置路径、常用查询模板 |
| `memory/prompts/`            | 可复用 prompt 模板 | 需要重复执行某类任务时读取                         | 可复用任务模板、检查清单、输出格式          |
| `memory/archive/`            | 过期或废弃内容       | 默认不读；只有追溯历史时读                         | 把不再活跃但可能有历史价值的内容移进去        |

这张表应该明确写进 `memory/README.md`，否则 agent 不会天然知道这些目录的语义。

---

# 3. 替换版 `global-agent.instructions.md`

这个文件负责“把全局 AGENTS 和 memory router 拉进上下文”。

## macOS / Windows 通用内容

放到：

```text
~/.copilot/instructions/global-agent.instructions.md
```

Windows 等价路径：

```text
%USERPROFILE%\.copilot\instructions\global-agent.instructions.md
```

内容：

```md
---
applyTo: "**"
---

# Global Agent Bootstrap

Use the global operating guide at [AGENTS](../AGENTS.md).

When the task involves prior context, user preferences, Jira, Confluence, repeated workflows, decisions, people, projects, or cross-session continuity, also read:

- [MEMORY](../MEMORY.md)
- [Memory Directory Guide](../memory/README.md)

Use the memory router in `memory/README.md` to decide which memory files to read or write.

Do not store credentials, API tokens, passwords, cookies, private keys, PATs, session IDs, or raw secrets in AGENTS.md, MEMORY.md, daily notes, prompts, logs, generated docs, or repository files.

For Jira and Confluence work, use the local CLI tools `jira` and `confluence`, always request `--json`, inspect command metadata before non-trivial calls, and use dry-run before write operations.
```

VS Code 文档里提到，`*.instructions.md` 可以放在 user profile 级别并跨 workspace 使用，也可以通过 Markdown links 引用上下文文件；如果没有生效，可以用 Chat diagnostics 检查加载了哪些 instructions。([Visual Studio Code][1])

---

# 4. 替换版 `memory/README.md`

这个文件最关键。它告诉 agent **什么时候读哪个目录、什么时候写哪个目录**。

````md
# Memory Directory Guide

This directory is the global Markdown memory system.

The agent must treat memory as explicit files on disk. If something is not written to disk, it should not be assumed to persist across sessions.

## Root files

### `../AGENTS.md`

Purpose:
- Global behavior rules.
- Tool usage policy.
- Security and credential rules.
- Memory routing summary.

Read:
- At the beginning of every meaningful task.
- Whenever instructions conflict.

Write:
- Only when the user explicitly asks to change global agent behavior.
- Do not write task notes here.
- Do not write project facts here unless they are global rules.

### `../MEMORY.md`

Purpose:
- Long-term, high-signal, curated memory.
- Stable preferences.
- Durable decisions.
- Long-lived tool defaults.
- Cross-project principles.

Read:
- When the task depends on previous user preferences.
- When the task involves recurring workflows.
- When the task asks "what did we decide before?"
- When the task touches Jira, Confluence, credential configuration, or memory behavior.
- When the user asks the agent to continue from prior context.

Write:
- Only promote durable facts from daily notes or explicit user instructions.
- Write concise entries with dates when possible.
- Prefer one fact per bullet.
- Include enough context to avoid ambiguity.

Do not write:
- Secrets.
- Raw credentials.
- Temporary debugging output.
- Unconfirmed assumptions.
- Large transcripts.
- One-off task details.

## `daily/`

Path:
- `memory/daily/YYYY-MM-DD.md`

Purpose:
- Short-term running notes.
- Today's task context.
- Observations.
- Temporary assumptions.
- Candidate memory items.
- Things to review later.

Read:
- At the start of any task that may continue previous work from today.
- Read today's note first.
- Read yesterday's note when the user says "continue", "from yesterday", "last time", "刚才", "昨天", or when today's note lacks context.

Write:
- Append during or after meaningful tasks.
- Write raw but useful notes.
- Record what changed, what was tried, what failed, and what should be followed up.
- Put long-term candidates under "Candidates to promote to MEMORY.md".

Do not write:
- Secrets.
- Full terminal logs unless summarized.
- Private tokens or credentials.
- Large copied content from Jira or Confluence unless summarized.

Daily note template:

```md
# YYYY-MM-DD

## Active context

## Actions taken

## Observations

## Decisions today

## Follow-ups

## Candidates to promote to MEMORY.md
````

## `projects/`

Purpose:

* Project-specific but durable context.
* Repository conventions.
* Product goals.
* Architecture notes.
* Important links by name, not secrets.
* Known constraints.

File naming:

* Use lowercase kebab-case.
* Example: `projects/mobile-app.md`, `projects/internal-portal.md`.

Read:

* When the user mentions a project, repo, service, product, codebase, workspace, or initiative.
* When Jira or Confluence issues clearly belong to a known project.
* When generating project-specific docs, plans, tickets, PRs, or reviews.

Write:

* When a stable project fact is learned.
* When a recurring convention is discovered.
* When the user confirms a project decision.
* When daily notes contain repeated project context worth preserving.

Do not write:

* Temporary implementation attempts.
* Credentials.
* Unconfirmed guesses.
* Sensitive raw data copied from private systems.

Recommended structure:

```md
# Project: <name>

## Summary

## Repositories / workspaces

## Conventions

## Architecture notes

## Jira / Confluence references

## Durable decisions

## Open loops
```

## `people/`

Purpose:

* Work-relevant collaboration context.
* Non-sensitive preferences.
* Roles, teams, and responsibilities when useful for work.

File naming:

* Use lowercase kebab-case.
* Example: `people/alex-chen.md`.
* If identity is ambiguous, use team or role instead of a person file.

Read:

* When the user asks about a person's responsibilities, prior discussions, preferences, or collaboration context.
* When drafting communication to a known person or team.

Write:

* Only when the information is work-relevant and safe.
* Prefer facts the user explicitly gave or approved.
* Use dates for context.

Do not write:

* Sensitive personal information.
* Health, family, finance, politics, private life, credentials, or personal identifiers not required for work.
* Speculation about people.

Recommended structure:

```md
# Person: <name>

## Role / team

## Work preferences

## Relevant context

## Interaction notes

## Last updated
```

## `decisions/`

Purpose:

* Durable decisions and rationale.
* Useful for answering "why did we choose this?"

File naming:

* `YYYY-MM-DD-short-title.md`
* Example: `2026-05-21-use-cli-for-atlassian.md`.

Read:

* When the user asks about previous decisions.
* When changing architecture, process, tooling, or conventions.
* When a task may conflict with earlier choices.

Write:

* Only after a decision is explicit or strongly confirmed.
* Include context, decision, alternatives, rationale, consequences, and date.

Do not write:

* Tentative ideas.
* Brainstorming.
* Unapproved recommendations.

Decision template:

```md
# Decision: <title>

Date: YYYY-MM-DD
Status: proposed | accepted | superseded | archived

## Context

## Decision

## Alternatives considered

## Rationale

## Consequences

## Related files
```

## `tools/`

Purpose:

* Operational playbooks for tools.
* Non-secret configuration references.
* Command patterns.
* Error handling notes.
* Jira / Confluence usage rules.

Read:

* Whenever using a tool from this directory.
* Before Jira or Confluence commands.
* Before debugging authentication, routing, or command errors.

Write:

* When a reusable command pattern is learned.
* When a non-secret instance name, default project, default space, or config path is confirmed.
* When an error pattern and fix are discovered.
* When the user updates tool preferences.

Do not write:

* Tokens.
* Passwords.
* Cookies.
* API keys.
* PAT values.
* Authorization headers.
* Full config files containing secrets.

Files:

* `tools/jira.md`
* `tools/confluence.md`
* `tools/credentials.md`

## `tools/credentials.md`

Purpose:

* Document where credentials are stored.
* Document how to validate auth.
* Document which commands to run.
* Never store actual credentials.

Read:

* Before auth testing.
* Before helping the user configure credentials.
* When auth fails.

Write:

* Non-secret config path.
* Non-secret instance names.
* Auth validation commands.
* Credential rotation procedure without secret values.

Allowed examples:

* "Jira credentials are read from `$ATLASSIAN_CONFIG` if set."
* "Default macOS/Linux path: `~/.config/atlassian/config.json`."
* "Default Windows path: `%APPDATA%\atlassian\config.json`."
* "Run `jira auth test --instance <name> --json`."

Forbidden examples:

* API tokens.
* PATs.
* Passwords.
* Cookies.
* Base64 auth strings.
* Authorization headers.

## `prompts/`

Purpose:

* Reusable prompt templates.
* Reusable output formats.
* Repeatable task checklists.

Read:

* When the user asks for repeated workflows such as ticket creation, review, planning, release notes, incident summary, meeting notes, or Confluence page drafting.

Write:

* When the user asks to save a reusable prompt.
* When a workflow becomes repeated enough to template.
* When a prompt template has been tested and improved.

Do not write:

* One-off chat history.
* Secrets.
* Large source documents.

## `archive/`

Purpose:

* Old memory that should not be loaded by default.
* Superseded project notes.
* Old decisions.
* Deprecated tool instructions.

Read:

* Only when tracing history.
* Only when current files do not answer the question.
* Only when the user asks for old context.

Write:

* Move outdated files here instead of deleting them when historical context may matter.
* Mark why the item was archived.

Do not write:

* Active memory.
* Current instructions.
* Current credentials guidance.

````

OpenClaw 的 memory 思路也是“写入 Markdown 文件才算持久记忆”，并区分 `MEMORY.md` 长期记忆和 `memory/YYYY-MM-DD.md` daily notes；这里我们只是把它扩展成更明确的全局目录路由。:contentReference[oaicite:2]{index=2}

---

# 5. 替换版 `AGENTS.md` 的 memory router

你可以把下面这一段放进 `~/.copilot/AGENTS.md`，替换之前比较粗的 memory 部分。

```md
# Global AGENTS.md

## Global memory and directory router

The global memory home is:

- macOS/Linux: `~/.copilot`
- Windows: `%USERPROFILE%\.copilot`

Important files:

- `AGENTS.md`: global rules
- `MEMORY.md`: long-term curated memory
- `memory/README.md`: memory directory guide
- `memory/daily/YYYY-MM-DD.md`: daily notes
- `memory/projects/`: project memory
- `memory/people/`: people/team memory
- `memory/decisions/`: durable decisions
- `memory/tools/`: tool playbooks
- `memory/prompts/`: reusable prompt templates
- `memory/archive/`: inactive historical memory

### Start-of-task memory protocol

For trivial one-shot questions, memory lookup is optional.

For any non-trivial task, first decide whether memory is relevant.

Memory is relevant if the task mentions or implies:

- prior context
- user preferences
- "continue", "last time", "yesterday", "之前", "上次", "刚才"
- Jira
- Confluence
- credentials
- a known project
- a known person or team
- previous decisions
- repeated workflows
- writing tickets, docs, plans, PRs, reviews, or summaries

If memory is relevant, read in this order:

1. `memory/README.md`
2. `MEMORY.md`
3. today's daily note: `memory/daily/YYYY-MM-DD.md`
4. yesterday's daily note if continuity is likely
5. topic-specific files:
   - Jira / Confluence / credentials: `memory/tools/*.md`
   - project-specific work: `memory/projects/*.md`
   - people/team context: `memory/people/*.md`
   - decision history: `memory/decisions/*.md`
   - reusable workflows: `memory/prompts/*.md`

Do not read `memory/archive/` unless explicitly needed.

### Write routing protocol

When new information appears, classify it before writing:

1. Temporary observation or task progress:
   - write to today's daily note.

2. Durable user preference:
   - write to `MEMORY.md` after user confirmation or clear instruction.

3. Durable project fact:
   - write to `memory/projects/<project>.md`.

4. Durable decision:
   - write to `memory/decisions/YYYY-MM-DD-short-title.md`.

5. Tool usage pattern:
   - write to `memory/tools/<tool>.md`.

6. Reusable prompt or workflow:
   - write to `memory/prompts/<workflow>.md`.

7. Person/team collaboration context:
   - write to `memory/people/<name-or-team>.md`, only if work-relevant and non-sensitive.

8. Deprecated but historically useful content:
   - move or summarize into `memory/archive/`.

### Daily note rules

Use daily notes as the first write target.

Append concise entries during meaningful tasks:

```md
## <time or task title>

### Context

### Actions

### Observations

### Follow-ups

### Candidates to promote
````

Daily notes may contain unconfirmed observations. Long-term memory must not.

### Promotion rules

Promote from daily notes to long-term memory only when the information is:

* stable
* reusable
* non-secret
* not merely a one-off detail
* not speculative
* useful across future sessions

Before promoting sensitive, personal, or surprising information, ask the user.

### Memory hygiene

Never store:

* credentials
* API tokens
* PATs
* passwords
* cookies
* private keys
* authorization headers
* raw confidential documents
* full Jira/Confluence payloads unless summarized safely
* large terminal logs
* unconfirmed assumptions as facts

Use dates and sources when possible.

Prefer concise bullets over long prose.

When memory conflicts with the current user instruction, follow the current user instruction and optionally note that memory may be stale.

## Jira and Confluence tool protocol

Use local CLI tools for Jira and Confluence.

Before using Jira:

1. Read `memory/tools/jira.md` if it exists.
2. Run `jira commands --json` if command availability is unclear.
3. Run `jira schema <command> --json` before non-trivial commands.
4. Always use `--json`.
5. Use `--dry-run` before write operations.
6. Use `--yes` only after explicit confirmation for destructive operations.

Before using Confluence:

1. Read `memory/tools/confluence.md` if it exists.
2. Run `confluence commands --json` if command availability is unclear.
3. Run `confluence schema <command> --json` before non-trivial commands.
4. Always use `--json`.
5. Use `--dry-run` before write operations.
6. Use `--yes` only after explicit confirmation for destructive operations.

Credentials are not memory. Read credential location guidance from `memory/tools/credentials.md`, but never write secret values there.

````

你给的 Jira/Confluence CLI 仓库本身也建议 agent 先检查 `commands --json`，再查 `schema <command> --json`，始终使用 `--json`，写操作先 `--dry-run`，破坏性操作才用 `--yes`；所以这些规则适合直接写进全局 `AGENTS.md`。:contentReference[oaicite:3]{index=3}

---

# 6. `MEMORY.md` 建议模板

`MEMORY.md` 不要变成垃圾桶。它应该只存长期、稳定、高价值的信息。

```md
# MEMORY.md

This is curated long-term memory.

Do not store credentials or raw secrets here.

## Stable user preferences

## Durable cross-project decisions

## Tooling defaults

### Jira / Confluence

- Credentials are handled by local tool configuration, not by memory.
- Use `jira` and `confluence` CLI commands with `--json`.
- Read `memory/tools/jira.md`, `memory/tools/confluence.md`, and `memory/tools/credentials.md` before related tasks.

## Reusable working style

## Long-term open loops

## Recently promoted from daily notes
````

---

# 7. `memory/tools/jira.md`

````md
# Jira Tool Guide

## Purpose

Use this file for Jira operating patterns, not secrets.

## Read this file when

- The user asks about Jira.
- The task involves creating, updating, searching, assigning, commenting, or transitioning issues.
- A Jira URL or issue key appears.
- Jira auth, instance routing, project defaults, or JQL is relevant.

## Write to this file when

- A reusable Jira command pattern is confirmed.
- A default Jira instance or project key is confirmed.
- A useful JQL template is discovered.
- A recurring error and fix is confirmed.

## Do not write

- API tokens.
- Passwords.
- PATs.
- Cookies.
- Authorization headers.
- Raw sensitive issue content.

## Standard protocol

1. Use `jira commands --json` when command availability is unclear.
2. Use `jira schema <command> --json` before non-trivial commands.
3. Always use `--json`.
4. Use full Jira issue URLs when the user provides URLs.
5. Use `--instance <name>` when routing is ambiguous.
6. Use `--dry-run` before write operations.
7. Use `--yes` only after explicit confirmation for destructive operations.
8. On errors, inspect `error.code` and `error.hint`.

## Known instances

Add non-secret instance names here.

```text
main: <description, no token>
````

## Common JQL templates

```text
project = <PROJECT> ORDER BY updated DESC
assignee = currentUser() AND statusCategory != Done ORDER BY updated DESC
text ~ "<keyword>" ORDER BY updated DESC
```

## Error notes

Add reusable non-secret fixes here.

````

---

# 8. `memory/tools/confluence.md`

```md
# Confluence Tool Guide

## Purpose

Use this file for Confluence operating patterns, not secrets.

## Read this file when

- The user asks about Confluence.
- The task involves searching, reading, creating, updating, moving, archiving, or deleting pages.
- A Confluence URL appears.
- Space, page hierarchy, permissions, or CQL is relevant.

## Write to this file when

- A reusable Confluence command pattern is confirmed.
- A default Confluence instance or space is confirmed.
- A useful CQL template is discovered.
- A recurring error and fix is confirmed.

## Do not write

- API tokens.
- Passwords.
- PATs.
- Cookies.
- Authorization headers.
- Raw sensitive page content.

## Standard protocol

1. Use `confluence commands --json` when command availability is unclear.
2. Use `confluence schema <command> --json` before non-trivial commands.
3. Always use `--json`.
4. Use full page URLs when the user provides URLs.
5. Use `--instance <name>` when routing is ambiguous.
6. Use `--dry-run` before write operations.
7. Use `--yes` only after explicit confirmation for destructive operations.
8. On errors, inspect `error.code` and `error.hint`.

## Known spaces

Add non-secret space names here.

```text
ENG: <description>
````

## Common CQL templates

```text
space = <SPACE> AND text ~ "<keyword>" ORDER BY lastmodified DESC
space = <SPACE> AND title ~ "<keyword>" ORDER BY lastmodified DESC
type = page AND lastmodified > now("-30d") ORDER BY lastmodified DESC
```

## Page drafting conventions

* Draft locally first when content is long.
* Use dry-run before creating or updating pages.
* Summarize large pages instead of copying raw content into memory.

## Error notes

Add reusable non-secret fixes here.

````

---

# 9. `memory/tools/credentials.md`

这个文件专门解决你说的“下次方便使用凭证”，但注意：**它只记录凭证读取方式，不记录凭证本身**。

```md
# Credential Handling Guide

## Purpose

This file documents where tools read credentials from and how to validate auth.

It must never contain actual secrets.

## Read this file when

- Jira or Confluence auth is needed.
- The user asks to configure, validate, rotate, or troubleshoot credentials.
- A CLI command returns auth-related errors.
- The agent needs to know which config path to use.

## Write to this file when

- A non-secret config path is confirmed.
- A non-secret instance name is confirmed.
- An auth validation command is confirmed.
- A credential rotation procedure is documented without secret values.

## Never write

- API tokens.
- PATs.
- Passwords.
- Cookies.
- Private keys.
- Authorization headers.
- Base64 credentials.
- Raw config files containing secrets.

## Config locations

Default macOS/Linux path:

```text
~/.config/atlassian/config.json
````

Default Windows path:

```text
%APPDATA%\atlassian\config.json
```

Environment override:

```text
ATLASSIAN_CONFIG
```

## Auth validation

Jira:

```bash
jira auth test --instance <instance> --json
```

Confluence:

```bash
confluence auth test --instance <instance> --json
```

## Safe troubleshooting

If auth fails:

1. Do not print secrets.
2. Report only the failure category.
3. Ask the user to rotate or refresh credentials if needed.
4. Re-run auth validation.

````

这和工具 README 里的配置路径是一致的：它支持 `ATLASSIAN_CONFIG` 覆盖，macOS/Linux 默认 `~/.config/atlassian/config.json`，Windows 默认 `%APPDATA%\atlassian\config.json`。:contentReference[oaicite:4]{index=4}

---

# 10. `memory/daily/YYYY-MM-DD.md` 模板

daily 是最重要的“先写入口”。不要一开始就把所有东西写进 `MEMORY.md`。

```md
# YYYY-MM-DD

## Active context

## Actions taken

## Observations

## Decisions today

## Follow-ups

## Candidates to promote to MEMORY.md

## Candidates to promote to projects/

## Candidates to promote to tools/

## Candidates to promote to decisions/
````

实际使用时可以这样写：

```md
## Jira ticket drafting workflow

### Context

User wants reusable Jira/Confluence support through global VS Code Copilot instructions.

### Actions taken

- Refined memory directory router.
- Added explicit read/write rules for each directory.
- Added tool guides for Jira, Confluence, and credentials.

### Observations

- Directory structure alone is insufficient.
- Agent needs explicit read/write triggers.

### Candidates to promote to MEMORY.md

- User prefers general reusable solutions, not project-specific naming.
- Credentials should be remembered through local config, not AI memory.
```

---

# 11. agent 的实际工作流应该长这样

## 场景 A：用户说“帮我查一下这个 Jira 单”

agent 应该：

```text
1. 读 AGENTS.md
2. 读 memory/README.md
3. 读 MEMORY.md
4. 读 memory/tools/jira.md
5. 读 memory/tools/credentials.md
6. 调用 jira commands --json 或直接使用已知命令
7. 调用 jira issue get ... --json
8. 把有价值但临时的信息写入 today's daily note
9. 如果发现可复用规则，再写入 tools/jira.md
```

## 场景 B：用户说“按上次的格式写一个 Confluence 页面”

agent 应该：

```text
1. 读 MEMORY.md
2. 读今天 daily
3. 读昨天 daily
4. 读 memory/tools/confluence.md
5. 读 memory/prompts/ 里相关模板
6. 草拟页面
7. 如果要发布，先 dry-run
8. 写入 daily：本次使用了哪个格式、页面草稿状态
```

## 场景 C：用户说“以后都记住我喜欢这种格式”

agent 应该：

```text
1. 判断这是长期偏好
2. 写入 MEMORY.md
3. 在今天 daily 记录：已将该偏好提升为长期记忆
```

## 场景 D：agent 发现了一个临时错误

比如 Jira 某个命令失败了。

agent 应该：

```text
1. 不写 MEMORY.md
2. 先写 today's daily note
3. 如果这个错误重复出现并且解决方案稳定，再写入 memory/tools/jira.md
```

---

# 12. 最小落地版：你真正需要复制的三份文件

你现在不需要一次建很多内容。最小可用是：

```text
~/.copilot/instructions/global-agent.instructions.md
~/.copilot/AGENTS.md
~/.copilot/memory/README.md
```

然后加：

```text
~/.copilot/MEMORY.md
~/.copilot/memory/daily/YYYY-MM-DD.md
~/.copilot/memory/tools/jira.md
~/.copilot/memory/tools/confluence.md
~/.copilot/memory/tools/credentials.md
```

这套规则补上的就是你指出的缺口：

```text
不是只告诉 agent “这里有目录”，
而是告诉它：
  什么任务触发读取；
  读取顺序是什么；
  新信息先写哪里；
  什么信息可以提升为长期记忆；
  什么信息必须禁止写入；
  Jira / Confluence / credentials 分别怎么处理。
```

最终效果应该是：**agent 遇到任务先判断是否需要 memory；需要时按 router 读取；任务中先写 daily；只有稳定、可复用、非敏感的信息才提升到 MEMORY、projects、tools 或 decisions。**

[1]: https://code.visualstudio.com/docs/copilot/customization/custom-instructions "Use custom instructions in VS Code"

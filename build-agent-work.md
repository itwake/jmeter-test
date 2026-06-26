
你刚刚已经通过 mobile-auto + BrowserStack 对当前 App 完整探索了一轮，并完成了登录流程。

现在不要继续盲目探索 App，也不要重新从零测试登录。你的任务是把刚才已经走过的路径、成功步骤、失败步骤、定位方式、mobile-auto 使用规则、恢复策略，沉淀成 VS Code Copilot Agent 后续能长期使用的 workspace Markdown 知识库。

目标：
让后续任何 VS Code Copilot Agent 进入这个 workspace 后，都能先读这些 Markdown 文件，理解这个 App 的登录路径、页面结构、mobile-auto 调用方式、常见失败恢复方式，并且在未来测试不顺畅时能自动更新这些文档，而不是重新消耗大量 token 探索。

请严格按下面要求执行。

============================================================
一、总体原则
============================================================

1. 只基于已经发生过的探索路径、当前会话中可见的终端输出、mobile-auto JSON 输出、workspace 里的日志、artifacts、测试脚本、README 和配置文件建立知识库。

2. 如果某些信息你记得但没有证据，请可以写入，但必须标记为：
   - confidence: low
   - source: agent memory / inferred
   - needs verification: true

3. 不要把 mobile-auto observe 返回的临时 ref 当成长期 locator。
   例如 obs-xxx:e17 这种 ref 只能用于当前 observation，不能写成长期操作规则。

4. 长期文档里只记录稳定定位策略：
   - screen identity
   - visible text
   - role
   - accessibility name
   - label
   - semantic meaning
   - fallback locator candidates
   - known timing/keyboard/network behavior

5. 不要把密码、token、BrowserStack key、测试账号密码、cookie、session id、access key 写进任何 Markdown。
   所有 secret 必须写成：
   - TEST_USERNAME
   - TEST_PASSWORD
   - BROWSERSTACK_USERNAME
   - BROWSERSTACK_ACCESS_KEY
   - <redacted>

6. 不要把完整 observe JSON 大段粘贴进文档。
   只摘录有用字段，例如 screen name、visible text、element candidates、失败 code、artifact path。

7. 不要把文档写成泛泛而谈。
   必须尽可能具体到这个 App：
   - 启动后第一屏是什么
   - 登录入口在哪里
   - 用户名输入框如何定位
   - 密码输入框如何定位
   - 登录按钮如何定位
   - 登录后如何判断成功
   - 哪些弹窗/权限/加载页/键盘遮挡/网络等待会出现
   - 出错后应该怎么恢复

8. 文档结构要适合 VS Code Copilot Agent 使用：
   - `.github/copilot-instructions.md` 只放简短常驻规则和索引，不要塞入巨量细节。
   - `.github/instructions/*.instructions.md` 放 mobile-auto 专项规则。
   - `.github/prompts/*.prompt.md` 放可复用任务 prompt。
   - `docs/mobile-auto/**` 放 App 具体知识库。

9. 生成完文档后，必须自检：
   - 文件存在
   - 链接路径正确
   - 没有 secret
   - 没有把 obs-xxx:e17 这类临时 ref 写成长期规则
   - 后续 agent 能从 `.github/copilot-instructions.md` 找到入口

============================================================
二、请创建或更新以下文件结构
============================================================

请创建或更新：

.github/
  copilot-instructions.md
  instructions/
    mobile-auto.instructions.md
  prompts/
    mobile-login-smoke.prompt.md
    mobile-auto-repair-runbook.prompt.md
    mobile-auto-learn-from-run.prompt.md

docs/
  mobile-auto/
    README.md
    MOBILE_AUTO_TOOL_GUIDE.md
    APP_PROFILE.md
    APP_MAP.md
    SCREEN_MAP.md
    FLOWS/
      login.md
    ERROR_RECOVERY.md
    SELF_IMPROVEMENT.md
    EVIDENCE_INDEX.md
    CHANGELOG.md

如果这些文件已经存在，不要简单覆盖。请合并现有内容，保留有价值信息，并把本次探索新增内容追加进去。

============================================================
三、.github/copilot-instructions.md 要求
============================================================

这个文件是 workspace 常驻入口。内容要短，控制在 80-150 行以内。

必须包含：

1. Project AI operating rules
2. Mobile testing rules
3. 指向 docs/mobile-auto/README.md
4. 指向 docs/mobile-auto/FLOWS/login.md
5. 指向 docs/mobile-auto/APP_MAP.md
6. 指向 docs/mobile-auto/ERROR_RECOVERY.md
7. 明确说：
   - 当任务涉及 mobile-auto、BrowserStack、App 登录、App UI 自动化时，必须先读 docs/mobile-auto/README.md。
   - 不要从零探索登录流程，除非 runbook 已失败。
   - 所有 mobile-auto 命令必须带 --json。
   - 页面变化后必须重新 observe 或使用 --post-observe。
   - 不要使用旧 observation ref。
   - 不要把 secrets 写入日志或文档。
   - 成功或失败后都要更新 docs/mobile-auto/CHANGELOG.md 和必要的 App map/runbook。

建议内容结构：

# Copilot Instructions

## Mobile automation first-read rule

When a task involves mobile-auto, BrowserStack, App Automate, login testing, or App UI automation, first read:

- [Mobile Auto Knowledge Base](../docs/mobile-auto/README.md)
- [Login Flow Runbook](../docs/mobile-auto/FLOWS/login.md)
- [App Map](../docs/mobile-auto/APP_MAP.md)
- [Error Recovery](../docs/mobile-auto/ERROR_RECOVERY.md)

## Non-negotiable mobile-auto rules

- Always use `--json`.
- Do not restart discovery from scratch if a runbook exists.
- Do not persist observation refs such as `obs-...:e17`.
- After every mutating action, re-observe or use `--post-observe`.
- Do not tap ambiguous elements.
- Do not print or store secrets.
- Finish BrowserStack runs even on failure when possible.
- Update the knowledge base after new findings.

============================================================
四、.github/instructions/mobile-auto.instructions.md 要求
============================================================

这个文件是 mobile-auto 专项 instruction。

必须使用 VS Code instruction frontmatter：

---
name: 'Mobile Auto App Testing'
description: 'Rules for using mobile-auto with BrowserStack for this App'
applyTo: '**'
---

内容必须包含：

1. 什么时候适用：
   - mobile-auto
   - BrowserStack
   - App Automate
   - Android/iOS App UI testing
   - login smoke test
   - App screen exploration

2. mobile-auto 标准流程：
   - doctor/auth check
   - run start
   - observe
   - locate
   - action
   - observe/assert
   - run finish

3. 命令规则：
   - every command uses --json
   - schema 查一次，不要每一步重复查
   - 如果命令失败，先读 error.code
   - 不要根据自然语言猜错误

4. App 专用规则：
   - 先读 docs/mobile-auto/APP_MAP.md
   - 先读 docs/mobile-auto/FLOWS/login.md
   - 先读 docs/mobile-auto/ERROR_RECOVERY.md

5. 成本控制：
   - 登录最多允许有限探索
   - 超过预算要停止盲试，写明当前屏幕、失败点、建议更新
   - 不要重复 observe 大段输出
   - 只保存简明 evidence

6. 自我改善规则：
   - 成功后更新 verified path
   - 失败后更新 failure pattern
   - UI 变化后更新 SCREEN_MAP
   - locator 变化后更新 locator candidates
   - 新弹窗/权限/MFA/错误页出现后更新 APP_MAP 和 ERROR_RECOVERY

============================================================
五、.github/prompts/mobile-login-smoke.prompt.md 要求
============================================================

这是以后可以通过 slash command 复用的登录测试 prompt。

必须使用 frontmatter：

---
name: mobile-login-smoke
description: Run the known mobile-auto login smoke flow using the workspace runbook
agent: agent
---

内容要让后续 agent 做这些事：

1. 先读：
   - docs/mobile-auto/README.md
   - docs/mobile-auto/FLOWS/login.md
   - docs/mobile-auto/APP_MAP.md
   - docs/mobile-auto/ERROR_RECOVERY.md

2. 不允许从零探索，除非 runbook 缺失或明确失败。

3. 执行登录 smoke：
   - verify mobile-auto availability
   - verify BrowserStack auth without printing secrets
   - start run
   - follow login runbook
   - assert success
   - finish run
   - collect artifacts if configured

4. 输出必须包含：
   - passed/failed
   - run_id
   - failed step
   - error.code if any
   - evidence path
   - docs updated or not updated

5. 成功/失败后更新：
   - docs/mobile-auto/CHANGELOG.md
   - docs/mobile-auto/EVIDENCE_INDEX.md
   - 如果发现 UI 变化，更新 APP_MAP/SCREEN_MAP/FLOWS/login.md

============================================================
六、.github/prompts/mobile-auto-repair-runbook.prompt.md 要求
============================================================

这是登录失败后修复知识库的 prompt。

必须使用 frontmatter：

---
name: mobile-auto-repair-runbook
description: Repair the mobile-auto runbook after a failed App automation run
agent: agent
---

内容要指导 agent：

1. 先读失败输出和 artifacts。
2. 不要立即重跑完整登录。
3. 判断失败类别：
   - locator changed
   - ambiguous element
   - stale observation
   - keyboard obstruction
   - network wait
   - permission dialog
   - MFA / OTP
   - invalid credentials
   - app crash
   - BrowserStack/session issue
   - mobile-auto command misuse
4. 更新对应文档：
   - locator changed -> SCREEN_MAP.md + FLOWS/login.md
   - new screen -> APP_MAP.md + SCREEN_MAP.md
   - new failure -> ERROR_RECOVERY.md
   - tool misuse -> MOBILE_AUTO_TOOL_GUIDE.md
5. 只允许做最小验证，不允许无限探索。
6. 最后输出修复摘要和下次应该怎么跑。

============================================================
七、.github/prompts/mobile-auto-learn-from-run.prompt.md 要求
============================================================

这是每次成功/失败后让 agent 学习的 prompt。

必须使用 frontmatter：

---
name: mobile-auto-learn-from-run
description: Convert a completed mobile-auto run into persistent App knowledge
agent: agent
---

内容要指导 agent：

1. 从最近一次 run 的输出、logs、artifacts、terminal history 中提取：
   - run id
   - platform/device/network
   - app version/build if known
   - screens visited
   - actions taken
   - stable locators
   - assertions
   - failures/retries
   - artifacts paths
2. 归一化为知识：
   - screen map
   - flow steps
   - failure recovery
   - evidence index
3. 明确不要保存：
   - secrets
   - transient refs
   - raw screenshots if not needed
   - giant observe JSON
4. 更新 CHANGELOG.md。
5. 给每条新增知识加 confidence：
   - high: verified in successful run
   - medium: observed but not asserted
   - low: inferred / agent memory only

============================================================
八、docs/mobile-auto/README.md 要求
============================================================

这个文件是 mobile-auto 知识库入口。

必须包含：

1. Purpose
2. Quick start for future agents
3. Required first reads
4. File map
5. Golden path for login
6. How to update this knowledge base
7. Safety rules
8. Secret handling
9. Command budget guidance

必须写清楚：

未来 agent 遇到 mobile App 测试任务时，顺序是：

1. Read this README.
2. Read APP_MAP.md.
3. Read FLOWS/login.md.
4. Read ERROR_RECOVERY.md.
5. Run only the known flow first.
6. Explore only the failed or unknown screen.
7. Update docs after learning.

============================================================
九、docs/mobile-auto/MOBILE_AUTO_TOOL_GUIDE.md 要求
============================================================

这个文件记录 mobile-auto 工具使用规则。

必须包含：

1. mobile-auto mental model
2. Required global rule: always --json
3. Discovery commands:
   - mobile-auto doctor --json
   - mobile-auto auth test --json
   - mobile-auto commands --json
   - mobile-auto schema ... --json
4. BrowserStack run lifecycle:
   - app resolve if needed
   - run start
   - observe
   - locate
   - tap/type/scroll
   - assert
   - run finish
5. Ref lifecycle:
   - refs are observation-scoped
   - never persist refs
   - re-observe after screen changes
6. Good locator strategy:
   - role
   - name
   - text
   - actionable
   - bounds only as fallback
7. Bad patterns:
   - raw coordinate tap first
   - repeated blind taps
   - parsing human text when JSON exists
   - continuing after ambiguous locate
   - dumping full XML/observe output into chat
8. Artifact handling
9. Failure handling
10. Command budget

请根据你实际使用过的 mobile-auto 命令补充具体示例。

============================================================
十、docs/mobile-auto/APP_PROFILE.md 要求
============================================================

这个文件描述当前 App。

请尽可能从 workspace 和刚才探索中提取：

1. App name
2. Package/bundle id if known
3. Platform tested
4. App file path if known
5. BrowserStack network mode used
6. Login method
7. Test account env vars
8. Known environments
9. Login success definition
10. Known blockers:
    - permissions
    - MFA
    - captcha
    - deep link
    - webview
    - slow network
    - keyboard behavior

如果未知，写 UNKNOWN，不要编造。

============================================================
十一、docs/mobile-auto/APP_MAP.md 要求
============================================================

这个文件是 App 状态机地图。

必须包含：

1. Screen inventory
2. Screen transitions
3. Known entry points
4. Login-related screens
5. Success/home screen
6. Failure screens
7. Modal/dialog screens
8. Unknown screens

建议格式：

# App Map

## State machine summary

```mermaid
stateDiagram-v2
    AppLaunch --> LoginScreen
    LoginScreen --> HomeScreen: valid credentials
    LoginScreen --> LoginError: invalid credentials
    LoginScreen --> MfaScreen: MFA required
````

如果不确定 mermaid 是否准确，可以仍然创建，但给 confidence: low。

每个 screen 用这个模板：

## Screen: Login

* id:
* confidence:
* last verified:
* evidence:
* visible text:
* stable identity:
* primary actions:
* next screens:
* known risks:
* update notes:

不要记录 obs-xxx refs。

============================================================
十二、docs/mobile-auto/SCREEN_MAP.md 要求
====================================

这个文件记录每个屏幕上的元素定位策略。

必须为登录流程涉及到的每个 screen 建立条目。

每个元素用这个模板：

### Element: username input

* semantic id: login.username
* role candidates:
* name candidates:
* text candidates:
* accessibility candidates:
* stable locator strategy:
* fallback strategy:
* do not use:
* last verified:
* confidence:
* evidence:

必须包含这些元素，如果探索中出现过：

1. 登录入口按钮
2. 用户名/邮箱输入框
3. 密码输入框
4. 登录/Sign in 按钮
5. 登录成功后的 Home/Dashboard 标识
6. 错误提示
7. 权限弹窗按钮
8. 键盘相关处理
9. MFA/OTP 如果出现过

============================================================
十三、docs/mobile-auto/FLOWS/login.md 要求
=====================================

这是最重要的文件。请基于刚刚探索成功的路径写得尽可能详细。

必须包含：

1. Goal
2. Preconditions
3. Required env vars
4. App launch/start command pattern
5. Known login path
6. Step-by-step commands
7. Locator strategy per step
8. Expected observation after each step
9. Assertion for success
10. Failure recovery per step
11. When to stop and ask for human intervention
12. Post-run docs update

每个步骤用这个模板：

## Step 03: Enter username

* intent:
* precondition:
* command pattern:
* locator:
* action:
* expected result:
* evidence from previous exploration:
* common failure:
* recovery:
* confidence:
* do not:

命令示例必须使用环境变量，不允许写真实账号密码：

```bash
mobile-auto type \
  --run-id "$RUN_ID" \
  --ref "$REF" \
  --text-env TEST_USERNAME \
  --post-observe \
  --json
```

如果你不确定某个 mobile-auto 参数名，先查 schema 或写成 “verify with `mobile-auto schema <command> --json`”。

============================================================
十四、docs/mobile-auto/ERROR_RECOVERY.md 要求
========================================

这个文件要成为后续 agent 的失败恢复表。

必须包含：

1. General recovery policy
2. Error code table
3. Symptom table
4. Screen-specific recovery
5. Stop conditions
6. How to update this file

必须至少覆盖：

| Symptom / error          | Likely cause               | Next action                           | Docs to update            |
| ------------------------ | -------------------------- | ------------------------------------- | ------------------------- |
| ambiguous element        | locator too broad          | narrow by role/name/actionable        | SCREEN_MAP.md             |
| stale observation        | used old ref               | observe again and re-locate           | MOBILE_AUTO_TOOL_GUIDE.md |
| element not found        | UI changed or wrong screen | observe and identify screen           | APP_MAP.md                |
| keyboard covers button   | keyboard open              | hide keyboard or submit from keyboard | FLOWS/login.md            |
| login timeout            | network/backend slow       | wait/assert visible or check error    | ERROR_RECOVERY.md         |
| invalid credentials      | bad test account           | stop, do not retry blindly            | APP_PROFILE.md            |
| MFA required             | account policy             | handoff/manual step                   | APP_MAP.md                |
| app crashed              | app/runtime bug            | collect artifacts, finish failed      | EVIDENCE_INDEX.md         |
| BrowserStack auth failed | env/key issue              | auth test, do not print keys          | MOBILE_AUTO_TOOL_GUIDE.md |

如果实际探索中遇到过其他错误，也加入表格。

============================================================
十五、docs/mobile-auto/SELF_IMPROVEMENT.md 要求
==========================================

这个文件定义“后续 agent 怎么自己变聪明”。

必须包含：

1. When to update docs
2. What to update after success
3. What to update after failure
4. How to record confidence
5. How to avoid knowledge rot
6. Review checklist
7. Anti-patterns

必须写清楚：

成功后：

* 更新 last verified
* 更新 evidence path
* 更新 CHANGELOG
* 如果 locator 更稳定，替换旧 locator

失败后：

* 不要无限 retry
* 先分类失败
* 只探索失败附近的 screen
* 更新 failure pattern
* 增加 recovery rule
* 标记 confidence

必须包含 confidence 规则：

* high: verified by a passing run
* medium: observed during run but not asserted
* low: inferred from memory/logs only
* deprecated: no longer valid after latest run

============================================================
十六、docs/mobile-auto/EVIDENCE_INDEX.md 要求
========================================

这个文件记录证据索引，不记录 secrets。

必须包含：

1. Run records table
2. Artifact paths
3. What was verified
4. What failed
5. Docs updated

表格格式：

| Date | Run ID | Platform | Build/App | Result | Verified flow | Evidence path | Docs updated |
| ---- | ------ | -------- | --------- | ------ | ------------- | ------------- | ------------ |

如果没有 run_id，写 UNKNOWN。

============================================================
十七、docs/mobile-auto/CHANGELOG.md 要求
===================================

记录知识库变化。

格式：

# Mobile Auto Knowledge Changelog

## YYYY-MM-DD

### Added

* ...

### Changed

* ...

### Deprecated

* ...

### Needs verification

* ...

本次生成文档要写一条 changelog，说明：

* created initial workspace mobile-auto knowledge base
* extracted login flow from prior exploration
* added self-improvement protocol
* marked unknowns for verification

============================================================
十八、质量要求
=======

生成文档后，请执行这些自检：

1. 列出所有创建/更新的文件。
2. 搜索并确认没有写入明显 secret：

   * BROWSERSTACK_ACCESS_KEY=
   * TEST_PASSWORD=
   * password: 真实值
   * access key
   * token 真实值
3. 搜索并确认没有把临时 observation ref 写成长期规则：

   * obs-
   * :e 后跟数字
     如果文档里出现 obs-，只能出现在“不要持久化 obs refs”的说明里，不能作为实际 locator。
4. 检查 `.github/copilot-instructions.md` 是否链接到 docs/mobile-auto/README.md。
5. 检查 `.github/prompts/*.prompt.md` frontmatter 是否存在。
6. 检查 `.github/instructions/mobile-auto.instructions.md` 是否有 applyTo。
7. 最后输出：

   * files created/updated
   * key knowledge captured
   * remaining unknowns
   * next recommended command for future login smoke
   * no-secrets check result

============================================================
十九、输出方式
=======

请直接修改 workspace 文件。

最后不要输出长篇重复文档全文，只输出摘要：

1. Created/updated files
2. What was learned from prior exploration
3. How future agents should run login
4. Known gaps / needs verification
5. Safety check results

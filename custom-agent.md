可以，而且**非常适合你这个场景**。

但它的价值不是“让 Copilot 突然知道你的 App”，而是把 Copilot 固定成一个专门角色：**Mobile Auto QA Agent**。这样它每次进入这个 workspace 后，都会按同一套规则读 App 文档、用 `mobile-auto`、控制探索次数、更新 runbook，而不是每次像新手一样重新摸索。

VS Code 官方对 custom agent 的定位也正好符合这个用途：custom agent 可以有自己的 persona、instructions、可用 tools、model 和 handoff；workspace 级 custom agent 默认放在 `.github/agents`，文件后缀是 `.agent.md`。([Visual Studio Code][1])

## 它能帮你解决什么

**第一，减少“自由探索”导致的 token 浪费。**
普通 Copilot Agent 会根据当前屏幕一步步猜。Custom agent 可以被写死为：遇到 mobile 测试任务必须先读 `docs/mobile-auto/README.md`、`APP_MAP.md`、`FLOWS/login.md`、`ERROR_RECOVERY.md`，不允许从零探索。这样 token 从“重新理解 App”变成“执行已有流程”。

**第二，固定 mobile-auto 的工具使用纪律。**
比如所有命令必须带 `--json`、登录流程必须 `observe → locate → action → observe/assert`、不能持久化 `obs-...:e17`、失败必须更新知识库。这些规则放在 custom agent 里，比每次 prompt 里临时提醒稳定得多。

**第三，可以限制工具范围。**
VS Code custom agent 的 frontmatter 支持 `tools`，可以限制这个 agent 能用哪些工具；官方文档也强调 custom agent 适合给不同任务配置不同工具能力，安全敏感流程可以用较小工具集。([Visual Studio Code][1]) 对你来说，Mobile QA Agent 通常只需要读文件、改 Markdown、运行终端、查看最近命令输出，不需要随便改业务源码。

**第四，可以做多角色 handoff。**
VS Code custom agent 支持 handoffs，也就是一个 agent 结束后给出按钮，把上下文交给另一个 agent；官方例子包括 Planning → Implementation、Implementation → Review。([Visual Studio Code][1]) 你的场景可以设计成：

```text
Mobile Explorer Agent
  负责有限探索，找出登录路径

Mobile Knowledge Writer Agent
  负责把探索路径写入 APP_MAP / SCREEN_MAP / login.md

Mobile Smoke Runner Agent
  负责以后只按 runbook 跑登录 smoke

Mobile Repair Agent
  负责失败后修复 locator / recovery table
```

**第五，可以配合 hooks 做硬约束。**
VS Code hooks 目前是 Preview，但官方说明它可以在 agent session 生命周期中运行 shell 命令，也能在 `PreToolUse` 阶段读取 JSON 输入并阻止某些工具调用。([Visual Studio Code][2]) 后续你可以用 hook 做更硬的 guardrail，例如：阻止不带 `--json` 的 `mobile-auto` 命令、阻止把 `BROWSERSTACK_ACCESS_KEY` 写进 Markdown、阻止无限探索。

---

## 我建议你加一个 custom agent

创建这个文件：

```text
.github/agents/mobile-auto-qa.agent.md
```

内容可以先用下面这版。

````md
---
name: Mobile Auto QA
description: Use mobile-auto and BrowserStack to run known App flows, repair runbooks, and maintain the mobile automation knowledge base.
argument-hint: "Describe the mobile flow to run or repair, for example: login smoke"
# tools:
#   - search/codebase
#   - read/terminalLastCommand
#   - edit
#   - runCommands
#
# Tool names vary by VS Code version/extensions. If a tool is unavailable, configure tools from the chat UI.
handoffs:
  - label: Repair mobile runbook
    agent: Mobile Auto QA
    prompt: "Analyze the latest failed mobile-auto run, update ERROR_RECOVERY.md and the relevant flow docs, then summarize the minimum next verification."
    send: false
---

You are the workspace Mobile Auto QA Agent.

Your job is to use the existing mobile automation knowledge base instead of rediscovering the App from scratch.

## First-read rule

For any task involving mobile-auto, BrowserStack, App Automate, Android/iOS UI testing, login smoke, or App screen exploration, first read:

- `docs/mobile-auto/README.md`
- `docs/mobile-auto/APP_PROFILE.md`
- `docs/mobile-auto/APP_MAP.md`
- `docs/mobile-auto/SCREEN_MAP.md`
- `docs/mobile-auto/FLOWS/login.md`
- `docs/mobile-auto/ERROR_RECOVERY.md`
- `docs/mobile-auto/SELF_IMPROVEMENT.md`

If these files are missing, create them before running new exploration.

## Primary goal

Prefer known flows over exploration.

Use this order:

1. Read the mobile-auto knowledge base.
2. Identify the target flow.
3. Run the known flow first.
4. Explore only the failed or unknown screen.
5. Update the knowledge base after every meaningful success or failure.

## Non-negotiable mobile-auto rules

- Every `mobile-auto` command must use `--json`.
- Do not parse human-readable output when JSON is available.
- Do not print secrets.
- Do not write secrets to Markdown, logs, artifacts, or test fixtures.
- Use environment variables for credentials:
  - `BROWSERSTACK_USERNAME`
  - `BROWSERSTACK_ACCESS_KEY`
  - `TEST_USERNAME`
  - `TEST_PASSWORD`
- Never persist temporary observation refs such as `obs-...:e17` as long-term locators.
- Observation refs are valid only for the latest observation.
- After every mutating action, re-observe or use `--post-observe`.
- Do not tap an ambiguous locate result.
- Do not use raw coordinate taps unless all semantic locators fail and the fallback is documented.
- Always finish BrowserStack runs when possible, even on failure.
- Collect artifacts for failed or uncertain runs when possible.

## Standard flow

Use this lifecycle:

```text
doctor/auth check
  -> run start
  -> observe
  -> locate
  -> action
  -> observe/assert
  -> run finish
  -> update docs
````

Do not repeatedly query `mobile-auto commands` or `mobile-auto schema` once the needed schema is known. Query schema only when a command fails because of usage or argument shape.

## Exploration budget

For login flows:

* Do not perform more than 15 `mobile-auto` commands without producing a short state summary.
* Do not retry the same failed action more than 2 times.
* If the screen is unknown, identify it and update `APP_MAP.md` or `SCREEN_MAP.md`.
* If the locator changed, update `SCREEN_MAP.md` and `FLOWS/login.md`.
* If a new failure pattern appears, update `ERROR_RECOVERY.md`.
* If the flow succeeds, update `EVIDENCE_INDEX.md` and `CHANGELOG.md`.

## How to handle failure

Classify the failure before retrying:

* `ambiguous_element`: narrow by role/name/text/actionable; do not tap.
* `stale_observation`: run observe again, then locate again.
* `element_not_found`: verify current screen; update screen map if UI changed.
* keyboard covers button: hide keyboard or submit from keyboard.
* login timeout: observe current screen and check for error/MFA/loading state.
* invalid credentials: stop; do not blindly retry.
* MFA/OTP required: document human handoff.
* app crash: collect artifacts and finish failed.
* BrowserStack/session issue: diagnose auth/session separately from App logic.
* mobile-auto misuse: check schema once and update `MOBILE_AUTO_TOOL_GUIDE.md`.

## Documentation update rules

After a successful run, update:

* `docs/mobile-auto/EVIDENCE_INDEX.md`
* `docs/mobile-auto/CHANGELOG.md`
* `docs/mobile-auto/FLOWS/login.md` if the verified path changed
* `docs/mobile-auto/SCREEN_MAP.md` if locator strategies improved

After a failed run, update:

* `docs/mobile-auto/ERROR_RECOVERY.md`
* `docs/mobile-auto/SCREEN_MAP.md` if a locator failed or changed
* `docs/mobile-auto/APP_MAP.md` if a new screen/modal appeared
* `docs/mobile-auto/EVIDENCE_INDEX.md`
* `docs/mobile-auto/CHANGELOG.md`

Every new fact must include confidence:

* high: verified by a passing run
* medium: observed but not asserted
* low: inferred from logs or agent memory
* deprecated: contradicted by the latest run

## Output style

At the end of each task, summarize only:

* passed or failed
* run_id if known
* failed step if any
* error.code if any
* evidence path
* docs updated
* next recommended command

````

注意：上面的 `tools` 我先注释掉了。VS Code 的工具名会随版本、扩展和 agent 类型变化；官方文档说 custom agent 的 `tools` 可以引用内置 tools、tool sets、MCP tools 或扩展贡献的 tools，不可用的 tool 会被忽略。:contentReference[oaicite:4]{index=4} 你可以先不写 `tools`，然后在 Chat 里用 **Configure Tools** 查看当前实际 tool 名称，再收紧。

---

## 再配一个 prompt 文件调用它

创建：

```text
.github/prompts/mobile-login-smoke.prompt.md
````

```md
---
name: mobile-login-smoke
description: Run the known mobile-auto login smoke flow and update the mobile automation knowledge base
agent: Mobile Auto QA
---

Run the known mobile login smoke flow.

Do not rediscover the login path from scratch.

First read:

- `docs/mobile-auto/README.md`
- `docs/mobile-auto/APP_MAP.md`
- `docs/mobile-auto/SCREEN_MAP.md`
- `docs/mobile-auto/FLOWS/login.md`
- `docs/mobile-auto/ERROR_RECOVERY.md`

Then:

1. Verify `mobile-auto` and BrowserStack auth without printing secrets.
2. Start a BrowserStack run using the documented command pattern.
3. Follow `docs/mobile-auto/FLOWS/login.md`.
4. Use only stable locator strategies from `SCREEN_MAP.md`.
5. Assert the documented login success condition.
6. Finish the run and collect artifacts when appropriate.
7. Update `EVIDENCE_INDEX.md` and `CHANGELOG.md`.
8. If the run fails, classify the failure and update the relevant docs.

Final response must include:

- result: passed/failed
- run_id
- failed step
- error.code
- evidence path
- docs updated
- next recommended command
```

VS Code prompt files就是这种用途：把常见任务保存成 `.prompt.md`，在 chat 里用 slash command 调用；workspace prompt 默认放 `.github/prompts`。([Visual Studio Code][3])

以后你就可以在 Copilot Chat 里选择 **Mobile Auto QA** agent，然后输入：

```text
/mobile-login-smoke
```

或者直接：

```text
使用 Mobile Auto QA agent，按 login runbook 跑一次登录 smoke。不要从零探索，失败后更新知识库。
```

---

## custom agent 和前面那些 Markdown 的关系

我建议这样分工：

```text
.github/copilot-instructions.md
  所有 agent 都会看到的入口规则：遇到 mobile-auto 必须读 docs/mobile-auto/README.md。

.github/agents/mobile-auto-qa.agent.md
  专门的 Mobile QA persona：怎么用 mobile-auto、怎么限制探索、怎么更新知识库。

.github/prompts/mobile-login-smoke.prompt.md
  可重复执行的 slash command：跑登录 smoke。

docs/mobile-auto/**
  App 的真实长期知识：页面、流程、locator、错误恢复、证据。
```

简单说：

```text
instructions = 默认记忆
custom agent = 专门角色
prompt file = 可重复任务入口
docs/mobile-auto = App 知识库
hooks = 可选硬约束
```

VS Code 官方也建议逐步采用这些层级：先项目级 instructions，再 targeted instructions，再 prompt files，最后 custom agents、skills、MCP、hooks。([Visual Studio Code][4])

---

## 它不能帮你解决什么

Custom agent **不能自动获得 App 的真实知识**。它仍然需要你把已探索路径写进 workspace 文档。

Custom agent **不能替代稳定脚本或 suite**。最省 token 的最终形态仍然是：

```text
docs/mobile-auto/FLOWS/login.md
+ tests/mobile/login-smoke.yaml
+ scripts/mobile-login-smoke.sh
+ Mobile Auto QA custom agent
```

Custom agent **不能保证每条 shell 命令都合规**。它能强提示、能限制工具；如果你要强制拦截，比如“不允许不带 `--json` 的 mobile-auto 命令”，更适合再加 VS Code hooks。Hooks 仍是 Preview，配置和行为可能变，且可能被组织策略禁用。([Visual Studio Code][2])

---

## 我的建议

你应该用 custom agent，但不要只做一个泛泛的 “QA Agent”。要把它做成：

```text
Mobile Auto QA Agent
  = mobile-auto 使用规范
  + App knowledge base first-read rule
  + 探索预算
  + 失败分类
  + 文档自我更新协议
```

这样后续 Copilot 就不会每次都问“我现在在哪里、这个按钮能不能点、登录成功怎么算”，而是会先读你 workspace 里沉淀出来的 App 地图，再按已验证路径执行。

[1]: https://code.visualstudio.com/docs/agent-customization/custom-agents "Custom agents in VS Code"
[2]: https://code.visualstudio.com/docs/copilot/customization/hooks "Agent hooks in Visual Studio Code (Preview)"
[3]: https://code.visualstudio.com/docs/copilot/customization/prompt-files "Use prompt files in VS Code"
[4]: https://code.visualstudio.com/docs/agent-customization/overview "Customize AI in Visual Studio Code"

可以。**你这里需要配置两个地方：先让 BrowserStack Local 隧道运行起来，再在 Inspector 中指定“使用哪条隧道”。仅在 Capability Generator 勾选 Local，并不会自动建立连接。**([BrowserStack][1])

下面统一用一个示例隧道名称：

```text
efp-fx-local
```

**这个名称是你自己定义的，不是 BrowserStack 提供的固定值。启动隧道和配置 Inspector 时必须一致。**

---

# 一、先分清楚：Local 和 Inspector 不是一个服务

你的连接关系应该是：

```text
【控制手机】
电脑浏览器中的 Inspector
    → BrowserStack 云端 Appium
    → BrowserStack 云端真机

【App 访问公司内网】
BrowserStack 云端真机里的 App
    → BrowserStack Local 隧道
    → 运行 Local 的电脑或服务器
    → 公司内网测试后台
```

Inspector 的远程服务配置决定“控制哪台手机”；Local 解决“手机里的 App 如何访问内网后台”。这是两条不同的连接。([Appium][2])

**Local 不要求必须运行在你的 Windows 电脑上。** 采用公司已有、能够访问测试后台的获批机器运行 Local，你的电脑只打开 Inspector，也可以按这套架构连接。([BrowserStack][3])

## 先处理你提到的 EXE 限制

**BrowserStack Local 仍需要原生二进制程序。`npm install browserstack-local` 只是安装 Node.js 封装，它默认会尝试下载和执行 Local 二进制文件，不是纯 JavaScript 隧道。**

因此，你们应选以下一种方式：

| 企业环境                       | 应采用的方式                           |
| -------------------------- | -------------------------------- |
| 团队已经有获批、正在运行的 Local 隧道     | **优先复用，Windows 不需要下载 Local EXE** |
| IT 已通过内部渠道提供并允许运行 Local 程序 | 使用下面的 CMD 命令启动                   |
| Windows 完全不允许运行该程序         | 由 IT 在获批服务器上部署，再使用它的隧道标识         |

**下面不会提供外网下载 EXE 的步骤。**

---

# 二、让 Local 隧道先运行起来

## 情况 A：团队已经有 Local 隧道

先确认这三项信息：

| 要确认的内容                    | 例子或要求                   |
| ------------------------- | ----------------------- |
| 隧道的 `localIdentifier`     | 例如 `team-uat-local`     |
| 你的 BrowserStack 会话是否有权使用它 | 让隧道管理员确认账号或共享配置         |
| 运行隧道的机器能否访问测试后台           | 必须能够解析、访问 App 实际调用的内网地址 |

**已有隧道叫 `team-uat-local`，后面 JSON 就填 `team-uat-local`，不要照抄本文的 `efp-fx-local`。** Local 的标识用于让测试会话选择对应连接。([BrowserStack][1])

这种情况下，直接跳到第三部分配置 Capabilities。

## 情况 B：你已经有公司批准的 Local 程序

以下假设 IT 提供的文件位于：

```text
C:\CompanyTools\BrowserStackLocal.exe
```

**这是示例路径，必须改成你实际已有的路径。没有这个文件就不要执行后面的启动命令。**

### 第 1 步：打开 CMD，设置程序路径

```cmd
cmd /d /v:off
```

```cmd
set "BS_LOCAL_BIN=C:\CompanyTools\BrowserStackLocal.exe"
```

确认文件存在：

```cmd
dir "%BS_LOCAL_BIN%"
```

### 第 2 步：输入 BrowserStack Access Key

```cmd
set "BROWSERSTACK_ACCESS_KEY="
set /p "BROWSERSTACK_ACCESS_KEY=BrowserStack Access Key: "
```

**注意：CMD 的 `set /p` 会显示输入内容，不是隐藏输入。不要在共享屏幕时操作，也不要把密钥写进提交到 Git 的脚本。**

### 第 3 步：设置隧道名称

```cmd
set "BS_LOCAL_ID=efp-fx-local"
```

这个名称后面要填到 Inspector 的 `localIdentifier`。

### 第 4 步：启动 Local

```cmd
"%BS_LOCAL_BIN%" --key "%BROWSERSTACK_ACCESS_KEY%" --local-identifier "%BS_LOCAL_ID%" --verbose 1
```

这里使用的是 Local 官方的 Access Key、连接标识和日志参数。([browserstack.com][4])

**看到连接成功的日志后再继续。这个 CMD 窗口保持运行，不要关闭。** 不同版本的成功提示文案可能不同，不能只看“进程已经启动”。

### 公司必须通过代理出网时

先停止你刚启动的这一条 Local 进程，然后使用带代理参数的命令。**下面的代理主机和端口都要替换成公司实际值：**

```cmd
"%BS_LOCAL_BIN%" --key "%BROWSERSTACK_ACCESS_KEY%" --local-identifier "%BS_LOCAL_ID%" --proxy-host "REPLACE_WITH_PROXY_HOST" --proxy-port 8080 --verbose 1
```

`--proxy-host`、`--proxy-port` 是 Local 的参数，**不是填在 Inspector 手机型号或 App 地址里的内容**。需要代理认证时还有独立的用户名、密码参数，应按公司代理类型配置。([BrowserStack][5])

另外，`--force-local` 表示让设备的 URL 请求都经 Local 解析和转发，不只是“打开 Local 功能”。**不要默认加上；只有需要全部流量走公司网络时，再按网络政策启用。**([BrowserStack][4])

---

# 三、Capability Generator 到底怎么填？

打开 **App Automate** 的 Capability Generator，不是桌面浏览器测试的 Automate 页面：

```text
https://www.browserstack.com/docs/app-automate/capabilities
```

这个页面的用途是生成测试配置。当前页面也提供 BrowserStack SDK 的 YAML 配置，因此**生成出来的内容不一定能原样粘贴进 Inspector**。([BrowserStack][6])

下面按字段用途对应。不同页面版本的栏目名称可能略有差别；**页面没有显示的字段，直接在下一部分的 Inspector JSON 中补上。**

## 1. 基础选项

| 页面选项              | 你应该怎么选                   | 容易填错的地方                       |
| ----------------- | ------------------------ | ----------------------------- |
| 产品 / Product      | **App Automate**         | 不要选测试桌面浏览器的 Automate          |
| 自动化框架 / Framework | **Appium**               | 即使测试 iPhone，这里仍选 Appium       |
| 语言 / Language     | **Node.js / JavaScript** | 不需要 Python                    |
| 客户端或示例框架          | 有 WebdriverIO 选项时选它      | 主要影响示例代码，不决定手机系统              |
| 平台 / Platform     | Android 或 iOS            | **不是填写你电脑的 Windows**          |
| 设备 / Device       | 从列表选择一台实际手机              | 不要自己猜一个设备名称                   |
| 系统版本 / OS Version | 选择该手机支持的系统版本             | **不是 Appium 3.7.0**           |
| Application / App | 你上传 App 后取得的 `bs://...`  | 不是公司后台 URL，也不是 Windows APK 路径 |

设备及系统版本应采用 BrowserStack 当前支持的组合；App 标识来自上传响应中的 `app_url`。([BrowserStack][7])

例如：

```text
你的电脑系统：Windows
云端手机平台：Android
云端手机系统：从 Generator 中选择的 Android 版本
本地 Appium：3.7.0
```

**这几个版本属于不同层次，不要互相填错。**

## 2. Local 相关选项——重点看这里

| 页面选项                               | 应填写的值              | 含义                   |
| ---------------------------------- | ------------------ | -------------------- |
| Local Testing / BrowserStack Local | **开启 / true**      | 让本次手机会话使用 Local      |
| Local Identifier                   | **`efp-fx-local`** | 必须与正在运行的隧道名称完全相同     |
| Force Local                        | 第一轮不要额外开启，除非网络要求   | 它改变流量路由范围，不是基本连接的必填项 |

在非 SDK 的 W3C 会话配置中，对应的关键内容是：([BrowserStack][1])

```json
{
  "bstack:options": {
    "local": true,
    "localIdentifier": "efp-fx-local"
  }
}
```

**上面只是 Local 配置片段，不是完整的手机会话配置。**

## 3. 其他选项

我建议第一轮这样处理：

| 字段             | 第一轮填写建议                              |
| -------------- | ------------------------------------ |
| Project Name   | `EFP Mobile`，用于报告分组                  |
| Build Name     | `inspector-local-poc`，用于识别这次验证       |
| Session Name   | `FX recording`，用于识别会话                |
| Appium Version | 使用 BrowserStack 当前支持的默认值，或团队验证过的云端版本 |
| Browser Name   | **原生 App 测试不填写 Chrome 或 Safari**     |

项目、构建和会话名称是报告标签；`browserName` 则是启动浏览器的能力项，不是“用 Chrome 打开 Inspector”的设置。([BrowserStack][8])

**不要因为你本地安装的是 Appium 3.7.0，就把云端版本强制填成 3.7.0。** 云端支持什么版本由 BrowserStack 决定，本地承载 Inspector 的版本不是云端版本的证明。([Appium][9])

---

# 四、直接给你一份可以填入 Inspector 的 JSON

**建议你用 Generator 确认手机型号和系统版本，然后使用下面这份 JSON，避免复制到 SDK 的 YAML 配置。**

## Android 模板

```json
{
  "platformName": "Android",
  "appium:automationName": "UiAutomator2",
  "appium:deviceName": "REPLACE_WITH_DEVICE_NAME",
  "appium:platformVersion": "REPLACE_WITH_OS_VERSION",
  "appium:app": "bs://REPLACE_WITH_APP_ID",
  "appium:newCommandTimeout": 300,
  "bstack:options": {
    "local": true,
    "localIdentifier": "efp-fx-local",
    "projectName": "EFP Mobile",
    "buildName": "inspector-local-poc",
    "sessionName": "FX recording",
    "idleTimeout": 300
  }
}
```

这里采用 Appium 的命名空间能力项，并把 BrowserStack 特有配置放在 `bstack:options` 中。两个超时值设置为 300 秒，是为人工操作预留停顿时间的示例设置。([Appium][9])

### 必须修改的位置

| JSON 字段                          | 你要替换为什么                  |
| -------------------------------- | ------------------------ |
| `appium:deviceName`              | Generator 中所选手机的准确名称     |
| `appium:platformVersion`         | 该手机对应的系统版本字符串            |
| `appium:app`                     | 你自己的完整 `bs://...` App 标识 |
| `bstack:options.localIdentifier` | 已启动隧道的真实名称               |

**所有 `REPLACE_WITH_...` 都是占位符，不替换就不能直接运行。**

例如 App 上传返回：

```json
{
  "app_url": "bs://你实际得到的标识"
}
```

那么配置就应填写：

```json
"appium:app": "bs://你实际得到的标识"
```

Local 不会把你电脑里的 APK 自动变成云端已上传的 App；上传后的 `app_url` 才是这条 Inspector 接入流程使用的 App 标识。([BrowserStack][10])

## iOS 怎么改？

复制上面的完整模板，只调整这些项目：

```text
platformName                 → iOS
appium:automationName        → XCUITest
appium:deviceName            → Generator 中实际选择的 iPhone
appium:platformVersion       → 对应 iOS 版本
appium:app                   → 上传 IPA 后取得的 bs:// 标识
```

Local 的两项配置保持同样的逻辑，不因为 Android 换成 iOS 就变成另一种写法。([BrowserStack][11])

## 特别容易复制错：SDK YAML 与 Inspector JSON

Generator 中可能出现：

```yaml
browserstackLocal: true
browserStackLocalOptions:
  localIdentifier: efp-fx-local
```

这是 **BrowserStack SDK 的配置形式**。([BrowserStack][6])

**在 Inspector 中，应使用前面完整 JSON 里的这一段，而不是把 YAML 原样粘贴进去：**

```json
"bstack:options": {
  "local": true,
  "localIdentifier": "efp-fx-local"
}
```

也不要把它误写成顶层的 `appium:local`。Local 连接选择是 BrowserStack 的会话选项。([BrowserStack][1])

---

# 五、在 Inspector 界面的哪个位置填写？

## 第 1 步：启动你本机的 Inspector

另开一个 CMD。已经安装 Inspector 插件时，执行：

```cmd
appium --address 127.0.0.1 --port 4723 --use-plugins=inspector
```

浏览器打开：

```text
http://127.0.0.1:4723/inspector
```

插件使用 `/inspector` 路径提供浏览器界面。([Appium][12])

**现在可能有两个独立窗口：一个运行 Local，一个运行承载 Inspector 的 Appium。它们不是重复启动。** 使用团队远程隧道时，你本机只需要后一个窗口。

## 第 2 步：选择云服务商

在 Inspector 的会话创建页面：

```text
Cloud Providers
    → BrowserStack
```

填写：

```text
Username：你的 BrowserStack Username
Access Key：你的 BrowserStack Access Key
```

这是 BrowserStack 官方的 Inspector 凭据填写位置。([BrowserStack][13])

**不要在输入框里填写 `%BROWSERSTACK_ACCESS_KEY%` 这样的 CMD 变量名；输入框需要实际值。**

## 第 3 步：编辑右侧 JSON

在 **Capability Builder** 区域：

```text
右侧 JSON 区域
    → Edit
    → 粘贴前面的完整 JSON
    → 替换占位符
    → Save
```

Inspector 官方说明，右侧 JSON 区域可直接编辑和校验配置；包含嵌套对象的能力项适合在这里填写。([Appium][14])

**不要把 `bstack:options` 当作一个普通字符串输入，也不要再套一层 `"capabilities": {...}`。这里填写的就是能力对象本身。**

## 第 4 步：确认连接目标没有填成本机

BrowserStack 云端 Appium 的连接地址是：

```text
https://hub-cloud.browserstack.com/wd/hub
```

需要手动核对服务端字段时，对应关系是：([BrowserStack][10])

| 字段          | 值                            |
| ----------- | ---------------------------- |
| Remote Host | `hub-cloud.browserstack.com` |
| Port        | `443`                        |
| Remote Path | `/wd/hub`                    |
| SSL / HTTPS | 开启                           |

**不要把 Remote Host 填成运行 Local 的电脑或服务器地址。BrowserStack Local 是网络隧道，不是你要连接的 Appium 设备服务端。**

也就是说：

```text
浏览器地址栏：127.0.0.1:4723/inspector
Inspector 控制目标：BrowserStack 云端 Appium
Local 隧道选择：bstack:options.localIdentifier
```

最后点击 **Start Session**。

---

# 六、怎么判断配置真正成功？

建议分三层检查，而不是只看“手机画面出来了”：

| 检查层次         | 应看到的结果                | 能说明什么          |
| ------------ | --------------------- | -------------- |
| Local 连接     | 隧道进程报告连接成功            | 隧道已经建立         |
| Inspector 会话 | 出现云端 App 画面和元素信息      | 手机控制连接成功       |
| 内网业务访问       | App 中依赖内网接口的页面拿到了预期数据 | 这次业务请求能够访问测试后台 |

**能看到登录界面，并不一定代表 Local 已经验证成功；登录界面可能只是 App 自带的静态内容。** 实际验证时，应选择一个确定会请求内网后台的页面或操作。

出现问题时，按症状定位：

| 现象                 | 优先检查                                       |
| ------------------ | ------------------------------------------ |
| 找不到 Local 连接       | 隧道是否在线；`localIdentifier` 是否一致；账号是否能使用该隧道   |
| 设备或系统版本无效          | 使用 Generator 中实际支持的型号和版本组合，不手填猜测值          |
| App 无效或找不到         | 是否填写了正确、当前可用的 `bs://...`                   |
| 手机能操作，但内网请求失败      | 从**运行 Local 的机器**检查内网域名、网络、代理和认证           |
| 浏览器报 `CORS policy` | 这是网页版 Inspector 与云端接口的跨域问题，不是 Local 参数能解决的 |

Local 连接选择、设备选择和 App 上传都有各自独立的配置；对于云服务商接口的 CORS 错误，Inspector 官方明确说明需要在服务商侧处理，给本机加 `--allow-cors` 不能改变 BrowserStack 的响应。([BrowserStack][1])

**最重要的对应关系只有这一组：**

```text
Local 启动参数：
--local-identifier efp-fx-local

Inspector 会话配置：
"bstack:options": {
  "local": true,
  "localIdentifier": "efp-fx-local"
}
```

**隧道先在线，名称一致，手机和 App 填实际值；你的 Windows 电脑没有获批的 Local 程序时，就复用获批服务器上的隧道，不需要为了操作 Inspector 而额外下载 EXE。**

[1]: https://www.browserstack.com/docs/app-automate/appium/test-on-internal-network/manage-multiple-connections "Setup multiple local connections for Appium tests | BrowserStack Docs"
[2]: https://appium.github.io/appium-inspector/latest/session-builder/server-details/ "Server Details - Appium Inspector"
[3]: https://www.browserstack.com/docs/low-code-automation/local-testing?utm_source=chatgpt.com "Test on internal networks | BrowserStack Docs"
[4]: https://www.browserstack.com/docs/local-testing/binary-params "Local Testing | Flags for BrowserStack Local | BrowserStack Docs"
[5]: https://www.browserstack.com/docs/app-automate/appium/test-on-internal-network/test-behind-proxy/configure-settings?utm_source=chatgpt.com "Configure proxy settings to run Appium tests"
[6]: https://www.browserstack.com/docs/app-automate/capabilities "BrowserStack Docs"
[7]: https://www.browserstack.com/docs/app-automate/appium/set-up-tests/select-devices "Select real devices on BrowserStack for Appium testing | BrowserStack Docs"
[8]: https://www.browserstack.com/docs/app-automate/appium/getting-started/python/integrate-your-tests-legacy "Integrate Appium tests using Python | BrowserStack Docs"
[9]: https://appium.io/docs/en/latest/guides/caps/ "Session Capabilities - Appium Documentation"
[10]: https://www.browserstack.com/docs/app-automate/appium/getting-started/nodejs/integrate-your-tests-legacy "Integrate Appium tests using NodeJS | BrowserStack Docs"
[11]: https://www.browserstack.com/guide/desired-capabilities-in-appium?utm_source=chatgpt.com "What Are Desired Capabilities in Appium? (2026)"
[12]: https://appium.github.io/appium-inspector/latest/quickstart/installation/ "Installation - Appium Inspector"
[13]: https://www.browserstack.com/docs/app-automate/appium/integrations/appium-desktop "Use Appium Inspector for automated app testing | BrowserStack Docs"
[14]: https://appium.github.io/appium-inspector/latest/session-builder/capability-builder/ "Capability Builder Tab - Appium Inspector"

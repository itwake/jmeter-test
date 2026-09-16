# BrowserStack + Appium Inspector：Windows CMD 完整流程

本指南所有终端命令都在 **CMD（cmd.exe）** 中执行，不调用 PowerShell，不需要修改 ExecutionPolicy。Inspector 的安装和录制仍有图形界面操作。

这是自编辅助验证包，不是 BrowserStack 官方产品。包内的 Python 脚本用于连接、配置、录制代码回放；不包含你们 App 的真实元素定位。仅使用已授权的 App Automate 账号、测试 App 和测试数据，第一轮不要提交真实资金交易。

一条命令报错就先停止，解决后再继续。不要把整份文档一次性粘贴到终端。

## 1. 打开 CMD 与安装 Python

按 `Win + R`，输入下面内容后回车：

```cmd
cmd /d /v:off
```

`/d` 不执行 CMD AutoRun；`/v:off` 关闭延迟变量展开。[1]

先检查 Python：

```cmd
py -3.12 --version
```

已有可用 Python 可复用；这里沿用原包的 Python 3.12。没有该版本时，在企业允许的安装范围内执行：

```cmd
winget install --exact --id Python.Python.3.12 --source winget
```

安装后关闭并重新打开 CMD，再检查版本。`winget` 不可用时，使用公司软件中心或 Python 官方 Windows 安装程序；没有必要为此改用另一种 shell。[2]

没有 `py` 启动器、但已有适用 Python 时，先执行 `python --version` 确认版本与来源，再把后文创建虚拟环境、解压用的 `py -3.12` 换成该 Python 的命令或完整路径。

## 2. 解压 CMD 版辅助包

把聊天里的 `browserstack-inspector-starter-cmd.zip` 下载到 Downloads。使用独立目录，避免覆盖上一份包里已经改好的录制代码：

```cmd
if not exist "%USERPROFILE%\projects\bs-cmd" mkdir "%USERPROFILE%\projects\bs-cmd"
```

```cmd
py -3.12 -m zipfile -e "%USERPROFILE%\Downloads\browserstack-inspector-starter-cmd.zip" "%USERPROFILE%\projects\bs-cmd"
```

```cmd
cd /d "%USERPROFILE%\projects\bs-cmd\browserstack-inspector-starter"
```

下载位置不在默认 Downloads 时，修改 ZIP 路径。不要重新解压覆盖已编辑的同名文件，先备份。

## 3. 创建环境与安装依赖

```cmd
py -3.12 -m venv .venv
```

```cmd
call .venv\Scripts\activate.bat
```

```cmd
python --version
```

```cmd
where python
```

应优先指向当前项目的 `.venv\Scripts\python.exe`。CMD 对应的是 `activate.bat`，不需要运行其他 shell 的激活脚本。[3]

```cmd
python -m pip install --upgrade pip
```

```cmd
python -m pip install -r requirements.txt
```

```cmd
python -m pip freeze > requirements.lock.txt
```

也可以完全不激活，直接使用 `.venv\Scripts\python.exe` 替代后续的 `python`。[3]

## 4. 安装 Inspector

下载器沿用原包固定版本 2026.7.1，按平台和架构选择官方安装包，核对包内固定 SHA-256；它不自动执行安装程序。版本固定是为了复现，不是“自动跟随最新版”。[4]

```cmd
python download_inspector.py
```

看到 `SHA-256 OK` 后打开下载目录：

```cmd
explorer downloads
```

双击其中对应的 `Appium-Inspector-2026.7.1-win-*.exe` 完成安装，再从开始菜单启动 **Appium Inspector**。安装提示应按企业政策处理，不要关闭安全检测或跳过哈希验证。

## 5. 隐藏输入凭据，进入本次测试 CMD

本包增加了 `cmd_session.py`。它用 Python `getpass` 隐藏输入密钥，再启动一个继承凭据的 CMD 子会话；不通过临时 .bat 文件或永久环境变量保存秘密。[5]

准备同时测试登录时执行：

```cmd
python cmd_session.py --test
```

依次输入：

```text
BrowserStack Username:
BrowserStack Access Key (hidden):
Test Username:
Test Password (hidden):
```

这里只配置 BrowserStack、暂时没有测试 App 账号时，改用 `python cmd_session.py`，不带 `--test`。后续需要测试凭据时，结束云端会话和隧道，输入 `exit` 返回，再运行带 `--test` 的命令。

成功后，同一个终端会出现 `[BS-CMD]` 提示符。继续在这个提示符下执行后续命令。它继承了虚拟环境，无需重新激活。

注意：

- 脚本不会把秘密写入磁盘、自己的启动参数或注册表；进程环境中仍有明文秘密，不等于凭据保险库。
- 后文 `curl --user`、Local `--key` 仍通过进程参数传递 BrowserStack 凭据，可能被有权限的本机工具读取。严格企业环境应改接批准的凭据管理机制。
- 不要执行裸 `set`、`echo %BROWSERSTACK_ACCESS_KEY%`，也不要把凭据写入脚本、Git、录屏或日志。
- 凭据只存在于本次子会话及继承它的程序中。`exit` 返回父 CMD 后不再拥有新输入的这些变量；另开窗口也不会自动取得它们。[1]

## 6. 获取设备列表

先检查 curl：

```cmd
curl.exe --version
```

没有 curl.exe 时，使用企业批准的安装方式。然后执行官方设备列表 API：[6]

```cmd
curl.exe --fail --silent --show-error --user "%BROWSERSTACK_USERNAME%:%BROWSERSTACK_ACCESS_KEY%" "https://api-cloud.browserstack.com/app-automate/devices.json" --output devices.json
```

```cmd
python -m json.tool devices.json
```

返回 401/403 时检查凭据和 App Automate 权限。列表是支持的设备目录，不是实时空闲库存。

## 7. 上传 APK，选择 Android 机型

把路径换成真实 APK，保留外层双引号：

```cmd
set "APP_FILE=C:\builds\your-test-app.apk"
```

```cmd
dir "%APP_FILE%"
```

确认存在后上传：[7]

```cmd
curl.exe --fail --silent --show-error --user "%BROWSERSTACK_USERNAME%:%BROWSERSTACK_ACCESS_KEY%" -X POST "https://api-cloud.browserstack.com/app-automate/upload" -F "file=@%APP_FILE%" --output upload.android.json
```

```cmd
python -m json.tool upload.android.json
```

成功响应应有真实的 `app_url`，形式为 `bs://...`。

```cmd
python configure.py --platform android --app-file upload.android.json --out caps.android.json
```

输入设备序号，按回车。脚本把实际机型、系统版本和 App 标识写入配置，不写入用户名或密钥。

```cmd
set "CAPS=caps.android.json"
```

```cmd
type "%CAPS%"
```

已有 App Automate 的真实 `bs://` 标识时，可以跳过上传，改用下面的生成命令（二选一，不要重复创建同名文件）：

```cmd
python configure.py --platform android --app "bs://REPLACE_WITH_REAL_APP_ID" --out caps.android.json
```

`configure.py` 默认拒绝覆盖已有配置；重新选择时使用新的 `--out` 文件名。

## 8. 可选：内网后台使用 BrowserStack Local

后台完全可从公网访问时跳过。需要内网、VPN 或本机服务时，运行隧道的电脑必须确实能够访问后台，并已获企业授权。

```cmd
if not exist bin mkdir bin
```

```cmd
if not exist downloads mkdir downloads
```

```cmd
curl.exe --fail --location --show-error "https://local-downloads.browserstack.com/BrowserStackLocal-win32.zip" --output downloads\BrowserStackLocal.zip
```

```cmd
python -m zipfile -e downloads\BrowserStackLocal.zip bin
```

```cmd
dir bin\BrowserStackLocal.exe
```

生成当前隧道标识：

```cmd
set "BS_LOCAL_IDENTIFIER=efp-cmd-%RANDOM%-%RANDOM%"
```

启动隧道：[8]

```cmd
bin\BrowserStackLocal.exe --key "%BROWSERSTACK_ACCESS_KEY%" --local-identifier "%BS_LOCAL_IDENTIFIER%" --daemon start --log-file "%CD%\browserstack-local.log"
```

```cmd
type browserstack-local.log
```

确认日志报告连接成功，不要只检查进程是否存在。若受代理、VPN 或企业 CA 限制，应按网络政策配置，不要通过禁用 TLS 校验绕过。

生成匹配同一隧道标识的配置：

```cmd
python configure.py --platform android --app-file upload.android.json --local-id "%BS_LOCAL_IDENTIFIER%" --out caps.android.local.json
```

```cmd
set "CAPS=caps.android.local.json"
```

不要在没有成功隧道时仅把 `local` 改成 true；后续一律使用 `%CAPS%` 指向的这份文件。

## 9. 连通性检查

```cmd
python run.py --caps "%CAPS%" --mode smoke
```

本包脚本会打开一个云端手机会话，获取页面结构和截图，再关闭会话。预期输出 `SMOKE OK`、`Session closed.`。

```cmd
dir /s /b artifacts
```

检查 `start.png`、`start.xml` 等文件。SMOKE OK 只证明基础连接完成，不代表登录或交易正确。如果创建会话超时，先在 BrowserStack 检查是否已有活动会话，再决定是否重试。

## 10. 在 Inspector 界面连接与录制

在 Inspector 选择 `Select Cloud Providers → BrowserStack`，填写同一个 Username、Access Key，使用 Capabilities 的 JSON 编辑器。[9]

先将配置复制到剪贴板：

```cmd
clip < "%CAPS%"
```

`clip` 可以把文件或命令输出复制到 Windows 剪贴板。[10] 粘贴到 Inspector 后核对内容，再点 **Start Session**。不要填写 localhost，也不要在 Inspector 中粘贴 `%BROWSERSTACK_ACCESS_KEY%` 指望它自动展开。

连接后开启 **Toggle Recorder**，选择 Python，优先 **Tap By Element**，输入用 **Send Keys**。录制首次流程建议只到登录、跳过引导、进入外币页面及选择币种，不加入真实交易提交。

Recorder 里隐藏会话模板代码（boilerplate），复制动作主体。[11]

在 CMD 打开文本文件：

```cmd
notepad recorded.raw.py
```

在记事本中粘贴、以 UTF-8 保存。CMD 不需要提供“读取剪贴板”的额外命令，直接在记事本粘贴即可。先保存录制，再结束 Inspector Session，避免占用并发位。

## 11. 整理步骤并回放

```cmd
notepad recorded_steps.py
```

将真实操作主体粘入 `run(driver)`，缩进 4 空格；保留所需 import；删掉占位的 `NotImplementedError`；把 `READY = False` 改成 `READY = True`。不要复制 driver 创建和 driver.quit()。

密码不能原样留在代码中；把录制的固定账号、密码换成运行时引用：

```python
username_element.send_keys(os.environ["TEST_USERNAME"])
password_element.send_keys(os.environ["TEST_PASSWORD"])
```

这里 `username_element`、`password_element` 是说明用变量名，必须对应你真实录制中的元素。没有真实定位值时不要照抄假设值。

币种也要先修改实际步骤，使其读取参数，再在 CMD 设置参数：

```cmd
set "TARGET_CURRENCY=USD"
```

例如，在 Python 中用 `os.environ["TARGET_CURRENCY"]` 构建实际币种定位并验证最终选择。仅设置环境变量不会自动把录制的 USD 变成动态参数。

补充必要的显式等待、页面检查和业务断言后执行：

```cmd
python -m py_compile recorded_steps.py
```

```cmd
python run.py --caps "%CAPS%" --mode replay --steps recorded_steps.py
```

Python 会新建会话，不会沿用 Inspector 的人工登录态。后台账号的状态、余额、OTP 或设备绑定也不会因为新建会话自动重置。

无异常完成不等于业务验证通过；应检查实际币种、金额和交易结果。交易提交后结果不确定时不要盲目重放。

## 12. 换机型验证

公网场景：

```cmd
python configure.py --platform android --app-file upload.android.json --out caps.android-2.json
```

选择另一台同系统手机，保持步骤文件不变：

```cmd
python run.py --caps caps.android-2.json --mode replay --steps recorded_steps.py
```

内网场景改用匹配隧道的配置：

```cmd
python configure.py --platform android --app-file upload.android.json --local-id "%BS_LOCAL_IDENTIFIER%" --out caps.android-2.local.json
```

```cmd
python run.py --caps caps.android-2.local.json --mode replay --steps recorded_steps.py
```

第二台失败时检查定位、等待、键盘与滚动差异，不要立即复制一整份脚本。并发还需独立账号和后台数据隔离，不包含在这一版单会话验证器中。

## 13. iOS 差异

仍在同一个本机 Inspector 中连接 BrowserStack 的 iPhone，不在 Windows 本机运行 iOS 模拟器。[9]

上传适用于真机的测试 IPA：

```cmd
set "IOS_APP_FILE=C:\builds\your-test-app.ipa"
```

```cmd
dir "%IOS_APP_FILE%"
```

```cmd
curl.exe --fail --silent --show-error --user "%BROWSERSTACK_USERNAME%:%BROWSERSTACK_ACCESS_KEY%" -X POST "https://api-cloud.browserstack.com/app-automate/upload" -F "file=@%IOS_APP_FILE%" --output upload.ios.json
```

```cmd
python configure.py --platform ios --app-file upload.ios.json --out caps.ios.json
```

```cmd
python run.py --caps caps.ios.json --mode smoke
```

```cmd
clip < caps.ios.json
```

内网版本在生成配置时带上 `--local-id "%BS_LOCAL_IDENTIFIER%"` 并选一个新输出文件名，之后 smoke 和 Inspector 都使用该文件。

不要默认 Android 原始录制步骤可以直接运行于 iOS。先分别录制、审核定位与平台操作，再抽取共享业务流程。

## 14. 停止与清理

先结束 Inspector 的 Session，并确认 Python 输出 `Session closed.`。异常时到 BrowserStack 确认残留设备会话。

只在实际启动过 Local 时执行：

```cmd
bin\BrowserStackLocal.exe --key "%BROWSERSTACK_ACCESS_KEY%" --local-identifier "%BS_LOCAL_IDENTIFIER%" --daemon stop
```

清除当前子会话中的秘密：

```cmd
set "BROWSERSTACK_ACCESS_KEY="
```

```cmd
set "TEST_PASSWORD="
```

```cmd
exit
```

`exit` 返回运行 `cmd_session.py` 前的 CMD。退出不等于强制撤销其他已启动进程里的环境变量，也不自动关闭云端设备或后台隧道，务必先完成上述清理。

截图、XML、录制源码和云端日志可能含敏感信息，按你们的规范保存和清理。

## CMD 常见写法

| 操作 | CMD |
|---|---|
| 当前窗口设置非秘密参数 | `set "TARGET_CURRENCY=USD"` |
| 引用参数 | `%TARGET_CURRENCY%` |
| 进入目录并切换盘符 | `cd /d "C:\path with spaces"` |
| 查看文件 | `type caps.android.json` |
| 复制文件内容到剪贴板 | `clip < caps.android.json` |
| 激活环境 | `call .venv\Scripts\activate.bat` |
| 读取凭据 | 本包的 `python cmd_session.py --test` |

这里所有终端命令均写成单行，避开续行符差异。CMD 多行命令使用行末 `^`，其后不能有空格。普通 `set /p` 的输入可见，不适合声称“隐藏输入密码”。

## 验证边界

本包含可离线执行的配置生成与凭据准备单元测试：

```cmd
python -m unittest -v test_configure test_cmd_session
```

本次在 Linux 环境完成 16 项离线测试及 Python 语法检查。未在真实 Windows CMD 中端到端运行，也未使用你们的 BrowserStack 凭据验证云端连接或 App 业务流程。安装器与依赖沿用原包版本；未在本次重新实装下载的官方 GUI 安装器。

## 官方参考

[1] Microsoft CMD：`https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/cmd`

[2] WinGet install：`https://learn.microsoft.com/en-us/windows/package-manager/winget/install`

[3] Python venv：`https://docs.python.org/3/library/venv.html`

[4] Inspector 固定版本：`https://github.com/appium/appium-inspector/releases/tag/v2026.7.1`

[5] Python getpass：`https://docs.python.org/3/library/getpass.html`

[6] BrowserStack 设备 API：`https://www.browserstack.com/docs/app-automate/api-reference/appium/devices`

[7] BrowserStack App 上传 API：`https://www.browserstack.com/docs/app-automate/api-reference/appium/apps`

[8] BrowserStack Local 参数：`https://www.browserstack.com/docs/local-testing/binary-params`

[9] BrowserStack + Inspector：`https://www.browserstack.com/docs/app-automate/appium/integrations/appium-desktop`

[10] Microsoft clip：`https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/clip`

[11] Inspector Recorder：`https://appium.github.io/appium-inspector/latest/session-inspector/recorder/`

可以。下面故意**不用 BrowserStack SDK 或 BrowserStack CLI**，只使用：

```text
curl 上传 APK
        ↓
Appium Python Client 创建远程会话
        ↓ HTTPS
BrowserStack Appium Server
        ↓
BrowserStack Android 真机
```

这样最容易理解 Appium 和 BrowserStack 的边界：

* BrowserStack API：上传、管理 App。
* Appium：查找元素、点击、输入、截图。
* BrowserStack：提供远程 Appium Server 和真实手机。

你的 Ubuntu 上**不需要安装**本地 Appium Server、Node.js、Android SDK、ADB、模拟器或 KVM。

---

# 1. 准备 BrowserStack 账号

你需要一个具有 **App Automate** 权限的 BrowserStack 账号，并取得：

```text
Username
Access Key
```

它们可以在 BrowserStack 的账户资料或 App Automate 页面中找到。BrowserStack 官方入门文档也将这两个凭据列为创建远程 Appium 会话的前提。([BrowserStack][1])

不要把 Access Key 写进代码或提交到 Git。

---

# 2. 在 Ubuntu 安装 Python 环境

```bash
sudo apt update

sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  curl
```

创建项目：

```bash
mkdir appium-browserstack-demo
cd appium-browserstack-demo

python3 -m venv .venv
source .venv/bin/activate
```

激活成功后，终端一般会显示：

```text
(.venv) user@ubuntu:~/appium-browserstack-demo$
```

安装 Appium Python 客户端：

```bash
python -m pip install --upgrade pip
python -m pip install "Appium-Python-Client==5.3.1"
```

`5.3.1` 是 2026 年 4 月发布的当前版本；该版本系列要求 Python 3.9 或更高版本。Appium Python Client 会同时安装所需的 Selenium 依赖。([PyPI][2])

检查安装：

```bash
python -c "import appium; print('Appium Python Client 安装成功')"
```

---

# 3. 设置 BrowserStack 凭据

把下面的值替换成你账户中的真实值：

```bash
export BROWSERSTACK_USERNAME='你的_USERNAME'
export BROWSERSTACK_ACCESS_KEY='你的_ACCESS_KEY'
```

检查 Username：

```bash
echo "$BROWSERSTACK_USERNAME"
```

不要执行：

```bash
echo "$BROWSERSTACK_ACCESS_KEY"
```

避免密钥出现在屏幕共享、终端记录或 CI 日志中。

这些环境变量只在当前 Shell 中有效。关闭终端后需要重新设置。

---

# 4. 上传 BrowserStack 官方示例 App

为了避免你现在还没有自己的 APK，先使用 BrowserStack 官方提供的 Wikipedia 示例 App。

执行：

```bash
curl -u "${BROWSERSTACK_USERNAME}:${BROWSERSTACK_ACCESS_KEY}" \
  -X POST "https://api-cloud.browserstack.com/app-automate/upload" \
  -F "url=https://www.browserstack.com/app-automate/sample-apps/android/WikipediaSample.apk"
```

成功后会得到类似结果：

```json
{
  "app_url": "bs://f7c874f21852ba57957a3fdc33f47514288c4ba4"
}
```

你实际得到的 `bs://...` 会不同。这个值表示：

> BrowserStack 中已经上传完成的某个 App。

BrowserStack 要求先上传 APK/IPA，再把返回的 `app_url` 放进 Appium 的 `app` capability。([BrowserStack][3])

设置环境变量，注意替换为你刚刚得到的真实值：

```bash
export BROWSERSTACK_APP_ID='bs://你实际得到的值'
```

检查：

```bash
echo "$BROWSERSTACK_APP_ID"
```

不要直接复制文档里的示例 `bs://f7...`，必须使用你自己的上传响应。

---

# 5. 创建 Appium 测试脚本

创建文件：

```bash
nano test_browserstack.py
```

复制以下完整代码：

```python
import os
import time

from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


REMOTE_URL = "https://hub.browserstack.com/wd/hub"


def require_env(name: str) -> str:
    """读取必须存在的环境变量，并提供清晰的错误信息。"""
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"缺少环境变量 {name}，请先在终端中执行 export {name}=..."
        )

    return value


# Capabilities 用于告诉 BrowserStack：
# 1. 使用什么平台
# 2. 使用哪台手机
# 3. 安装哪个 App
# 4. 使用哪个 BrowserStack 账号
capabilities = {
    "platformName": "Android",

    # 使用 Android 的 UiAutomator2 自动化驱动
    "appium:automationName": "UiAutomator2",

    # 选择 BrowserStack 上的真机和 Android 版本
    "appium:deviceName": "Google Pixel 7",
    "appium:platformVersion": "13.0",

    # 使用刚才上传后得到的 bs://...
    "appium:app": require_env("BROWSERSTACK_APP_ID"),

    # BrowserStack 专属参数
    "bstack:options": {
        "userName": require_env("BROWSERSTACK_USERNAME"),
        "accessKey": require_env("BROWSERSTACK_ACCESS_KEY"),

        # 以下名称会显示在 BrowserStack Dashboard
        "projectName": "Appium BrowserStack 入门",
        "buildName": "first-build",
        "sessionName": "Wikipedia 搜索测试",

        # 在 Dashboard 中记录每条命令对应的截图
        "debug": True,
    },
}


options = UiAutomator2Options().load_capabilities(capabilities)

driver = None

try:
    # 连接 BrowserStack 的远程 Appium Server。
    # 这里不是连接 localhost，因此本机不需要启动 appium 命令。
    driver = webdriver.Remote(
        command_executor=REMOTE_URL,
        options=options,
    )

    print(f"已连接 BrowserStack，Session ID: {driver.session_id}")

    # 等待并查找 “Search Wikipedia” 元素
    search_button = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable(
            (
                AppiumBy.ACCESSIBILITY_ID,
                "Search Wikipedia",
            )
        )
    )

    # 模拟用户点击
    search_button.click()
    print("已点击 Search Wikipedia")

    # 查找搜索输入框
    search_input = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable(
            (
                AppiumBy.ID,
                "org.wikipedia.alpha:id/search_src_text",
            )
        )
    )

    # 模拟用户输入
    search_input.send_keys("BrowserStack")
    print("已输入 BrowserStack")

    # 等待搜索结果加载
    time.sleep(5)

    # 获取当前页面中的文本控件
    search_results = driver.find_elements(
        AppiumBy.CLASS_NAME,
        "android.widget.TextView",
    )

    if not search_results:
        raise AssertionError("页面中没有找到任何搜索结果控件")

    # 从远程真机截图，并保存到当前 Ubuntu 目录
    driver.save_screenshot("browserstack-result.png")

    print(f"页面中找到 {len(search_results)} 个文本控件")
    print("测试成功：已完成连接、点击、输入和截图")

finally:
    # 无论测试成功还是失败，都必须释放 BrowserStack 真机
    if driver is not None:
        driver.quit()
        print("BrowserStack Session 已关闭")
```

保存并退出 `nano`：

```text
Ctrl + O
Enter
Ctrl + X
```

其中使用的 `Search Wikipedia` 和 `org.wikipedia.alpha:id/search_src_text` 是 BrowserStack 官方 Python 示例采用的元素定位方式。([GitHub][4])

---

# 6. 运行测试

确认当前目录中有文件：

```bash
ls
```

应该看到：

```text
test_browserstack.py
```

运行：

```bash
python test_browserstack.py
```

正常情况下会看到类似输出：

```text
已连接 BrowserStack，Session ID: abcdef123456
已点击 Search Wikipedia
已输入 BrowserStack
页面中找到 20 个文本控件
测试成功：已完成连接、点击、输入和截图
BrowserStack Session 已关闭
```

当前目录中还会产生：

```text
browserstack-result.png
```

检查截图：

```bash
file browserstack-result.png
```

在有桌面环境的 Ubuntu 中可以打开：

```bash
xdg-open browserstack-result.png
```

即使 Ubuntu 或 K8s 完全没有 UI，截图文件仍然可以正常生成。

---

# 7. 在 BrowserStack 页面查看执行过程

打开 BrowserStack 的 **App Automate Dashboard**，应该能看到：

```text
Project: Appium BrowserStack 入门
Build: first-build
Session: Wikipedia 搜索测试
```

进入 Session 后，可以看到真实手机上的执行视频、Appium 命令、截图和设备日志。BrowserStack 官方入门流程同样要求在 App Automate Dashboard 中查看测试结果。([BrowserStack][1])

---

# 8. 这段代码实际上做了什么

## 创建一台远程手机会话

```python
driver = webdriver.Remote(
    command_executor="https://hub.browserstack.com/wd/hub",
    options=options,
)
```

这一步相当于向 BrowserStack 发出请求：

```text
请分配一台 Google Pixel 7
Android 13
安装指定的 APK
启动 Appium UiAutomator2
返回一个 Session ID
```

BrowserStack 返回 Session 后，`driver` 就代表那台远程手机。

## 查找控件

```python
driver.find_element(
    AppiumBy.ACCESSIBILITY_ID,
    "Search Wikipedia",
)
```

相当于：

```text
在手机当前页面中，查找 accessibility id 为
Search Wikipedia 的控件
```

## 模拟点击

```python
search_button.click()
```

这条 Appium 命令经 BrowserStack 发送给真实手机。

## 模拟输入

```python
search_input.send_keys("BrowserStack")
```

相当于在真实手机的输入框里输入文本。

## 截图

```python
driver.save_screenshot("browserstack-result.png")
```

截图发生在 BrowserStack 手机上，图片通过 Appium 返回到 Ubuntu。

所以，在 UI 操作层面，BrowserStack 没有替换 Appium：

```text
find_element
click
send_keys
get_attribute
page_source
swipe
W3C Actions
截图
切换 WebView
```

这些仍然是标准 Appium 命令。

---

# 9. 换成你自己的 APK

假设你的 APK 位于：

```text
/home/user/app-debug.apk
```

上传：

```bash
curl -u "${BROWSERSTACK_USERNAME}:${BROWSERSTACK_ACCESS_KEY}" \
  -X POST "https://api-cloud.browserstack.com/app-automate/upload" \
  -F "file=@/home/user/app-debug.apk"
```

得到：

```json
{
  "app_url": "bs://新的值"
}
```

更新环境变量：

```bash
export BROWSERSTACK_APP_ID='bs://新的值'
```

测试脚本中的这些定位器也要换成你自己 App 的控件：

```python
AppiumBy.ACCESSIBILITY_ID, "Search Wikipedia"
```

以及：

```python
AppiumBy.ID, "org.wikipedia.alpha:id/search_src_text"
```

例如你的登录按钮 resource-id 是：

```text
com.example.myapp:id/login_button
```

可以写成：

```python
login_button = WebDriverWait(driver, 30).until(
    EC.element_to_be_clickable(
        (
            AppiumBy.ID,
            "com.example.myapp:id/login_button",
        )
    )
)

login_button.click()
```

---

# 10. 常见错误

### `401 Unauthorized`

通常是 Username 或 Access Key 错误：

```bash
echo "$BROWSERSTACK_USERNAME"
```

重新设置：

```bash
export BROWSERSTACK_USERNAME='正确值'
export BROWSERSTACK_ACCESS_KEY='正确值'
```

### `Invalid app capability`

检查：

```bash
echo "$BROWSERSTACK_APP_ID"
```

它必须类似：

```text
bs://xxxxxxxxxxxxxxxx
```

并且必须是你上传 App 后实际返回的值。

### `No matching device`

说明当前账号或设备池中没有：

```text
Google Pixel 7 / Android 13.0
```

把脚本中的两项替换为 BrowserStack Dashboard 中可选的设备组合：

```python
"appium:deviceName": "Samsung Galaxy S22",
"appium:platformVersion": "12.0",
```

### `TimeoutException: Search Wikipedia`

通常表示：

* 上传的不是 BrowserStack Wikipedia 示例 App；
* App 启动失败；
* 页面加载速度较慢；
* 示例 App 的 UI 发生了变化。

先在 BrowserStack Session 中查看视频和 Appium 日志。

### `ModuleNotFoundError: No module named 'appium'`

重新激活虚拟环境：

```bash
cd appium-browserstack-demo
source .venv/bin/activate

python test_browserstack.py
```

---

# 11. 放进 K8s 时需要什么

本地跑通后，放进 K8s 时测试代码基本不变。Pod 只需要：

```text
Python
Appium-Python-Client
测试脚本
三个环境变量
能够访问 BrowserStack 的 HTTPS 出站网络
```

三个环境变量是：

```text
BROWSERSTACK_USERNAME
BROWSERSTACK_ACCESS_KEY
BROWSERSTACK_APP_ID
```

不需要：

```text
/dev/kvm
privileged
Android Emulator
ADB
Android SDK
本地 Appium Server
```

只有当你的 App 需要访问 K8s 内部域名、内网测试服务或 VPN 后端时，才需要额外建立 BrowserStack Local 隧道。([BrowserStack][5])

这套示例跑通后，你已经完成了一条完整链路：

```text
Ubuntu Python 程序
→ Appium Client
→ BrowserStack Appium Server
→ BrowserStack Android 真机
→ 查找控件
→ 点击
→ 输入
→ 截图
→ 释放设备
```

[1]: https://www.browserstack.com/docs/app-automate/appium/getting-started/python "Run Appium tests using Python | BrowserStack Docs"
[2]: https://pypi.org/project/Appium-Python-Client/?utm_source=chatgpt.com "Appium-Python-Client"
[3]: https://www.browserstack.com/docs/app-automate/appium/upload-app-using-public-url "Upload app using public URL to test on Appium | BrowserStack Docs"
[4]: https://raw.githubusercontent.com/browserstack/python-appium-app-browserstack/master/android/browserstack_sample.py "raw.githubusercontent.com"
[5]: https://www.browserstack.com/docs/app-automate/appium/getting-started/python/local-testing?utm_source=chatgpt.com "Enable Local Testing for Python Appium SDK Tests on App ..."

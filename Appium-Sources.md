可以，而且 **Android 和 iOS 都能获取**。但在 BrowserStack 上不再直接运行：

```bash
adb exec-out uiautomator dump /dev/tty
```

而是连接 BrowserStack 的远程 Appium Session，然后调用：

```python
source = driver.page_source
```

它对应标准 Appium/WebDriver 接口：

```http
GET /session/{sessionId}/source
```

返回当前页面的 XML 或 HTML 页面树。([Appium][1])

## 对应关系

```text
本地 Android
adb exec-out uiautomator dump
        ↓
Android UI XML

BrowserStack Android
Appium driver.page_source
        ↓
UiAutomator2 生成的 Android UI XML

BrowserStack iOS
Appium driver.page_source
        ↓
XCUITest / WebDriverAgent 生成的 iOS UI XML
```

BrowserStack 支持在其真实 Android 和 iOS 设备上运行 Appium，所以 `Get Page Source`、元素查找、点击和输入等标准 Appium 命令都可以通过远程 Session 执行。([BrowserStack][2])

## 最简单的代码

在前面创建 BrowserStack Driver 的代码之后：

```python
driver = webdriver.Remote(
    command_executor="https://hub.browserstack.com/wd/hub",
    options=options,
)
```

加入：

```python
from pathlib import Path

source = driver.page_source

Path("source_aos.xml").write_text(
    source,
    encoding="utf-8",
)

print(source[:2000])
print("页面 Source 已保存到 source_aos.xml")
```

完整核心流程是：

```python
from pathlib import Path

from appium import webdriver
from appium.options.android import UiAutomator2Options


capabilities = {
    "platformName": "Android",
    "appium:automationName": "UiAutomator2",
    "appium:deviceName": "Google Pixel 7",
    "appium:platformVersion": "13.0",
    "appium:app": "bs://你的APP_ID",

    "bstack:options": {
        "userName": "你的_BROWSERSTACK_USERNAME",
        "accessKey": "你的_BROWSERSTACK_ACCESS_KEY",
        "projectName": "获取页面 Source",
        "buildName": "source-demo",
        "sessionName": "dump Android source",
    },
}

options = UiAutomator2Options().load_capabilities(capabilities)

driver = webdriver.Remote(
    command_executor="https://hub.browserstack.com/wd/hub",
    options=options,
)

try:
    # 获取当前设备页面的 UI hierarchy
    source = driver.page_source

    # 保存到运行脚本的 Ubuntu/K8s 容器中
    Path("source_aos.xml").write_text(
        source,
        encoding="utf-8",
    )

    print(source[:2000])
    print("获取成功：source_aos.xml")

finally:
    driver.quit()
```

运行结束后，文件保存在执行脚本的机器或 K8s 容器中：

```text
source_aos.xml
```

不是保存在 BrowserStack 手机上。

## iOS 的代码是否不同

获取 Source 这一行完全相同：

```python
source = driver.page_source
```

只是创建 Session 时改成 iOS：

```python
from appium.options.ios import XCUITestOptions

capabilities = {
    "platformName": "iOS",
    "appium:automationName": "XCUITest",
    "appium:deviceName": "iPhone 15",
    "appium:platformVersion": "17",
    "appium:app": "bs://你的_IOS_APP_ID",

    "bstack:options": {
        "userName": "你的_USERNAME",
        "accessKey": "你的_ACCESS_KEY",
    },
}

options = XCUITestOptions().load_capabilities(capabilities)
```

保存时可以写：

```python
Path("source_ios.xml").write_text(
    driver.page_source,
    encoding="utf-8",
)
```

iOS Source 中通常会看到类似：

```xml
<XCUIElementTypeApplication ...>
    <XCUIElementTypeWindow ...>
        <XCUIElementTypeButton
            name="Login"
            label="Login"
            enabled="true"
            visible="true"
            x="20"
            y="500"
            width="300"
            height="50"/>
    </XCUIElementTypeWindow>
</XCUIElementTypeApplication>
```

Android 则可能包含：

```xml
<android.widget.Button
    text="登录"
    resource-id="com.example:id/login_button"
    content-desc="登录"
    clickable="true"
    enabled="true"
    bounds="[20,500][320,550]" />
```

具体节点格式和属性会随 Appium Driver、平台版本和应用实现变化。

## 获取 Source 后定位元素

假设 XML 中有：

```xml
<android.widget.Button
    resource-id="com.example:id/login_button"
    content-desc="登录"
    text="登录"/>
```

优先使用 `resource-id`：

```python
from appium.webdriver.common.appiumby import AppiumBy

login_button = driver.find_element(
    AppiumBy.ID,
    "com.example:id/login_button",
)

login_button.click()
```

或者使用 `content-desc`：

```python
login_button = driver.find_element(
    AppiumBy.ACCESSIBILITY_ID,
    "登录",
)

login_button.click()
```

最后才考虑 XPath：

```python
login_button = driver.find_element(
    AppiumBy.XPATH,
    '//android.widget.Button[@text="登录"]',
)
```

推荐顺序一般是：

```text
Accessibility ID
    ↓
Resource ID
    ↓
Android UIAutomator / iOS Predicate
    ↓
XPath
```

XPath 基于完整页面树，容易受页面层级变化影响，而且通常比 ID 定位慢。

## 解析 XML

也可以像处理 `uiautomator dump` 一样解析：

```python
import xml.etree.ElementTree as ET
from pathlib import Path


source = driver.page_source
Path("source_aos.xml").write_text(source, encoding="utf-8")

root = ET.fromstring(source)

for node in root.iter():
    attrs = node.attrib

    resource_id = attrs.get("resource-id", "")
    content_desc = attrs.get("content-desc", "")
    text = attrs.get("text", "")
    name = attrs.get("name", "")
    label = attrs.get("label", "")
    bounds = attrs.get("bounds", "")

    if resource_id or content_desc or text or name or label:
        print(
            {
                "tag": node.tag,
                "resource_id": resource_id,
                "content_desc": content_desc,
                "text": text,
                "name": name,
                "label": label,
                "bounds": bounds,
            }
        )
```

这段代码同时兼容 Android 和 iOS 的常见属性，但不同平台实际拥有的属性不同。

## 与 `uiautomator dump` 的重要差异

### 1. 输出不会保证完全一致

下面两种方式都表示 UI hierarchy：

```bash
adb exec-out uiautomator dump /dev/tty
```

以及：

```python
driver.page_source
```

但 XML 的：

* 节点标签
* 属性名称
* 属性数量
* 不可见节点处理方式
* 系统窗口处理方式
* 节点顺序

都可能不同。因此不要要求两份 XML 按字节完全一致。应当以稳定属性，例如 `resource-id`、`content-desc`、`name`、`label` 作为定位依据。

### 2. BrowserStack 通常不能执行任意 ADB Shell

BrowserStack 出于安全原因没有为 Appium 开启任意 Shell 执行权限，因此不能依赖：

```python
driver.execute_script(
    "mobile: shell",
    {
        "command": "uiautomator",
        "args": ["dump", "/dev/tty"],
    },
)
```

BrowserStack 只对部分明确支持的 ADB 用例提供定制能力；获取页面树应使用标准的 `driver.page_source`。([BrowserStack][3])

### 3. 必须存在活动中的 Appium Session

正确顺序是：

```text
创建 BrowserStack Appium Session
        ↓
等待 App 启动
        ↓
调用 driver.page_source
        ↓
保存 XML
        ↓
查找、点击元素
        ↓
driver.quit()
```

`driver.quit()` 后设备 Session 已释放，不能再读取该页面的 Source。因此需要在 Session 结束前保存 XML。

## Hybrid App 要注意 Context

对于原生页面：

```python
print(driver.current_context)
```

应该是：

```text
NATIVE_APP
```

此时：

```python
driver.page_source
```

返回原生 Android/iOS XML。

如果切换到 WebView：

```python
print(driver.contexts)

driver.switch_to.context("WEBVIEW_com.example")
source = driver.page_source
```

这时返回的通常是 WebView 中的 HTML DOM，而不是原生 UI XML。

可以显式切回：

```python
driver.switch_to.context("NATIVE_APP")
```

## Source 中可能缺少元素

`page_source` 不是对屏幕像素进行识别，它读取的是 Android/iOS 自动化框架能够暴露的元素树。因此以下内容可能不完整：

* Canvas 自绘控件
* OpenGL、游戏画面
* `SurfaceView`
* 没有配置语义或无障碍信息的自定义控件
* 部分 Flutter、自绘 React Native 控件
* 不可见或被过滤的元素
* iOS 中嵌套过深的元素

如果 BrowserStack Android Source 中缺少元素，可以在创建 Session 时尝试：

```python
capabilities = {
    # 其他 capabilities...

    "appium:disableSuppressAccessibilityService": False,
    "appium:settings[allowInvisibleElements]": True,
}
```

BrowserStack 文档说明，前者可避免抑制 Accessibility Service，后者可以把 `displayed=false` 的元素加入 XML Source。([BrowserStack][4])

不过不建议默认长期打开 `allowInvisibleElements`，因为它可能导致：

* Source 节点显著增多
* 获取 Source 变慢
* 定位到屏幕上实际不可交互的控件

iOS 中如果元素嵌套特别深，Apple XCTest 可能无法把它包含在标准 Source 中；BrowserStack另外提供了 `appium:bstackPageSource` 的网格采样能力来处理此类情况。([BrowserStack][5])

## 可以直接可视化查看

除了代码保存 XML，还可以使用 **Appium Inspector 连接 BrowserStack**。它会同时展示：

```text
左侧：手机截图
右侧：页面 Source 树
选中节点：ID、文本、名称、坐标等属性
```

BrowserStack 官方支持 Appium Inspector 连接其 Android 和 iOS 真机，适合第一次分析页面以及生成定位器。([BrowserStack][6])

## 最终结论

你的原有流程可以直接迁移：

```text
原来的方案
ADB → uiautomator dump → XML → 解析 → 定位 → 操作

BrowserStack 方案
远程 Appium Session
→ driver.page_source
→ XML
→ 解析或生成定位器
→ driver.find_element
→ 点击/输入
```

最核心的替换只有这一处：

```bash
adb exec-out uiautomator dump /dev/tty > source_aos.xml
```

替换为：

```python
from pathlib import Path

Path("source_aos.xml").write_text(
    driver.page_source,
    encoding="utf-8",
)
```

**所以答案是：可以，而且这是 BrowserStack 上获取 Android/iOS 页面元素树的标准方式。**

[1]: https://appium.io/docs/en/3.3/reference/api/webdriver/ "WebDriver Protocol - Appium Documentation"
[2]: https://www.browserstack.com/docs/app-automate/api-reference/appium/overview "Test apps using Appium via App Automate API | BrowserStack Docs"
[3]: https://www.browserstack.com/docs/app-automate/appium/advanced-features/adb-commands?utm_source=chatgpt.com "Test using adb - App Automate"
[4]: https://www.browserstack.com/docs/app-automate/appium/sdk-faqs/generic/appium-inspector "Identify elements using Appium Inspector | BrowserStack Docs"
[5]: https://www.browserstack.com/docs/app-automate/appium/troubleshooting/deeply-nested-elements-ios "Find deeply nested elements in iOS | BrowserStack Docs"
[6]: https://www.browserstack.com/docs/app-automate/appium/integrations/appium-desktop "Use Appium inspector for automated app testing | BrowserStack Docs"

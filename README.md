# 星布谷地 · 格连商店助手

一个只围绕**格连商店**这一页的小工具：自动走到店门口、进店、把当天刷新的家具
连拍下来拼成一张总览图。附带「桌面分身」——把游戏放到 Windows 第二个桌面上跑，
主屏幕用一个小窗看着它，**小窗里点击不会动你的真实鼠标**。

不读写游戏内存、不注入进程、不装键鼠钩子：只用系统截图接口看画面，用模拟输入走路。

![界面](docs/screenshot.png)

## 功能（就这些）

| 区域            | 功能                                                         |
| --------------- | ------------------------------------------------------------ |
| 桌面分身        | 创建分身并启动游戏；打开分身小窗；结束分身                   |
| 小窗：寻路+截图 | 一键：走到门口 → 按 F 进店 → 等画面亮起 → 转圈连拍 → 拼 `sheet.png` |
| 分开执行        | 启用寻路（只走路）/ 启用截图（假设已在店里）                 |
| 一键停止        | 停下寻路与截图（分身和游戏不关）                             |
| 参数            | 输出标签、截图张数、每张转角、时间预算、最大步数             |
| 窗口            | 只认 `PetitPlanet.exe`；找不到可手动选窗口                   |

**没有**：自动拾取、自动钓鱼、自动演奏、好友列表、跨桥、自动领奖。

## 工作原理

1. **蓝点定位**：导航点是固定颜色的 UI 叠加层（青环 + 暗环 + 米色圆心），
   颜色掩膜 + 连通域 + 三点半径校验，像素级定位。
2. **朝向**：蓝点偏离中心超过阈值才转向（限幅小步），走路不会被转向打断。
3. **遮挡**：看不到蓝点就继续向前走几步，再小幅转身搜索，**绝不重开地图**
   （重开会把蓝点关掉）。
4. **卡住判定**：以 HUD「N 米」为准；数字不再变小且蓝点位置也没动才判定卡住。
5. **脱困**：跳 → 后退 → 换边 → 45°/100° 转向，统计成功率并优先用有效的组合。
6. **进店拍照**：检测到「F 进入格连商店」→ 按 F → 等画面由黑转亮（亮度阈值）
   → 向里走一点避开门口 → 拍 `shots` 张并每张转 `turn` 像素 → 拼 2 列总览图。
7. **分身**：`CreateDesktopW` 新建桌面 + `CreateProcess(lpDesktop)` 起游戏；
   截图用 `PrintWindow`（非活跃桌面唯一可行）；小窗里的点击用 `PostMessage`。

## 环境要求

- Windows 10 / 11 x64
- Python 3.10+（含 `numpy`）
- 游戏窗口化 / 无边框，客户区 **1920×1080**
- 创建分身需要**管理员权限**（游戏是高完整性进程）

## 快速开始

```powershell
py -3 -m pip install numpy
```

双击 `启动格连商店.bat`（会自动挑一个装了 numpy 的 Python 3）。

流程建议：

1. 先不开分身：在游戏里走到商店附近，点「启用寻路」验证识别与走路正常；
2. 再创建分身：点「创建分身并启动游戏」→ 等窗口出现（自动打开小窗）→
   点「小窗：寻路+截图」；
3. 想停就点「一键停止」。

命令行（不用界面）：

```powershell
py -3 run_shop_console.py --phase both --shots 6 --turn 1500 --label today
py -3 run_shop_console.py --phase nav      # 只寻路
py -3 run_shop_console.py --phase shop     # 只进店拍照
```

## 输出

```
reports/runs/<标签>/
├── inside_01.png … inside_06.png   进店后的家具截图
├── sheet.png                       2 列总览图（发群/对比用）
├── walk_only_end.png               寻路结束时的画面
└── STUCK_stop.png                  卡住放弃时的画面（如有）
```

`reports/stuck_stats.json` 记录每种脱困组合的成功率。

## 分身小窗

- 画面：`PrintWindow` 抓另一个桌面上的窗口，缩放到小窗大小，约 12 fps；
- 点击：小窗内的点击/拖拽翻译成 `WM_MOUSEMOVE / WM_LBUTTONDOWN/UP` 发给游戏窗口，
  **真实鼠标和键盘不动**，你可以在外面正常干别的事；
- 关闭小窗会自动解除绑定，输入恢复正常。

## 已知限制

- 客户区不是 1920×1080 时，居中与转向判断会偏；
- 游戏完全最小化时截图拿不到画面（被遮挡但没最小化可以，走 `PrintWindow`）；
- 分身里游戏能否持续渲染取决于游戏引擎，请先切过去看一眼（本机制在记事本/Edge 上已验证）；
- 窄桥、贴墙等极端地形可能耗时较长或失败；
- 不同画质下 UI 颜色可能有细微差异，找不到蓝点时先截图确认。

## 免责声明（重要）

本项目**仅供学习与研究**，不得用于商业代练/代肝或违反游戏用户协议的行为。

- 使用第三方自动化工具通常违反游戏用户协议，**存在封号风险**，风险自负；
- 不读写内存、不注入，**不代表游戏方认可**；
- 作者不对账号、数据或财产损失负责。

完整条款见 [DISCLAIMER.md](DISCLAIMER.md)。

## 许可与致谢

GPL-3.0（见 [LICENSE](LICENSE)）。算法参考 [BetterPetitPlanet](https://github.com/Sonder9999/better-petit-planet)（GPL-3.0），
未复制其代码；`assets/templates/` 中少量 UI 图标仅用于识别定位。详见 [NOTICE.md](NOTICE.md)。

## 目录结构

```
pnav-shop/
├── pnavshop/
│   ├── walker.py           寻路 + 进店截图主流程
│   ├── gui.py              界面
│   ├── preview.py          分身小窗
│   ├── virtual_desktop.py  第二桌面
│   ├── gamepath.py         自动查找游戏安装位置
│   ├── detect.py distance.py uid.py escape.py avoid.py
│   ├── pose.py gridmap.py plan.py nav.py
│   ├── pngio.py imgproc.py win32.py
│   └── spec.py
├── assets/templates/       识别用的小图标与数字模板
├── tests/                  62 个离线测试
├── docs/                   使用说明、发布说明、视频文案、截图
├── run_shop.pyw / run_shop_console.py
└── 启动格连商店.bat
```

## 开发

```powershell
py -3 -m unittest discover -s tests
```

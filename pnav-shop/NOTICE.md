# NOTICE / 第三方说明

## 算法参考

参考了 [BetterPetitPlanet](https://github.com/Sonder9999/better-petit-planet)（GPL-3.0）的
公开实现与文档：

- 「N 米」距离文本的读取思路（数字模板 + 可信度过滤）；
- 脱困动作的枚举顺序和成功率统计；
- 蓝色导航点的三层颜色判据。

**未复制其源代码**；本项目为独立实现（Python + numpy，无 OpenCV 依赖），按 GPL-3.0 分发。

## 游戏素材

`assets/templates/` 的图标与数字模板截取自本机游戏画面，仅用于识别定位。
版权归《星布谷地》权利方所有；本项目不主张权利，权利方要求即移除。

## 依赖

- Python 标准库（含 `tkinter`）
- [numpy](https://numpy.org/)（BSD-3-Clause）

## 许可

GPL-3.0-or-later，见 `LICENSE`。

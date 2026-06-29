# NeuroVis · CNN 可视化

交互式卷积神经网络教学 App。在浏览器里跑**真实**的 MNIST CNN 前向传播，逐层、逐通道、逐像素地把卷积、ReLU、池化、Softmax 讲清楚。移动端优先，可装进 iOS WebView 当原生 App 用。

> 灵感来自 [CNN Explainer](https://poloclub.github.io/cnn-explainer/)（arXiv:2004.15004），但权重、前向计算、可视化与交互全部重写，并针对触屏与中文教学重做。

## 特性

- **真实计算，非动画占位**：每个特征图都是纯 JS 实时卷积算出来的（`src/cnn/engine.js`），权重来自预训练的 MNIST 模型。
- **可调网络结构**：卷积层数（1–2）、卷积核大小（3/5/7）、通道数（1–10）、激活函数（ReLU / Sigmoid / Tanh），改设置即时重算。
- **逐层详情视图**：
  - 第一层卷积 — 单通道弹性视图：卷积核在输入上滑动，逐格生成中间结果。
  - 第二层卷积 — 多通道堆叠视图：多个输入通道各自卷积 → 求和 → 加 bias → 输出，输入/中间结果/输出**逐格同步动画**生成。
  - 卷积公式视图、ReLU、最大池化、Softmax 各自的交互讲解。
- **中 / 英双语**、多种配色（蓝橙 / viridis / 灰度）。
- **移动端自适应**：手机尺寸自动切换全屏 App 模式；桌面端按内容缩放居中。

## 本地运行

需要一个静态服务器（页面会 `fetch` 权重和样本 JSON，直接双击 `index.html` 用 `file://` 打开会被浏览器拦截）。

```bash
python3 -m http.server 8123
# 浏览器打开 http://127.0.0.1:8123/index.html
# 手机版预览: http://127.0.0.1:8123/index.html?app=1
```

### iOS 模拟器预览

仓库内含一个 WKWebView 壳工程，双击 `打开iPhone预览.command`（或运行 `scripts/run-ios-webview.sh`）会起本地服务器并在 iOS 模拟器里加载。需要 Xcode + 已启动的模拟器。

## 技术栈

- **React 18**（UMD）+ **Babel standalone** —— 浏览器内即时编译 JSX，**无构建步骤**，改完刷新即生效。
- **Canvas 2D** 做所有特征图 / 动画渲染。
- 前向传播为手写纯 JS，无任何 ML 框架依赖。
- 预训练权重按 `层数 × 核大小 × 激活函数` 预生成多套（`assets/mnist_weights_l{L}_k{K}_{fn}.json`）。

## 目录结构

```
index.html                 入口 + 视口缩放逻辑 + 脚本加载顺序
src/
  app.jsx                  根组件、路由、全局设置状态
  data.js                  网络层定义、按设置重算所有类别数据
  cnn/
    engine.js              真实 CNN 前向传播 + 权重加载
    inputs.js              MNIST 样本加载
  screens/
    screens-l1.jsx         主可视化（逐层 column）
    detail-views.jsx       各层详情弹窗（卷积/ReLU/池化/Softmax）
    screens-rest.jsx       图片选择、设置、教程等屏
  components/              iOS 外框、tweaks 面板、通用组件
  styles/tokens.css        设计 token
assets/                    预训练权重、MNIST 样本、模型
ios/                       iOS WKWebView 壳工程
scripts/                   iOS 模拟器预览脚本
```

## 说明

- `卷积第二层可视化-精简 (1).html` 是第二层卷积堆叠视图的设计稿，已实现进 `src/screens/detail-views.jsx`，保留作参考。
- `.claude/settings.local.json` 为本地工具权限设置，本地改动不随提交推送。

// CNN 数据层：Tiny VGG 9 层，10 个类别。
// 真实激活由 src/cnn/engine.js 跑 TF.js 前向产出，
// 本文件负责把张量整理成 UI 期望的形状: NETWORK_DATA[catIndex][layerId][channelIndex][y][x]

window.CATEGORIES = [
  { id: 'd0', zh: '0', en: 'Zero',  digit: '0', emoji: '0', hue: 0 },
  { id: 'd1', zh: '1', en: 'One',   digit: '1', emoji: '1', hue: 0 },
  { id: 'd2', zh: '2', en: 'Two',   digit: '2', emoji: '2', hue: 0 },
  { id: 'd3', zh: '3', en: 'Three', digit: '3', emoji: '3', hue: 0 },
  { id: 'd4', zh: '4', en: 'Four',  digit: '4', emoji: '4', hue: 0 },
  { id: 'd5', zh: '5', en: 'Five',  digit: '5', emoji: '5', hue: 0 },
  { id: 'd6', zh: '6', en: 'Six',   digit: '6', emoji: '6', hue: 0 },
  { id: 'd7', zh: '7', en: 'Seven', digit: '7', emoji: '7', hue: 0 },
  { id: 'd8', zh: '8', en: 'Eight', digit: '8', emoji: '8', hue: 0 },
  { id: 'd9', zh: '9', en: 'Nine',  digit: '9', emoji: '9', hue: 0 },
];

// Tiny VGG 架构
// 根据 (numLayers, kernelSize) 生成 LAYERS。所有数据驱动 UI 都通过 window.LAYERS 取,
// 调 recomputeAll 时会重新赋值。
window.buildLayers = function (L, K) {
  const layers = [
    { id: 'input', name: 'input', zh: '输入图像', ch: 1, size: 28, kind: 'input' },
  ];
  let cur = 28;
  for (let i = 1; i <= L; i++) {
    cur = cur - K + 1;
    layers.push({ id: `conv_1_${i}`, name: 'conv', zh: '卷积', ch: 10, size: cur, kind: 'conv' });
    layers.push({ id: `relu_1_${i}`, name: 'relu', zh: 'ReLU', ch: 10, size: cur, kind: 'relu' });
  }
  cur = Math.floor(cur / 2);
  layers.push({ id: 'pool_1', name: 'max_pool', zh: '最大池化', ch: 10, size: cur, kind: 'pool' });
  layers.push({ id: 'output', name: 'softmax', zh: 'Softmax', ch: 10, size: 1, kind: 'softmax' });
  return layers;
};

window.LAYERS = window.buildLayers(1, 3); // 默认 1 层 conv, K=3

// ─── 下采样: H×W 2D 数组 → target×target (区块平均) ─────────────────────
// UI 的 Heatmap 渲染细到 16x16 就够了，大尺寸激活图下采样以保性能
const RENDER_SIZE = 16;
function downsample(map2d, target) {
  const n = map2d.length;
  if (n <= target) return map2d.map(r => r.slice());
  const out = [];
  for (let y = 0; y < target; y++) {
    const row = [];
    const y0 = Math.floor(y * n / target);
    const y1 = Math.max(y0 + 1, Math.floor((y + 1) * n / target));
    for (let x = 0; x < target; x++) {
      const x0 = Math.floor(x * n / target);
      const x1 = Math.max(x0 + 1, Math.floor((x + 1) * n / target));
      let sum = 0, count = 0;
      for (let i = y0; i < y1; i++) {
        for (let j = x0; j < x1; j++) {
          sum += map2d[i][j];
          count++;
        }
      }
      row.push(count > 0 ? sum / count : 0);
    }
    out.push(row);
  }
  return out;
}

// 从 HxWxC 张量抽出第 c 个 channel，得到 HxW
function extractChannel(tensor3d, c) {
  const H = tensor3d.length;
  const out = new Array(H);
  for (let y = 0; y < H; y++) {
    const W = tensor3d[y].length;
    const row = new Array(W);
    for (let x = 0; x < W; x++) row[x] = tensor3d[y][x][c];
    out[y] = row;
  }
  return out;
}

// 按层归一化到 [-1, 1]: 每层 channels 共享一个 abs_max，保留 channel 之间的相对强度
function normalizeChannels(channels) {
  let max = 0;
  for (const m of channels) for (const r of m) for (const v of r) {
    const av = Math.abs(v);
    if (av > max) max = av;
  }
  if (max === 0) return;
  for (const m of channels) for (const r of m) for (let x = 0; x < r.length; x++) r[x] /= max;
}

// ─── 单类别真实激活 ──────────────────────────────────────────────────
// N: 限制 conv 输出 channel 数 (1..10). engine 会 mask 多余 channel.
// K: kernel size, L: 卷积层数. engine 按 (L, K, fn) 加载对应权重。
window.makeNetworkData = async function (catIndex, N = 5, K = 3, activationFn = 'relu', L = 1) {
  const input = window.CNN_INPUTS.getInputTensor(catIndex);
  if (!input) throw new Error(`MNIST sample for catIndex=${catIndex} not loaded`);
  const raw = await window.CNN.runForward(input, activationFn, N, K, L);

  const result = { maxAbs: {} };
  for (const layer of window.LAYERS) {
    if (layer.kind === 'input') {
      // input 层填 RGB 三个 channel 的下采样,归一化到 [-1, 1]
      const rgb = window.CNN_INPUTS.getInputRGB(catIndex);
      const channels = [];
      for (let c = 0; c < 3; c++) {
        const ch2d = rgb.map(row => row.map(px => px[c] / 127.5 - 1));
        channels.push(downsample(ch2d, RENDER_SIZE));
      }
      result[layer.id] = channels;
      result.maxAbs[layer.id] = 1.0; // input 已经归一化
      continue;
    }
    if (layer.kind === 'softmax') {
      // 真实 softmax 概率(不再人为 boost 正确类),与详情页公式视图一致
      const probs = raw.output.slice();
      result.output = [probs];
      result.outputLogits = raw.dense_1.slice();
      result.maxAbs.output = Math.max(...probs);
      // 同时把 flatten 数据存起来,供 softmax 详情浮层用(主屏不画这一列)
      if (raw.flatten) {
        result.flatten = raw.flatten.slice();
        let fm = 0;
        for (const v of raw.flatten) { const a = Math.abs(v); if (a > fm) fm = a; }
        result.maxAbs.flatten = fm || 1;
      }
      continue;
    }
    // conv / relu / pool: 取 N 个 channel
    const t = raw[layer.id];
    const channels = [];
    const channelCount = Math.min(N, t[0][0].length);
    for (let c = 0; c < channelCount; c++) {
      const ch2d = extractChannel(t, c);
      channels.push(downsample(ch2d, RENDER_SIZE));
    }
    // 记录归一化前的 abs max,用于色阶图例
    let max = 0;
    for (const m of channels) for (const r of m) for (const v of r) {
      const av = Math.abs(v);
      if (av > max) max = av;
    }
    result.maxAbs[layer.id] = max;
    normalizeChannels(channels);
    result[layer.id] = channels;
  }
  // 详情页(公式视图)需要真实数值,而不是上面下采样+归一化后的展示用图。
  // 把完整前向张量和原始输入(28×28×1, 0..1)原样挂上,供 detail-views.jsx 取真值。
  result.raw = raw;          // { conv_1_i, relu_1_i, pool_1, flatten, dense_1, output }
  result.inputTensor = input; // [28][28][1]
  result.thumb = window.makeThumb(catIndex, window.CATEGORIES[catIndex].hue);
  return result;
};

// 设置改变(channel 数 / K / 激活函数 / 层数) 时,重算所有 10 个类别的网络数据。
// L 或 K 改变会同时更新 window.LAYERS(影响 UI 渲染的 column 结构)。
window.recomputeAll = async function (N, K = 3, activationFn = 'relu', L = 1) {
  window.LAYERS = window.buildLayers(L, K);
  const all = await Promise.all(
    window.CATEGORIES.map((_, i) => window.makeNetworkData(i, N, K, activationFn, L))
  );
  for (let i = 0; i < all.length; i++) window.NETWORK_DATA[i] = all[i];
  return all;
};

// 缩略图保留旧的伪随机色块画法 (装饰用，跟真实卷积无关)
function mulberry32(seed) {
  return function () {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
window.makeThumb = function (seed, hue) {
  const rng = mulberry32(seed * 1000 + 7);
  const cells = [];
  for (let i = 0; i < 64; i++) cells.push(0.3 + rng() * 0.7);
  return { cells, hue };
};

// ─── 启动时: 加载真实权重 + MNIST 样本, 然后算好所有 10 个类别 ──────
window.NETWORK_DATA = new Array(window.CATEGORIES.length).fill(null);
window.NETWORK_READY = (async function () {
  const t0 = performance.now();
  // 先并行加载权重 + 样本 JSON
  await Promise.all([
    window.CNN.loadWeights(),
    window.CNN_INPUTS.loadSamples(),
  ]);
  console.log('[CNN] weights + samples loaded in', Math.round(performance.now() - t0), 'ms');
  const t1 = performance.now();
  // 真实 forward 算每个类别的所有层激活
  const all = await Promise.all(
    window.CATEGORIES.map((_, i) => window.makeNetworkData(i))
  );
  for (let i = 0; i < all.length; i++) window.NETWORK_DATA[i] = all[i];
  console.log('[CNN] forward all categories in', Math.round(performance.now() - t1), 'ms');
})();

// ─── 颜色映射保留 ─────────────────────────────────────────────────
// Viridis LUT — Xiaohongshu 风:亮青蓝(0)→深紫(max),适合深色背景的激活可视化
window.VIRIDIS = [
  [125, 220, 248], [98,  200, 242], [72,  175, 232], [55,  145, 215],
  [50,  115, 195], [55,  88,  170], [70,  62,  145], [85,  42,  120],
  [92,  28,  100], [82,  18,  80],  [62,  10,  62],
];
// Plasma LUT — 暖色版:浅粉(0)→深紫红(max)
window.PLASMA = [
  [255, 235, 220], [253, 210, 190], [250, 175, 165], [240, 135, 155],
  [220, 95,  150], [195, 65,  150], [165, 40,  150], [130, 25,  145],
  [95,  15,  130], [62,  5,   110], [38,  0,   80],
];

function _lutLookup(lut, t) {
  t = Math.max(0, Math.min(1, t));
  const idx = t * (lut.length - 1);
  const i0 = Math.floor(idx), i1 = Math.min(i0 + 1, lut.length - 1);
  const f = idx - i0;
  const c0 = lut[i0], c1 = lut[i1];
  return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
}

// 把激活值映射成颜色字符串。
// colormap: 'viridis' | 'plasma' | 'gray' | 'rwb'
window.actColor = function (v, dark = false, colormap = 'viridis') {
  if (colormap === 'rwb') {
    const t = Math.max(-1, Math.min(1, v));
    const base = dark ? 28 : 255;
    if (t >= 0) {
      const r = Math.round(base + (37  - base) * t);
      const g = Math.round(base + (99  - base) * t);
      const b = Math.round(base + (235 - base) * t);
      return `rgb(${r},${g},${b})`;
    } else {
      const r = Math.round(base + (220 - base) * -t);
      const g = Math.round(base + (38  - base) * -t);
      const b = Math.round(base + (38  - base) * -t);
      return `rgb(${r},${g},${b})`;
    }
  }
  if (colormap === 'gray') {
    const t = Math.min(1, Math.abs(v));
    const base = dark ? 28 : 255;
    const c = Math.round(base + (0 - base) * t);
    return `rgb(${c},${c},${c})`;
  }
  // viridis / plasma
  const lut = colormap === 'plasma' ? window.PLASMA : window.VIRIDIS;
  const [r, g, b] = _lutLookup(lut, Math.min(1, Math.abs(v)));
  return `rgb(${r|0},${g|0},${b|0})`;
};

window.colormapGradient = function (name) {
  if (name === 'rwb') return 'linear-gradient(90deg, #dc2626, #fff 50%, #2563eb)';
  if (name === 'gray') return 'linear-gradient(90deg, #fff, #000)';
  const lut = name === 'plasma' ? window.PLASMA : window.VIRIDIS;
  return 'linear-gradient(90deg, ' + lut.map(([r,g,b]) => `rgb(${r},${g},${b})`).join(',') + ')';
};

// 权重色: 黄 → 白 → 绿
window.weightColor = function (v) {
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) {
    return `rgb(${Math.round(255 + (16 - 255) * t)},${Math.round(255 + (185 - 255) * t)},${Math.round(255 + (129 - 255) * t)})`;
  } else {
    return `rgb(${Math.round(255 + (234 - 255) * (-t))},${Math.round(255 + (179 - 255) * (-t))},${Math.round(255 + (8 - 255) * (-t))})`;
  }
};


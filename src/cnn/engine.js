// 真实 MNIST CNN forward pass,纯 JS,权重从 assets/mnist_weights.json 加载。
//
// 网络结构(跟 model_training/train.py 一致):
//   input  (28, 28, 1)
//   conv_1_1: Conv2D 3x3 ×10  →  (26, 26, 10)
//   relu_1_1: ReLU            →  (26, 26, 10)
//   pool_1:   MaxPool 2x2     →  (13, 13, 10)
//   flatten:                  →  1690
//   dense_1:  Dense →10       →  10 logits
//   output:   Softmax         →  10 probs
//
// 数据格式: 所有特征图都是 [H][W][C] 嵌套 JS 数组。
// 权重格式: conv kernel [kH][kW][inCh][outCh] + bias [outCh],
//          dense kernel [in][out] + bias [out]。

(function () {
  // ── 卷积 (3x3, VALID padding, stride=1, 带 bias) ──────────────
  function conv2d(input, kernel, bias) {
    const H = input.length;
    const W = input[0].length;
    const inCh = input[0][0].length;
    const outCh = kernel[0][0][0].length;
    const K = kernel.length;
    const outH = H - K + 1;
    const outW = W - K + 1;
    const out = new Array(outH);
    for (let y = 0; y < outH; y++) {
      const row = new Array(outW);
      for (let x = 0; x < outW; x++) {
        const acc = new Float32Array(outCh);
        for (let ky = 0; ky < K; ky++) {
          for (let kx = 0; kx < K; kx++) {
            const ipx = input[y + ky][x + kx];
            const kpx = kernel[ky][kx];
            for (let ic = 0; ic < inCh; ic++) {
              const v = ipx[ic];
              const kic = kpx[ic];
              for (let oc = 0; oc < outCh; oc++) {
                acc[oc] += v * kic[oc];
              }
            }
          }
        }
        if (bias) for (let oc = 0; oc < outCh; oc++) acc[oc] += bias[oc];
        row[x] = Array.from(acc);
      }
      out[y] = row;
    }
    return out;
  }

  // 激活函数: relu / sigmoid / tanh
  function applyActivation(input, fn) {
    const H = input.length;
    const W = input[0].length;
    const ch = input[0][0].length;
    const out = new Array(H);
    for (let y = 0; y < H; y++) {
      const row = new Array(W);
      for (let x = 0; x < W; x++) {
        const px = new Array(ch);
        const ipx = input[y][x];
        for (let c = 0; c < ch; c++) {
          const v = ipx[c];
          if (fn === 'sigmoid') px[c] = 1 / (1 + Math.exp(-v));
          else if (fn === 'tanh') px[c] = Math.tanh(v);
          else px[c] = v > 0 ? v : 0; // relu
        }
        row[x] = px;
      }
      out[y] = row;
    }
    return out;
  }

  function maxpool2x2(input) {
    const H = input.length;
    const W = input[0].length;
    const ch = input[0][0].length;
    const outH = Math.floor(H / 2);
    const outW = Math.floor(W / 2);
    const out = new Array(outH);
    for (let y = 0; y < outH; y++) {
      const row = new Array(outW);
      for (let x = 0; x < outW; x++) {
        const px = new Array(ch).fill(-Infinity);
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const ipx = input[y * 2 + dy][x * 2 + dx];
            for (let c = 0; c < ch; c++) {
              if (ipx[c] > px[c]) px[c] = ipx[c];
            }
          }
        }
        row[x] = px;
      }
      out[y] = row;
    }
    return out;
  }

  function flatten(input) {
    const H = input.length;
    const W = input[0].length;
    const ch = input[0][0].length;
    const out = new Array(H * W * ch);
    let i = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const px = input[y][x];
        for (let c = 0; c < ch; c++) out[i++] = px[c];
      }
    }
    return out;
  }

  function dense(input, weight, bias) {
    const M = weight[0].length;
    const out = new Float32Array(M);
    if (bias) for (let m = 0; m < M; m++) out[m] = bias[m];
    for (let n = 0; n < input.length; n++) {
      const v = input[n];
      const row = weight[n];
      for (let m = 0; m < M; m++) out[m] += v * row[m];
    }
    return Array.from(out);
  }

  function softmax(input) {
    let max = -Infinity;
    for (const v of input) if (v > max) max = v;
    const exp = input.map(v => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(v => v / sum);
  }

  // ── 权重加载: 18 套 (L=1/2 × K=3/5/7 × relu/sigmoid/tanh), 按需 fetch + 缓存 ──
  const SUPPORTED_L = [1, 2];
  const SUPPORTED_K = [3, 5, 7];
  const SUPPORTED_FN = ['relu', 'sigmoid', 'tanh'];
  const DEFAULT_L = 1;
  const DEFAULT_K = 3;
  const DEFAULT_FN = 'relu';
  const _weightsByKey = {};
  const _weightsPromiseByKey = {};
  function loadWeights(L = DEFAULT_L, K = DEFAULT_K, fn = DEFAULT_FN) {
    if (!SUPPORTED_L.includes(L)) L = DEFAULT_L;
    if (!SUPPORTED_K.includes(K)) K = DEFAULT_K;
    if (!SUPPORTED_FN.includes(fn)) fn = DEFAULT_FN;
    const key = `${L}_${K}_${fn}`;
    if (_weightsByKey[key]) return Promise.resolve(_weightsByKey[key]);
    if (_weightsPromiseByKey[key]) return _weightsPromiseByKey[key];
    _weightsPromiseByKey[key] = fetch(`assets/mnist_weights_l${L}_k${K}_${fn}.json`)
      .then(r => r.json())
      .then(raw => {
        _weightsByKey[key] = raw;
        return raw;
      });
    return _weightsPromiseByKey[key];
  }
  function getCachedWeights(L, K, fn) {
    return _weightsByKey[`${L}_${K}_${fn}`] || null;
  }
  function maskChannels(map3d, n) {
    if (n >= 10) return;
    const H = map3d.length, W = map3d[0].length;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const px = map3d[y][x];
        for (let c = n; c < 10; c++) px[c] = 0;
      }
    }
  }

  // ── 正向推理 ─────────────────────────────────────────────────
  // 输入: image [28][28][1],值域 [0,1] (灰度,0=黑,1=白)
  // numChannels: 1..10, 限制 conv 输出实际使用的 channel 数(多余置零)
  // K: 3/5/7, activationFn: relu/sigmoid/tanh, numLayers: 1/2 → 切换对应权重
  async function runForward(image, activationFn = DEFAULT_FN, numChannels = 10,
                            K = DEFAULT_K, numLayers = DEFAULT_L) {
    const W = await loadWeights(numLayers, K, activationFn);
    const result = {};
    let cur = image;
    for (let i = 1; i <= numLayers; i++) {
      const convId = `conv_1_${i}`;
      const reluId = `relu_1_${i}`;
      cur = conv2d(cur, W[convId].kernel, W[convId].bias);
      maskChannels(cur, numChannels);
      result[convId] = cur;
      cur = applyActivation(cur, activationFn);
      result[reluId] = cur;
    }
    const pool_1 = maxpool2x2(cur);
    result.pool_1 = pool_1;
    const flat = flatten(pool_1);
    result.flatten = flat;
    const dense_1 = dense(flat, W.dense_1.kernel, W.dense_1.bias);
    result.dense_1 = dense_1;
    result.output = softmax(dense_1);
    return result;
  }

  // 弹窗演示拉 kernel 切片(inChIdx: 第一层是 0,后续层可以取任意 input channel)
  // 不强求传 (L,K,fn) — 智能从所有已加载缓存里找含 layerId 的权重(L=2 时 conv_1_2 只在 L=2 缓存里)。
  function getKernelSlice(layerId, outChIdx, inChIdx = 0) {
    let W = null;
    for (const key in _weightsByKey) {
      const cand = _weightsByKey[key];
      if (cand && cand[layerId]) { W = cand; break; }
    }
    if (!W || !W[layerId]) return null;
    const kernel = W[layerId].kernel;
    if (!Array.isArray(kernel) || !Array.isArray(kernel[0])) return null;
    const kSize = kernel.length;
    const slice = [];
    for (let y = 0; y < kSize; y++) {
      const row = [];
      for (let x = 0; x < kSize; x++) {
        const v = kernel[y][x] && kernel[y][x][inChIdx] && kernel[y][x][inChIdx][outChIdx];
        row.push(typeof v === 'number' ? v : 0);
      }
      slice.push(row);
    }
    return slice;
  }

  // 拿 conv 层的 bias[channelIdx]
  function getBias(layerId, channelIdx) {
    for (const key in _weightsByKey) {
      const cand = _weightsByKey[key];
      if (cand && cand[layerId] && cand[layerId].bias) {
        return cand[layerId].bias[channelIdx] ?? 0;
      }
    }
    return 0;
  }

  // 拿 conv 层真实的 input channel 数 (kernel_shape = [H, W, inCh, outCh])
  function getInChannels(layerId) {
    for (const key in _weightsByKey) {
      const cand = _weightsByKey[key];
      if (cand && cand[layerId] && cand[layerId].kernel_shape) {
        return cand[layerId].kernel_shape[2];
      }
    }
    return 1;
  }

  window.CNN = {
    runForward,
    getKernelSlice,
    getBias,
    getInChannels,
    loadWeights,
  };
})();

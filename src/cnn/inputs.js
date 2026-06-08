// 真实 MNIST 测试集样本(10 张,每个数字一张),从 assets/mnist_samples.json 加载。
// JSON 格式: [{digit: 0, pixels: [[…28×28 int 0-255…]]}, …]
//
// 提供两个 view:
//   getInputTensor(catIndex) → [28][28][1] 单通道,值域 [0,1] (喂给 CNN.runForward)
//   getInputRGB(catIndex)    → [28][28][3] RGB,值域 0-255 (灰度复制到三通道,UI 兼容)

(function () {
  const SIZE = 28;
  let _samples = null;             // [{digit, pixels: 28x28 0-255}, ...] 按 digit 排序
  let _samplesPromise = null;
  const tensorCache = new Map();
  const rgbCache = new Map();

  function loadSamples() {
    if (_samples) return Promise.resolve(_samples);
    if (_samplesPromise) return _samplesPromise;
    _samplesPromise = fetch('assets/mnist_samples.json')
      .then(r => r.json())
      .then(raw => {
        // 按 digit 排序,索引 = 数字
        const byDigit = new Array(10).fill(null);
        for (const s of raw) byDigit[s.digit] = s.pixels;
        _samples = byDigit;
        return _samples;
      });
    return _samplesPromise;
  }

  function getPixels(catIndex) {
    if (!_samples) return null;
    return _samples[catIndex] || null;
  }

  // [28][28][1] 浮点 0-1, 喂 forward
  function getInputTensor(catIndex) {
    if (tensorCache.has(catIndex)) return tensorCache.get(catIndex);
    const px = getPixels(catIndex);
    if (!px) return null;
    const out = new Array(SIZE);
    for (let y = 0; y < SIZE; y++) {
      const row = new Array(SIZE);
      for (let x = 0; x < SIZE; x++) {
        row[x] = [px[y][x] / 255];
      }
      out[y] = row;
    }
    tensorCache.set(catIndex, out);
    return out;
  }

  // [28][28][3] uint8 0-255, 给现有 UI(它假设 RGB 三通道展示输入图)。
  // 灰度值复制到 R/G/B 三通道, 视觉上还是黑白。
  function getInputRGB(catIndex) {
    if (rgbCache.has(catIndex)) return rgbCache.get(catIndex);
    const px = getPixels(catIndex);
    if (!px) return null;
    const rows = new Array(SIZE);
    for (let y = 0; y < SIZE; y++) {
      const row = new Array(SIZE);
      for (let x = 0; x < SIZE; x++) {
        const v = px[y][x];
        row[x] = [v, v, v];
      }
      rows[y] = row;
    }
    rgbCache.set(catIndex, rows);
    return rows;
  }

  window.CNN_INPUTS = {
    SIZE,
    loadSamples,
    getInputTensor,
    getInputRGB,
  };
})();

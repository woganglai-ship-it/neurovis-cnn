/* global React, NETWORK_DATA, LAYERS, CATEGORIES, ImageThumb, haptic */
// 第二页:层详情。对齐 CNN Explainer 论文 (Sect 6.2 Elastic Explanation / 6.3 Interactive Formula)。
//
// 两级视图(逐级深入):
//   Level 1  ConvElasticView   — 多输入通道 → 各自卷积 → 中间结果 → Σ+bias → 输出
//   Level 2  ConvFormulaView   — 单个像素的真实数值点积公式 (Fig 6A)
// ReLU / MaxPool / Softmax 无多通道分解,直接进各自 FormulaView (Fig 6 B/C/D)。
//
// 所有数字来自 engine.js 真实前向 (NETWORK_DATA[cat].raw / .inputTensor),不放假数字。
// 配色: 激活 = R-W-B (window.actColor), 权重/偏置 = 黄绿 (window.weightColor)。
(function () {
  const { useState, useEffect, useRef, useMemo } = React;

  const HL = '#E8B92C';            // kernel / 当前格 高亮色 (琥珀)
  const fmt = (v) => (v >= 0 ? '+' : '') + v.toFixed(2);

  // ── 真值取数 ───────────────────────────────────────────────────
  function rawChannel(tensor3d, c) {
    if (!tensor3d || !tensor3d.length) return null;
    const H = tensor3d.length, W = tensor3d[0].length;
    const out = new Array(H);
    for (let y = 0; y < H; y++) {
      const row = new Array(W);
      for (let x = 0; x < W; x++) row[x] = tensor3d[y][x][c];
      out[y] = row;
    }
    return out;
  }
  function pctMaxAbs(m, p = 0.75) {
    if (!m) return 1;
    const s = m.flat().map(Math.abs).sort((a, b) => a - b);
    return Math.max(s[Math.floor(s.length * p)] || 0, 1e-4);
  }
  // valid 卷积 (stride 1),返回 raw 值(不归一化)。与 engine.conv2d 单通道一致。
  function convolveValid(inMap, kern) {
    if (!inMap || !kern || !kern.length) return null;
    const k = kern.length, H = inMap.length, W = inMap[0].length;
    const oh = H - k + 1, ow = W - k + 1;
    if (oh <= 0 || ow <= 0) return null;
    const out = [];
    for (let y = 0; y < oh; y++) {
      const row = [];
      for (let x = 0; x < ow; x++) {
        let s = 0;
        for (let dy = 0; dy < k; dy++)
          for (let dx = 0; dx < k; dx++) s += inMap[y + dy][x + dx] * kern[dy][dx];
        row.push(s);
      }
      out.push(row);
    }
    return out;
  }

  // ── 矩阵热图 (canvas, pixelated) + 高亮窗口 + 指针拾取 ──────────
  // grayscale=true: 把 0..1 当灰度 (黑→白),用于 MNIST 原图。
  // 否则按 maxAbs 归一化到 [-1,1] 走 R-W-B 激活色阶。
  function MatrixView({ data, maxAbs, size, colormap = 'rwb', grayscale = false,
                        win = null, winColor = HL, onPick = null, flow = false }) {
    const ref = useRef(null);
    const rows = data.length, cols = data[0].length;
    useEffect(() => {
      const c = ref.current; if (!c) return;
      c.width = cols; c.height = rows;
      const ctx = c.getContext('2d');
      const M = maxAbs || 1;
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const v = data[y][x];
          if (grayscale) {
            const g = Math.round(Math.max(0, Math.min(1, v)) * 255);
            ctx.fillStyle = `rgb(${g},${g},${g})`;
          } else {
            ctx.fillStyle = window.actColor(v / M, false, colormap);
          }
          ctx.fillRect(x, y, 1, 1);
        }
      }
    }, [data, maxAbs, colormap, grayscale, rows, cols]);

    const cw = size / cols, chh = size / rows;
    const pick = (e) => {
      if (!onPick) return;
      const r = e.currentTarget.getBoundingClientRect();
      const cx = Math.max(0, Math.min(cols - 1, Math.floor((e.clientX - r.left) / cw)));
      const cy = Math.max(0, Math.min(rows - 1, Math.floor((e.clientY - r.top) / chh)));
      onPick(cx, cy);
    };
    return (
      <div
        onPointerDown={onPick ? (e) => { e.stopPropagation(); pick(e); } : undefined}
        onPointerMove={onPick ? (e) => { if (e.pointerType === 'mouse' || e.buttons) { e.stopPropagation(); pick(e); } } : undefined}
        style={{
          position: 'relative', width: size, height: size,
          background: grayscale ? '#000' : '#FAF7F0',
          boxShadow: 'inset 0 0 0 0.5px rgba(0,0,0,0.18)', borderRadius: 3,
          cursor: onPick ? 'crosshair' : 'default', touchAction: 'none',
        }}>
        <canvas ref={ref} style={{
          width: size, height: size, imageRendering: 'pixelated',
          display: 'block', borderRadius: 3,
        }} />
        {win && (
          <div style={{
            position: 'absolute',
            left: win.x * cw, top: win.y * chh,
            width: win.w * cw, height: win.h * chh,
            border: `1.5px solid ${winColor}`,
            background: 'rgba(232,185,44,0.14)',
            borderRadius: 1, pointerEvents: 'none', zIndex: 2,
            boxShadow: '0 0 0 1px rgba(232,185,44,0.4)',
            transition: 'left .25s cubic-bezier(.55,.05,.25,1), top .25s cubic-bezier(.55,.05,.25,1)',
          }} />
        )}
      </div>
    );
  }

  // 3×3 (或 k×k) kernel 小热图 — 黄绿权重色,跟激活图区分
  function KernelMini({ kern, size = 30 }) {
    if (!kern || !kern.length) return <div style={{ width: size, height: size, background: '#eee', borderRadius: 2 }} />;
    const k = kern.length;
    const cell = size / k;
    let m = 0; for (const r of kern) for (const v of r) m = Math.max(m, Math.abs(v));
    m = m || 1;
    return (
      <div style={{
        position: 'relative', width: size, height: size, borderRadius: 2,
        boxShadow: 'inset 0 0 0 0.5px rgba(0,0,0,0.2)', overflow: 'hidden',
      }}>
        {kern.flatMap((row, y) => row.map((v, x) => (
          <div key={`${y}-${x}`} style={{
            position: 'absolute', left: x * cell, top: y * cell, width: cell, height: cell,
            background: window.weightColor(v / m),
          }} />
        )))}
      </div>
    );
  }

  // 公式里的数值小方块: 激活值用激活色背景, 权重用黄绿背景
  function NumCell({ v, kind = 'act', maxAbs = 1, colormap = 'rwb', w = 40 }) {
    const bg = kind === 'weight'
      ? window.weightColor(v / (maxAbs || 1))
      : window.actColor(v / (maxAbs || 1), false, colormap);
    const norm = Math.min(1, Math.abs(v / (maxAbs || 1)));
    const light = kind === 'weight' ? norm < 0.55 : norm < 0.5;
    return (
      <span style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        minWidth: w, height: 16, padding: '0 4px', borderRadius: 2,
        background: bg, color: light ? 'rgba(0,0,0,0.8)' : '#fff',
        fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600,
        boxShadow: 'inset 0 0 0 0.5px rgba(0,0,0,0.12)',
      }}>{v.toFixed(2)}</span>
    );
  }

  // ── Modal 外壳 ─────────────────────────────────────────────────
  function Modal({ onClose, children, bg = '#fff', maxW = '96vw' }) {
    return (
      <div onClick={onClose} style={{
        position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.22)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 500, animation: 'sheetFade 200ms ease-out both', padding: 10,
      }}>
        <div onClick={(e) => e.stopPropagation()} style={{
          background: bg, borderRadius: 14, boxShadow: '0 20px 60px rgba(0,0,0,0.28)',
          maxWidth: maxW, maxHeight: '94%', overflow: 'auto',
          display: 'flex', flexDirection: 'column',
        }}>{children}</div>
      </div>
    );
  }

  function CloseBtn({ onClose }) {
    return (
      <button onClick={onClose} aria-label="close" style={{
        width: 22, height: 22, border: 'none', background: 'transparent', cursor: 'pointer',
        color: 'var(--text-3)', fontSize: 17, lineHeight: 1, padding: 0, flexShrink: 0,
      }}>×</button>
    );
  }

  function PlayBtn({ playing, onToggle }) {
    return (
      <button onClick={(e) => { e.stopPropagation(); onToggle(); }} aria-label={playing ? 'pause' : 'play'} style={{
        width: 26, height: 26, borderRadius: '50%', border: 'none',
        background: playing ? 'transparent' : 'rgba(0,0,0,0.06)',
        cursor: 'pointer', color: 'var(--text-2)', flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 0,
      }}>
        {playing
          ? <svg width="10" height="10" viewBox="0 0 12 12"><rect x="3" y="2.4" width="2.2" height="7.2" rx="0.7" fill="currentColor" /><rect x="6.8" y="2.4" width="2.2" height="7.2" rx="0.7" fill="currentColor" /></svg>
          : <svg width="10" height="10" viewBox="0 0 12 12"><path d="M3.5 2.2l6 3.8-6 3.8z" fill="currentColor" /></svg>}
      </button>
    );
  }

  // ════════════════════════════════════════════════════════════════
  //  Conv 详情 — 管理 Level1 (Elastic) ↔ Level2 (Formula) 切换
  // ════════════════════════════════════════════════════════════════
  function ConvDetail({ layer, layerIdx, channelIdx, catIndex, onClose, lang, colormap, numChannels }) {
    const zh = lang === 'zh';
    const data = NETWORK_DATA[catIndex];
    const prev = LAYERS[layerIdx - 1];
    const prevIsInput = prev && prev.kind === 'input';

    // 输入通道 (真值): conv_1_1 → 灰度输入 1 通道; conv_1_2 → 上一层 relu 的 numChannels 通道
    const inputChannels = useMemo(() => {
      if (prevIsInput) return [rawChannel(data.inputTensor, 0)];
      const src = data.raw[prev.id];
      const n = Math.min(numChannels, src[0][0].length);
      return Array.from({ length: n }, (_, c) => rawChannel(src, c));
    }, [catIndex, layerIdx, numChannels]);

    const oc = channelIdx;                       // 这次点开的输出通道
    const outputMap = rawChannel(data.raw[layer.id], oc);
    const bias = window.CNN.getBias(layer.id, oc);
    const kernels = inputChannels.map((_, ic) => window.CNN.getKernelSlice(layer.id, oc, ic));
    const intermediates = inputChannels.map((inMap, ic) => convolveValid(inMap, kernels[ic]));

    // Level 2: 点中间结果/输出 → 进入公式视图,带上是哪条输入通道
    const [formula, setFormula] = useState(null); // { ic } | null

    if (formula) {
      return (
        <ConvFormulaView
          inputMap={inputChannels[formula.ic]}
          inputIsGray={prevIsInput}
          kernel={kernels[formula.ic]}
          interMap={intermediates[formula.ic]}
          outputMap={outputMap}
          bias={bias}
          singleChannel={inputChannels.length === 1}
          channelLabel={formula.ic}
          oc={oc}
          colormap={colormap}
          lang={lang}
          onBack={() => setFormula(null)}
          onClose={onClose}
        />
      );
    }

    // 第二层卷积 (输入为多通道) → 堆叠视图; 第一层 (灰度单通道) → 弹性视图
    if (inputChannels.length > 1) {
      return (
        <ConvStackView
          inputChannels={inputChannels}
          kernels={kernels}
          intermediates={intermediates}
          outputMap={outputMap}
          bias={bias}
          oc={oc}
          prevLayer={prev}
          layer={layer}
          numChannels={numChannels}
          lang={lang}
          onClose={onClose}
          onEnterFormula={(ic) => { haptic(); setFormula({ ic }); }}
        />
      );
    }

    return (
      <ConvElasticView
        inputChannels={inputChannels}
        inputIsGray={prevIsInput}
        kernels={kernels}
        intermediates={intermediates}
        outputMap={outputMap}
        bias={bias}
        oc={oc}
        colormap={colormap}
        lang={lang}
        onClose={onClose}
        onEnterFormula={(ic) => { haptic(); setFormula({ ic }); }}
      />
    );
  }

  // ─────────── Level 1: Conv 弹性视图 (Claude Design UI · 接真实 MNIST 数据) ───────────
  // 用到的 useState/useEffect/useRef/useMemo / Modal / CloseBtn 都来自本文件已有作用域。
  function ConvElasticView({ inputChannels, inputIsGray, kernels, intermediates,
                             outputMap, bias, oc, colormap, lang, onClose, onEnterFormula }) {
    const zh = lang === 'zh';
    const input  = inputChannels[0] || [];
    const kernel = kernels[0] || [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const omap   = outputMap || intermediates[0] || [];
    const inN    = input.length || 28;
    const outN   = omap.length || 26;

    const inRef = useRef(null), midRef = useRef(null), inBoxRef = useRef(null), midBoxRef = useRef(null);
    const [playing, setPlaying] = useState(true);
    const [speed, setSpeed]     = useState(1);
    const [runId, setRunId]     = useState(0);
    const playRef = useRef(true), speedRef = useRef(1);
    useEffect(() => { playRef.current = playing; }, [playing]);
    useEffect(() => { speedRef.current = speed; }, [speed]);

    const SIZE = 168, inLeft = 150, inTop = 66, midLeft = 470, midTop = 66;
    const cellIn = SIZE / inN, cellMid = SIZE / outN;

    // 输出归一为橙色强度 (|conv| → gamma 0.7)
    const onorm = useMemo(() => {
      let mx = 1e-6;
      for (let r = 0; r < outN; r++) for (let c = 0; c < outN; c++) { const v = Math.abs(omap[r][c]); if (v > mx) mx = v; }
      const g = [];
      for (let r = 0; r < outN; r++) { const row = []; for (let c = 0; c < outN; c++) row.push(Math.pow(Math.abs(omap[r][c]) / mx, 0.7)); g.push(row); }
      return g;
    }, [omap, outN]);

    // 实时数据 ref(loop 里读最新, 避免对象引用变化触发重启)
    const onormRef = useRef(onorm); onormRef.current = onorm;
    // 稳定数据签名: 仅当数字/通道/尺寸真正变化时才变 (字符串值, 同数据时 === 相等 → 不重启动画)
    const sig = useMemo(() => {
      let acc = 0;
      for (let r = 0; r < outN; r++) for (let c = 0; c < outN; c++) acc += Math.abs(omap[r][c]) * (r * 31 + c + 1);
      return outN + '|' + oc + '|' + Math.round(acc * 1000);
    }, [omap, outN, oc]);

    // 输入白点 = 自身最大值 (兼容 0–1 或 0–255)
    const inMx = useMemo(() => {
      let mx = 1e-6; for (let r = 0; r < inN; r++) for (let c = 0; c < inN; c++) { if (input[r][c] > mx) mx = input[r][c]; }
      return mx;
    }, [input, inN]);

    // ① 画输入: 28×28 离散像素 + 网格线
    useEffect(() => {
      const cv = inRef.current; if (!cv) return;
      const ctx = cv.getContext('2d'), cw = SIZE / inN;
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, SIZE, SIZE);
      for (let r = 0; r < inN; r++) for (let c = 0; c < inN; c++) {
        const p = Math.round(Math.max(0, Math.min(1, input[r][c] / inMx)) * 255);
        ctx.fillStyle = 'rgb(' + p + ',' + p + ',' + p + ')';
        ctx.fillRect(c * cw, r * cw, cw, cw);
      }
      ctx.strokeStyle = 'rgba(150,150,150,0.22)'; ctx.lineWidth = 1; ctx.beginPath();
      for (let i = 0; i <= inN; i++) { const x = Math.round(i * cw) + 0.5; ctx.moveTo(x, 0); ctx.lineTo(x, SIZE); ctx.moveTo(0, x); ctx.lineTo(SIZE, x); }
      ctx.stroke();
    }, [input, inN, inMx]);

    // ② 输出逐格扫描动画 + 滑窗同步
    useEffect(() => {
      const mc = midRef.current; if (!mc) return;
      const mctx = mc.getContext('2d');
      const inBox = inBoxRef.current, midBox = midBoxRef.current;
      let idx = 0, acc = 0, hold = 0, last = 0, raf, dwell = 26;
      function pos(r, c) {
        if (inBox) { inBox.style.left = (inLeft + c * cellIn) + 'px'; inBox.style.top = (inTop + r * cellIn) + 'px'; inBox.style.width = (3 * cellIn) + 'px'; inBox.style.height = (3 * cellIn) + 'px'; }
        if (midBox) { midBox.style.left = (midLeft + c * cellMid) + 'px'; midBox.style.top = (midTop + r * cellMid) + 'px'; midBox.style.width = cellMid + 'px'; midBox.style.height = cellMid + 'px'; }
      }
      function reset() { idx = 0; acc = 0; hold = 0; mctx.clearRect(0, 0, outN, outN); pos(0, 0); }
      function adv(step) {
        if (idx >= outN * outN) { hold += step; if (hold > 900) reset(); return; }
        const k = idx, r = (k / outN) | 0, c = k % outN, row = onormRef.current[r] || [], v = Math.min(1, (row[c] || 0) * 1.18);
        mctx.fillStyle = 'rgba(232,117,76,' + v.toFixed(3) + ')'; mctx.fillRect(c, r, 1, 1); pos(r, c); idx++;
      }
      function loop(t) {
        if (!last) last = t; const dt = Math.min(80, t - last); last = t;
        if (playRef.current) { const step = dwell / speedRef.current; acc += dt; let g = 0; while (acc >= step && g < 80) { acc -= step; adv(step); g++; } }
        raf = requestAnimationFrame(loop);
      }
      reset(); raf = requestAnimationFrame(loop);
      return () => cancelAnimationFrame(raf);
    }, [sig, runId]);

    // kernel 色块: 真实核 → 红蓝
    const km = Math.max(0.001, Math.max.apply(null, kernel.flat().map(Math.abs)));
    function rdbu(t) {
      t = Math.max(0, Math.min(1, t)); let r, g, b;
      if (t < 0.5) { const u = t / 0.5; r = 178 + (255 - 178) * u; g = 24 + (255 - 24) * u; b = 43 + (255 - 43) * u; }
      else { const u = (t - 0.5) / 0.5; r = 255 - (255 - 33) * u; g = 255 - (255 - 102) * u; b = 255 - (255 - 172) * u; }
      return 'rgb(' + (r | 0) + ',' + (g | 0) + ',' + (b | 0) + ')';
    }

    const L = zh
      ? { input: '输入', inter: '中间结果', kernel: '卷积核', cap1: '卷积核在输入上滑动', cap1b: '得到中间结果', cap2: '每个输出像素 = 3×3 区块', cap2b: '与卷积核加权求和', foot: `conv · 通道 ${oc} · 3×3 · 步长 1` }
      : { input: 'input', inter: 'intermediate', kernel: 'Kernel', cap1: 'Slide the kernel over the input', cap1b: 'to get the intermediate result', cap2: 'Each output pixel = sum of the 3×3', cap2b: 'patch × the kernel weights', foot: `conv · ch ${oc} · 3×3 · stride 1` };

    const btn = { height: 38, border: '1px solid #dcdcdc', background: '#fafafa', borderRadius: 8, cursor: 'pointer', color: '#333', display: 'flex', alignItems: 'center', justifyContent: 'center' };
    const cap = { position: 'absolute', fontFamily: 'Georgia,serif', fontStyle: 'italic', lineHeight: 1.5 };

    return (
      <Modal onClose={onClose} bg="#fff" maxW="96vw">
        <CloseBtn onClose={onClose} />
        <div style={{ width: 760, height: 392, fontFamily: "'Helvetica Neue',Helvetica,Arial,sans-serif", color: '#2b2b2b', position: 'relative', padding: '6px 20px 10px' }}>
          <div style={{ position: 'relative', width: 720, height: 372, margin: '0 auto' }}>
            <svg width="720" height="372" viewBox="0 0 720 372" style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none' }}>
              <line x1="318" y1="146" x2="470" y2="146" stroke="#c4c4c4" strokeWidth="1.5" strokeDasharray="6 5" />
              <polygon points="470,146 462,142 462,150" fill="#c4c4c4" />
              <path d="M 96 124 C 110 142, 128 142, 146 108" fill="none" stroke="#b6b6b6" strokeWidth="1.4" strokeDasharray="4 4" />
              <polygon points="146,108 139,116 148,117" fill="#b6b6b6" />
            </svg>

            <div style={{ position: 'absolute', left: 150, top: 22, fontSize: 19, fontWeight: 700 }}>{L.input}</div>
            <div style={{ position: 'absolute', left: 150, top: 44, fontSize: 12.5, color: '#9a9a9a' }}>({inN}, {inN}, 1)</div>
            <div style={{ position: 'absolute', left: 470, top: 22, fontSize: 19, fontWeight: 700 }}>{L.inter}</div>
            <div style={{ position: 'absolute', left: 470, top: 44, fontSize: 12.5, color: '#9a9a9a' }}>({outN}, {outN}, 1)</div>

            <div style={{ position: 'absolute', left: 18, top: 84, fontSize: 13.5, fontStyle: 'italic', fontFamily: 'Georgia,serif', color: '#444' }}>{L.kernel}</div>
            <div style={{ position: 'absolute', left: 74, top: 82, width: 20, height: 20, display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 1, border: '1px solid #cfcfcf', background: '#cfcfcf' }}>
              {kernel.flat().map((v, i) => <div key={i} style={{ background: rdbu(0.5 + 0.5 * v / km) }} />)}
            </div>

            <div style={{ position: 'absolute', left: 150, top: 66, width: 168, height: 168, border: '1px solid #d8d8d8' }}>
              <canvas ref={inRef} width="168" height="168" style={{ width: 168, height: 168, display: 'block' }} />
            </div>
            <div style={{ position: 'absolute', left: 470, top: 66, width: 168, height: 168, border: '1px solid #d8d8d8', background: '#fff', cursor: onEnterFormula ? 'pointer' : 'default' }}
                 onClick={() => onEnterFormula && onEnterFormula(0)}>
              <canvas ref={midRef} width={outN} height={outN} style={{ width: 168, height: 168, display: 'block', imageRendering: 'pixelated' }} />
            </div>

            <div ref={inBoxRef} style={{ position: 'absolute', left: 150, top: 66, width: 21, height: 21, border: '1.6px solid #e8754c', background: 'rgba(232,117,76,0.14)', pointerEvents: 'none' }} />
            <div ref={midBoxRef} style={{ position: 'absolute', left: 470, top: 66, width: 8, height: 8, border: '1.4px solid #e8754c', pointerEvents: 'none' }} />

            <div style={{ ...cap, left: 150, top: 248, width: 260, fontSize: 14, color: '#666' }}>{L.cap1}<br />{L.cap1b}</div>
            <div style={{ ...cap, left: 470, top: 248, width: 230, fontSize: 13, color: '#9a9a9a' }}>{L.cap2}<br />{L.cap2b}</div>

            <div style={{ position: 'absolute', left: 150, top: 322, display: 'flex', alignItems: 'center', gap: 10 }}>
              <button style={{ ...btn, width: 38, fontSize: 16, color: '#555' }} onClick={() => { setRunId(x => x + 1); setPlaying(true); }}>↺</button>
              <button style={{ ...btn, width: 54, fontSize: 14 }} onClick={() => setPlaying(p => !p)}>{playing ? '❚❚' : '▶'}</button>
              <button style={{ ...btn, padding: '0 14px', fontSize: 13, fontWeight: 600 }} onClick={() => setSpeed(s => ({ 0.5: 1, 1: 2, 2: 4, 4: 0.5 }[s]))}>{speed}×</button>
              <div style={{ marginLeft: 8, fontSize: 12, color: '#a8a8a8', fontFamily: 'Georgia,serif', fontStyle: 'italic' }}>{L.foot}</div>
            </div>
          </div>
        </div>
      </Modal>
    );
  }


  // ─────────── Level 1 (多通道): Conv 堆叠视图 — 第二层卷积 (像素级复刻设计稿) ───────────
  // 设计稿原生坐标系 1300×540, 整体按 SCALE 缩放放进 Modal。三块数据画布接真实 MNIST 数据,
  // 其余布局 / 配色 / 连线 / 注释 / 装饰核图标均与设计稿一致。
  function ConvStackView({ inputChannels, kernels, intermediates, outputMap,
                           bias, oc, prevLayer, layer, numChannels, lang, onClose, onEnterFormula }) {
    const zh = lang === 'zh';
    const nv   = Math.min(3, inputChannels.length);
    const inN  = (inputChannels[0] && inputChannels[0].length) || 28;
    const outN = (outputMap && outputMap.length) || (inN - 2);
    const OUT  = outN;
    const cpIn = 108 / inN, cpOut = 108 / outN;
    const SCALE = 0.56, W = Math.round(1300 * SCALE), H = Math.round(540 * SCALE);

    const inRefs = useRef([]), intRefs = useRef([]), kerRefs = useRef([]), curRefs = useRef([]), outRef = useRef(null);

    // 双色映射 (设计稿原函数): v<0 → 橙, v>0 → 蓝
    function color(v) {
      const r0 = 250, g0 = 248, b0 = 245;
      if (v < 0) { const t = Math.min(1, -v / 0.7);  return [Math.round(r0 + (233 - r0) * t), Math.round(g0 + (150 - g0) * t), Math.round(b0 + (118 - b0) * t)]; }
      if (v > 0) { const t = Math.min(1, v / 0.72); return [Math.round(r0 + (95 - r0) * t),  Math.round(g0 + (150 - g0) * t), Math.round(b0 + (205 - b0) * t)]; }
      return [r0, g0, b0];
    }
    const maxAbs = (maps) => { let m = 1e-6; for (const mp of maps) for (let r = 0; r < mp.length; r++) for (let c = 0; c < mp[r].length; c++) { const a = Math.abs(mp[r][c]); if (a > m) m = a; } return m; };
    const inMax  = useMemo(() => maxAbs(inputChannels.slice(0, nv)), [inputChannels, nv]);
    const intMax = useMemo(() => maxAbs(intermediates.slice(0, nv)), [intermediates, nv]);
    const outMax = useMemo(() => maxAbs([outputMap]), [outputMap]);

    // 预计算中间结果完整像素 (供逐格揭示)
    const intFull = useMemo(() => intermediates.slice(0, nv).map((mp) => {
      const d = new Uint8ClampedArray(outN * outN * 4);
      for (let y = 0; y < outN; y++) for (let x = 0; x < outN; x++) {
        const rgb = color((mp[y][x] / intMax) * 0.72), p = (y * outN + x) * 4;
        d[p] = rgb[0]; d[p + 1] = rgb[1]; d[p + 2] = rgb[2]; d[p + 3] = 255;
      }
      return d;
    }), [intermediates, nv, intMax, outN]);

    // 预计算输出完整像素 (与中间结果同步逐格揭示)
    const outFull = useMemo(() => {
      const d = new Uint8ClampedArray(outN * outN * 4);
      for (let y = 0; y < outN; y++) for (let x = 0; x < outN; x++) {
        const rgb = color((outputMap[y][x] / outMax) * 0.72), p = (y * outN + x) * 4;
        d[p] = rgb[0]; d[p + 1] = rgb[1]; d[p + 2] = rgb[2]; d[p + 3] = 255;
      }
      return d;
    }, [outputMap, outMax, outN]);

    const sig = useMemo(() => {
      let acc = 0; for (let r = 0; r < outN; r++) for (let c = 0; c < outN; c++) acc += Math.abs(outputMap[r][c]) * (r * 31 + c + 1);
      return inN + '|' + outN + '|' + oc + '|' + nv + '|' + Math.round(acc * 1000);
    }, [outputMap, inN, outN, oc, nv]);

    // 静态: 输入通道
    useEffect(() => {
      for (let i = 0; i < nv; i++) {
        const cv = inRefs.current[i]; if (!cv) continue;
        const ctx = cv.getContext('2d'), img = ctx.createImageData(inN, inN), mp = inputChannels[i];
        for (let y = 0; y < inN; y++) for (let x = 0; x < inN; x++) {
          const rgb = color((mp[y][x] / inMax) * 0.72), p = (y * inN + x) * 4;
          img.data[p] = rgb[0]; img.data[p + 1] = rgb[1]; img.data[p + 2] = rgb[2]; img.data[p + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
      }
    }, [sig, inMax]);

    // 动画: 卷积核滑窗 + 中间结果逐格揭示 (设计稿原逻辑)
    useEffect(() => {
      const bg = [250, 248, 245], total = OUT * OUT;
      let scan = 0, hold = 0, timer;
      function frame() {
        const done = Math.min(scan, total), oi = Math.min(done, total - 1), ox = oi % OUT, oy = (oi / OUT) | 0;
        for (let i = 0; i < nv; i++) {
          const ker = kerRefs.current[i], cur = curRefs.current[i];
          if (ker) ker.style.transform = 'translate(' + (ox * cpIn) + 'px,' + (oy * cpIn) + 'px)';
          if (cur) cur.style.transform = 'translate(' + (ox * cpOut) + 'px,' + (oy * cpOut) + 'px)';
        }
        for (let i = 0; i < nv; i++) {
          const cv = intRefs.current[i]; if (!cv) continue;
          const ctx = cv.getContext('2d'), img = ctx.createImageData(outN, outN), full = intFull[i];
          for (let y = 0; y < outN; y++) for (let x = 0; x < outN; x++) {
            const p = (y * outN + x) * 4;
            if ((oy * OUT + ox) >= (y * OUT + x)) { img.data[p] = full[p]; img.data[p + 1] = full[p + 1]; img.data[p + 2] = full[p + 2]; img.data[p + 3] = 255; }
            else { img.data[p] = bg[0]; img.data[p + 1] = bg[1]; img.data[p + 2] = bg[2]; img.data[p + 3] = 255; }
          }
          ctx.putImageData(img, 0, 0);
        }
        const o = outRef.current;
        if (o) {
          const ctx = o.getContext('2d'), img = ctx.createImageData(outN, outN);
          for (let y = 0; y < outN; y++) for (let x = 0; x < outN; x++) {
            const p = (y * outN + x) * 4;
            if ((oy * OUT + ox) >= (y * OUT + x)) { img.data[p] = outFull[p]; img.data[p + 1] = outFull[p + 1]; img.data[p + 2] = outFull[p + 2]; img.data[p + 3] = 255; }
            else { img.data[p] = bg[0]; img.data[p + 1] = bg[1]; img.data[p + 2] = bg[2]; img.data[p + 3] = 255; }
          }
          ctx.putImageData(img, 0, 0);
        }
      }
      frame();
      timer = setInterval(() => {
        frame();
        if (hold > 0) hold--;
        else { scan += 1.5; if (scan >= total) { scan = total; hold = 45; frame(); scan = 0; } }
      }, 33);
      return () => clearInterval(timer);
    }, [sig, intFull, outFull]);

    const prevId = (prevLayer && prevLayer.id) || 'input';
    const convId = (layer && layer.id) || 'conv';
    const tops = [96, 232, 396], icoTops = [98, 234, 398];
    const KICON = [
      ['#bcd4ec', '#f4f4f4', '#f4f4f4', '#cfe0f0'],
      ['#5bbfa3', '#eef6f2', '#dff0ea', '#7fcdb8'],
      ['#e8c25a', '#bcd4ec', '#f4ead0', '#cfe0f0'],
    ];
    const chans = [0, 1, 2].slice(0, nv);

    const A1 = zh
      ? ['卷积核在输入通道上', '滑动得到', '中间结果', 'Click', ' 了解更多']
      : ['Slide kernel over input', 'channel to get', 'intermediate result', 'Click', ' to learn more'];
    const A2 = zh
      ? ['每个输入通道', '用不同的卷积核', 'Hover', ' 查看数值！']
      : ['Each input channel', 'gets a different kernel', 'Hover over', ' to see value!'];
    const A3 = zh ? ['把所有中间结果相加', '再加上 bias'] : ['Add up all intermediate', 'results and then add bias'];
    const interLbl = zh ? '中间结果' : 'intermediate';

    return (
      <Modal onClose={onClose} bg="#fff" maxW="96vw">
        <div style={{ position: 'relative', width: W, height: H }}>
          <button onClick={onClose} aria-label="close" style={{ position: 'absolute', top: 6, right: 8, width: 22, height: 22, border: 'none', background: 'transparent', cursor: 'pointer', color: '#bbb', fontSize: 17, lineHeight: 1, padding: 0, zIndex: 5 }}>×</button>
          <div style={{ position: 'absolute', left: 0, top: 0, width: 1300, height: 540, transform: 'scale(' + SCALE + ')', transformOrigin: 'top left', background: '#fff', fontFamily: "-apple-system,'Helvetica Neue',Helvetica,Arial,sans-serif" }}>

            {/* connectors */}
            <svg width="1300" height="540" style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none' }}>
              <line x1="416" y1="152" x2="592" y2="152" stroke="#bdbdbd" strokeWidth="1.4" strokeDasharray="5 4" />
              <line x1="416" y1="288" x2="592" y2="288" stroke="#bdbdbd" strokeWidth="1.4" strokeDasharray="5 4" />
              <line x1="416" y1="452" x2="592" y2="452" stroke="#bdbdbd" strokeWidth="1.4" strokeDasharray="5 4" />
              <path d="M704,152 C 790,152 840,300 872,300" fill="none" stroke="#bdbdbd" strokeWidth="1.3" strokeDasharray="5 4" />
              <path d="M704,288 C 800,288 845,300 872,300" fill="none" stroke="#bdbdbd" strokeWidth="1.3" strokeDasharray="5 4" />
              <path d="M704,452 C 800,452 845,300 872,300" fill="none" stroke="#bdbdbd" strokeWidth="1.3" strokeDasharray="5 4" />
              <path d="M690,360 C 800,360 845,302 872,302" fill="none" stroke="#dcdcdc" strokeWidth="1" strokeDasharray="3 4" />
              <path d="M690,378 C 800,378 845,304 872,304" fill="none" stroke="#dcdcdc" strokeWidth="1" strokeDasharray="3 4" />
              <line x1="908" y1="300" x2="1082" y2="300" stroke="#bdbdbd" strokeWidth="1.4" strokeDasharray="5 4" />
              <line x1="890" y1="318" x2="890" y2="332" stroke="#bdbdbd" strokeWidth="1.3" strokeDasharray="4 3" />
              <path d="M268,150 C 282,148 292,150 301,152" fill="none" stroke="#9a9a9a" strokeWidth="1.3" />
              <path d="M301,152 l -8,-3 M301,152 l -5,6" fill="none" stroke="#9a9a9a" strokeWidth="1.3" />
              <path d="M268,292 C 282,290 292,290 301,290" fill="none" stroke="#9a9a9a" strokeWidth="1.3" />
              <path d="M301,290 l -8,-2 M301,290 l -6,6" fill="none" stroke="#9a9a9a" strokeWidth="1.3" />
              <path d="M952,258 C 938,268 918,282 905,294" fill="none" stroke="#9a9a9a" strokeWidth="1.3" />
              <path d="M905,294 l 10,-1 M905,294 l 1,-8" fill="none" stroke="#9a9a9a" strokeWidth="1.3" />
            </svg>

            {/* headers */}
            <div style={{ position: 'absolute', left: 260, top: 32, width: 200, textAlign: 'center', fontSize: 23, fontWeight: 700, color: '#2b2b2b' }}>{prevId}</div>
            <div style={{ position: 'absolute', left: 260, top: 62, width: 200, textAlign: 'center', fontSize: 15, color: '#9a9a9a' }}>({inN}, {inN}, {numChannels})</div>
            <div style={{ position: 'absolute', left: 548, top: 32, width: 200, textAlign: 'center', fontSize: 23, fontWeight: 700, color: '#2b2b2b' }}>{interLbl}</div>
            <div style={{ position: 'absolute', left: 548, top: 62, width: 200, textAlign: 'center', fontSize: 15, color: '#9a9a9a' }}>({outN}, {outN}, {numChannels})</div>
            <div style={{ position: 'absolute', left: 1040, top: 32, width: 200, textAlign: 'center', fontSize: 23, fontWeight: 700, color: '#2b2b2b' }}>{convId}</div>
            <div style={{ position: 'absolute', left: 1040, top: 62, width: 200, textAlign: 'center', fontSize: 15, color: '#9a9a9a' }}>({outN}, {outN}, {numChannels})</div>

            {/* per-channel: kernel icon + input + sliding kernel + intermediate + cursor */}
            {chans.map((i) => (
              <React.Fragment key={i}>
                <div style={{ position: 'absolute', left: 272, top: icoTops[i], width: 16, height: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr', gap: 1, background: '#cccccc' }}>
                  {KICON[i].map((c, k) => <div key={k} style={{ background: c }} />)}
                </div>
                <canvas ref={(el) => (inRefs.current[i] = el)} width={inN} height={inN} style={{ position: 'absolute', left: 304, top: tops[i], width: 108, height: 108, border: '2px solid #b3b3b3', background: '#faf8f5' }} />
                <div ref={(el) => (kerRefs.current[i] = el)} style={{ position: 'absolute', left: 304, top: tops[i], width: 3 * cpIn, height: 3 * cpIn, border: '1.5px solid rgba(50,50,50,0.9)', background: 'rgba(110,150,205,0.22)', boxSizing: 'border-box', pointerEvents: 'none', willChange: 'transform' }} />
                <canvas ref={(el) => (intRefs.current[i] = el)} width={outN} height={outN}
                        onClick={() => onEnterFormula && onEnterFormula(i)}
                        style={{ position: 'absolute', left: 592, top: tops[i], width: 108, height: 108, border: '2px solid #b3b3b3', background: '#faf8f5', cursor: onEnterFormula ? 'pointer' : 'default' }} />
                <div ref={(el) => (curRefs.current[i] = el)} style={{ position: 'absolute', left: 592, top: tops[i], width: cpOut, height: cpOut, border: '1.4px solid rgba(50,50,50,0.85)', boxSizing: 'border-box', pointerEvents: 'none', willChange: 'transform' }} />
              </React.Fragment>
            ))}

            {/* omitted-channel ellipsis */}
            <div style={{ position: 'absolute', left: 280, top: 352, fontSize: 30, lineHeight: 0.5, letterSpacing: 2, color: '#c4c4c4', fontWeight: 700 }}>⋮</div>
            <div style={{ position: 'absolute', left: 360, top: 352, fontSize: 30, lineHeight: 0.5, color: '#c4c4c4', fontWeight: 700 }}>⋮</div>
            <div style={{ position: 'absolute', left: 648, top: 352, fontSize: 30, lineHeight: 0.5, color: '#c4c4c4', fontWeight: 700 }}>⋮</div>

            {/* conv selected output */}
            <canvas ref={outRef} width={outN} height={outN}
                    onClick={() => onEnterFormula && onEnterFormula(0)}
                    style={{ position: 'absolute', left: 1084, top: 244, width: 112, height: 112, border: '3px solid #585858', background: '#faf8f5', cursor: onEnterFormula ? 'pointer' : 'default' }} />

            {/* summation + bias */}
            <div style={{ position: 'absolute', left: 872, top: 282, width: 36, height: 36, border: '1.5px solid #8a8a8a', borderRadius: 2, background: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 26, color: '#666', lineHeight: 1 }}>+</div>
            <div style={{ position: 'absolute', left: 881, top: 332, width: 18, height: 18, borderRadius: '50%', background: '#34b3a0' }} />
            <div style={{ position: 'absolute', left: 855, top: 354, width: 70, textAlign: 'center', fontSize: 15, fontStyle: 'italic', fontFamily: 'Georgia,serif', color: '#8a8a8a' }}>Bias</div>

            {/* annotations */}
            <div style={{ position: 'absolute', left: 40, top: 96, width: 222, textAlign: 'right', fontFamily: 'Georgia,serif', fontStyle: 'italic', fontSize: 15.5, lineHeight: 1.4, color: '#8a8a8a' }}>
              <div style={{ fontStyle: 'normal', color: '#6a6a6a', fontSize: 14, marginBottom: 3 }}>Kernel <span style={{ display: 'inline-block', width: 14, height: 12, verticalAlign: 'middle', background: 'linear-gradient(90deg,#bcd4ec 50%,#f0f0f0 50%)', border: '1px solid #c8c8c8' }} /></div>
              {A1[0]}<br />{A1[1]}<br />{A1[2]}<br /><b style={{ color: '#555' }}>{A1[3]}</b>{A1[4]}
            </div>
            <div style={{ position: 'absolute', left: 30, top: 248, width: 232, textAlign: 'right', fontFamily: 'Georgia,serif', fontStyle: 'italic', fontSize: 15.5, lineHeight: 1.4, color: '#8a8a8a' }}>
              {A2[0]}<br />{A2[1]}<br /><b style={{ color: '#555' }}>{A2[2]}</b>{A2[3]}
            </div>
            <div style={{ position: 'absolute', left: 948, top: 200, width: 170, textAlign: 'left', fontFamily: 'Georgia,serif', fontStyle: 'italic', fontSize: 15.5, lineHeight: 1.4, color: '#8a8a8a' }}>
              {A3[0]}<br />{A3[1]}
            </div>

          </div>
        </div>
      </Modal>
    );
  }

  // ─────────── Level 2: Convolutional Interactive Formula (Fig 6A) ───────────
  function ConvFormulaView({ inputMap, inputIsGray, kernel, interMap, outputMap, bias,
                             singleChannel, channelLabel, oc, colormap, lang, onBack, onClose }) {
    const zh = lang === 'zh';
    const k = (kernel && kernel.length) || 3;
    // 右侧矩阵: 单通道时直接是真实输出 (含bias); 多通道时是该通道的中间结果
    const rightMap = singleChannel ? outputMap : interMap;
    const rN = rightMap ? rightMap.length : 0;
    const inN = inputMap ? inputMap.length : 0;

    // 默认窗口落在中心(角落多为 0),让公式一开始就有真实数字
    const [pos, setPos] = useState({ x: Math.floor(rN / 2), y: Math.floor(rN / 2) });
    const [playing, setPlaying] = useState(false);
    useEffect(() => {
      if (!playing || rN === 0) return;
      let i = pos.y * rN + pos.x;
      const id = setInterval(() => {
        i = (i + 1) % (rN * rN);
        setPos({ x: i % rN, y: Math.floor(i / rN) });
      }, 600);
      return () => clearInterval(id);
    }, [playing, rN]);
    const ox = Math.min(pos.x, Math.max(0, rN - 1));
    const oy = Math.min(pos.y, Math.max(0, rN - 1));

    const inMax = inputIsGray ? 1 : pctMaxAbs(inputMap, 0.8);
    const rMax = pctMaxAbs(rightMap, 0.7);
    let kMax = 0; if (kernel) for (const r of kernel) for (const v of r) kMax = Math.max(kMax, Math.abs(v));
    kMax = kMax || 1;

    // 9 项: a × w
    const terms = [];
    let dot = 0;
    for (let dy = 0; dy < k; dy++) {
      for (let dx = 0; dx < k; dx++) {
        const a = (inputMap[oy + dy] && inputMap[oy + dy][ox + dx]) || 0;
        const w = kernel[dy][dx];
        dot += a * w;
        terms.push({ a, w });
      }
    }
    const result = singleChannel ? dot + bias : dot;

    const MAT = 124;
    const L = zh ? {
      tag: '卷积运算 · 公式', back: '返回结构图',
      input: '输入', out: singleChannel ? '输出' : '中间结果',
      note: singleChannel
        ? '9 个乘积相加,再加偏置 = 输出像素'
        : `这是第 ${channelLabel} 个输入通道的贡献 (中间结果);偏置在上一级求和时统一加`,
      hint: '在任一矩阵上 hover / 拖动 → 窗口同步, 公式实时更新',
      bias: '偏置',
    } : {
      tag: 'CONVOLUTION · FORMULA', back: 'structure view',
      input: 'input', out: singleChannel ? 'output' : 'intermediate',
      note: singleChannel
        ? '9 products summed, then add bias = output pixel'
        : `contribution of input channel ${channelLabel} (intermediate); bias is added once when summing`,
      hint: 'hover / drag either matrix → window syncs, formula updates live',
      bias: 'bias',
    };

    return (
      <Modal onClose={onClose} bg="#fff" maxW="94vw">
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
          padding: '12px 16px 8px', borderBottom: '0.5px solid rgba(0,0,0,0.08)',
        }}>
          <button onClick={(e) => { e.stopPropagation(); onBack(); }} style={{
            display: 'flex', alignItems: 'center', gap: 5, border: 'none', background: 'transparent',
            cursor: 'pointer', color: 'var(--text-2)', fontSize: 11, fontFamily: 'var(--font-mono)', padding: 0,
          }}>
            <svg width="8" height="12" viewBox="0 0 10 16"><path d="M8 1L2 8l6 7" stroke="currentColor" strokeWidth="1.6" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg>
            {L.back}
          </button>
          <span style={{
            padding: '2px 8px', borderRadius: 4, background: 'rgba(232,185,44,0.15)',
            fontSize: 9, fontWeight: 700, color: '#8a6b14', letterSpacing: '0.12em',
            textTransform: 'uppercase', fontFamily: 'var(--font-mono)',
          }}>{L.tag}</span>
          <span style={{ flex: 1 }} />
          <PlayBtn playing={playing} onToggle={() => setPlaying(p => !p)} />
          <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>
            ({ox},{oy})
          </span>
          <CloseBtn onClose={onClose} />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '6px 18px' }}>
          {/* 输入矩阵 */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
            <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{L.input} {inN}×{inN}</span>
            <MatrixView data={inputMap} maxAbs={inMax} size={MAT} colormap={colormap} grayscale={inputIsGray}
              win={{ x: ox, y: oy, w: k, h: k }}
              onPick={(cx, cy) => { setPlaying(false); setPos({ x: Math.min(cx, rN - 1), y: Math.min(cy, rN - 1) }); }} />
          </div>

          {/* 公式 */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0, minWidth: 146 }}>
            {terms.map((tm, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 3, height: 14 }}>
                <span style={{ width: 7, textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-3)' }}>
                  {i === 0 ? ' ' : '+'}
                </span>
                <NumCell v={tm.a} kind="act" maxAbs={inMax} colormap={colormap} w={38} />
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--text-3)' }}>×</span>
                <NumCell v={tm.w} kind="weight" maxAbs={kMax} w={38} />
              </div>
            ))}
            {singleChannel && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 3, height: 14 }}>
                <span style={{ width: 7, textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-3)' }}>+</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--text-3)' }}>{L.bias}</span>
                <NumCell v={bias} kind="weight" maxAbs={Math.max(0.5, Math.abs(bias))} w={38} />
              </div>
            )}
            <div style={{ height: 1, background: 'var(--text-2)', margin: '2px 0' }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              <span style={{ color: 'var(--text-3)' }}>=</span>
              <b style={{ fontSize: 13, color: 'var(--text)' }}>{result.toFixed(2)}</b>
            </div>
          </div>

          {/* 输出/中间矩阵 */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
            <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{L.out} {rN}×{rN}</span>
            <MatrixView data={rightMap} maxAbs={rMax} size={MAT} colormap={colormap}
              win={{ x: ox, y: oy, w: 1, h: 1 }}
              onPick={(cx, cy) => { setPlaying(false); setPos({ x: cx, y: cy }); }} />
          </div>
        </div>

        <div style={{ padding: '0 18px 12px', display: 'flex', flexDirection: 'column', gap: 3 }}>
          <span style={{ fontSize: 10, fontFamily: 'Georgia, serif', fontStyle: 'italic', color: 'var(--text-2)' }}>{L.note}</span>
          <span style={{ fontSize: 9.5, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{L.hint}</span>
        </div>
      </Modal>
    );
  }

  // ─────────── Level 2: 激活函数视图 (Claude Design UI · 接真实数据) ───────────
  // 替换原 ActivationFormulaView。relu: max(0,x)=y; sigmoid: σ(x)=y; tanh: tanh(x)=y。
  // 配色沿用 Claude Design: 负→橙 / 正→蓝 / 零→白。useState/useEffect/useRef/useMemo / Modal / CloseBtn 均来自本文件作用域。
  function ActivationFormulaView({ layer, layerIdx, channelIdx, catIndex, onClose, lang, colormap, activationFn }) {
    const zh = lang === 'zh';
    const data = NETWORK_DATA[catIndex];
    const prev = LAYERS[layerIdx - 1];
    const inputMap = rawChannel(data.raw[prev.id], channelIdx);   // 激活前(带符号)
    const outputMap = rawChannel(data.raw[layer.id], channelIdx); // 激活后
    const n = outputMap ? outputMap.length : 0;
    const fn = activationFn || 'relu';

    const [pos, setPos] = useState(() => {        // 默认落在最负的像素 → 一进来就演示 max(0,负)=0
      let mn = Infinity, mx = (n >> 1) || 0, my = (n >> 1) || 0;
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { const v = (inputMap[r] && inputMap[r][c]) || 0; if (v < mn) { mn = v; mx = c; my = r; } }
      return { x: mx, y: my };
    });
    const [playing, setPlaying] = useState(false);
    useEffect(() => {
      if (!playing || n === 0) return;
      let i = pos.y * n + pos.x;
      const id = setInterval(() => { i = (i + 1) % (n * n); setPos({ x: i % n, y: Math.floor(i / n) }); }, 240);
      return () => clearInterval(id);
    }, [playing, n]);
    const x = Math.min(pos.x, Math.max(0, n - 1)), y = Math.min(pos.y, Math.max(0, n - 1));
    const inVal = (inputMap && inputMap[y] && inputMap[y][x]) || 0;
    const outVal = (outputMap && outputMap[y] && outputMap[y][x]) || 0;

    const inRef = useRef(null), outRef = useRef(null);

    // 稳定数据签名: 仅当通道/数字/层变化时才重画(避免每次渲染重画/重启)
    const sig = useMemo(() => {
      let a = 0;
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) a += (inputMap[r][c] || 0) * (r * 31 + c + 1);
      return n + '|' + channelIdx + '|' + (layer && layer.id) + '|' + Math.round(a * 1000);
    }, [inputMap, n, channelIdx, layer]);

    // CD 配色饱和点(自适应, 85 分位), 负橙正蓝
    const scale = useMemo(() => {
      const negs = [], poss = [];
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { const v = inputMap[r][c]; if (v < 0) negs.push(-v); else if (v > 0) poss.push(v); }
      negs.sort((a, b) => a - b); poss.sort((a, b) => a - b);
      return { negS: Math.max(0.4, negs[Math.floor(negs.length * 0.85)] || 0.6), posS: Math.max(0.25, poss[Math.floor(poss.length * 0.85)] || 0.4) };
    }, [sig]);

    function color(v) {
      if (v < 0) { const t = Math.min(1, -v / scale.negS); return [Math.round(252 + (233 - 252) * t), Math.round(252 + (150 - 252) * t), Math.round(252 + (118 - 252) * t)]; }
      if (v > 0) { const t = Math.min(1, v / scale.posS); return [Math.round(252 + (168 - 252) * t), Math.round(252 + (200 - 252) * t), Math.round(252 + (230 - 252) * t)]; }
      return [252, 252, 252];
    }
    function paint(canvas, M) {
      if (!canvas || !M.length) return;
      const ctx = canvas.getContext('2d'); const img = ctx.createImageData(n, n);
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { const a = color(M[r][c]); const i = (r * n + c) * 4; img.data[i] = a[0]; img.data[i + 1] = a[1]; img.data[i + 2] = a[2]; img.data[i + 3] = 255; }
      ctx.putImageData(img, 0, 0);
    }
    useEffect(() => { paint(inRef.current, inputMap); paint(outRef.current, outputMap); }, [sig]);

    // 选像素(hover / 点 / 拖, 兼容触摸)
    const pickFrom = (e, ref) => {
      const el = ref.current; if (!el) return;
      const rc = el.getBoundingClientRect();
      const cx = Math.max(0, Math.min(n - 1, Math.floor((e.clientX - rc.left) / (rc.width / n))));
      const cy = Math.max(0, Math.min(n - 1, Math.floor((e.clientY - rc.top) / (rc.height / n))));
      setPlaying(false); setPos({ x: cx, y: cy });
    };

    const title = zh
      ? ({ relu: 'ReLU 激活', sigmoid: 'Sigmoid 激活', tanh: 'Tanh 激活' }[fn])
      : ({ relu: 'ReLU Activation', sigmoid: 'Sigmoid Activation', tanh: 'Tanh Activation' }[fn]);
    const hint = zh
      ? { a: '在矩阵上', b: '滑动/点击', c: '切换像素。' }
      : { a: '', b: 'Hover over', c: ' the matrices to change pixel.' };

    const S = 0.74, IMG = 297, cell = IMG / n;
    const fmtNum = (v) => v.toFixed(2).replace('-', '−');
    const cbtn = { width: 34, height: 34, borderRadius: '50%', background: '#c7cdd2', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' };
    const numBox = (txt, hl) => (
      <span style={{ width: 46, height: 42, border: '1px solid ' + (hl ? '#e3b29c' : '#d6d6d6'), borderRadius: 2, background: hl ? '#f6d4c6' : '#fbfbfb', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13, color: hl ? '#6a5a52' : '#7a7a7a' }}>{txt}</span>
    );
    const marker = (
      <div style={{ position: 'absolute', left: Math.round(x * cell), top: Math.round(y * cell), width: cell, height: cell, border: '1.5px solid #333', boxShadow: '0 0 0 1px #fff', pointerEvents: 'none', boxSizing: 'border-box' }} />
    );
    const heat = (ref, M) => (
      <div style={{ position: 'absolute', left: 46, top: 143, width: IMG, height: IMG, border: '1px solid #eee', cursor: 'crosshair' }}
        onPointerDown={(e) => pickFrom(e, ref)} onPointerMove={(e) => { if (e.buttons || e.pressure) pickFrom(e, ref); }} onMouseMove={(e) => pickFrom(e, ref)}>
        <canvas ref={ref} width={n} height={n} style={{ width: IMG, height: IMG, display: 'block', imageRendering: 'pixelated' }} />
        {marker}
      </div>
    );

    return (
      <Modal onClose={onClose} bg="#fff" maxW="96vw">
        <div style={{ width: 1024 * S, height: 530 * S, overflow: 'hidden' }}>
          <div style={{ width: 1024, height: 530, transformOrigin: 'top left', transform: 'scale(' + S + ')', fontFamily: "-apple-system,'Helvetica Neue',Helvetica,Arial,sans-serif", position: 'relative' }}>
            <div style={{ position: 'absolute', left: 10, top: 10, width: 1004, height: 510, background: '#fff', borderRadius: 16, border: '1px solid rgba(0,0,0,0.05)', boxShadow: '0 1px 3px rgba(0,0,0,0.05),0 10px 34px rgba(0,0,0,0.09)' }}>

              <div style={{ position: 'absolute', left: 0, top: 30, width: 1004, textAlign: 'center', fontSize: 36, fontWeight: 400, color: '#3f4e5d', letterSpacing: 0.2 }}>{title}</div>

              <div style={{ position: 'absolute', right: 24, top: 30, display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={cbtn}><svg width="18" height="18" viewBox="0 0 24 24"><circle cx="12" cy="6.5" r="1.7" fill="#fff" /><rect x="10.6" y="10" width="2.8" height="9" rx="1.3" fill="#fff" /></svg></div>
                <div style={cbtn} onClick={() => setPlaying(p => !p)}>
                  {playing
                    ? <svg width="14" height="14" viewBox="0 0 24 24"><rect x="6" y="5" width="4" height="14" rx="1" fill="#fff" /><rect x="14" y="5" width="4" height="14" rx="1" fill="#fff" /></svg>
                    : <svg width="16" height="16" viewBox="0 0 24 24"><path d="M7 4.5 L19 12 L7 19.5 Z" fill="#fff" /></svg>}
                </div>
                <div style={cbtn} onClick={onClose}><svg width="14" height="14" viewBox="0 0 24 24"><path d="M5 5 L19 19 M19 5 L5 19" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" /></svg></div>
              </div>

              <div style={{ position: 'absolute', left: 46, top: 113, fontSize: 21, color: '#3f4e5d' }}>{zh ? '输入' : 'Input'} <span style={{ color: '#9aa6b0', fontSize: 16 }}>({n}, {n})</span></div>
              <div style={{ position: 'absolute', left: 681, top: 113, fontSize: 21, color: '#3f4e5d' }}>{zh ? '输出' : 'Output'} <span style={{ color: '#9aa6b0', fontSize: 16 }}>({n}, {n})</span></div>

              {heat(inRef, inputMap)}
              <div style={{ position: 'absolute', left: 681, top: 143, width: IMG, height: IMG, border: '1px solid #eee', cursor: 'crosshair' }}
                onPointerDown={(e) => pickFrom(e, outRef)} onMouseMove={(e) => pickFrom(e, outRef)}>
                <canvas ref={outRef} width={n} height={n} style={{ width: IMG, height: IMG, display: 'block', imageRendering: 'pixelated' }} />
                <div style={{ position: 'absolute', left: Math.round(x * cell), top: Math.round(y * cell), width: cell, height: cell, border: '1.5px solid #333', boxShadow: '0 0 0 1px #fff', pointerEvents: 'none', boxSizing: 'border-box' }} />
              </div>

              <div style={{ position: 'absolute', left: 353, top: 248, width: 318, height: 48, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7, fontSize: 33, fontWeight: 400, color: '#3f4e5d' }}>
                {fn === 'relu' && <><span>max(</span>{numBox('0', false)}<span>,</span>{numBox(fmtNum(inVal), true)}<span>)</span></>}
                {fn === 'sigmoid' && <><span>σ(</span>{numBox(fmtNum(inVal), true)}<span>)</span></>}
                {fn === 'tanh' && <><span>tanh(</span>{numBox(fmtNum(inVal), true)}<span>)</span></>}
                <span>=</span>{numBox(fmtNum(outVal), false)}
              </div>

              <div style={{ position: 'absolute', left: 0, top: 462, width: 1004, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
                <div style={{ width: 36, height: 36, borderRadius: '50%', background: '#dce7f4', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <svg width="18" height="20" viewBox="0 0 24 26"><path d="M5 2 L5 18 L9 14 L12 21 L15 20 L12 13 L18 13 Z" fill="#5b6b7a" stroke="#fff" strokeWidth="1" strokeLinejoin="round" /></svg>
                </div>
                <div style={{ fontSize: 21, fontStyle: 'italic', color: '#7a8893' }}>{hint.a}<span style={{ fontWeight: 700, color: '#3f4e5d' }}>{hint.b}</span>{hint.c}</div>
              </div>

            </div>
          </div>
        </div>
      </Modal>
    );
  }


  // ─────────── Level 2: 最大池化视图 (Claude Design UI · 接真实数据) ───────────
  // 替换原 MaxPoolFormulaView。输入=ReLU 输出(非负, 单色蓝), 2×2 取最大 → 输出尺寸减半。
  // useState/useEffect/useRef/useMemo / Modal / CloseBtn / rawChannel 均来自本文件作用域。
  function MaxPoolFormulaView({ layer, layerIdx, channelIdx, catIndex, onClose, lang, colormap }) {
    const zh = lang === 'zh';
    const data = NETWORK_DATA[catIndex];
    const prev = LAYERS[layerIdx - 1];
    const inputMap = rawChannel(data.raw[prev.id], channelIdx);   // ReLU 输出
    const outputMap = rawChannel(data.raw[layer.id], channelIdx); // 池化输出
    const inN = inputMap ? inputMap.length : 0;
    const outN = outputMap ? outputMap.length : 0;

    const [pos, setPos] = useState(() => {            // 默认落在最强的池化像素
      let mx = -Infinity, px = 0, py = 0;
      for (let r = 0; r < outN; r++) for (let c = 0; c < outN; c++) { const v = (outputMap[r] && outputMap[r][c]) || 0; if (v > mx) { mx = v; px = c; py = r; } }
      return { x: px, y: py };
    });
    const [playing, setPlaying] = useState(false);
    useEffect(() => {
      if (!playing || outN === 0) return;
      let i = pos.y * outN + pos.x;
      const id = setInterval(() => { i = (i + 1) % (outN * outN); setPos({ x: i % outN, y: Math.floor(i / outN) }); }, 320);
      return () => clearInterval(id);
    }, [playing, outN]);
    const ox = Math.min(pos.x, Math.max(0, outN - 1)), oy = Math.min(pos.y, Math.max(0, outN - 1));

    // 2×2 窗口真值 [TL,TR,BL,BR] + 最大值
    const vals = [];
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) vals.push((inputMap[oy * 2 + dy] && inputMap[oy * 2 + dy][ox * 2 + dx]) || 0);
    const result = Math.max.apply(null, vals);

    const inRef = useRef(null), outRef = useRef(null);

    // 稳定签名: 仅当通道/数字/层变化时才重画
    const sig = useMemo(() => {
      let a = 0;
      for (let r = 0; r < inN; r++) for (let c = 0; c < inN; c++) a += (inputMap[r][c] || 0) * (r * 31 + c + 1);
      return inN + '|' + channelIdx + '|' + (layer && layer.id) + '|' + Math.round(a * 1000);
    }, [inputMap, inN, channelIdx, layer]);

    // 单色蓝(CD), 自适应饱和点(92 分位)
    const posS = useMemo(() => {
      const ps = [];
      for (let r = 0; r < inN; r++) for (let c = 0; c < inN; c++) { const v = inputMap[r][c]; if (v > 0) ps.push(v); }
      ps.sort((a, b) => a - b);
      return Math.max(0.25, ps[Math.floor(ps.length * 0.92)] || 0.5);
    }, [sig]);
    function blue(v) { const t = Math.min(1, Math.max(0, v / posS)); return [Math.round(252 + (168 - 252) * t), Math.round(252 + (200 - 252) * t), Math.round(252 + (230 - 252) * t)]; }
    function paint(canvas, M, n) {
      if (!canvas || !M.length) return;
      const ctx = canvas.getContext('2d'); const img = ctx.createImageData(n, n);
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { const a = blue(M[r][c]); const i = (r * n + c) * 4; img.data[i] = a[0]; img.data[i + 1] = a[1]; img.data[i + 2] = a[2]; img.data[i + 3] = 255; }
      ctx.putImageData(img, 0, 0);
    }
    useEffect(() => { paint(inRef.current, inputMap, inN); paint(outRef.current, outputMap, outN); }, [sig]);

    // 选窗口: 在输入上 → 落到所在 2×2 块; 在输出上 → 直接选像素
    const pickIn = (e) => { const el = inRef.current; if (!el) return; const rc = el.getBoundingClientRect(); const cx = Math.floor((e.clientX - rc.left) / (rc.width / inN)), cy = Math.floor((e.clientY - rc.top) / (rc.height / inN)); setPlaying(false); setPos({ x: Math.max(0, Math.min(outN - 1, cx >> 1)), y: Math.max(0, Math.min(outN - 1, cy >> 1)) }); };
    const pickOut = (e) => { const el = outRef.current; if (!el) return; const rc = el.getBoundingClientRect(); const cx = Math.floor((e.clientX - rc.left) / (rc.width / outN)), cy = Math.floor((e.clientY - rc.top) / (rc.height / outN)); setPlaying(false); setPos({ x: Math.max(0, Math.min(outN - 1, cx)), y: Math.max(0, Math.min(outN - 1, cy)) }); };

    const S = 0.70, IW = 304, IH = 308;
    const cInX = IW / inN, cInY = IH / inN, cOutX = IW / outN, cOutY = IH / outN;
    const fmt = (v) => v.toFixed(2);
    const cbtn = { width: 34, height: 34, borderRadius: '50%', background: '#c7cdd2', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' };
    const title = zh ? '最大池化' : 'Max Pooling';
    const hint = zh ? { b: '滑动/点击', c: '矩阵移动 2×2 窗口。' } : { b: 'Hover over', c: ' the matrices to change kernel position.' };

    return (
      <Modal onClose={onClose} bg="#fff" maxW="96vw">
        <div style={{ width: 1102 * S, height: 582 * S, overflow: 'hidden' }}>
          <div style={{ width: 1102, height: 582, transformOrigin: 'top left', transform: 'scale(' + S + ')', position: 'relative', overflow: 'hidden', fontFamily: "-apple-system,'Helvetica Neue',Helvetica,Arial,sans-serif" }}>
            <div style={{ position: 'absolute', left: -60, top: -80, width: 360, height: 300, background: 'radial-gradient(closest-side,rgba(120,170,225,0.18),transparent)', filter: 'blur(8px)' }} />
            <div style={{ position: 'absolute', left: 430, top: -120, width: 340, height: 300, background: 'radial-gradient(closest-side,rgba(225,150,140,0.16),transparent)', filter: 'blur(8px)' }} />
            <div style={{ position: 'absolute', right: -40, top: -70, width: 340, height: 300, background: 'radial-gradient(closest-side,rgba(120,170,225,0.16),transparent)', filter: 'blur(8px)' }} />

            <div style={{ position: 'absolute', left: 30, top: 38, width: 1042, height: 506, background: '#fff', borderRadius: 18, border: '1px solid rgba(0,0,0,0.04)', boxShadow: '0 1px 3px rgba(0,0,0,0.05),0 14px 44px rgba(0,0,0,0.12)' }}>

              <div style={{ position: 'absolute', left: 0, top: 22, width: 1042, textAlign: 'center', fontSize: 36, fontWeight: 400, color: '#3f4e5d', letterSpacing: 0.2 }}>{title}</div>

              <div style={{ position: 'absolute', right: 24, top: 22, display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={cbtn}><svg width="18" height="18" viewBox="0 0 24 24"><circle cx="12" cy="6.5" r="1.7" fill="#fff" /><rect x="10.6" y="10" width="2.8" height="9" rx="1.3" fill="#fff" /></svg></div>
                <div style={cbtn} onClick={() => setPlaying(p => !p)}>
                  {playing
                    ? <svg width="16" height="16" viewBox="0 0 24 24"><path d="M7 4.5 L19 12 L7 19.5 Z" fill="#fff" /></svg>
                    : <svg width="16" height="16" viewBox="0 0 24 24"><rect x="6.5" y="5" width="4" height="14" rx="1.2" fill="#fff" /><rect x="13.5" y="5" width="4" height="14" rx="1.2" fill="#fff" /></svg>}
                </div>
                <div style={cbtn} onClick={onClose}><svg width="16" height="16" viewBox="0 0 24 24"><path d="M6 6 L18 18 M18 6 L6 18" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" /></svg></div>
              </div>

              <div style={{ position: 'absolute', left: 48, top: 84, width: 304, textAlign: 'center', fontSize: 28, color: '#5a6a78' }}>{zh ? '输入' : 'Input'} ({inN}, {inN})</div>
              <div style={{ position: 'absolute', left: 657, top: 84, width: 304, textAlign: 'center', fontSize: 28, color: '#5a6a78' }}>{zh ? '输出' : 'Output'} ({outN}, {outN})</div>

              <div style={{ position: 'absolute', left: 48, top: 120, width: IW, height: IH, cursor: 'crosshair' }}
                onPointerDown={pickIn} onPointerMove={(e) => { if (e.buttons || e.pressure) pickIn(e); }} onMouseMove={pickIn}>
                <canvas ref={inRef} width={inN} height={inN} style={{ width: IW, height: IH, display: 'block', imageRendering: 'pixelated' }} />
                <div style={{ position: 'absolute', left: Math.round(ox * 2 * cInX), top: Math.round(oy * 2 * cInY), width: Math.round(2 * cInX), height: Math.round(2 * cInY), border: '1.6px solid #2c2c2c', boxSizing: 'border-box', pointerEvents: 'none' }}>
                  <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: '#2c2c2c' }} />
                  <div style={{ position: 'absolute', top: '50%', left: 0, right: 0, height: 1, background: '#2c2c2c' }} />
                </div>
              </div>

              <div style={{ position: 'absolute', left: 657, top: 120, width: IW, height: IH, cursor: 'crosshair' }}
                onPointerDown={pickOut} onMouseMove={pickOut}>
                <canvas ref={outRef} width={outN} height={outN} style={{ width: IW, height: IH, display: 'block', imageRendering: 'pixelated' }} />
                <div style={{ position: 'absolute', left: Math.round(ox * cOutX), top: Math.round(oy * cOutY), width: Math.round(cOutX), height: Math.round(cOutY), border: '1.6px solid #2c2c2c', boxSizing: 'border-box', pointerEvents: 'none' }} />
              </div>

              <div style={{ position: 'absolute', left: 352, top: 240, width: 305, height: 110, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 9, fontSize: 33, fontWeight: 400, color: '#3f4e5d' }}>
                <span>max(</span>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr', gap: 1, background: '#c4c4c4', border: '1px solid #c4c4c4', width: 75, height: 75 }}>
                  {vals.map((v, i) => (
                    <div key={i} style={{ background: v === result ? 'rgba(168,200,230,0.55)' : (i < 2 ? '#fff' : '#f3f3f3'), display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13, color: '#333' }}>{fmt(v)}</div>
                  ))}
                </div>
                <span>)</span><span>=</span>
                <div style={{ width: 46, height: 52, border: '1.5px solid #8a8a8a', borderRadius: 1, background: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, color: '#333' }}>{fmt(result)}</div>
              </div>

              <div style={{ position: 'absolute', left: 0, top: 452, width: 1042, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
                <div style={{ width: 36, height: 36, borderRadius: '50%', background: '#dce7f4', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <svg width="18" height="20" viewBox="0 0 24 26"><path d="M5 2 L5 18 L9 14 L12 21 L15 20 L12 13 L18 13 Z" fill="#5b6b7a" stroke="#fff" strokeWidth="1" strokeLinejoin="round" /></svg>
                </div>
                <div style={{ fontSize: 21, fontStyle: 'italic', color: '#7a8893' }}><span style={{ fontWeight: 700, color: '#3f4e5d' }}>{hint.b}</span>{hint.c}</div>
              </div>

            </div>
          </div>
        </div>
      </Modal>
    );
  }

  // ════════════════════════════════════════════════════════════════
  //  Softmax Formula (Fig 6D) — logits 圆 + 双向联动
  // ════════════════════════════════════════════════════════════════
  function SoftmaxFormulaView({ channelIdx, catIndex, onClose, lang }) {
    const zh = lang === 'zh';
    const data = NETWORK_DATA[catIndex];
    const logits = data.raw.dense_1;
    const probs = data.raw.output;
    const [sel, setSel] = useState(channelIdx);
    const [hover, setHover] = useState(null);     // 联动高亮的类别 idx
    useEffect(() => { setSel(channelIdx); }, [channelIdx]);
    const active = hover != null ? hover : sel;

    const maxL = Math.max(...logits), minL = Math.min(...logits);
    const span = Math.max(1e-4, maxL - minL);
    // 浅橙→深橙
    const orange = (v) => {
      const t = (v - minL) / span;
      const r = Math.round(255 - t * 40), g = Math.round(220 - t * 130), b = Math.round(170 - t * 130);
      return `rgb(${r},${g},${b})`;
    };
    const expVals = logits.map(v => Math.exp(v - maxL));
    const denom = expVals.reduce((a, b) => a + b, 0);
    const prob = expVals[active] / denom;

    const title = zh ? `Softmax 分类 · 类别 ${active}` : `Softmax · class ${active}`;

    return (
      <Modal onClose={onClose}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 18px 8px', borderBottom: '0.5px solid rgba(0,0,0,0.08)' }}>
          <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--text)' }}>{title}</span>
          <span style={{ flex: 1 }} />
          <CloseBtn onClose={onClose} />
        </div>

        <div style={{ display: 'flex', gap: 18, padding: '14px 20px', alignItems: 'center' }}>
          {/* logits 圆 (竖排 10 个) */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {logits.map((v, i) => {
              const on = i === active;
              return (
                <div key={i}
                  onPointerEnter={() => setHover(i)}
                  onPointerLeave={() => setHover(null)}
                  onClick={(e) => { e.stopPropagation(); setSel(i); }}
                  style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
                  <span style={{ width: 10, textAlign: 'right', fontSize: 10, fontFamily: 'var(--font-mono)', color: on ? 'var(--text)' : 'var(--text-3)', fontWeight: on ? 700 : 400 }}>{i}</span>
                  <div style={{
                    width: 18, height: 18, borderRadius: 9, background: orange(v),
                    border: on ? '2px solid #c2410c' : '1px solid rgba(0,0,0,0.15)',
                    transform: on ? 'scale(1.15)' : 'none', transition: 'transform 120ms ease',
                  }} />
                  <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', width: 34 }}>{v.toFixed(2)}</span>
                  <div style={{ width: 60, height: 6, background: 'rgba(0,0,0,0.05)', borderRadius: 1, overflow: 'hidden' }}>
                    <div style={{ width: `${probs[i] * 100}%`, height: '100%', background: on ? '#c2410c' : 'rgba(60,60,67,0.4)' }} />
                  </div>
                  <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: on ? 'var(--text)' : 'var(--text-3)', width: 36, textAlign: 'right' }}>{(probs[i] * 100).toFixed(1)}%</span>
                </div>
              );
            })}
          </div>

          {/* 公式 */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-2)', marginBottom: 8 }}>
              {zh ? `类别 ${active} 的概率` : `probability of class ${active}`}
            </div>
            <div style={{ textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text)' }}>
              <span style={{ color: '#c2410c', fontWeight: 700 }}>exp({logits[active].toFixed(2)})</span>
            </div>
            <div style={{ height: 1, background: 'var(--text-2)', margin: '4px 0' }} />
            <div style={{ textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: 9.5, lineHeight: 1.7, color: 'var(--text-2)' }}>
              {logits.map((v, i) => (
                <span key={i}
                  onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}
                  style={{
                    cursor: 'pointer',
                    color: i === active ? '#c2410c' : 'var(--text-2)',
                    fontWeight: i === active ? 700 : 400,
                    background: i === active ? 'rgba(232,185,44,0.18)' : 'transparent',
                    borderRadius: 2, padding: '0 1px',
                  }}>exp({v.toFixed(2)}){i < logits.length - 1 ? ' + ' : ''}</span>
              ))}
            </div>
            <div style={{ textAlign: 'center', marginTop: 8, fontFamily: 'var(--font-mono)', fontSize: 12 }}>
              = <b>{prob.toFixed(4)}</b> <span style={{ color: 'var(--text-3)' }}>({(prob * 100).toFixed(1)}%)</span>
            </div>
          </div>
        </div>
        <div style={{ padding: '0 20px 12px', textAlign: 'center', fontSize: 9.5, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>
          {zh ? 'hover 圆 → 公式高亮对应项;hover 公式 → 圆高亮 (双向)' : 'hover a circle → its term highlights; hover a term → its circle highlights'}
        </div>
      </Modal>
    );
  }

  // ════════════════════════════════════════════════════════════════
  //  Input 数字选择器
  // ════════════════════════════════════════════════════════════════
  function InputPickerView({ catIndex, onClose, onChangeCat, lang }) {
    const zh = lang === 'zh';
    return (
      <Modal onClose={onClose}>
        <div style={{ padding: '16px 20px', width: 560 }}>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: 14 }}>
            <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--text)', flex: 1 }}>{zh ? '输入图像 · 28×28' : 'Input · 28×28'}</span>
            <CloseBtn onClose={onClose} />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
              <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{zh ? '当前' : 'current'}</span>
              <ImageThumb catIndex={catIndex} size={150} radius={4} grid />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', marginBottom: 8 }}>{zh ? '点一个数字切换输入' : 'tap a digit to switch'}</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6 }}>
                {CATEGORIES.map((c, i) => (
                  <button key={c.id} onClick={(e) => { e.stopPropagation(); haptic(); onChangeCat(i); }} style={{
                    padding: 2, border: 'none', background: 'transparent', cursor: 'pointer', lineHeight: 0,
                    outline: i === catIndex ? '1.5px solid var(--text)' : '0.5px solid var(--separator)',
                    outlineOffset: 1, borderRadius: 4,
                  }}>
                    <ImageThumb catIndex={i} size={56} radius={2} grid />
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </Modal>
    );
  }

  // ════════════════════════════════════════════════════════════════
  //  调度器
  // ════════════════════════════════════════════════════════════════
  function LayerDetailOverlay({ layer, layerIdx, channelIdx, catIndex, onClose, onChangeCat,
                                lang = 'zh', colormap = 'rwb', activationFn = 'relu', numChannels = 10 }) {
    if (!layer) return null;
    const data = NETWORK_DATA[catIndex];
    if (!data || !data.raw) return null;

    if (layer.kind === 'input')
      return <InputPickerView catIndex={catIndex} onClose={onClose} onChangeCat={onChangeCat} lang={lang} />;
    if (layer.kind === 'conv')
      return <ConvDetail layer={layer} layerIdx={layerIdx} channelIdx={channelIdx} catIndex={catIndex}
        onClose={onClose} lang={lang} colormap={colormap} numChannels={numChannels} />;
    if (layer.kind === 'relu')
      return <ActivationFormulaView layer={layer} layerIdx={layerIdx} channelIdx={channelIdx} catIndex={catIndex}
        onClose={onClose} lang={lang} colormap={colormap} activationFn={activationFn} />;
    if (layer.kind === 'pool')
      return <MaxPoolFormulaView layer={layer} layerIdx={layerIdx} channelIdx={channelIdx} catIndex={catIndex}
        onClose={onClose} lang={lang} colormap={colormap} />;
    if (layer.kind === 'softmax')
      return <SoftmaxFormulaView channelIdx={channelIdx} catIndex={catIndex} onClose={onClose} lang={lang} />;
    return null;
  }

  Object.assign(window, { LayerDetailOverlay });
})();

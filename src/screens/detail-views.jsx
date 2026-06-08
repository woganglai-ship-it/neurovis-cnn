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

  // ─────────── Level 1: Elastic Explanation View ───────────
  function ConvElasticView({ inputChannels, inputIsGray, kernels, intermediates,
                             outputMap, bias, oc, colormap, lang, onClose, onEnterFormula }) {
    const zh = lang === 'zh';
    const N = inputChannels.length;
    const outN = outputMap ? outputMap.length : 0;
    const inN = inputChannels[0] ? inputChannels[0].length : 0;
    const kSize = (kernels[0] && kernels[0].length) || 3;

    const [pos, setPos] = useState({ x: 0, y: 0 });
    const [playing, setPlaying] = useState(true);
    const [speed, setSpeed] = useState(1);
    useEffect(() => {
      if (!playing || outN === 0) return;
      let i = pos.y * outN + pos.x;
      const id = setInterval(() => {
        i = (i + 1) % (outN * outN);
        setPos({ x: i % outN, y: Math.floor(i / outN) });
      }, Math.round(520 / speed));
      return () => clearInterval(id);
    }, [playing, speed, outN]);
    const px = Math.min(pos.x, Math.max(0, outN - 1));
    const py = Math.min(pos.y, Math.max(0, outN - 1));

    // 色阶上限 (各自 75 分位)
    const inMax = inputChannels.map(m => inputIsGray ? 1 : pctMaxAbs(m, 0.8));
    const interMax = intermediates.map(m => pctMaxAbs(m, 0.7));
    const outMax = pctMaxAbs(outputMap, 0.7);

    // tile 尺寸:行数越多越小,允许列纵向滚动
    const TILE = Math.max(24, Math.min(64, Math.floor((250 - (N - 1) * 6) / N)));
    const KMINI = Math.round(TILE * 0.5);

    const L = zh ? {
      tag: '卷积层 · 第一级',
      sub: `${N} 输入通道 · ${N} 卷积核 · 1 输出`,
      input: '输入', inter: '中间结果', out: '输出',
      sumNote: '把所有中间结果相加,再加偏置', bias: '偏置',
      tapHint: '点中间结果或输出 → 看单个像素公式',
    } : {
      tag: 'CONVOLUTION · LEVEL 1',
      sub: `${N} input ${N === 1 ? 'channel' : 'channels'} · ${N} ${N === 1 ? 'kernel' : 'kernels'} · 1 output`,
      input: 'input', inter: 'intermediate', out: 'output',
      sumNote: 'sum all intermediate results, then add bias', bias: 'bias',
      tapHint: 'tap an intermediate or the output → per-pixel formula',
    };

    const flowStyle = playing
      ? { strokeDasharray: '4 3', animation: 'flowDash 0.5s linear infinite' }
      : { strokeDasharray: 'none' };

    return (
      <Modal onClose={onClose} bg="#F4F1EA" maxW="94vw">
        {/* ── 顶栏: 单行自适应, 标签+副标题在左, 控件在右 ── */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
          padding: '12px 16px 10px', borderBottom: '0.5px solid rgba(0,0,0,0.08)',
        }}>
          <span style={{
            padding: '2px 8px', borderRadius: 4, background: 'rgba(232,185,44,0.15)',
            fontSize: 9, fontWeight: 700, color: '#8a6b14', letterSpacing: '0.12em',
            textTransform: 'uppercase', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap',
          }}>{L.tag}</span>
          <span style={{
            fontSize: 12, color: 'var(--text-2)', fontFamily: 'Georgia, serif',
            fontStyle: 'italic', whiteSpace: 'nowrap',
          }}>{L.sub}</span>
          <span style={{ flex: 1, minWidth: 8 }} />
          <PlayBtn playing={playing} onToggle={() => setPlaying(p => !p)} />
          <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', whiteSpace: 'nowrap' }}>
            {zh ? '步' : 'step'} <b style={{ color: 'var(--text)' }}>{String(py * outN + px + 1).padStart(2, '0')}</b>/{outN * outN}
          </span>
          <div style={{ display: 'flex', border: '0.5px solid var(--separator-strong)', borderRadius: 11, padding: 2, background: '#fff' }}>
            {[0.5, 1, 2].map(s => (
              <button key={s} onClick={(e) => { e.stopPropagation(); setSpeed(s); }} style={{
                padding: '1px 7px', border: 'none', borderRadius: 9,
                background: speed === s ? '#1c1c1e' : 'transparent',
                color: speed === s ? '#fff' : 'var(--text-3)',
                fontSize: 9, fontFamily: 'var(--font-mono)', fontWeight: 600, cursor: 'pointer',
              }}>{s}×</button>
            ))}
          </div>
          <CloseBtn onClose={onClose} />
        </div>

        {/* ── 主体: [通道列 (可滚)] [⊕bias] [输出] ── */}
        <div style={{ display: 'flex', alignItems: 'stretch', gap: 0, padding: '6px 16px 12px' }}>
          {/* 通道列 — 列标题固定, 内容纵向滚动 */}
          <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <div style={{
              display: 'flex', gap: 8, paddingLeft: KMINI + 8, paddingBottom: 4,
              fontFamily: 'Georgia, serif', fontStyle: 'italic', fontSize: 11.5, color: 'var(--text-2)',
              position: 'sticky', top: 0, background: '#F4F1EA', zIndex: 3,
            }}>
              <span style={{ width: TILE, textAlign: 'center' }}>{L.input}</span>
              <span style={{ width: 22 }} />
              <span style={{ width: TILE, textAlign: 'center' }}>{L.inter}</span>
            </div>
            <div style={{ maxHeight: 264, overflowY: 'auto', overflowX: 'hidden' }}>
              {inputChannels.map((inMap, ic) => {
                const inter = intermediates[ic];
                const mN = inter ? inter.length : 0;
                return (
                  <div key={ic} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                    {/* kernel mini */}
                    <KernelMini kern={kernels[ic]} size={KMINI} />
                    {/* input tile + 滑窗 */}
                    <MatrixView data={inMap} maxAbs={inMax[ic]} size={TILE} colormap={colormap}
                      grayscale={inputIsGray}
                      win={{ x: px, y: py, w: kSize, h: kSize }} />
                    {/* 连线 (动画时流动虚线) */}
                    <svg width="22" height={TILE} style={{ flexShrink: 0 }}>
                      <line x1="0" y1={TILE / 2} x2="22" y2={TILE / 2}
                        stroke="#94918a" strokeWidth="0.8" {...flowStyle} />
                    </svg>
                    {/* intermediate tile (可点 → 公式) */}
                    <div onClick={(e) => { e.stopPropagation(); onEnterFormula(ic); }}
                      style={{ position: 'relative', cursor: 'pointer' }}
                      title={zh ? '点开公式' : 'open formula'}>
                      {inter
                        ? <MatrixView data={inter} maxAbs={interMax[ic]} size={TILE} colormap={colormap}
                            win={mN ? { x: Math.min(px, mN - 1), y: Math.min(py, mN - 1), w: 1, h: 1 } : null} />
                        : <div style={{ width: TILE, height: TILE, background: '#eee', borderRadius: 3 }} />}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 汇聚: 扇形虚线 → ⊕ */}
          <div style={{ position: 'relative', width: 70, alignSelf: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="70" height="120" viewBox="0 0 70 120" style={{ position: 'absolute' }}>
              {[12, 36, 60, 84, 108].map((y, i) => (
                <path key={i} d={`M 0 ${y} C 30 ${y}, 40 60, 52 60`}
                  stroke="rgba(91,95,107,0.28)" strokeWidth="0.7" fill="none" {...flowStyle} />
              ))}
            </svg>
            {/* ⊕ + bias */}
            <div style={{ position: 'relative', zIndex: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
              <div style={{
                width: 30, height: 30, borderRadius: 15, background: '#FAF7F0',
                border: '1px solid var(--text)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: 'Georgia, serif', fontStyle: 'italic', fontSize: 16, color: 'var(--text)',
              }}>+</div>
              <div style={{
                width: 9, height: 9, borderRadius: 5, background: window.weightColor(bias),
                boxShadow: 'inset 0 0 0 0.5px rgba(0,0,0,0.25)',
              }} title="bias" />
              <span style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>
                {L.bias} {fmt(bias)}
              </span>
            </div>
          </div>

          {/* 输出大图 (可点 → 公式, 默认走第 0 输入通道) */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 6, paddingLeft: 8 }}>
            <span style={{ fontFamily: 'Georgia, serif', fontStyle: 'italic', fontSize: 11.5, color: 'var(--text-2)' }}>
              {L.out} <span style={{ fontSize: 8, opacity: 0.5, fontFamily: 'var(--font-mono)', fontStyle: 'normal' }}>{outN}×{outN}</span>
            </span>
            <div onClick={(e) => { e.stopPropagation(); onEnterFormula(0); }} style={{ cursor: 'pointer' }}>
              <MatrixView data={outputMap} maxAbs={outMax} size={132} colormap={colormap}
                win={outN ? { x: px, y: py, w: 1, h: 1 } : null} />
            </div>
            <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>
              ({px},{py}) = <b>{fmt(outputMap ? outputMap[py][px] : 0)}</b>
            </span>
          </div>
        </div>

        {/* 注释 + 提示 */}
        <div style={{ padding: '0 16px 12px', display: 'flex', justifyContent: 'space-between', gap: 12 }}>
          <span style={{ fontSize: 10, fontFamily: 'Georgia, serif', fontStyle: 'italic', color: 'var(--text-3)' }}>{L.sumNote}</span>
          <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{L.tapHint}</span>
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

  // ════════════════════════════════════════════════════════════════
  //  ReLU / Sigmoid / Tanh Formula (Fig 6B)
  // ════════════════════════════════════════════════════════════════
  function ActivationFormulaView({ layer, layerIdx, channelIdx, catIndex, onClose, lang, colormap, activationFn }) {
    const zh = lang === 'zh';
    const data = NETWORK_DATA[catIndex];
    const prev = LAYERS[layerIdx - 1];
    const inputMap = rawChannel(data.raw[prev.id], channelIdx);
    const outputMap = rawChannel(data.raw[layer.id], channelIdx);
    const n = outputMap ? outputMap.length : 0;

    const [pos, setPos] = useState({ x: 0, y: 0 });
    const [playing, setPlaying] = useState(false);
    useEffect(() => {
      if (!playing || n === 0) return;
      let i = pos.y * n + pos.x;
      const id = setInterval(() => { i = (i + 1) % (n * n); setPos({ x: i % n, y: Math.floor(i / n) }); }, 280);
      return () => clearInterval(id);
    }, [playing, n]);
    const x = Math.min(pos.x, Math.max(0, n - 1)), y = Math.min(pos.y, Math.max(0, n - 1));
    const inVal = (inputMap && inputMap[y] && inputMap[y][x]) || 0;
    const outVal = (outputMap && outputMap[y] && outputMap[y][x]) || 0;

    const inMax = pctMaxAbs(inputMap, 0.8), outMax = pctMaxAbs(outputMap, 0.8);
    const fn = activationFn || 'relu';
    const title = zh
      ? { relu: 'ReLU 激活', sigmoid: 'Sigmoid 激活', tanh: 'Tanh 激活' }[fn]
      : { relu: 'ReLU Activation', sigmoid: 'Sigmoid Activation', tanh: 'Tanh Activation' }[fn];
    const MAT = 150;
    const pick = (cx, cy) => { setPlaying(false); setPos({ x: cx, y: cy }); };

    const Formula = () => {
      if (fn === 'sigmoid') return (<><span style={{ fontWeight: 600 }}>σ(</span><NumCell v={inVal} maxAbs={inMax} colormap={colormap} /><span>)</span></>);
      if (fn === 'tanh') return (<><span style={{ fontWeight: 600 }}>tanh(</span><NumCell v={inVal} maxAbs={inMax} colormap={colormap} /><span>)</span></>);
      return (<><span style={{ fontWeight: 600 }}>max(</span><NumCell v={0} maxAbs={1} colormap={colormap} /><span>,</span><NumCell v={inVal} maxAbs={inMax} colormap={colormap} /><span>)</span></>);
    };

    return (
      <Modal onClose={onClose}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 18px 8px', borderBottom: '0.5px solid rgba(0,0,0,0.08)' }}>
          <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--text)' }}>{title}</span>
          <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>· {zh ? '逐元素' : 'element-wise'} · {n}×{n}</span>
          <span style={{ flex: 1 }} />
          <PlayBtn playing={playing} onToggle={() => setPlaying(p => !p)} />
          <CloseBtn onClose={onClose} />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 14, padding: '14px 20px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
            <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{zh ? '输入' : 'input'}</span>
            <MatrixView data={inputMap} maxAbs={inMax} size={MAT} colormap={colormap}
              win={{ x, y, w: 1, h: 1 }} onPick={pick} />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-2)', padding: '10px 12px', background: 'var(--bg-grouped)', borderRadius: 8 }}>
            <Formula />
            <span style={{ fontWeight: 700, color: 'var(--text)' }}>=</span>
            <NumCell v={outVal} maxAbs={outMax} colormap={colormap} w={48} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
            <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{zh ? '输出' : 'output'}</span>
            <MatrixView data={outputMap} maxAbs={outMax} size={MAT} colormap={colormap}
              win={{ x, y, w: 1, h: 1 }} onPick={pick} />
          </div>
        </div>
        <div style={{ padding: '0 20px 12px', textAlign: 'center', fontSize: 9.5, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>
          {zh ? '在任一矩阵上点/拖任意像素,两边同步高亮,公式显示真实数字' : 'tap/drag any pixel on either matrix — both highlight, formula shows real numbers'}
        </div>
      </Modal>
    );
  }

  // ════════════════════════════════════════════════════════════════
  //  Max Pooling Formula (Fig 6C) — 尺寸减半
  // ════════════════════════════════════════════════════════════════
  function MaxPoolFormulaView({ layer, layerIdx, channelIdx, catIndex, onClose, lang, colormap }) {
    const zh = lang === 'zh';
    const data = NETWORK_DATA[catIndex];
    const prev = LAYERS[layerIdx - 1];
    const inputMap = rawChannel(data.raw[prev.id], channelIdx);
    const outputMap = rawChannel(data.raw[layer.id], channelIdx);
    const inN = inputMap ? inputMap.length : 0;
    const outN = outputMap ? outputMap.length : 0;

    const [pos, setPos] = useState({ x: 0, y: 0 });   // 输出坐标
    const [playing, setPlaying] = useState(false);
    useEffect(() => {
      if (!playing || outN === 0) return;
      let i = pos.y * outN + pos.x;
      const id = setInterval(() => { i = (i + 1) % (outN * outN); setPos({ x: i % outN, y: Math.floor(i / outN) }); }, 420);
      return () => clearInterval(id);
    }, [playing, outN]);
    const ox = Math.min(pos.x, Math.max(0, outN - 1)), oy = Math.min(pos.y, Math.max(0, outN - 1));

    const inMax = pctMaxAbs(inputMap, 0.8), outMax = pctMaxAbs(outputMap, 0.8);
    // 2×2 窗口真值
    const vals = [];
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      vals.push((inputMap[oy * 2 + dy] && inputMap[oy * 2 + dy][ox * 2 + dx]) || 0);
    }
    const result = Math.max(...vals);
    const MAT = 150;
    const title = zh ? '最大池化' : 'Max Pooling';

    return (
      <Modal onClose={onClose}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 18px 8px', borderBottom: '0.5px solid rgba(0,0,0,0.08)' }}>
          <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--text)' }}>{title}</span>
          <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>
            · {inN}×{inN} → {outN}×{outN} ({zh ? '尺寸减半' : 'halved'})
          </span>
          <span style={{ flex: 1 }} />
          <PlayBtn playing={playing} onToggle={() => setPlaying(p => !p)} />
          <CloseBtn onClose={onClose} />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 14, padding: '14px 20px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
            <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{zh ? '输入' : 'input'} {inN}×{inN}</span>
            <MatrixView data={inputMap} maxAbs={inMax} size={MAT} colormap={colormap}
              win={{ x: ox * 2, y: oy * 2, w: 2, h: 2 }}
              onPick={(cx, cy) => { setPlaying(false); setPos({ x: Math.min(Math.floor(cx / 2), outN - 1), y: Math.min(Math.floor(cy / 2), outN - 1) }); }} />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-2)', padding: '10px 12px', background: 'var(--bg-grouped)', borderRadius: 8 }}>
            <span style={{ fontWeight: 600 }}>max</span>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
              {vals.map((v, i) => (
                <span key={i} style={{ outline: v === result ? '1.5px solid ' + HL : 'none', outlineOffset: -1, borderRadius: 2 }}>
                  <NumCell v={v} maxAbs={inMax} colormap={colormap} w={40} />
                </span>
              ))}
            </div>
            <span style={{ fontWeight: 700, color: 'var(--text)' }}>=</span>
            <NumCell v={result} maxAbs={outMax} colormap={colormap} w={48} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
            <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{zh ? '输出' : 'output'} {outN}×{outN}</span>
            <MatrixView data={outputMap} maxAbs={outMax} size={MAT} colormap={colormap}
              win={{ x: ox, y: oy, w: 1, h: 1 }}
              onPick={(cx, cy) => { setPlaying(false); setPos({ x: cx, y: cy }); }} />
          </div>
        </div>
        <div style={{ padding: '0 20px 12px', textAlign: 'center', fontSize: 9.5, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>
          {zh ? '每 2×2 区域取最大值;点输入或输出双向联动' : 'max of each 2×2 patch; tap input or output for two-way linking'}
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

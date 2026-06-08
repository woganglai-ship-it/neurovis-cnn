/* global React */
const { useState, useEffect, useRef, useMemo, useCallback, Fragment } = React;

// ─── Heatmap tile (square) ────────────────────────────────────────────
function Heatmap({ data, size = 60, h = null, radius = 6, dark = false, animateKey = null, delay = 0, colormap = 'viridis' }) {
  const n = data.length;
  const height = h == null ? size : h;
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    const c = ref.current;
    const ctx = c.getContext('2d');
    c.width = n; c.height = n;
    c.style.width = size + 'px'; c.style.height = height + 'px';
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        ctx.fillStyle = window.actColor(data[y][x], dark, colormap);
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }, [data, size, height, dark, animateKey, colormap]);

  return (
    <div style={{
      width: size, height: height, borderRadius: radius, overflow: 'hidden',
      boxShadow: 'inset 0 0 0 0.5px rgba(0,0,0,0.12)',
      background: dark ? '#1c1c1e' : '#fff',
      animation: animateKey != null ? `tilePulse 600ms ${delay}ms ease-out both` : 'none',
    }}>
      <canvas ref={ref} style={{ imageRendering: 'pixelated', display: 'block' }} />
    </div>
  );
}

// ─── Color legend bar (red-white-blue) ────────────────────────────────
function ColorLegend({ kind = 'activation', compact = false, dark = false }) {
  const isAct = kind === 'activation';
  const grad = isAct
    ? `linear-gradient(90deg, #dc2626, ${dark ? '#1c1c1e' : '#fff'} 50%, #2563eb)`
    : `linear-gradient(90deg, #eab308, ${dark ? '#1c1c1e' : '#fff'} 50%, #10b981)`;
  const left = isAct ? '−1' : '−';
  const right = isAct ? '+1' : '+';
  const label = isAct ? '激活值' : '权重';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: compact ? 10 : 11, color: 'var(--text-3)' }}>
      {!compact && <span style={{ fontFamily: 'var(--font-cn)' }}>{label}</span>}
      <span className="mono" style={{ minWidth: 16, textAlign: 'right' }}>{left}</span>
      <div style={{
        width: compact ? 70 : 100, height: 8, borderRadius: 4,
        background: grad,
        boxShadow: 'inset 0 0 0 0.5px rgba(0,0,0,0.18)',
      }} />
      <span className="mono">{right}</span>
    </div>
  );
}

// ─── Layer kind chip ──────────────────────────────────────────────────
function LayerChip({ kind, name, zh }) {
  const map = {
    input:   { bg: 'var(--layer-input)',   edge: 'var(--layer-input-edge)',   label: 'INPUT' },
    conv:    { bg: 'var(--layer-conv)',    edge: 'var(--layer-conv-edge)',    label: 'CONV' },
    relu:    { bg: 'var(--layer-relu)',    edge: 'var(--layer-relu-edge)',    label: 'RELU' },
    pool:    { bg: 'var(--layer-pool)',    edge: 'var(--layer-pool-edge)',    label: 'POOL' },
    softmax: { bg: 'var(--layer-softmax)', edge: 'var(--layer-softmax-edge)', label: 'SOFTMAX' },
  }[kind] || { bg: '#eee', edge: '#888', label: kind };
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '3px 8px 3px 6px', borderRadius: 6,
      background: map.bg, color: map.edge,
      fontSize: 10, fontWeight: 700, letterSpacing: 0.5,
      fontFamily: 'var(--font-mono)',
    }}>
      <span style={{ width: 5, height: 5, borderRadius: 1, background: map.edge }} />
      {map.label}
    </span>
  );
}

// ─── Image thumb (procedural) ─────────────────────────────────────────
function ImageThumb({ catIndex, size = 56, radius = 6, grid = false }) {
  const cat = window.CATEGORIES[catIndex];
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    const c = ref.current;
    const ctx = c.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    c.width = size * dpr; c.height = size * dpr;
    c.style.width = size + 'px'; c.style.height = size + 'px';
    ctx.scale(dpr, dpr);

    // 渲染数字到 28×28 后 pixelated 放大到 `size`(黑底白字)
    const MNIST = 28;
    const small = document.createElement('canvas');
    small.width = MNIST;
    small.height = MNIST;
    const sctx = small.getContext('2d');
    sctx.fillStyle = '#000';
    sctx.fillRect(0, 0, MNIST, MNIST);
    sctx.fillStyle = '#fff';
    sctx.font = '700 22px "Kalam", "Caveat", "Comic Sans MS", cursive';
    sctx.textAlign = 'center';
    sctx.textBaseline = 'middle';
    sctx.fillText(cat.digit, MNIST / 2, MNIST / 2 + 5);

    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(small, 0, 0, MNIST, MNIST, 0, 0, size, size);

    // 像素网格 — 跟参考图(MNIST 灰度 + 明显格子)一致
    if (grid && size >= 60) {
      const cell = size / MNIST;
      ctx.strokeStyle = 'rgba(255,255,255,0.32)';
      ctx.lineWidth = 0.7;
      ctx.beginPath();
      for (let i = 1; i < MNIST; i++) {
        const p = i * cell;
        ctx.moveTo(p, 0); ctx.lineTo(p, size);
        ctx.moveTo(0, p); ctx.lineTo(size, p);
      }
      ctx.stroke();
    }
  }, [catIndex, size, grid]);

  return (
    <div style={{
      width: size, height: size, borderRadius: radius, overflow: 'hidden',
      position: 'relative', flexShrink: 0,
      background: '#000',
      boxShadow: 'inset 0 0 0 0.5px rgba(0,0,0,0.6)',
      lineHeight: 0,
    }}>
      <canvas ref={ref} style={{ display: 'block' }} />
    </div>
  );
}

// ─── Tap haptic stub ──────────────────────────────────────────────────
function haptic() {
  if (navigator.vibrate) try { navigator.vibrate(8); } catch (e) {}
}

Object.assign(window, {
  Heatmap, ColorLegend, LayerChip, ImageThumb, haptic,
});

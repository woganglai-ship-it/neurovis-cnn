/* global React, CATEGORIES, LAYERS, NETWORK_DATA, Heatmap, ColorLegend, LayerChip, ImageThumb, haptic */
const { useState: uS3, useEffect: uE3, useRef: uR3, useMemo: uM3 } = React;

// ─── S4 L1 Overview — landscape, horizontal CNN flow (CNN Explainer style) ──
function ScreenL1({ catIndex, onChangeCat, onTapNeuron, onOpenImagePicker, onOpenTutorial, onOpenSettings, computeWave, showDetail = true, lang = 'zh', numChannels = 10, kernelSize = 3, activationFn = 'relu', recomputeTick, isRecomputing, colormap = 'viridis' }) {
  const data = NETWORK_DATA[catIndex];
  const cat = CATEGORIES[catIndex];
  const probs = data.output[0];
  const topIdx = probs.indexOf(Math.max(...probs));

  // 当前被聚焦的方块 ({ layerIdx, channelIdx } or null)
  const [focused, setFocused] = uS3(null);
  // 详情浮层 ({ layerIdx, channelIdx } or null)
  const [detail, setDetail] = uS3(null);

  // 当 numChannels 改变时清掉可能越界的 focused
  uE3(() => { setFocused(null); setDetail(null); }, [numChannels]);

  // i18n strings
  const t = lang === 'zh' ? {
    tagline: '卷积神经网络 · MNIST',
    input: '输入', digit: '数字',
    prediction: '预测',
    hintIdle: '点任意方块,高亮它的连接线',
    hintFocused: '再点焦点方块 → L2 详情',
    recomputing: '重新计算中…',
  } : {
    tagline: 'Tiny VGG · CNN · MNIST',
    input: 'input', digit: 'digit',
    prediction: 'prediction',
    hintIdle: 'tap any tile to highlight its connections',
    hintFocused: 'tap focused tile again → L2 detail',
    recomputing: 'recomputing…',
  };

  const handleTileTap = (layerIdx, channelIdx) => {
    haptic();
    // 单次点击 → 直接打开详情浮层(所有 kind 都支持)
    setDetail({ layerIdx, channelIdx });
    setFocused(null);
  };

  // ── 自动 forward wave: 每 4.5 秒触发一次,每层依次"激活脉动" ──
  const [autoWave, setAutoWave] = uS3({});
  uE3(() => {
    const trigger = () => {
      const stamp = Date.now();
      const next = {};
      const layers = window.LAYERS || [];
      layers.forEach((_, i) => { next[i] = stamp + i * 220; });
      setAutoWave(next);
    };
    trigger();
    const id = setInterval(trigger, 4500);
    return () => clearInterval(id);
  }, []);

  // 容器宽度自适应:用 ResizeObserver 测主区可用宽,COL_W / COL_GAP 按列数算,
  // 保证 1 层(5 列)和 2 层(7 列)都铺满单屏不滚动。
  const mainRef = uR3(null);
  const [containerW, setContainerW] = uS3(0);
  uE3(() => {
    if (!mainRef.current) return;
    const measure = () => { if (mainRef.current) setContainerW(mainRef.current.clientWidth); };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(mainRef.current);
    return () => ro.disconnect();
  }, []);

  // Layout constants
  const TILE_GAP = 2;
  const LABEL_H = 36;
  const MAX_ROWS = 10;
  const SOFTMAX_TILE = 22;
  const TILE_AREA_H = MAX_ROWS * (SOFTMAX_TILE + TILE_GAP) - TILE_GAP;

  const colsAll = LAYERS;
  const HORIZ_PAD = 28;

  // block-aware gap: conv↔relu / relu↔pool 紧贴(同 block), 其他较松
  const isTightPair = (a, b) => {
    if (!a || !b) return false;
    if (a.kind === 'conv' && b.kind === 'relu') return true;
    if (a.kind === 'relu' && b.kind === 'pool') return true;
    return false;
  };
  const TIGHT_RATIO = 0.20;
  const LOOSE_RATIO = 0.75;
  // 算 effective gap units = sum of (tight*TIGHT + loose*LOOSE)
  let gapUnits = 0;
  for (let i = 0; i < colsAll.length - 1; i++) {
    gapUnits += isTightPair(colsAll[i], colsAll[i + 1]) ? TIGHT_RATIO : LOOSE_RATIO;
  }
  const availW = Math.max((containerW || 360) - HORIZ_PAD * 2, 240);
  const COL_W = Math.max(26, Math.floor(availW / (colsAll.length + gapUnits)));
  const TIGHT_GAP = Math.floor(COL_W * TIGHT_RATIO);
  const LOOSE_GAP = Math.floor(COL_W * LOOSE_RATIO);
  // 累加坐标
  const colXArr = [0];
  for (let i = 1; i < colsAll.length; i++) {
    const prevX = colXArr[i - 1];
    const gap = isTightPair(colsAll[i - 1], colsAll[i]) ? TIGHT_GAP : LOOSE_GAP;
    colXArr.push(prevX + COL_W + gap);
  }
  const colX = (i) => colXArr[i];
  const totalW = colsAll.length > 0 ? colXArr[colsAll.length - 1] + COL_W : 0;
  const totalH = LABEL_H + TILE_AREA_H;

  // 每列的实际行数:input=1,softmax=10,conv/relu/pool=numChannels
  const getColumnRows = (layer) => {
    if (layer.kind === 'input') return 1;
    if (layer.kind === 'softmax') return MAX_ROWS;
    return data[layer.id].length;
  };

  // 自适应 tile 大小:N 越少,每个方块越大
  const getTileSize = (layer) => {
    if (layer.kind === 'softmax') return SOFTMAX_TILE;
    if (layer.kind === 'input') return Math.min(COL_W - 12, 72);
    const rows = data[layer.id].length;
    return Math.min(80, Math.floor((TILE_AREA_H - (rows - 1) * TILE_GAP) / rows));
  };

  // 垂直居中
  const colYOffset = (layer) => {
    const rows = getColumnRows(layer);
    const tile = getTileSize(layer);
    const usedH = rows * (tile + TILE_GAP) - TILE_GAP;
    return (TILE_AREA_H - usedH) / 2;
  };
  const tileY = (layer, j) => LABEL_H + colYOffset(layer) + j * (getTileSize(layer) + TILE_GAP);

  // Connection lines —— 按 nextLayer.kind 决定连法:
  //   relu / pool: 一对一(只在 a===b 时连,因为只是 element-wise 激活 / 空间降采样)
  //   conv / softmax: 多对多(每个 input channel → 每个 output)
  const connections = [];
  for (let i = 0; i < colsAll.length - 1; i++) {
    const layer = colsAll[i];
    const nextLayer = colsAll[i + 1];
    const x1 = colX(i) + COL_W;
    const x2 = colX(i + 1);
    const dx = (x2 - x1) * 0.45;
    const fromCount = getColumnRows(layer);
    const toCount = getColumnRows(nextLayer);
    const isOneToOne = nextLayer.kind === 'relu' || nextLayer.kind === 'pool';
    for (let a = 0; a < fromCount; a++) {
      for (let b = 0; b < toCount; b++) {
        if (isOneToOne && a !== b) continue;
        const y1 = layer.kind === 'input'
          ? LABEL_H + TILE_AREA_H / 2
          : tileY(layer, a) + getTileSize(layer) / 2;
        const y2 = nextLayer.kind === 'input'
          ? LABEL_H + TILE_AREA_H / 2
          : tileY(nextLayer, b) + getTileSize(nextLayer) / 2;
        let stroke, width;
        if (focused) {
          const matchA = focused.layerIdx === i && focused.channelIdx === a;
          const matchB = focused.layerIdx === i + 1 && focused.channelIdx === b;
          if (matchA || matchB) {
            stroke = 'rgba(40,40,50,0.65)';
            width = 0.9;
          } else {
            stroke = 'rgba(60,60,67,0.10)';
            width = 0.4;
          }
        } else {
          stroke = 'rgba(60,60,67,0.36)';
          width = 0.6;
        }
        connections.push(
          <path key={`${i}-${a}-${b}`}
            d={`M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`}
            stroke={stroke} strokeWidth={width} fill="none"
          />
        );
      }
    }
  }

  return (
    <div onClick={() => { setFocused(null); }}
      style={{ width: '100%', height: '100%', background: 'var(--bg)', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      {/* Detail overlay */}
      {detail && (
        <LayerDetailOverlay
          layer={LAYERS[detail.layerIdx]}
          layerIdx={detail.layerIdx}
          channelIdx={detail.channelIdx}
          catIndex={catIndex}
          onClose={() => setDetail(null)}
          onChangeCat={(i) => { onChangeCat && onChangeCat(i); setDetail(null); }}
          lang={lang}
          colormap={colormap}
          activationFn={activationFn}
          numChannels={numChannels}
        />
      )}
      {/* Top bar */}
      <div style={{
        padding: '12px 56px 6px 56px',
        display: 'flex', alignItems: 'center', gap: 14,
      }} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexShrink: 0 }}>
          <span style={{ fontSize: 20, fontWeight: 600, color: 'var(--text)', letterSpacing: 0.3, fontFamily: 'var(--font-mono)' }}>NeuroVis</span>
          <span style={{ width: 1, height: 13, background: 'var(--separator-strong)' }} />
          <span style={{ fontSize: 14, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', letterSpacing: 0.4 }}>{t.tagline}</span>
        </div>

        {/* Digit picker strip removed — input column is the picker */}
        <div style={{ flex: 1 }} />

        <div style={{ display: 'flex', gap: 0, flexShrink: 0 }}>
          <button onClick={(e) => { e.stopPropagation(); haptic(); onOpenTutorial && onOpenTutorial(); }} aria-label="教程" style={{
            width: 28, height: 28, border: 'none', background: 'transparent', cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="9" stroke="#3a3a40" strokeWidth="1.3"/>
              <path d="M9.5 9.5a2.5 2.5 0 0 1 5 0c0 1.5-2.5 2-2.5 3.5" stroke="#3a3a40" strokeWidth="1.3" strokeLinecap="round" fill="none"/>
              <circle cx="12" cy="17" r="0.8" fill="#3a3a40"/>
            </svg>
          </button>
          <button onClick={(e) => { e.stopPropagation(); haptic(); onOpenSettings && onOpenSettings(); }} aria-label="设置" style={{
            width: 28, height: 28, border: 'none', background: 'transparent', cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3a3a40" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 0 0 2.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 0 0 1.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 0 0-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 0 0-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 0 0-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 0 0-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 0 0 1.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065Z"/>
              <circle cx="12" cy="12" r="3"/>
            </svg>
          </button>
        </div>
      </div>

      {/* Active digit label */}
      <div style={{ padding: '0 56px 4px', display: 'flex', alignItems: 'baseline', gap: 8 }}>
        {isRecomputing && (
          <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', fontStyle: 'italic' }}>{t.recomputing}</span>
        )}
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{t.prediction}</span>
        <span style={{ fontSize: 10, color: 'var(--text)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
          {topIdx} · {(probs[topIdx] * 100).toFixed(1)}%
        </span>
      </div>

      {/* Main viz — 容器自适应宽度,内部列宽按列数自动算 */}
      <div ref={mainRef} style={{
        flex: 1, padding: `0 ${HORIZ_PAD}px`,
        overflowX: 'hidden', overflowY: 'hidden',
        display: 'flex', justifyContent: 'center', alignItems: 'flex-start',
      }}>
        <div style={{ position: 'relative', width: totalW, height: totalH }}>
          <svg style={{
            position: 'absolute', top: 0, left: 0,
            width: totalW, height: totalH,
            pointerEvents: 'none', overflow: 'visible',
          }}>
            {connections}
          </svg>

          {colsAll.map((layer, i) => {
            const x = colX(i);
            const yOffset = colYOffset(layer);
            const maxAbs = (data.maxAbs && data.maxAbs[layer.id]) || 1;
            return (
              <div key={layer.id} style={{
                position: 'absolute', left: x, top: 0,
                width: COL_W,
                display: 'flex', flexDirection: 'column', alignItems: 'center',
              }}>
                {/* Column header — 简约,跟原版 CNN Explainer 一致 */}
                <div style={{
                  height: LABEL_H, display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'flex-end', paddingBottom: 10,
                }}>
                  <div style={{
                    fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 600,
                    color: 'var(--text-2)', letterSpacing: 0.2,
                    whiteSpace: 'nowrap', textTransform: 'lowercase',
                  }}>{
                    layer.kind === 'relu'
                      ? activationFn
                      : layer.kind === 'softmax'
                      ? 'softmax'
                      : layer.kind === 'pool'
                      ? 'max_pool'
                      : layer.kind === 'conv'
                      ? 'conv'
                      : 'input'
                  }</div>
                  {showDetail && (
                    <div style={{
                      fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-3)',
                      marginTop: 1, opacity: 0.8,
                    }}>
                      ({layer.size},{layer.size},{
                        layer.kind === 'input' ? 1 :
                        layer.kind === 'softmax' ? 10 :
                        numChannels
                      })
                    </div>
                  )}
                </div>

                {/* Column body — fixed TILE_AREA_H, content vertically centered */}
                <div style={{ width: '100%', height: TILE_AREA_H, position: 'relative' }}>
                  {layer.kind === 'input' && (
                    <div style={{
                      position: 'absolute', top: TILE_AREA_H / 2, left: '50%',
                      transform: 'translate(-50%, -50%)',
                    }}>
                      <button onClick={(e) => { e.stopPropagation(); haptic(); setDetail({ layerIdx: 0, channelIdx: 0 }); setFocused(null); }}
                        style={{ border: 'none', padding: 0, background: 'transparent', cursor: 'pointer' }}>
                        <ImageThumb catIndex={catIndex} size={Math.min(COL_W - 12, 84)} radius={3} grid />
                      </button>
                    </div>
                  )}

                  {layer.kind === 'softmax' && (
                    <div style={{
                      position: 'absolute', top: yOffset, left: 0, right: 0,
                      display: 'flex', flexDirection: 'column', gap: TILE_GAP,
                    }}>
                      {probs.map((p, j) => {
                        const isTop = j === topIdx;
                        const isFocused = focused && focused.layerIdx === i && focused.channelIdx === j;
                        return (
                          <div key={j}
                            onClick={(e) => { e.stopPropagation(); handleTileTap(i, j); }}
                            style={{
                              width: COL_W, height: SOFTMAX_TILE,
                              display: 'flex', alignItems: 'center', gap: 3,
                              cursor: 'pointer',
                              outline: isFocused ? '1.5px solid var(--text)' : 'none',
                              outlineOffset: 1, borderRadius: 2,
                            }}>
                            <span style={{
                              fontSize: 10, fontFamily: 'var(--font-mono)',
                              color: isTop ? 'var(--text)' : 'var(--text-3)',
                              fontWeight: isTop ? 700 : 400,
                              width: 9, textAlign: 'right', flexShrink: 0,
                            }}>{j}</span>
                            <div style={{
                              flex: 1, height: 6, borderRadius: 1,
                              background: 'rgba(0,0,0,0.05)', position: 'relative', overflow: 'hidden',
                            }}>
                              <div style={{
                                width: `${p * 100}%`, height: '100%',
                                background: isTop ? 'var(--text)' : 'rgba(60,60,67,0.4)',
                                transition: 'width 600ms cubic-bezier(.2,.8,.2,1)',
                              }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {layer.kind !== 'input' && layer.kind !== 'softmax' && layer.kind !== 'dense' && (() => {
                    const tile = getTileSize(layer);
                    return (
                      <div style={{
                        position: 'absolute', top: yOffset, left: '50%', transform: 'translateX(-50%)',
                        display: 'flex', flexDirection: 'column', gap: TILE_GAP,
                      }}>
                        {data[layer.id].map((m, c) => {
                          const isFocused = focused && focused.layerIdx === i && focused.channelIdx === c;
                          const dimmed = focused && !isFocused;
                          const waveStamp = autoWave[i];
                          return (
                            <button key={`${c}-${waveStamp || 0}`}
                              onClick={(e) => { e.stopPropagation(); handleTileTap(i, c); }}
                              style={{
                                border: 'none', padding: 0, background: 'transparent', cursor: 'pointer', lineHeight: 0,
                                outline: isFocused ? '1.5px solid var(--text)' : 'none',
                                outlineOffset: 1,
                                opacity: dimmed ? 0.35 : 1,
                                transition: 'opacity 180ms ease',
                                animation: waveStamp ? `wavePulse 700ms ${c * 55}ms ease-out both` : 'none',
                              }}>
                              <Heatmap data={m} size={tile} radius={2}
                                colormap={colormap} />
                            </button>
                          );
                        })}
                      </div>
                    );
                  })()}
                </div>

              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// 第二页(层详情 / 公式视图)已抽到 src/screens/detail-views.jsx,
// 通过 window.LAYERS / window.LayerDetailOverlay 协作。本文件只保留主屏 ScreenL1。
window.ScreenL1 = ScreenL1;

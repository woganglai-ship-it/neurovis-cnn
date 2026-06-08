/* global React, CATEGORIES, ImageThumb, haptic */
const { useState: uS6, useEffect: uE6 } = React;

// ─── S8 Image Picker (modal sheet) ───────────────────────────────────
function ScreenImagePicker({ catIndex, onPick, onClose }) {
  return (
    <div style={{
      position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.32)',
      display: 'flex', flexDirection: 'column', justifyContent: 'flex-end',
      animation: 'sheetFade 250ms ease-out both', zIndex: 100,
    }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{
        background: 'var(--bg)', borderTopLeftRadius: 28, borderTopRightRadius: 28,
        padding: '12px 16px 40px', maxHeight: '80%', overflowY: 'auto',
        animation: 'sheetIn 320ms cubic-bezier(.2,.8,.2,1) both',
      }}>
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 6 }}>
          <div style={{ width: 36, height: 5, borderRadius: 3, background: 'var(--separator-strong)' }} />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 4px 14px' }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>选一张图试试</div>
            <div style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 2 }}>整个网络会重新计算</div>
          </div>
          <button onClick={onClose} style={{
            width: 32, height: 32, borderRadius: 16, border: 'none',
            background: 'var(--bg-grouped)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <svg width="12" height="12" viewBox="0 0 12 12"><path d="M3 3l6 6M9 3l-6 6" stroke="#3a3a40" strokeWidth="2" fill="none" strokeLinecap="round"/></svg>
          </button>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
          {CATEGORIES.map((c, i) => (
            <button key={c.id} onClick={() => { haptic(); onPick && onPick(i); }} style={{
              border: 'none', padding: 8, borderRadius: 14, background: 'transparent',
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6,
              outline: i === catIndex ? '2px solid var(--brand-accent)' : 'none',
              outlineOffset: -2,
              backgroundColor: i === catIndex ? 'rgba(245,158,11,0.08)' : 'transparent',
            }}>
              <ImageThumb catIndex={i} size={70} radius={10} />
              <div style={{ fontSize: 12, color: 'var(--text)', fontWeight: i === catIndex ? 600 : 500 }}>{c.zh}</div>
              <div style={{ fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{c.en}</div>
            </button>
          ))}
        </div>
        <button style={{
          marginTop: 16, width: '100%', padding: 14, borderRadius: 14, border: '1px dashed var(--separator-strong)',
          background: 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
          color: 'var(--text-2)', fontSize: 14,
        }}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <rect x="1.5" y="2.5" width="13" height="11" rx="2" stroke="currentColor" strokeWidth="1.4"/>
            <circle cx="5.5" cy="6" r="1" fill="currentColor"/>
            <path d="M2 11l3.5-3 3 2.5L11 8l3 3" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinejoin="round"/>
          </svg>
          从相册上传
        </button>
      </div>
    </div>
  );
}

// ─── S9 Tutorial — minimal landscape, matches main screen ─────────────
function ScreenTutorial({ onBack, lang = 'zh' }) {
  const t = lang === 'zh' ? {
    section: 'tutorial',
    readTime: '阅读时长', readMin: '约 5 分钟', chapters: '共 3 章',
    chapter: '第',
    c1Title: '为什么需要卷积神经网络?',
    c1p1: '想象你要教一台计算机识别一张手写数字。最直觉的办法是把图片每一个像素都喂给它 —— 但一张 64×64 的图就有 4096 个数字。如果直接连起来,参数量爆炸,模型还学不会「数字 9 写在角落里也是 9」这件事。',
    c1p2: ['', '卷积', '给出了一个聪明的答案:用一个小窗口在图上滑动,反复检测「这一小块像不像我要找的图案」。这个小窗口叫做 ', 'kernel', ' —— 一个能扫描整张图的小放大镜。'],
    c2Title: 'ReLU:简单粗暴的非线性',
    c2p1: '一堆线性运算叠加起来,数学上还是一个线性运算 —— 网络再深也学不会复杂图案。',
    c2p2: ['ReLU 干的事简单到离谱:', '负数变 0,正数不变', '。但正是这个「掐掉一半」的小操作,让神经网络具备了「灵活」的本事。'],
    c3Title: '为什么要池化?',
    c3p1: ['特征图越大,越费算力。', '最大池化', '每 2×2 区域留下一个最强信号,相当于「这一片有没有我要的图案」的概要 —— 保留信息精华,扔掉位置细节,顺便让网络能看到更大范围的图案。'],
    nextLabel: 'next',
    nextHeading: '回到可视化,点一个方块试试',
    returnBtn: 'return →',
    footerLeft: 'NeuroVis', footerMid: 'tutorial · CNN 基础',
  } : {
    section: 'tutorial',
    readTime: 'read time', readMin: '~5 min', chapters: '3 chapters',
    chapter: 'chapter',
    c1Title: 'Why do we need convolutional networks?',
    c1p1: 'Imagine you want to teach a computer to recognize a handwritten digit. The naive approach is to feed every pixel directly to a fully-connected net — but even a 64×64 image gives you 4,096 numbers. That blows up the parameter count, and the model still wouldn\'t learn that "a 9 in the corner is still a 9".',
    c1p2: ['', 'Convolution', ' gives a clever answer: slide a small window across the image and repeatedly ask "does this little patch look like the pattern I\'m hunting for?" That small window is called a ', 'kernel', ' — a tiny magnifier that scans the whole image.'],
    c2Title: 'ReLU: brutally simple non-linearity',
    c2p1: 'Stacking linear operations is still a linear operation — no matter how deep the net is, it can\'t learn complex patterns without a non-linear step.',
    c2p2: ['ReLU\'s job is hilariously simple: ', 'negatives become zero, positives pass through', '. That tiny "chop off half" operation is what gives neural networks their flexibility.'],
    c3Title: 'Why pool?',
    c3p1: ['Larger feature maps cost more compute. ', 'Max pooling', ' keeps the strongest signal in every 2×2 patch — a summary of "is the pattern present anywhere here?" It preserves the important info, drops position detail, and lets the network see a larger receptive field.'],
    nextLabel: 'next',
    nextHeading: 'Back to the viz — tap a tile and see what happens',
    returnBtn: 'return →',
    footerLeft: 'NeuroVis', footerMid: 'tutorial · CNN basics',
  };

  const inlineMix = (parts) => (
    <span>
      {parts.map((p, i) => i % 2 === 1
        ? <strong key={i}>{p}</strong>
        : <span key={i}>{p}</span>
      )}
    </span>
  );

  return (
    <div style={{ width: '100%', height: '100%', background: 'var(--bg)', display: 'flex', flexDirection: 'column' }}>
      {/* Top bar */}
      <div style={{
        padding: '12px 56px 6px 56px',
        display: 'flex', alignItems: 'center', gap: 14,
      }}>
        <button onClick={() => { haptic(); onBack && onBack(); }} aria-label="back" style={{
          width: 28, height: 28, border: 'none', background: 'transparent', cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
        }}>
          <svg width="9" height="14" viewBox="0 0 10 16"><path d="M8 1L2 8l6 7" stroke="#3a3a40" strokeWidth="1.6" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>
        </button>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexShrink: 0 }}>
          <span style={{ fontSize: 18, fontWeight: 600, color: 'var(--text)', letterSpacing: 0.3, fontFamily: 'var(--font-mono)' }}>NeuroVis</span>
          <span style={{ width: 1, height: 9, background: 'var(--separator-strong)' }} />
          <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', letterSpacing: 0.4 }}>{t.section}</span>
        </div>
        <div style={{ flex: 1 }} />
      </div>

      {/* Section label echo */}
      <div style={{ padding: '0 56px 4px', display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{t.readTime}</span>
        <span style={{ fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>{t.readMin}</span>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{t.chapters}</span>
      </div>

      {/* Body — centered column for landscape */}
      <div style={{
        flex: 1, overflowY: 'auto',
        padding: '8px 56px 16px',
      }}>
        <div style={{ maxWidth: 580, margin: '0 auto', borderTop: '0.5px solid var(--separator)', paddingTop: 18 }}>
          <Chapter num="01" title={t.c1Title} chapter={t.chapter}>
            <p>{t.c1p1}</p>
            <p>{inlineMix(t.c1p2)}</p>
          </Chapter>

          <Chapter num="02" title={t.c2Title} chapter={t.chapter}>
            <p>{t.c2p1}</p>
            <p>{inlineMix(t.c2p2)}</p>
          </Chapter>

          <Chapter num="03" title={t.c3Title} chapter={t.chapter}>
            <p>{inlineMix(t.c3p1)}</p>
          </Chapter>

          <div style={{
            marginTop: 24, padding: '14px 0', borderTop: '0.5px solid var(--separator)',
            display: 'flex', alignItems: 'center', gap: 12,
          }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', letterSpacing: 0.4 }}>{t.nextLabel}</div>
              <div style={{ fontSize: 13, color: 'var(--text)', marginTop: 2, fontWeight: 500 }}>{t.nextHeading}</div>
            </div>
            <button onClick={() => { haptic(); onBack && onBack(); }} style={{
              padding: '6px 14px', borderRadius: 4, border: '0.5px solid var(--text)',
              background: 'var(--text)', color: '#fff',
              fontSize: 11, fontFamily: 'var(--font-mono)', cursor: 'pointer', letterSpacing: 0.4,
            }}>
              {t.returnBtn}
            </button>
          </div>
        </div>
      </div>

      {/* Bottom footer — matches L1 */}
      <div style={{
        padding: '6px 56px 8px', display: 'flex', alignItems: 'center', gap: 10,
        borderTop: '0.5px solid var(--separator)',
      }}>
        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', letterSpacing: 0.4 }}>{t.footerLeft}</span>
        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{t.footerMid}</span>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>inspired by CNN Explainer</span>
      </div>
    </div>
  );
}

function Chapter({ num, title, chapter, children }) {
  return (
    <div style={{ marginBottom: 22 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 8 }}>
        <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', letterSpacing: 0.6 }}>{chapter} {num}</span>
        <span style={{ width: 1, height: 9, background: 'var(--separator-strong)' }} />
        <span style={{ fontSize: 16, fontWeight: 500, color: 'var(--text)', letterSpacing: -0.1 }}>{title}</span>
      </div>
      <div style={{ fontSize: 13, color: 'var(--text-2)', lineHeight: 1.7, paddingLeft: 0 }}>
        {React.Children.map(children, (child, i) => (
          <div key={i} style={{ marginTop: i === 0 ? 0 : 8 }}>{child}</div>
        ))}
      </div>
    </div>
  );
}

function SuperParamsWidget() { return null; /* removed — content now lives inline */ }

// ─── S10 Settings — minimal, matches main screen aesthetic ─────────────
function ScreenSettings({ onBack, showDetail, setShowDetail, lang = 'zh', setLang, numChannels = 5, setNumChannels, kernelSize = 3, setKernelSize, numLayers = 1, setNumLayers, activationFn = 'relu', setActivationFn, colormap = 'viridis', setColormap }) {
  const t = lang === 'zh' ? {
    section: 'settings', preferences: '偏好', tagline: 'NeuroVis · v0.1',
    showDetailTitle: '显示详细信息',
    showDetailDesc: 'show detail · 显示每层张量维度',
    languageTitle: '语言 / Language',
    languageDesc: '中文 · English · 双语切换',
    channelsTitle: '通道数量',
    channelsDesc: 'number of channels · 每层 conv 的输出通道数',
    kernelTitle: '卷积核大小',
    kernelDesc: 'kernel size · 3×3 是 VGG/ResNet 标准,5/7 感知野更大',
    layersTitle: '卷积层数',
    layersDesc: 'conv layers · 1 浅 / 2 深(更深更准但慢)',
    activationTitle: '激活函数',
    activationDesc: 'activation function · 非线性变换',
    colormapTitle: '颜色映射',
    colormapDesc: 'colormap · 把激活数值翻译成颜色的方案',
    footerLeft: 'NeuroVis', footerMid: 'CNN 可视化',
  } : {
    section: 'settings', preferences: 'preferences', tagline: 'NeuroVis · v0.1',
    showDetailTitle: 'Show detail',
    showDetailDesc: 'Display tensor shapes per layer',
    languageTitle: 'Language / 语言',
    languageDesc: 'Switch between Chinese and English',
    channelsTitle: 'Number of channels',
    channelsDesc: 'How many output filters per conv layer (2-10)',
    kernelTitle: 'Kernel size',
    kernelDesc: '3×3 is VGG/ResNet standard; 5/7 see larger receptive field',
    layersTitle: 'Conv layers',
    layersDesc: '1 shallow / 2 deep (deeper = more accurate but slower)',
    activationTitle: 'Activation function',
    activationDesc: 'Non-linear transform applied after conv',
    colormapTitle: 'Colormap',
    colormapDesc: 'How activation values map to color',
    footerLeft: 'NeuroVis', footerMid: 'CNN visualizer',
  };

  return (
    <div style={{ width: '100%', height: '100%', background: 'var(--bg)', display: 'flex', flexDirection: 'column' }}>
      {/* Top bar — same proportions as L1 */}
      <div style={{
        padding: '12px 56px 6px 56px',
        display: 'flex', alignItems: 'center', gap: 14,
      }}>
        <button onClick={() => { haptic(); onBack && onBack(); }} aria-label="back" style={{
          width: 28, height: 28, border: 'none', background: 'transparent', cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
        }}>
          <svg width="9" height="14" viewBox="0 0 10 16"><path d="M8 1L2 8l6 7" stroke="#3a3a40" strokeWidth="1.6" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>
        </button>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexShrink: 0 }}>
          <span style={{ fontSize: 18, fontWeight: 600, color: 'var(--text)', letterSpacing: 0.3, fontFamily: 'var(--font-mono)' }}>NeuroVis</span>
          <span style={{ width: 1, height: 9, background: 'var(--separator-strong)' }} />
          <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', letterSpacing: 0.4 }}>{t.section}</span>
        </div>
        <div style={{ flex: 1 }} />
      </div>

      {/* Active label echo — matches L1's "input digit X / prediction" row */}
      <div style={{ padding: '0 56px 4px', display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{t.preferences}</span>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{t.tagline}</span>
      </div>

      {/* Body */}
      <div style={{
        flex: 1, padding: '20px 56px', overflowY: 'auto',
        display: 'flex', flexDirection: 'column', alignItems: 'stretch',
      }}>
        <div style={{
          maxWidth: 480, width: '100%', margin: '0 auto',
          borderTop: '0.5px solid var(--separator)',
        }}>
          {/* Show detail toggle */}
          <div style={{
            padding: '16px 4px', borderBottom: '0.5px solid var(--separator)',
            display: 'flex', alignItems: 'center', gap: 12,
          }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>{t.showDetailTitle}</div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 3, letterSpacing: 0.2 }}>
                {t.showDetailDesc}
              </div>
            </div>
            <MinimalSwitch on={showDetail} onChange={setShowDetail} />
          </div>

          {/* Language picker */}
          <div style={{
            padding: '16px 4px', borderBottom: '0.5px solid var(--separator)',
            display: 'flex', alignItems: 'center', gap: 12,
          }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>{t.languageTitle}</div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 3, letterSpacing: 0.2 }}>
                {t.languageDesc}
              </div>
            </div>
            <SegmentedPicker
              value={lang}
              options={[['zh', '中文'], ['en', 'EN']]}
              onChange={(v) => { haptic(); setLang && setLang(v); }}
            />
          </div>

          {/* Channels picker */}
          <div style={{
            padding: '16px 4px', borderBottom: '0.5px solid var(--separator)',
            display: 'flex', alignItems: 'center', gap: 12,
          }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>{t.channelsTitle}</div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 3, letterSpacing: 0.2 }}>
                {t.channelsDesc}
              </div>
            </div>
            <ChannelStepper value={numChannels} onChange={(v) => { haptic(); setNumChannels && setNumChannels(v); }} min={2} max={10} />
          </div>

          {/* Kernel size picker */}
          <div style={{
            padding: '16px 4px', borderBottom: '0.5px solid var(--separator)',
            display: 'flex', alignItems: 'center', gap: 12,
          }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>{t.kernelTitle}</div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 3, letterSpacing: 0.2 }}>
                {t.kernelDesc}
              </div>
            </div>
            <SegmentedPicker
              value={String(kernelSize)}
              options={[['3', '3×3'], ['5', '5×5'], ['7', '7×7']]}
              onChange={(v) => { haptic(); setKernelSize && setKernelSize(parseInt(v, 10)); }}
            />
          </div>

          {/* Layer count picker */}
          <div style={{
            padding: '16px 4px', borderBottom: '0.5px solid var(--separator)',
            display: 'flex', alignItems: 'center', gap: 12,
          }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>{t.layersTitle}</div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 3, letterSpacing: 0.2 }}>
                {t.layersDesc}
              </div>
            </div>
            <SegmentedPicker
              value={String(numLayers)}
              options={[['1', '1 层'], ['2', '2 层']]}
              onChange={(v) => { haptic(); setNumLayers && setNumLayers(parseInt(v, 10)); }}
            />
          </div>

          {/* Activation function picker */}
          <div style={{
            padding: '16px 4px', borderBottom: '0.5px solid var(--separator)',
            display: 'flex', alignItems: 'center', gap: 12,
          }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>{t.activationTitle}</div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 3, letterSpacing: 0.2 }}>
                {t.activationDesc}
              </div>
            </div>
            <SegmentedPicker
              value={activationFn}
              options={[['relu', 'ReLU'], ['sigmoid', 'Sigmoid'], ['tanh', 'Tanh']]}
              onChange={(v) => { haptic(); setActivationFn && setActivationFn(v); }}
            />
          </div>

          {/* Colormap picker */}
          <div style={{
            padding: '16px 4px', borderBottom: '0.5px solid var(--separator)',
            display: 'flex', alignItems: 'center', gap: 12,
          }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>{t.colormapTitle}</div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 3, letterSpacing: 0.2 }}>
                {t.colormapDesc}
              </div>
              {/* Preview gradients */}
              <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
                {[['viridis', 'Viridis'], ['plasma', 'Plasma'], ['gray', lang === 'zh' ? '灰度' : 'Gray'], ['rwb', lang === 'zh' ? '红蓝' : 'R-W-B']].map(([k, label]) => (
                  <div key={k} style={{ display: 'flex', flexDirection: 'column', gap: 2, opacity: colormap === k ? 1 : 0.4 }}>
                    <div style={{
                      width: 48, height: 6,
                      background: window.colormapGradient(k),
                      borderRadius: 1,
                      boxShadow: 'inset 0 0 0 0.5px rgba(0,0,0,0.18)',
                    }} />
                    <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{label}</span>
                  </div>
                ))}
              </div>
            </div>
            <SegmentedPicker
              value={colormap}
              options={[['viridis', 'Viridis'], ['plasma', 'Plasma'], ['gray', lang === 'zh' ? '灰度' : 'Gray'], ['rwb', 'R-W-B']]}
              onChange={(v) => { haptic(); setColormap && setColormap(v); }}
            />
          </div>
        </div>
      </div>

      {/* Bottom legend / hint — same style as L1's bottom legend row */}
      <div style={{
        padding: '6px 56px 8px', display: 'flex', alignItems: 'center', gap: 10,
        borderTop: '0.5px solid var(--separator)',
      }}>
        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', letterSpacing: 0.4 }}>{t.footerLeft}</span>
        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>{t.footerMid}</span>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>inspired by CNN Explainer</span>
      </div>
    </div>
  );
}

// Minimal monochrome toggle — flat, no shadows, matches main screen
function MinimalSwitch({ on, onChange }) {
  return (
    <button onClick={() => { haptic(); onChange && onChange(!on); }} aria-pressed={on}
      style={{
        width: 38, height: 22, borderRadius: 11, padding: 2,
        border: '0.5px solid ' + (on ? 'var(--text)' : 'var(--separator-strong)'),
        background: on ? 'var(--text)' : 'transparent',
        display: 'flex', alignItems: 'center', cursor: 'pointer',
        transition: 'background 180ms ease, border-color 180ms ease',
        flexShrink: 0,
      }}>
      <div style={{
        width: 16, height: 16, borderRadius: 8,
        background: on ? '#fff' : 'var(--text-2)',
        transform: `translateX(${on ? 16 : 0}px)`,
        transition: 'transform 180ms cubic-bezier(.2,.8,.2,1), background 180ms ease',
      }} />
    </button>
  );
}

// Channel count stepper — minimal − / value / + control
function ChannelStepper({ value, onChange, min = 2, max = 10 }) {
  const dec = () => onChange && value > min && onChange(value - 1);
  const inc = () => onChange && value < max && onChange(value + 1);
  return (
    <div style={{
      display: 'flex', alignItems: 'center',
      border: '0.5px solid var(--separator-strong)', borderRadius: 4,
      overflow: 'hidden', flexShrink: 0,
    }}>
      <button onClick={dec} disabled={value <= min} style={{
        width: 24, height: 22, border: 'none', background: 'transparent',
        color: value <= min ? 'var(--text-4)' : 'var(--text-2)',
        cursor: value <= min ? 'default' : 'pointer',
        fontSize: 14, fontFamily: 'var(--font-mono)', lineHeight: 1,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>−</button>
      <div style={{
        minWidth: 28, height: 22, padding: '0 6px',
        borderLeft: '0.5px solid var(--separator-strong)',
        borderRight: '0.5px solid var(--separator-strong)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text)', fontWeight: 600,
      }}>{value}</div>
      <button onClick={inc} disabled={value >= max} style={{
        width: 24, height: 22, border: 'none', background: 'transparent',
        color: value >= max ? 'var(--text-4)' : 'var(--text-2)',
        cursor: value >= max ? 'default' : 'pointer',
        fontSize: 14, fontFamily: 'var(--font-mono)', lineHeight: 1,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>+</button>
    </div>
  );
}

// Minimal segmented control — 2-3 options, flat, monochrome
function SegmentedPicker({ value, options, onChange }) {
  return (
    <div style={{
      display: 'flex', border: '0.5px solid var(--separator-strong)', borderRadius: 4,
      overflow: 'hidden', flexShrink: 0,
    }}>
      {options.map(([k, label], i) => {
        const active = value === k;
        return (
          <button key={k} onClick={() => onChange && onChange(k)} style={{
            padding: '4px 10px',
            border: 'none',
            borderLeft: i > 0 ? '0.5px solid var(--separator-strong)' : 'none',
            background: active ? 'var(--text)' : 'transparent',
            color: active ? '#fff' : 'var(--text-2)',
            fontSize: 11, fontFamily: 'var(--font-mono)',
            cursor: 'pointer', letterSpacing: 0.3,
            transition: 'background 160ms ease, color 160ms ease',
          }}>{label}</button>
        );
      })}
    </div>
  );
}

function SectionHeader({ title }) {
  return (
    <div style={{
      fontSize: 12, color: 'var(--text-3)', textTransform: 'uppercase',
      letterSpacing: 0.5, padding: '20px 16px 8px', fontFamily: 'var(--font-mono)',
    }}>{title}</div>
  );
}

function Toggle({ on, onChange }) {
  return (
    <button onClick={() => { haptic(); onChange && onChange(!on); }} style={{
      width: 50, height: 30, borderRadius: 15, border: 'none', padding: 2,
      background: on ? '#34c759' : '#e5e5ea',
      transition: 'background 200ms ease',
      display: 'flex', alignItems: 'center',
    }}>
      <div style={{
        width: 26, height: 26, borderRadius: 13, background: '#fff',
        boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
        transform: `translateX(${on ? 20 : 0}px)`,
        transition: 'transform 200ms ease',
      }} />
    </button>
  );
}

Object.assign(window, { ScreenImagePicker, ScreenTutorial, ScreenSettings });

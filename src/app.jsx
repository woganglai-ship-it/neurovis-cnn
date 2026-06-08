/* global React, ReactDOM, IOSDevice, ScreenL1, ScreenImagePicker, ScreenTutorial, ScreenSettings,
   TweaksPanel, useTweaks, TweakSection, TweakRadio, TweakToggle, TweakSlider, LAYERS */
const { useState: uSA, useEffect: uEA, useRef: uRA } = React;

const TWEAK_DEFAULS = /*EDITMODE-BEGIN*/{
  "animSpeed": "medium"
}/*EDITMODE-END*/;

// App mode: full-bleed render (no fake iPhone frame, no debug panels).
// Triggered by ?app=1, or auto-on when viewport looks like a phone.
function useAppMode() {
  const PHONE_PORTRAIT_MQ = '(max-width: 540px)';
  const PHONE_LANDSCAPE_MQ = '(max-height: 540px) and (max-width: 980px)';
  const detect = () => {
    if (typeof window === 'undefined') return false;
    const q = new URLSearchParams(window.location.search).get('app');
    if (q === '1') return true;
    if (q === '0') return false;
    return window.matchMedia(PHONE_PORTRAIT_MQ).matches || window.matchMedia(PHONE_LANDSCAPE_MQ).matches;
  };
  const [on, setOn] = uSA(detect);
  uEA(() => {
    const portraitMq = window.matchMedia(PHONE_PORTRAIT_MQ);
    const landscapeMq = window.matchMedia(PHONE_LANDSCAPE_MQ);
    const h = () => setOn(detect());
    portraitMq.addEventListener('change', h);
    landscapeMq.addEventListener('change', h);
    window.addEventListener('resize', h);
    return () => {
      portraitMq.removeEventListener('change', h);
      landscapeMq.removeEventListener('change', h);
      window.removeEventListener('resize', h);
    };
  }, []);
  uEA(() => {
    document.body.classList.toggle('app-mode', on);
  }, [on]);
  return on;
}

function App() {
  const appMode = useAppMode();
  const [tweaks, setTweak] = useTweaks(TWEAK_DEFAULS);
  // 单一入口:打开就是 L1 主可视化。没有 splash / onboarding / home / cnn entry。
  const [route, setRoute] = uSA({ name: 'l1' });
  const [history, setHistory] = uSA([]);
  const [catIndex, setCatIndex] = uSA(0);
  const [pickerOpen, setPickerOpen] = uSA(false);
  const [computeWave, setComputeWave] = uSA({});
  const [showDetail, setShowDetail] = uSA(false); // 默认隐藏张量维度
  const [lang, setLang] = uSA('zh');
  const [numChannels, setNumChannels] = uSA(10); // 对齐 CNN Explainer: 默认 10 个通道/卷积核
  const [kernelSize, setKernelSize] = uSA(3); // 卷积核大小,3/5/7
  const [numLayers, setNumLayers] = uSA(1); // 卷积层数, 1 或 2
  const [activationFn, setActivationFn] = uSA('relu'); // 'relu' | 'sigmoid' | 'tanh'
  const [colormap, setColormap] = uSA('rwb'); // 默认蓝橙(对应规格中蓝负橙正) // 'viridis' | 'gray' | 'rwb'
  const [recomputeTick, setRecomputeTick] = uSA(0);
  const [isRecomputing, setIsRecomputing] = uSA(false);
  const isFirstMountRef = uRA(true);

  // 当 numChannels / kernelSize / numLayers / activationFn 改变时,重算 NETWORK_DATA
  uEA(() => {
    if (isFirstMountRef.current) {
      isFirstMountRef.current = false;
      return;
    }
    let cancelled = false;
    setIsRecomputing(true);
    (async () => {
      await window.recomputeAll(numChannels, kernelSize, activationFn, numLayers);
      if (!cancelled) {
        setRecomputeTick(t => t + 1);
        setIsRecomputing(false);
      }
    })();
    return () => { cancelled = true; };
  }, [numChannels, kernelSize, activationFn, numLayers]);

  const go = (name, params = {}) => {
    setHistory(h => [...h, route]);
    setRoute({ name, ...params });
  };
  const back = () => {
    setHistory(h => {
      if (h.length === 0) { setRoute({ name: 'l1' }); return h; }
      const prev = h[h.length - 1];
      setRoute(prev);
      return h.slice(0, -1);
    });
  };

  // Trigger recompute wave when category changes
  const triggerWave = () => {
    LAYERS.forEach((_, i) => {
      setTimeout(() => {
        setComputeWave(prev => ({ ...prev, [i]: Date.now() }));
      }, i * 90);
    });
    setTimeout(() => setComputeWave({}), 1500 + LAYERS.length * 90);
  };

  const handlePickCat = (i) => {
    setCatIndex(i);
    setPickerOpen(false);
    triggerWave();
  };

  // 横屏固定尺寸 (844×390) — 桌面预览展示为一台横放的 iPhone
  const FRAME_W = 844, FRAME_H = 390;

  const frameStyle = appMode ? {
    width: '100vw', height: '100vh', minHeight: '100dvh',
    overflow: 'hidden', position: 'relative',
    background: 'var(--bg)',
  } : {
    width: FRAME_W, height: FRAME_H, borderRadius: 48,
    overflow: 'hidden', position: 'relative',
    background: 'var(--bg)',
    boxShadow: '0 40px 80px rgba(0,0,0,0.18), 0 0 0 1px rgba(0,0,0,0.12), 0 0 0 12px #1a1a1c',
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: appMode ? 0 : 12 }}>
      <div style={frameStyle}>
        {/* Dynamic island — landscape: vertical pill on the LEFT edge */}
        {!appMode && (
          <div style={{
            position: 'absolute', left: 11, top: '50%', transform: 'translateY(-50%)',
            width: 37, height: 126, borderRadius: 24, background: '#000', zIndex: 200,
          }} />
        )}

        {/* Screens */}
        <div style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden' }}>
          {route.name === 'l1' && (
            <ScreenL1
              catIndex={catIndex}
              onChangeCat={handlePickCat}
              onOpenImagePicker={() => setPickerOpen(true)}
              onOpenTutorial={() => go('tutorial')}
              onOpenSettings={() => go('settings')}
              computeWave={computeWave}
              showDetail={showDetail}
              lang={lang}
              numChannels={numChannels}
              kernelSize={kernelSize}
              activationFn={activationFn}
              recomputeTick={recomputeTick}
              isRecomputing={isRecomputing}
              colormap={colormap}
            />
          )}
          {route.name === 'tutorial' && <ScreenTutorial onBack={back} lang={lang} />}
          {route.name === 'settings' && <ScreenSettings onBack={back} showDetail={showDetail} setShowDetail={setShowDetail} lang={lang} setLang={setLang} numChannels={numChannels} setNumChannels={setNumChannels} kernelSize={kernelSize} setKernelSize={setKernelSize} numLayers={numLayers} setNumLayers={setNumLayers} activationFn={activationFn} setActivationFn={setActivationFn} colormap={colormap} setColormap={setColormap} />}

          {pickerOpen && <ScreenImagePicker catIndex={catIndex} onPick={handlePickCat} onClose={() => setPickerOpen(false)} />}
        </div>

        {/* Home indicator — landscape: vertical pill on the RIGHT edge */}
        {!appMode && (
          <div style={{
            position: 'absolute', right: 0, top: 0, bottom: 0,
            width: 34, zIndex: 250, pointerEvents: 'none',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <div style={{
              width: 5, height: 139, borderRadius: 100,
              background: 'rgba(0,0,0,0.25)', marginRight: 8,
            }} />
          </div>
        )}
      </div>

      {!appMode && <TweaksPanel title="Tweaks">
        <TweakSection title="可视化">
          <TweakRadio label="动画速度" value={tweaks.animSpeed} options={[['slow','慢'],['medium','中'],['fast','快']]} onChange={v => setTweak('animSpeed', v)} />
        </TweakSection>
      </TweaksPanel>}
    </div>
  );
}

// 等 CNN 把 10 个类别的真实激活算完再 mount。
// 同时挂一个超时/错误兜底——VS Code Simple Browser 等弱 WebView 跑不了 TF.js 时给出明确提示。
const _root = ReactDOM.createRoot(document.getElementById('root'));
function splash(msg, sub) {
  _root.render(
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 12,
      height: '100vh', padding: 24, textAlign: 'center',
      fontFamily: 'var(--font-mono)', color: '#555',
    }}>
      <div style={{ fontSize: 14 }}>{msg}</div>
      {sub && <div style={{ fontSize: 12, color: '#999', maxWidth: 320, lineHeight: 1.5 }}>{sub}</div>}
    </div>
  );
}
splash('正在跑 CNN 前向…');

let _settled = false;
// Babel standalone 用 XHR 单独加载 .jsx，不遵循 async=false 顺序，
// 所以 app.jsx 可能比 data.js 先执行 → window.NETWORK_READY 还没定义。
// 轮询等它出现。
function attachWhenReady() {
  if (window.NETWORK_READY) {
    window.NETWORK_READY
      .then(() => { _settled = true; _root.render(<App />); })
      .catch(err => {
        _settled = true;
        console.error('[CNN] 初始化失败:', err);
        splash('CNN 初始化失败', String(err && err.message || err));
      });
  } else {
    setTimeout(attachWhenReady, 20);
  }
}
attachWhenReady();

// 把任何未捕获的错误显示到 splash，便于在没开 DevTools 时也能看到挂点
window.addEventListener('error', e => {
  if (_settled) return;
  _settled = true;
  splash('运行时错误', (e.message || 'unknown') + '  @  ' + (e.filename || '') + ':' + (e.lineno || ''));
});
window.addEventListener('unhandledrejection', e => {
  if (_settled) return;
  _settled = true;
  splash('Promise rejection', String(e.reason && e.reason.message || e.reason));
});

// 5s 超时兜底：把当前运行时状态显示出来，方便定位
setTimeout(() => {
  if (_settled) return;
  const status = [
    'CNN: ' + (window.CNN ? 'loaded' : '✗ 缺失'),
    'CNN_INPUTS: ' + (window.CNN_INPUTS ? 'loaded' : '✗ 缺失'),
    'NETWORK_READY: ' + (window.NETWORK_READY ? 'pending' : '✗ 未定义'),
    'NETWORK_DATA[0]: ' + (window.NETWORK_DATA && window.NETWORK_DATA[0] ? 'ready' : 'null'),
  ].join(' · ');
  splash('CNN 加载超时（5s）', status);
}, 5000);

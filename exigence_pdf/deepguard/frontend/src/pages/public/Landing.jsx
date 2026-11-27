/* Landing page — the exact design of the approved single-file build
 * (DeepGuard_Project p05_public.js home + p01_head.html CSS, now scoped
 * under `.dg5`): sticky nav, hero with the browser mockup and procedural
 * face, logo strip, features, steps, live demo, tech card and CTA band.
 * The live-demo cards run REAL inference against the Django backend
 * (POST /predict) instead of the old simulated timer. */
import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext.jsx'
import { useTheme } from '../../context/ThemeContext.jsx'
import { predict } from '../../daf/api.js'

/* ---------- icons (verbatim from the approved build) ---------- */
const IC = {
  shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
  shieldCheck: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
  scan: '<path d="M3 7V5a2 2 0 0 1 2-2h2m10 0h2a2 2 0 0 1 2 2v2m0 10v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4m11.4-11.4 1.4-1.4"/>',
  moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  zap: '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
  arrowRight: '<path d="M5 12h14m-6-7 7 7-7 7"/>',
  home: '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 22V12h6v10"/>',
  cpu: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v2m6-2v2M9 20v2m6-2v2M2 9h2m-2 6h2m16-6h2m-2 6h2"/>',
  layers: '<path d="m12 2 10 5-10 5L2 7l10-5z"/><path d="m2 17 10 5 10-5M2 12l10 5 10-5"/>',
  database: '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
  activity: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
  server: '<rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><path d="M6 6h.01M6 18h.01"/>',
  globe: '<circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15 15 0 0 1 0 20 15 15 0 0 1 0-20z"/>',
  target: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
  clock: '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
  users: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8"/>',
  lock: '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
  alert: '<path d="m21.7 18.3-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-2.7z"/><path d="M12 9v4m0 4h.01"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4m0-4h.01"/>',
}
const icon = (name, cls = '') => (
  <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
    dangerouslySetInnerHTML={{ __html: IC[name] || IC.info }} />
)

/* extra icons used by the auth pages (same source) */
Object.assign(IC, {
  mail: '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 6L2 7"/>',
  eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>',
  eyeOff: '<path d="m10.7 5.1A9.8 9.8 0 0 1 12 5c6.5 0 10 7 10 7a17.4 17.4 0 0 1-3 3.9M6.6 6.6A17 17 0 0 0 2 12s3.5 7 10 7a9.9 9.9 0 0 0 4.4-1M9.9 9.9a3 3 0 0 0 4.2 4.2"/><path d="m2 2 20 20"/>',
  key: '<path d="m21 2-2 2m-7.6 7.6a5.5 5.5 0 1 1-7.8 7.8 5.5 5.5 0 0 1 7.8-7.8zm0 0L15.5 7.5m0 0 3 3L22 7l-3-3m-3.5 3.5L19 4"/>',
  user: '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5m-5 5V3"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
})

/* ---------- procedural face generator (verbatim from the approved build) ---------- */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
function drawFace(canvas, seed, fake, label) {
  const ctx = canvas.getContext('2d')
  const W = (canvas.width = 320), H = (canvas.height = 320)
  const rng = mulberry32(seed)
  const bg = ctx.createLinearGradient(0, 0, W, H)
  const hue = Math.floor(rng() * 40) + (fake ? 190 : 210)
  bg.addColorStop(0, `hsl(${hue},42%,22%)`)
  bg.addColorStop(1, `hsl(${hue + 26},48%,12%)`)
  ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H)
  ctx.fillStyle = `hsl(${hue + 10},30%,26%)`
  ctx.beginPath(); ctx.ellipse(W / 2, H * 1.18, W * 0.42, H * 0.3, 0, 0, 7); ctx.fill()
  ctx.fillStyle = `hsl(28,38%,${52 + rng() * 10}%)`
  ctx.fillRect(W / 2 - 30, H * 0.62, 60, 70)
  ctx.fillStyle = `hsl(30,42%,${56 + rng() * 12}%)`
  ctx.beginPath(); ctx.ellipse(W / 2, H * 0.42, W * 0.23, H * 0.29, 0, 0, 7); ctx.fill()
  ctx.fillStyle = `hsl(${Math.floor(rng() * 30)},30%,${14 + rng() * 22}%)`
  ctx.beginPath(); ctx.ellipse(W / 2, H * 0.42 - H * 0.16, W * 0.235, H * 0.15, 0, Math.PI, 0); ctx.fill()
  const ey = H * 0.38, ex = W * 0.155
  ;[-1, 1].forEach(s => {
    ctx.fillStyle = '#fdfdfb'
    ctx.beginPath(); ctx.ellipse(W / 2 + s * ex, ey, 17, 10, 0, 0, 7); ctx.fill()
    ctx.fillStyle = 'hsl(205,45%,35%)'
    ctx.beginPath(); ctx.arc(W / 2 + s * ex + (fake ? s * 3 : 0), ey, 5.4, 0, 7); ctx.fill()
  })
  ctx.strokeStyle = 'hsl(25,28%,28%)'; ctx.lineWidth = 5; ctx.lineCap = 'round'
  ;[-1, 1].forEach(s => { ctx.beginPath(); ctx.moveTo(W / 2 + s * (ex - 14), ey - 16); ctx.lineTo(W / 2 + s * (ex + 13), ey - 19); ctx.stroke() })
  ctx.strokeStyle = 'hsla(25,30%,40%,.7)'; ctx.lineWidth = 4
  ctx.beginPath(); ctx.moveTo(W / 2, ey + 8); ctx.quadraticCurveTo(W / 2 + 7, H * 0.47, W / 2 - 2, H * 0.485); ctx.stroke()
  ctx.strokeStyle = 'hsl(6,52%,42%)'; ctx.lineWidth = 6
  ctx.beginPath()
  if (fake) ctx.moveTo(W / 2 - 26, H * 0.53)
  ctx.moveTo(W / 2 - 24, H * 0.535); ctx.quadraticCurveTo(W / 2, H * 0.56, W / 2 + 24, H * 0.535)
  ctx.stroke()
  if (fake) {
    ctx.globalAlpha = 0.16
    ctx.strokeStyle = '#c7f9ff'; ctx.lineWidth = 1
    for (let i = 1; i < 8; i++) { ctx.beginPath(); ctx.moveTo((W / 8) * i, H * 0.15); ctx.lineTo((W / 8) * i, H * 0.9); ctx.stroke() }
    for (let i = 1; i < 8; i++) { ctx.beginPath(); ctx.moveTo(W * 0.1, (H / 8) * i); ctx.lineTo(W * 0.9, (H / 8) * i); ctx.stroke() }
    ctx.globalAlpha = 0.1; ctx.fillStyle = '#9ff3ff'
    ctx.fillRect(W * 0.28, 0, W * 0.05, H); ctx.fillRect(0, H * 0.3, W, H * 0.04)
    ctx.globalAlpha = 1
  }
  if (label) {
    ctx.fillStyle = 'rgba(255,255,255,.85)'; ctx.font = '600 13px Inter,sans-serif'
    ctx.fillText(label, 12, 20)
  }
}

/* ---------- shared chrome (exported for the auth pages) ---------- */
export const p05icon = icon
export const p05IC = IC

export function Brand({ to = '/' }) {
  return (
    <Link to={to} className="brand" aria-label="DeepGuard home">
      <span className="logo">{icon('shield')}</span>
      <span>DeepGuard<small>AdaptiveFusion</small></span>
    </Link>
  )
}

/* ---------- live demo (real backend inference) ---------- */
const DEMO_SAMPLES = [
  { seed: 12, name: 'Sample A' },
  { seed: 31, name: 'Sample B' },
  { seed: 55, name: 'Sample C' },
  { seed: 84, name: 'Sample D' },
]

function LiveDemo() {
  const navigate = useNavigate()
  const { user } = useAuth()
  const canvasRefs = useRef([])
  const [state, setState] = useState({ phase: 'idle' }) // idle | loading | done | error

  useEffect(() => {
    DEMO_SAMPLES.forEach((s, i) => {
      if (canvasRefs.current[i]) drawFace(canvasRefs.current[i], s.seed, false)
    })
  }, [])

  const run = async (i) => {
    if (state.phase === 'loading') return
    const canvas = canvasRefs.current[i]
    if (!canvas) return
    setState({ phase: 'loading' })
    try {
      const blob = await new Promise((r) => canvas.toBlob(r, 'image/jpeg', 0.92))
      const file = new File([blob], `sample-${DEMO_SAMPLES[i].seed}.jpg`, { type: 'image/jpeg' })
      const { data, clientMs } = await predict(file)
      setState({ phase: 'done', data, clientMs, name: DEMO_SAMPLES[i].name })
    } catch (e) {
      setState({ phase: 'error', message: e.message || 'Inference failed.' })
    }
  }

  const result = state.phase === 'done' ? state.data : null
  const pFake = result?.prob_fake ?? 0
  const isFake = (result?.label || 'real') === 'fake'

  return (
    <div className="card card-pad" style={{ maxWidth: 860, margin: '0 auto' }}>
      <div className="ex-grid">
        {DEMO_SAMPLES.map((s, i) => (
          <div key={s.seed} className="ex-item" role="button" tabIndex={0}
            onClick={() => run(i)} onKeyDown={(e) => e.key === 'Enter' && run(i)}>
            <canvas ref={(el) => (canvasRefs.current[i] = el)} />
            <span className="cap">{s.name}</span>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 18 }}>
        {state.phase === 'loading' && (
          <div className="inf-load">
            <div className="ring" />
            <div className="st">running dual-stream inference...</div>
            <div className="stages">preprocess · spatial · frequency · fusion — POST /predict</div>
          </div>
        )}
        {state.phase === 'error' && (
          <div className="form-alert err">{icon('alert')}<span>{state.message}</span></div>
        )}
        {state.phase === 'done' && result && (
          <>
            <div className={`verdict-hero ${isFake ? 'fake' : 'real'}`}>
              {icon(isFake ? 'alert' : 'shieldCheck')}
              <div>
                <div className="big">{isFake ? 'Likely FAKE' : 'Likely REAL'}</div>
                <div style={{ fontSize: '.78rem', color: 'var(--muted)' }}>
                  {state.name} — live result from the ONNX backend
                </div>
              </div>
              <div className="conf">
                <b>{(result.confidence * 100).toFixed(1)}%</b>
                <span>confidence</span>
              </div>
            </div>
            <div style={{ marginTop: 16 }}>
              <div className="prob-row">
                <div className="lbl"><span>P(real)</span><span className="pct">{((1 - pFake) * 100).toFixed(1)}%</span></div>
                <div className="prog prog-ok"><i style={{ width: `${(1 - pFake) * 100}%` }} /></div>
              </div>
              <div className="prob-row">
                <div className="lbl"><span>P(fake)</span><span className="pct">{(pFake * 100).toFixed(1)}%</span></div>
                <div className="prog prog-bad"><i style={{ width: `${pFake * 100}%` }} /></div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 6 }}>
              <span className="lat-chip">{icon('zap')}server <b>{(result.latency?.total_server_ms ?? 0).toFixed(1)} ms</b></span>
              <span className="lat-chip">fusion <b>{(result.fusion_weights?.spatial ?? 0).toFixed(2)} spatial</b></span>
              <span className="badge badge-brand" style={{ alignSelf: 'center' }}>
                {icon('info')}Real model — no simulation
              </span>
              {!user && (
                <button className="btn btn-primary btn-sm" style={{ marginLeft: 'auto' }} onClick={() => navigate('/signup')}>
                  Create account
                </button>
              )}
            </div>
          </>
        )}
        {state.phase === 'idle' && (
          <p style={{ textAlign: 'center', fontSize: '.8rem', color: 'var(--muted)', padding: '14px 0 2px' }}>
            Click a face to run the real dual-stream model — verdict in a second.
          </p>
        )}
      </div>
    </div>
  )
}

/* ---------- page ---------- */
const FEATURES = [
  ['shieldCheck', 'ic-brand', 'Dual-stream engine',
    'An EfficientNetV2-RW-S spatial stream and an FFT frequency stream vote through a learned softmax fusion gate, catching both visual and spectral GAN artefacts.'],
  ['target', 'ic-freq', 'Explainable verdicts',
    'Grad-CAM overlays show exactly which regions drove the decision, with a class switcher and opacity control — no black-box scores.'],
  ['zap', 'ic-warn', 'Sub-100 ms inference',
    'ONNX Runtime serving with a 2-input / 3-output contract, per-request server timing and client-side latency tracking on every call.'],
  ['clock', 'ic-ok', 'Analysis history',
    'Every scan is stored with verdict, confidence, fusion weights and latency — searchable, filterable and exportable to CSV.'],
  ['users', 'ic-info', 'Team administration',
    'Admins manage members, roles and status from a full console with audit logging of every sensitive action.'],
  ['lock', 'ic-bad', 'Secure by default',
    'Session auth, role-guarded routes, API keys per analyst and a complete audit trail — designed to bolt onto the FastAPI backend.'],
]

const STEPS = [
  ['1', 'Add a face image', 'Drag and drop a file, pick a sample, or frame yourself with the webcam. Client-side validation mirrors the server rules before anything is sent.'],
  ['2', 'Dual-stream inference', 'The image and its FFT magnitude travel through the two streams; the fusion gate weighs them and emits a single calibrated probability.'],
  ['3', 'Read and keep the verdict', 'Animated probabilities, fusion weights, a Grad-CAM heatmap and latency — saved to your history with one click.'],
]

const TECH = [
  ['Model', 'AdaptiveFusionDetector'],
  ['Backbone', 'efficientnetv2_rw_s (~31.9M params)'],
  ['Serving', 'ONNX Runtime, opset 17'],
  ['API', 'FastAPI contract — served by Django'],
  ['Pipeline', 'DVC prepare-train-eval-export'],
  ['Tracking', 'MLflow experiments R1-R4'],
  ['Deploy', 'Docker Compose x3 services'],
  ['Frontend', 'React 18 + Vite + Tailwind'],
]

export default function Landing() {
  const { user } = useAuth()
  const { theme, toggle } = useTheme()
  const nav = useNavigate()
  const mockFace = useRef(null)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    if (mockFace.current) drawFace(mockFace.current, 77, true)
  }, [])

  return (
    <div className="dg5">
      {/* nav */}
      <header className="nav">
        <div className="nav-in">
          <Brand />
          <nav className="nav-links" aria-label="Main">
            <a href="#features">Features</a>
            <a href="#how">How it works</a>
            <a href="#demo">Live demo</a>
            <a href="#tech">Tech</a>
          </nav>
          <div className="nav-cta">
            <button className="theme-toggle" aria-label="Toggle theme" onClick={toggle}>
              {icon(theme === 'dark' ? 'sun' : 'moon')}
            </button>
            {user ? (
              <button className="btn btn-primary btn-sm" onClick={() => nav(user.role === 'admin' ? '/admin/overview' : '/user/dashboard')}>
                {icon('home')}Dashboard
              </button>
            ) : (
              <>
                <button className="btn btn-ghost btn-sm" onClick={() => nav('/login')}>Sign in</button>
                <button className="btn btn-primary btn-sm" onClick={() => nav('/signup')}>Get started</button>
              </>
            )}
          </div>
          <button className="burger" aria-label="Open menu" onClick={() => setMenuOpen(o => !o)}>
            {icon(menuOpen ? 'x' : 'menu')}
          </button>
        </div>
      </header>

      {/* mobile menu */}
      {menuOpen && (
        <div className="mobile-menu" style={{ display: 'block', position: 'fixed', inset: 0, zIndex: 130, background: 'var(--bg)', padding: 20, overflowY: 'auto' }}>
          <Brand />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 24 }}>
            <a href="#features" className="btn btn-outline" onClick={() => setMenuOpen(false)}>Features</a>
            <a href="#how" className="btn btn-outline" onClick={() => setMenuOpen(false)}>How it works</a>
            {user
              ? <button className="btn btn-primary" onClick={() => nav('/user/dashboard')}>Open dashboard</button>
              : <>
                  <button className="btn btn-primary" onClick={() => nav('/signup')}>Create account</button>
                  <button className="btn btn-outline" onClick={() => nav('/login')}>Sign in</button>
                </>}
          </div>
        </div>
      )}

      {/* hero */}
      <section className="hero">
        <div className="hero-bg">
          <span className="orb o1" /><span className="orb o2" /><span className="orb o3" />
          <div className="grid-lines" />
        </div>
        <div className="hero-in">
          <div>
            <span className="eyebrow"><span className="dot" />Dual-stream detection engine</span>
            <h1>Spot AI-generated faces <span className="grad-text">before they fool anyone</span></h1>
            <p className="lead">
              DeepGuard pairs a spatial CNN with a frequency-domain network and a learned fusion gate,
              then explains every verdict with Grad-CAM heatmaps. Upload, webcam or samples — a verdict
              in under 100&nbsp;ms.
            </p>
            <div className="hero-ctas">
              {user ? (
                <button className="btn btn-primary btn-lg" onClick={() => nav('/analyze')}>
                  {icon('scan')}Run an analysis
                </button>
              ) : (
                <>
                  <button className="btn btn-primary btn-lg" onClick={() => nav('/analyze')}>
                    {icon('zap')}Run an analysis — no card
                  </button>
                  <button className="btn btn-outline btn-lg" onClick={() => nav('/login')}>
                    {icon('arrowRight')}Sign in
                  </button>
                </>
              )}
            </div>
            <div className="hero-stats">
              <div><b>0.952</b><span>external-test AUC (notebook-reported)</span></div>
              <div><b>&lt;100ms</b><span>ONNX inference budget</span></div>
              <div><b>2 streams</b><span>spatial + FFT frequency</span></div>
            </div>
          </div>

          {/* mockup */}
          <div className="mock">
            <div className="mock-hd">
              <span className="dots"><i /><i /><i /></span>
              <span className="url">deepguard.io/analyze</span>
            </div>
            <div className="mock-img">
              <canvas ref={mockFace} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              <div className="mock-scanline" />
              <span className="mock-verdict badge badge-bad">{icon('alert')}FAKE 96.4%</span>
            </div>
            <div className="mock-row"><span>P(real)</span><b>3.6%</b></div>
            <div className="prog prog-ok" style={{ marginBottom: 10 }}><i style={{ width: '4%' }} /></div>
            <div className="mock-row"><span>P(fake)</span><b>96.4%</b></div>
            <div className="prog prog-bad"><i style={{ width: '96%' }} /></div>
            <div className="mock-row" style={{ marginTop: 12 }}><span>Spatial weight</span><b>0.62</b></div>
            <div className="fusion-bar"><span className="sp" style={{ width: '62%' }} /><span className="fq" style={{ width: '38%' }} /></div>
            <div className="mock-row" style={{ marginTop: 10 }}><span>Server latency</span><b>31.2 ms</b></div>
          </div>
        </div>
      </section>

      {/* logos */}
      <div className="logo-strip">
        <span>{icon('cpu')}ONNX Runtime</span>
        <span>{icon('layers')}timm / PyTorch</span>
        <span>{icon('database')}DVC</span>
        <span>{icon('activity')}MLflow</span>
        <span>{icon('server')}FastAPI contract</span>
        <span>{icon('globe')}React + Vite</span>
      </div>

      {/* features */}
      <section className="section" id="features">
        <div className="section-hd">
          <span className="eyebrow">Platform</span>
          <h2>Everything a verification team needs</h2>
          <p>From a single scan to fleet-wide monitoring — the workflow is covered end to end.</p>
        </div>
        <div className="grid g-3">
          {FEATURES.map(([ic, cls, title, text]) => (
            <div key={title} className="card feat">
              <span className={`ic ${cls}`}>{icon(ic)}</span>
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* how */}
      <section className="section" id="how" style={{ paddingTop: 0 }}>
        <div className="section-hd">
          <span className="eyebrow">Workflow</span>
          <h2>Three steps to a verdict</h2>
          <p>The same pipeline whether you verify one profile picture or batch a whole dataset.</p>
        </div>
        <div className="steps">
          {STEPS.map(([n, title, text]) => (
            <div key={n} className="card step">
              <span className="n">{n}</span>
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* live demo */}
      <section className="section" id="demo" style={{ paddingTop: 0 }}>
        <div className="section-hd">
          <span className="eyebrow">Live demo</span>
          <h2>Try it right here</h2>
          <p>Real inference on generated samples — every verdict below comes from the ONNX backend, not a mock.</p>
        </div>
        <LiveDemo />
      </section>

      {/* tech */}
      <section className="section" id="tech" style={{ paddingTop: 0 }}>
        <div className="section-hd">
          <span className="eyebrow">Architecture</span>
          <h2>Built on a real MLOps stack</h2>
          <p>The interface maps one-to-one onto the AdaptiveFusion project backend: FastAPI contract, ONNX, DVC, MLflow and Docker Compose.</p>
        </div>
        <div className="card card-pad">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(210px,1fr))' }}>
            {TECH.map(([k, v]) => (
              <div key={k} className="kv-row">
                <span>{k}</span>
                <b className="mono">{v}</b>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* cta */}
      <section className="section" style={{ paddingTop: 0 }}>
        <div className="cta-band">
          <span className="eyebrow"><span className="dot" />Ready when you are</span>
          <h2>Give your trust a second opinion</h2>
          <p>Create an account and get the full workspace: analysis history, explainability overlays and team administration.</p>
          <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
            <button className="btn btn-primary btn-lg" onClick={() => nav(user ? '/analyze' : '/signup')}>
              {icon('arrowRight')}{user ? 'Run an analysis' : 'Create free account'}
            </button>
            <button className="btn btn-outline btn-lg" onClick={() => nav('/login')}
              style={{ background: 'rgba(255,255,255,.08)', borderColor: 'rgba(255,255,255,.25)', color: '#e6edf7' }}>
              Sign in
            </button>
          </div>
        </div>
      </section>

      {/* footer */}
      <footer className="footer">
        <div>{icon('shield')} <b>DeepGuard</b> — AdaptiveFusion deepfake detection research platform</div>
        <nav>
          <a href="#features">Features</a>
          <a href="#tech">Tech</a>
          <button onClick={() => nav('/login')} style={{ background: 'none', border: 0, cursor: 'pointer', color: 'inherit', font: 'inherit' }}>Sign in</button>
          <button onClick={() => nav('/signup')} style={{ background: 'none', border: 0, cursor: 'pointer', color: 'inherit', font: 'inherit' }}>Create account</button>
        </nav>
        <div className="mono" style={{ fontSize: '.72rem' }}>Research prototype — not production</div>
      </footer>
    </div>
  )
}

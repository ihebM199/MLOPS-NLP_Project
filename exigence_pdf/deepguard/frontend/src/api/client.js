/* ============ DeepGuard React — data layer ============
   Two halves:
   1. Api    — the REAL model endpoints of the Django backend (exact contract
               of the original FastAPI service: /predict, /predict/visualize,
               /health, /model/metadata).
   2. Store  — the platform state (accounts, history, notifications, audit
               log, settings) in localStorage, as in the approved prototype.
               Same schema (deepguard_v1), so data created with the
               single-file UI carries over. */

/* ---------------- helpers ---------------- */
export const uid = (p) => (p || 'id') + '_' + Math.random().toString(36).slice(2, 9)
export const clamp = (v, a, b) => Math.min(b, Math.max(a, v))

export function hashString(str) { // FNV-1a (stable pseudo-randomness, NOT cryptographic)
  let h = 2166136261
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619) }
  return h >>> 0
}
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
export function fmtDate(ts) {
  return new Date(ts).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })
}
export function fmtDateTime(ts) {
  const d = new Date(ts)
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }) + ' ' +
    d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
}
export function timeAgo(ts) {
  const s = Math.max(1, Math.floor((Date.now() - ts) / 1000))
  if (s < 60) return s + 's ago'
  if (s < 3600) return Math.floor(s / 60) + 'm ago'
  if (s < 86400) return Math.floor(s / 3600) + 'h ago'
  return Math.floor(s / 86400) + 'd ago'
}
export const AV_COLORS = ['#1f47f5', '#7c3aed', '#0ea5e9', '#10b981', '#f59e0b', '#f43f5e', '#8b5cf6', '#14b8a6', '#e11d48', '#2563eb']

export function dataURLtoBlob(dataURL) {
  const parts = String(dataURL).split(',')
  const mime = (parts[0].match(/^data:([^;]+)/) || [null, 'image/jpeg'])[1]
  const bin = atob(parts[1] || '')
  const buf = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i)
  return new Blob([buf], { type: mime })
}

/** Shrink an image dataURL to a ~40px JPEG thumbnail for history storage. */
export function makeThumb(dataURL, size = 44) {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      const cv = document.createElement('canvas')
      cv.width = size; cv.height = size
      const cx = cv.getContext('2d')
      const s = Math.min(img.width, img.height)
      cx.drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, size, size)
      try { resolve(cv.toDataURL('image/jpeg', 0.62)) } catch { resolve(null) }
    }
    img.onerror = () => resolve(null)
    img.src = dataURL
  })
}

/* ---------------- real API client (exact original contract) ---------------- */
export class ApiError extends Error {
  constructor(detail, status, code) {
    super(detail)
    this.status = status
    this.code = code
  }
}

async function apiCall(input, init) {
  let res
  try { res = await fetch(input, init) }
  catch (cause) {
    throw new ApiError('API unreachable — start the Django backend (python manage.py runserver) and keep the Vite dev server proxying to port 8000.', 0, 'NETWORK')
  }
  let payload = null
  try { payload = await res.json() } catch (e) { /* non-JSON (PNG) */ }
  if (!res.ok) {
    throw new ApiError((payload && (payload.detail || payload.error)) || ('HTTP ' + res.status), res.status, payload && payload.error)
  }
  return payload
}

export const Api = {
  /** GET /health — {status, model_loaded, model_version, gpu_available, onnxruntime_providers, gradcam_available, uptime_s, detail} */
  health() { return apiCall('/health', { headers: { Accept: 'application/json' } }) },
  /** GET /model/metadata — model card: name, architecture, backbone, mode, image_size, classes, decision_rule, threshold, parameters, metrics, trained_at, model_version, git_sha, limitations */
  metadata() { return apiCall('/model/metadata', { headers: { Accept: 'application/json' } }) },
  /** POST /predict — multipart 'file'. Returns the exact PredictionResponse. */
  predict(blob, name) {
    const fd = new FormData()
    fd.append('file', blob, name || 'capture.jpg')
    return apiCall('/predict', { method: 'POST', body: fd, headers: { Accept: 'application/json' } })
  },
  /** POST /predict/visualize?target=&alpha=&format=json — Grad-CAM overlay (base64 PNG). */
  visualize(blob, name, target, alpha) {
    const fd = new FormData()
    fd.append('file', blob, name || 'capture.jpg')
    const q = new URLSearchParams({ target: target || 'fake', alpha: String(alpha == null ? 1 : alpha), format: 'json' })
    return apiCall('/predict/visualize?' + q.toString(), { method: 'POST', body: fd, headers: { Accept: 'application/json' } })
  },
}

/** Built-in sample faces served by Django (media/samples). */
export const SAMPLES = [
  { id: 'sample_real_1', url: '/media/samples/sample_real_1.jpg', name: 'real-face-01.jpg', truth: 'real' },
  { id: 'sample_real_2', url: '/media/samples/sample_real_2.jpg', name: 'real-face-02.jpg', truth: 'real' },
  { id: 'sample_real_3', url: '/media/samples/sample_real_3.jpg', name: 'real-face-03.jpg', truth: 'real' },
  { id: 'sample_real_4', url: '/media/samples/sample_real_4.jpg', name: 'real-face-04.jpg', truth: 'real' },
  { id: 'sample_fake_1', url: '/media/samples/sample_fake_1.jpg', name: 'synth-face-01.jpg', truth: 'fake' },
  { id: 'sample_fake_2', url: '/media/samples/sample_fake_2.jpg', name: 'synth-face-02.jpg', truth: 'fake' },
  { id: 'sample_fake_3', url: '/media/samples/sample_fake_3.jpg', name: 'synth-face-03.jpg', truth: 'fake' },
  { id: 'sample_fake_4', url: '/media/samples/sample_fake_4.jpg', name: 'synth-face-04.jpg', truth: 'fake' },
  { id: 'sample_fake_5', url: '/media/samples/sample_fake_5.jpg', name: 'synth-face-05.jpg', truth: 'fake' },
].map(s => ({ ...s, description: s.truth === 'real' ? 'Camera portrait — original capture.' : 'AI-generated synthetic portrait.' }))

/** Fetch a sample image as a Blob so it can be POSTed like an upload. */
export async function fetchSampleBlob(sample) {
  const res = await fetch(sample.url)
  if (!res.ok) throw new ApiError('Sample media unavailable — the Django backend must be running.', res.status)
  return res.blob()
}

/* ---------------- store (localStorage, schema deepguard_v1) ---------------- */
const LS_KEY = 'deepguard_v1'
const DAY = 86400000
let DB = null
const listeners = new Set()

export function saveDB() {
  try { localStorage.setItem(LS_KEY, JSON.stringify(DB)) } catch (e) { /* storage full/blocked */ }
  listeners.forEach(cb => { try { cb() } catch { /* listener error */ } })
}
export function subscribeStore(cb) { listeners.add(cb); return () => listeners.delete(cb) }

function loadDB() {
  try {
    const raw = localStorage.getItem(LS_KEY)
    if (raw) { const d = JSON.parse(raw); if (d && d.v === 1) return d }
  } catch (e) { /* corrupted -> reseed */ }
  return null
}

/* seed: deterministic demo data, relative to "now" (same as the approved prototype) */
function seedDB() {
  const now = Date.now()
  const rng = mulberry32(20261124)
  const db = {
    v: 1, session: null, theme: 'dark',
    settings: {
      platform: 'DeepGuard', threshold: 0.5, maintenance: false,
      minPwd: 8, sessionTimeout: 60, twoFA: false, signupOpen: true, gradcamOpacity: 0.55,
    },
    users: [], history: [], logs: [], notifs: [], model: {},
  }

  const mkUser = (name, email, pwd, role, status, joinedAgo, activeAgo) => ({
    id: uid('u'), name, email, pass: hashString(pwd + '|dg'), role, status,
    joined: now - joinedAgo * DAY, lastActive: now - activeAgo * DAY, org: '',
    color: AV_COLORS[Math.floor(rng() * AV_COLORS.length)],
    apiKey: 'dg_live_' + hashString(email).toString(36).padStart(10, '0') + Math.floor(rng() * 1e6).toString(36),
  })
  db.users.push(mkUser('Amina Belkacem', 'admin@deepguard.io', 'Admin@2026', 'admin', 'active', 92, 0.01))
  db.users.push(mkUser('Daniel Reyes', 'demo@deepguard.io', 'Demo@2026', 'user', 'active', 61, 0.04))

  const NAMES = [
    ['Sofia Marchetti', 'sofia.marchetti'], ["Liam O'Connor", 'liam.oconnor'], ['Yuki Tanaka', 'yuki.tanaka'],
    ['Noah Fischer', 'noah.fischer'], ['Priya Sharma', 'priya.sharma'], ['Lucas Moreau', 'lucas.moreau'],
    ['Emma Van Dijk', 'emma.vandijk'], ['Omar Haddad', 'omar.haddad'], ['Chloe Bernard', 'chloe.bernard'],
    ['Ethan Kowalski', 'ethan.kowalski'], ['Ines Ferreira', 'ines.ferreira'], ['Marco Rossi', 'marco.rossi'],
    ['Amara Diallo', 'amara.diallo'], ['Viktor Petrov', 'viktor.petrov'], ['Hana Kim', 'hana.kim'],
    ['Jonas Lindqvist', 'jonas.lindqvist'], ['Rania Cherif', 'rania.cherif'], ['Diego Alvarez', 'diego.alvarez'],
    ['Freya Nilsson', 'freya.nilsson'], ['Kofi Mensah', 'kofi.mensah'], ['Isabella Ricci', 'isabella.ricci'],
    ['Arjun Mehta', 'arjun.mehta'], ['Celine Dubois', 'celine.dubois'], ['Tomas Novak', 'tomas.novak'],
    ['Leila Bensalem', 'leila.bensalem'], ['Ryan Mitchell', 'ryan.mitchell'], ['Zara Ahmed', 'zara.ahmed'],
    ['Felix Wagner', 'felix.wagner'], ['Nadia Yamamoto', 'nadia.yamamoto'],
  ]
  const STATUSES = ['active', 'active', 'active', 'active', 'invited', 'suspended']
  NAMES.forEach((n, i) => {
    db.users.push(mkUser(n[0], n[1] + '@example.com', 'Example@2026', 'user', STATUSES[i % STATUSES.length], 3 + rng() * 85, rng() * 12))
  })

  // history for the demo + admin users
  const SRC = ['upload', 'webcam', 'sample']
  const NAMES2 = ['selfie-check.jpg', 'id-verification.png', 'profile-crop.jpg', 'webcam-frame.jpg', 'team-photo.png', 'avatar-scan.jpg', 'interview-frame.jpg', 'linkedin-thumb.png']
  const mkHist = (user, count, offsetDays) => {
    for (let i = 0; i < count; i++) {
      const r = rng()
      const fake = r < 0.42
      const conf = fake ? 0.74 + rng() * 0.24 : 0.78 + rng() * 0.2
      db.history.push({
        id: uid('h'), userId: user.id, ts: now - (offsetDays + i * (rng() * 1.6 + 0.6)) * DAY,
        source: SRC[Math.floor(rng() * 3)], name: NAMES2[(i + offsetDays) % NAMES2.length],
        verdict: fake ? 'fake' : 'real', confidence: conf,
        pFake: fake ? conf : 1 - conf,
        wS: 0.4 + rng() * 0.45, latency: 28 + Math.floor(rng() * 42), thumb: null,
        modelVersion: '1.0.0',
      })
    }
  }
  mkHist(db.users[1], 22, 0.2)
  mkHist(db.users[0], 13, 1.4)
  db.history.sort((a, b) => b.ts - a.ts)

  // audit logs
  const ACTIONS = [
    ['Signed in', 'info'], ['Prediction run', 'info'], ['Password changed', 'warn'],
    ['User suspended', 'bad'], ['User created', 'ok'], ['Model promoted', 'ok'],
    ['Failed sign-in attempt', 'bad'], ['API key regenerated', 'warn'], ['Settings updated', 'warn'],
    ['CSV export', 'info'], ['Account deleted', 'bad'], ['Role changed', 'warn'],
  ]
  for (let i = 0; i < 46; i++) {
    const u = db.users[Math.floor(rng() * db.users.length)]
    const a = ACTIONS[Math.floor(rng() * ACTIONS.length)]
    db.logs.push({
      id: uid('l'), ts: now - (0.1 + i * (rng() * 3 + 0.5)) * DAY, userId: u.id, userName: u.name,
      action: a[0], severity: a[1], ip: '10.0.' + Math.floor(rng() * 250) + '.' + Math.floor(rng() * 250),
      detail: a[0] === 'Prediction run' ? 'POST /predict' : (a[0] === 'Signed in' ? 'session created' : 'admin action'),
    })
  }
  db.logs.sort((a, b) => b.ts - a.ts)

  // notifications for the demo user
  const mkNotif = (userId, type, severity, title, message, agoH, link) => ({
    id: uid('n'), userId, type, severity, title, message, link: link || null, ts: now - agoH * 3600000, read: false,
  })
  db.notifs.push(
    mkNotif(db.users[1].id, 'analysis', 'success', 'Analysis complete', 'webcam-frame.jpg — likely REAL (92.4% confidence).', 2.1, '/user/history'),
    mkNotif(db.users[1].id, 'analysis', 'critical', 'High-confidence fake detected', 'profile-crop.jpg flagged FAKE at 96.1% — review the Grad-CAM overlay.', 5.4, '/user/history'),
    mkNotif(db.users[1].id, 'model', 'info', 'Model health check', 'AdaptiveFusion serving normally — ONNX Runtime, CPUExecutionProvider.', 26, '/admin/model'),
    mkNotif(db.users[1].id, 'account', 'info', 'Welcome to DeepGuard', 'Your analyst workspace is ready — run your first analysis.', 61 * 24, null),
  )
  db.notifs.push(
    mkNotif(db.users[0].id, 'system', 'warning', 'Threshold review due', 'The decision threshold has not been recalibrated in 30 days.', 30, '/admin/settings'),
    mkNotif(db.users[0].id, 'model', 'info', 'Model metadata refreshed', 'GET /model/metadata answered 200 — model card in sync.', 1.2, '/admin/model'),
  )

  db.model = {
    name: 'AdaptiveFusionDetector', backbone: 'efficientnetv2_rw_s', params: '-',
    threshold: 0.5, status: 'serving', prodVersion: '1.0.0',
    versions: [
      { v: '1.0.0', date: now - 21 * DAY, stage: 'production' },
    ],
  }
  return db
}

DB = loadDB() || seedDB()
saveDB()

/* ---------------- store API ---------------- */
export function getDB() { return DB }

export function resetDemoData() {
  DB = seedDB()
  saveDB()
  return DB
}

export function me() { return DB.session ? DB.users.find(u => u.id === DB.session.userId) || null : null }
export function isAdmin() { const u = me(); return !!u && u.role === 'admin' }

export function logEvent(action, severity, detail) {
  const u = me()
  DB.logs.unshift({
    id: uid('l'), ts: Date.now(), userId: u ? u.id : 'anon', userName: u ? u.name : 'Anonymous',
    action, severity: severity || 'info', ip: '10.0.0.' + (5 + (hashString(action) % 200)), detail: detail || '-',
  })
  DB.logs = DB.logs.slice(0, 300)
  saveDB()
}

export function pushNotif(userId, type, severity, title, message, link) {
  DB.notifs.unshift({ id: uid('n'), userId, type, severity, title, message, link: link || null, ts: Date.now(), read: false })
  DB.notifs = DB.notifs.slice(0, 200)
  saveDB()
}

export function myHistory() {
  const u = me()
  return u ? DB.history.filter(h => h.userId === u.id) : []
}

export function addHistory(entry) {
  DB.history.unshift({ ...entry, id: uid('h'), userId: me()?.id, ts: Date.now() })
  DB.history = DB.history.slice(0, 400)
  saveDB()
}

export function deleteHistory(id) {
  DB.history = DB.history.filter(h => h.id !== id)
  saveDB()
}

/** Verdict at the locally adjustable threshold (server default 0.50). */
export function verdictAt(pFake, threshold) {
  return pFake >= (threshold == null ? DB.settings.threshold : threshold) ? 'fake' : 'real'
}

/* ---------------- auth (localStorage accounts, as in the approved prototype) ---------------- */
export function findUserByEmail(email) {
  const e = String(email || '').trim().toLowerCase()
  return DB.users.find(u => u.email === e) || null
}

export function loginStore(email, password) {
  const user = findUserByEmail(email)
  if (!user || user.pass !== hashString(password + '|dg')) {
    logEvent('Failed sign-in attempt', 'bad', String(email || '').toLowerCase())
    throw new ApiError('Invalid email or password.', 401)
  }
  if (user.status === 'suspended') throw new ApiError('This account is suspended. Contact an administrator.', 403)
  if (user.status === 'invited') throw new ApiError('This account has not been activated yet. Check your invitation email.', 403)
  user.lastActive = Date.now()
  DB.session = { userId: user.id }
  saveDB()
  logEvent('Signed in', 'info', 'session created')
  return user
}

export function signupStore({ full_name, email, org, password }) {
  if (DB.settings.maintenance) throw new ApiError('Sign-ups are temporarily disabled (maintenance mode).', 503)
  if (!DB.settings.signupOpen) throw new ApiError('Sign-ups are closed. Contact an administrator.', 403)
  if (findUserByEmail(email)) throw new ApiError('An account with this email already exists.', 409)
  if (String(password).length < DB.settings.minPwd) {
    throw new ApiError(`Password must be at least ${DB.settings.minPwd} characters.`, 422)
  }
  const now = Date.now()
  const user = {
    id: uid('u'), name: full_name.trim(), email: email.trim().toLowerCase(), org: (org || '').trim(),
    pass: hashString(password + '|dg'), role: 'user', status: 'active',
    joined: now, lastActive: now, color: AV_COLORS[hashString(email) % AV_COLORS.length],
    apiKey: 'dg_live_' + hashString(email + now).toString(36).padStart(10, '0'),
  }
  DB.users.push(user)
  DB.session = { userId: user.id }
  saveDB()
  logEvent('User created', 'ok', user.email)
  pushNotif(user.id, 'account', 'info', 'Welcome to DeepGuard', 'Your analyst workspace is ready — run your first analysis.', null)
  return user
}

export function logoutStore() {
  DB.session = null
  saveDB()
}

export function changePassword(current, next) {
  const u = me()
  if (!u) throw new ApiError('Not signed in.', 401)
  if (u.pass !== hashString(current + '|dg')) throw new ApiError('Current password is incorrect.', 403)
  if (String(next).length < DB.settings.minPwd) throw new ApiError(`New password must be at least ${DB.settings.minPwd} characters.`, 422)
  u.pass = hashString(next + '|dg')
  saveDB()
  logEvent('Password changed', 'warn', u.email)
}

export function regenerateApiKey() {
  const u = me()
  if (!u) throw new ApiError('Not signed in.', 401)
  u.apiKey = 'dg_live_' + hashString(u.email + Date.now()).toString(36).padStart(10, '0')
  saveDB()
  logEvent('API key regenerated', 'warn', u.email)
  return u.apiKey
}

export function updateUser(patch) {
  const u = me()
  if (!u) throw new ApiError('Not signed in.', 401)
  Object.assign(u, patch)
  saveDB()
  return u
}

/* ---------------- settings ---------------- */
export function updateSettings(patch) {
  Object.assign(DB.settings, patch)
  saveDB()
  logEvent('Settings updated', 'warn', Object.keys(patch).join(', '))
  return DB.settings
}

/* ---------------- CSV export ---------------- */
export function downloadCsv(filename, headers, rows) {
  const esc = (v) => {
    const s = v == null ? '' : String(v)
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s
  }
  const csv = [headers.map(esc).join(','), ...rows.map(r => r.map(esc).join(','))].join('\n')
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
  logEvent('CSV export', 'info', filename)
}

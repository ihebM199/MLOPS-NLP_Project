/* Sign-in — the approved single-file build's auth design (p05 login):
 * two-column layout with the pitch panel and the glassy form card,
 * powered by the platform's real session store. */
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext.jsx'
import { useToast } from '../../context/ToastContext.jsx'
import { p05icon as icon, Brand } from './Landing.jsx'

export default function Login() {
  const { login } = useAuth()
  const pushToast = useToast()
  const nav = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [show, setShow] = useState(false)
  const [remember, setRemember] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [fieldErrors, setFieldErrors] = useState({})

  const setErr = (which, msg) =>
    setFieldErrors(fe => ({ ...fe, [which]: msg }))

  const submit = async (e) => {
    e?.preventDefault()
    setError(''); setFieldErrors({})
    let ok = true
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(email.trim())) { setErr('email', 'Enter a valid email address.'); ok = false }
    if (!password) { setErr('password', 'Password is required.'); ok = false }
    if (!ok) return
    setBusy(true)
    try {
      const user = await login(email.trim(), password)
      if (!remember) localStorage.removeItem('dg_remember')
      else localStorage.setItem('dg_remember', '1')
      pushToast(`Welcome back, ${user.name.split(' ')[0]}`,
        user.role === 'admin' ? 'Opening the admin console.' : 'Opening your workspace.', 'ok')
      nav(user.role === 'admin' ? '/admin/overview' : '/user/dashboard')
    } catch (err) {
      setError(err.message || 'Sign-in failed.')
    } finally { setBusy(false) }
  }

  const fillDemo = (role) => {
    setEmail(role === 'admin' ? 'admin@deepguard.io' : 'demo@deepguard.io')
    setPassword(role === 'admin' ? 'Admin@2026' : 'Demo@2026')
    setError(''); setFieldErrors({})
  }

  const fe = (k) => fieldErrors[k] && (
    <div className="field-msg">{icon('alert')}{fieldErrors[k]}</div>
  )
  const errCls = (k) => fieldErrors[k] ? ' input err' : ' input'

  return (
    <div className="dg5">
      <div className="auth">
        <div className="auth-side">
          <span className="glow" /><span className="glow g2" />
          <div className="auth-side-in">
            <span className="eyebrow" style={{ background: 'rgba(255,255,255,.08)', borderColor: 'rgba(255,255,255,.2)', color: '#9db8ff' }}>DeepGuard platform</span>
            <h2>Welcome back to the verification desk</h2>
            <p>Pick up where you left off — your analysis history, saved fusion weights and team settings are waiting.</p>
            <div className="auth-perks">
              <div>{icon('shieldCheck')}<span>Session-based sign-in with role-guarded routes</span></div>
              <div>{icon('scan')}<span>Dual-stream engine with Grad-CAM explanations</span></div>
              <div>{icon('users')}<span>Full admin console for team owners</span></div>
            </div>
            <div className="auth-quote">"Reproduce, explain, monitor — the three words this platform was designed around."</div>
          </div>
        </div>
        <div className="auth-main">
          <div className="auth-card">
            <Brand />
            <h1>Sign in</h1>
            <p className="sub">Enter your credentials to open your workspace.</p>

            {error && <div className="form-alert err">{icon('alert')}<span>{error}</span></div>}

            <div className="demo-fill">
              <button type="button" className="chip clickable" onClick={() => fillDemo('admin')}>{icon('key')}Admin demo</button>
              <button type="button" className="chip clickable" onClick={() => fillDemo('user')}>{icon('user')}User demo</button>
            </div>

            <form onSubmit={submit} noValidate>
              <div className="field">
                <label className="label" htmlFor="lg-email">Email</label>
                <div className="input-wrap">
                  {icon('mail', 'lead')}
                  <input id="lg-email" className={errCls('email')} type="email" autoComplete="email"
                    placeholder="you@company.com" value={email}
                    onChange={e => { setEmail(e.target.value); setErr('email', null) }} />
                </div>
                {fe('email')}
              </div>
              <div className="field">
                <label className="label" htmlFor="lg-pwd">Password</label>
                <div className="input-wrap">
                  {icon('lock', 'lead')}
                  <input id="lg-pwd" className={errCls('password')} type={show ? 'text' : 'password'}
                    autoComplete="current-password" placeholder="Your password" value={password}
                    onChange={e => { setPassword(e.target.value); setErr('password', null) }} />
                  <button type="button" className="eye-btn" aria-label="Show password" onClick={() => setShow(s => !s)}>
                    {icon(show ? 'eyeOff' : 'eye')}
                  </button>
                </div>
                {fe('password')}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
                <label className="checkbox">
                  <input type="checkbox" checked={remember} onChange={e => setRemember(e.target.checked)} />Remember me
                </label>
                <Link to="/forgot" style={{ fontSize: '.84rem', fontWeight: 600 }}>Forgot password?</Link>
              </div>
              <button className="btn btn-primary btn-block btn-lg" type="submit" disabled={busy}>
                {busy ? <><span className="loading-spin" />Signing in...</> : 'Sign in'}
              </button>
            </form>

            <p className="auth-alt">New to DeepGuard? <Link to="/signup" style={{ fontWeight: 600 }}>Create an account</Link></p>
            <p className="auth-alt" style={{ marginTop: 8 }}>
              <Link to="/analyze" style={{ fontSize: '.8rem' }}>Try the analyzer first — no account needed</Link>
            </p>
            <p className="auth-alt" style={{ marginTop: 8 }}>
              <Link to="/" style={{ fontSize: '.8rem' }}>Back to homepage</Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

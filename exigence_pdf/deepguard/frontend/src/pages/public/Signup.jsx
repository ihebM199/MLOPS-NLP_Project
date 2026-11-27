/* Sign-up — the approved single-file build's auth design (p05 signup):
 * pitch panel + form card with the password strength meter, powered by the
 * platform's real account store. */
import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext.jsx'
import { useToast } from '../../context/ToastContext.jsx'
import { p05icon as icon, Brand } from './Landing.jsx'

function passStrength(pw) {
  let s = 0
  if (pw.length >= 8) s++
  if (pw.length >= 12) s++
  if (/[A-Z]/.test(pw) && /[a-z]/.test(pw)) s++
  if (/\d/.test(pw)) s++
  if (/[^A-Za-z0-9]/.test(pw)) s++
  return Math.min(s, 5)
}
const meter = (s) => s <= 1 ? ['Very weak', '#f43f5e', '12'] : s === 2 ? ['Weak', '#f59e0b', '32']
  : s === 3 ? ['Fair', '#eab308', '55'] : s === 4 ? ['Strong', '#10b981', '80'] : ['Excellent', '#10b981', '100']

export default function Signup() {
  const { signup } = useAuth()
  const pushToast = useToast()
  const nav = useNavigate()
  const [form, setForm] = useState({ full_name: '', org: '', email: '', password: '', confirm: '' })
  const [show, setShow] = useState(false)
  const [terms, setTerms] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [fieldErrors, setFieldErrors] = useState({})

  const strength = useMemo(() => passStrength(form.password), [form.password])
  const m = meter(strength)

  const set = (k) => (e) => {
    setForm(f => ({ ...f, [k]: e.target.value }))
    setFieldErrors(fe => ({ ...fe, [k]: null }))
  }
  const setErr = (which, msg) => setFieldErrors(fe => ({ ...fe, [which]: msg }))
  const fe = (k) => fieldErrors[k] && <div className="field-msg">{icon('alert')}{fieldErrors[k]}</div>
  const errCls = (k) => 'input' + (fieldErrors[k] ? ' err' : '')

  const submit = async (e) => {
    e?.preventDefault()
    setError(''); setFieldErrors({})
    let ok = true
    if (form.full_name.trim().length < 2) { setErr('full_name', 'Please enter your full name.'); ok = false }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(form.email.trim())) { setErr('email', 'Enter a valid email address.'); ok = false }
    if (form.password.length < 8) { setErr('password', 'Password must be at least 8 characters.'); ok = false }
    if (form.confirm !== form.password || !form.confirm) { setErr('confirm', 'Passwords do not match.'); ok = false }
    if (!terms) { setErr('terms', 'Please accept the terms to continue.'); ok = false }
    if (!ok) return
    setBusy(true)
    try {
      const user = await signup({
        full_name: form.full_name.trim(), email: form.email.trim().toLowerCase(),
        org: form.org.trim(), password: form.password,
      })
      pushToast(`Welcome to DeepGuard, ${user.name.split(' ')[0]}`, 'Your workspace is ready.', 'ok')
      nav('/user/dashboard')
    } catch (err) {
      setError(err.message || 'Sign-up failed.')
    } finally { setBusy(false) }
  }

  return (
    <div className="dg5">
      <div className="auth">
        <div className="auth-side">
          <span className="glow" /><span className="glow g2" />
          <div className="auth-side-in">
            <span className="eyebrow" style={{ background: 'rgba(255,255,255,.08)', borderColor: 'rgba(255,255,255,.2)', color: '#9db8ff' }}>Join DeepGuard</span>
            <h2>Your verification workspace, in one minute</h2>
            <p>Free accounts include the full analysis workbench, saved history and CSV export. Admin features unlock with a team plan.</p>
            <div className="auth-perks">
              <div>{icon('scan')}<span>Unlimited dual-stream analyses</span></div>
              <div>{icon('target')}<span>Grad-CAM explainability on every scan</span></div>
              <div>{icon('download')}<span>History with CSV export</span></div>
            </div>
          </div>
        </div>
        <div className="auth-main">
          <div className="auth-card">
            <Brand />
            <h1>Create your account</h1>
            <p className="sub">Start verifying faces in under a minute.</p>

            {error && <div className="form-alert err">{icon('alert')}<span>{error}</span></div>}

            <form onSubmit={submit} noValidate>
              <div className="field">
                <label className="label" htmlFor="su-name">Full name</label>
                <div className="input-wrap">
                  {icon('user', 'lead')}
                  <input id="su-name" className={errCls('full_name')} autoComplete="name" placeholder="Jane Cooper"
                    value={form.full_name} onChange={set('full_name')} />
                </div>
                {fe('full_name')}
              </div>
              <div className="field">
                <label className="label" htmlFor="su-org">Organisation <span className="hint">optional</span></label>
                <div className="input-wrap">
                  {icon('users', 'lead')}
                  <input id="su-org" className="input" autoComplete="organization" placeholder="Company or lab"
                    value={form.org} onChange={set('org')} />
                </div>
              </div>
              <div className="field">
                <label className="label" htmlFor="su-email">Work email</label>
                <div className="input-wrap">
                  {icon('mail', 'lead')}
                  <input id="su-email" className={errCls('email')} type="email" autoComplete="email" placeholder="jane@company.com"
                    value={form.email} onChange={set('email')} />
                </div>
                {fe('email')}
              </div>
              <div className="field">
                <label className="label" htmlFor="su-pwd">Password <span className="hint">min 8 characters</span></label>
                <div className="input-wrap">
                  {icon('lock', 'lead')}
                  <input id="su-pwd" className={errCls('password')} type={show ? 'text' : 'password'} autoComplete="new-password"
                    placeholder="Create a strong password" value={form.password} onChange={set('password')} />
                  <button type="button" className="eye-btn" aria-label="Show password" onClick={() => setShow(s => !s)}>
                    {icon(show ? 'eyeOff' : 'eye')}
                  </button>
                </div>
                <div className="pass-meter"><i style={{ width: form.password ? m[2] + '%' : 0, background: m[1] }} /></div>
                <div className="pass-hint">{form.password ? `Strength: ${m[0]}` : 'Use 12+ chars with mixed case, a number and a symbol.'}</div>
                {fe('password')}
              </div>
              <div className="field">
                <label className="label" htmlFor="su-pwd2">Confirm password</label>
                <div className="input-wrap">
                  {icon('lock', 'lead')}
                  <input id="su-pwd2" className={errCls('confirm')} type="password" autoComplete="new-password"
                    placeholder="Repeat the password" value={form.confirm} onChange={set('confirm')} />
                </div>
                {fe('confirm')}
              </div>
              <label className="checkbox" style={{ marginBottom: 18 }}>
                <input type="checkbox" checked={terms} onChange={e => { setTerms(e.target.checked); setErr('terms', null) }} />
                I agree to the <Link to="/" onClick={e => e.preventDefault()}>Terms of Service</Link> and <Link to="/" onClick={e => e.preventDefault()}>Privacy Policy</Link>
              </label>
              {fieldErrors.terms && <div className="field-msg" style={{ marginTop: -10, marginBottom: 12 }}>{icon('alert')}{fieldErrors.terms}</div>}
              <button className="btn btn-primary btn-block btn-lg" type="submit" disabled={busy}>
                {busy ? <><span className="loading-spin" />Creating account...</> : 'Create account'}
              </button>
            </form>

            <p className="auth-alt">Already have an account? <Link to="/login" style={{ fontWeight: 600 }}>Sign in</Link></p>
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

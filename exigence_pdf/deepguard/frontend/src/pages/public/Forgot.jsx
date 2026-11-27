/* Password reset — the approved single-file build's auth design (p05 forgot).
 * No real email transport exists in this prototype, exactly as the original
 * card states; the flow is logged to the audit trail. */
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { logEvent } from '../../api/client.js'
import { p05icon as icon, Brand } from './Landing.jsx'

export default function Forgot() {
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [fieldError, setFieldError] = useState('')
  const [busy, setBusy] = useState(false)
  const [sent, setSent] = useState(false)

  const submit = (e) => {
    e?.preventDefault()
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(email.trim())) {
      setFieldError('Enter a valid email address.')
      return
    }
    setFieldError(''); setBusy(true)
    setTimeout(() => {
      setBusy(false); setSent(true)
      logEvent('Password reset requested', 'warn', email.trim().toLowerCase())
    }, 800)
  }

  return (
    <div className="dg5">
      <div className="auth">
        <div className="auth-side">
          <span className="glow" />
          <div className="auth-side-in">
            <span className="eyebrow" style={{ background: 'rgba(255,255,255,.08)', borderColor: 'rgba(255,255,255,.2)', color: '#9db8ff' }}>Account recovery</span>
            <h2>Locked out? Let&apos;s fix that</h2>
            <p>Enter the email tied to your account and we will send a single-use reset link. Links expire after 30 minutes and can only be used once.</p>
          </div>
        </div>
        <div className="auth-main">
          <div className="auth-card">
            <Brand />
            {!sent ? (
              <div id="forgot-box">
                <h1>Reset your password</h1>
                <p className="sub">We will email you a secure reset link.</p>
                {error && <div className="form-alert err">{icon('alert')}<span>{error}</span></div>}
                <form onSubmit={submit} noValidate>
                  <div className="field">
                    <label className="label" htmlFor="fg-email">Email</label>
                    <div className="input-wrap">
                      {icon('mail', 'lead')}
                      <input id="fg-email" className={'input' + (fieldError ? ' err' : '')} type="email"
                        placeholder="you@company.com" value={email}
                        onChange={(e) => { setEmail(e.target.value); setFieldError('') }} />
                    </div>
                    {fieldError && <div className="field-msg">{icon('alert')}{fieldError}</div>}
                  </div>
                  <button className="btn btn-primary btn-block btn-lg" type="submit" disabled={busy}>
                    {busy ? <><span className="loading-spin" />Sending...</> : 'Send reset link'}
                  </button>
                </form>
                <p className="auth-alt"><Link to="/login" style={{ fontWeight: 600 }}>Back to sign in</Link></p>
              </div>
            ) : (
              <div id="forgot-ok">
                <div style={{ textAlign: 'center', padding: '10px 0 4px' }}>
                  <span className="card-title-ic" style={{ display: 'inline-flex', width: 52, height: 52, margin: '0 auto 16px', borderRadius: 14, background: 'var(--ok-soft)', color: 'var(--ok)', alignItems: 'center', justifyContent: 'center' }}>
                    {icon('mail')}
                  </span>
                  <h1>Check your inbox</h1>
                  <p className="sub" style={{ margin: '10px 0 20px' }}>
                    If an account exists for <b className="mono">{email.trim()}</b>, a reset link is on its way. It expires in 30 minutes.
                  </p>
                  <div className="form-alert ok">{icon('info')}<span>Prototype mode — no real email is sent.</span></div>
                  <Link to="/login" className="btn btn-outline btn-block" style={{ marginTop: 8 }}>Back to sign in</Link>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

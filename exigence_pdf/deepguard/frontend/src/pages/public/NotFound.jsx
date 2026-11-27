/* 404 — the approved build's not-found page (p05): centered eyebrow,
 * gradient 404, two actions. Rendered inside the landing design system. */
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext.jsx'
import { p05icon as icon } from './Landing.jsx'

export default function NotFound() {
  const { user } = useAuth()
  const nav = useNavigate()
  return (
    <div className="dg5" style={{ maxWidth: 640, margin: '0 auto', padding: '90px 24px', textAlign: 'center' }}>
      <span className="eyebrow">Error 404</span>
      <h1 className="grad-text mono" style={{ fontSize: 'clamp(3rem,9vw,5rem)', margin: '18px 0' }}>404</h1>
      <p style={{ color: 'var(--muted)', marginBottom: 28 }}>
        The page you are looking for was moved, removed, or perhaps never existed.
      </p>
      <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
        <button className="btn btn-primary" onClick={() => nav('/')}>{icon('home')}Back to homepage</button>
        <button className="btn btn-outline" onClick={() => nav(user ? (user.role === 'admin' ? '/admin/overview' : '/user/dashboard') : '/analyze')}>
          {user ? 'Go to dashboard' : 'Open the analyzer'}
        </button>
      </div>
      <p style={{ marginTop: 24 }}>
        <Link to="/analyze" style={{ fontSize: '.8rem' }}>Deepfake Adaptive Fusion analyzer</Link>
      </p>
    </div>
  )
}

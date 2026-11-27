import { Navigate, Route, Routes, useLocation, useNavigate, NavLink } from 'react-router-dom'
import { useEffect, useRef, useState } from 'react'
import { useAuth } from './context/AuthContext.jsx'
import { useNotifications } from './context/NotificationsContext.jsx'
import { useTheme } from './context/ThemeContext.jsx'
import { useToast } from './context/ToastContext.jsx'
import { Icon, P, Badge, Spinner } from './components/ui.jsx'
import Landing from './pages/public/Landing.jsx'
import Login from './pages/public/Login.jsx'
import Signup from './pages/public/Signup.jsx'
import Forgot from './pages/public/Forgot.jsx'
import NotFound from './pages/public/NotFound.jsx'
import UserDashboard from './pages/user/Dashboard.jsx'
import Analyze from './pages/user/Analyze.jsx'
import History from './pages/user/History.jsx'
import Profile from './pages/user/Profile.jsx'
import UserNotifications from './pages/user/Notifications.jsx'
import AdminOverview from './pages/admin/Overview.jsx'
import AdminUsers from './pages/admin/Users.jsx'
import AdminModel from './pages/admin/Model.jsx'
import AdminAudit from './pages/admin/Audit.jsx'
import AdminSettings from './pages/admin/Settings.jsx'

/* ---------- health pill (real GET /health, polled every 12 s) ----------- */
function HealthPill() {
  const [ok, setOk] = useState(null)
  useEffect(() => {
    let alive = true
    const check = () => fetch('/health')
      .then(r => alive && setOk(r.ok))
      .catch(() => alive && setOk(false))
    check()
    const t = setInterval(check, 12000)
    return () => { alive = false; clearInterval(t) }
  }, [])
  return (
    <span className="chip mute hidden md:inline-flex" title={ok ? 'API online' : 'API unreachable'}>
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? 'bg-ok animate-pulse-soft' : 'bg-bad'}`} />
      <span className="num text-[10px]">{ok == null ? '…' : ok ? 'API ONLINE' : 'API OFFLINE'}</span>
    </span>
  )
}

/* ---------- notification bell + dropdown -------------------------------- */
function BellMenu() {
  const { unread, latest, markRead, markAllRead } = useNotifications()
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  const nav = useNavigate()
  useEffect(() => {
    const h = (e) => { if (!ref.current?.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [])
  const sev = { critical: 'bad', warning: 'warn', success: 'ok', info: 'brand' }
  return (
    <div className="relative" ref={ref}>
      <button onClick={() => setOpen(o => !o)} aria-label="Notifications"
        className="relative h-9 w-9 rounded-xl border border-line hover:border-brand/50 text-mute hover:text-ink transition grid place-items-center">
        <Icon d={P.bell} />
        {unread > 0 && (
          <span className="absolute -top-1.5 -right-1.5 min-w-[1.15rem] h-[1.15rem] px-1 rounded-full bg-bad
                           text-white text-[10px] font-bold num grid place-items-center border-2 border-bg">
            {unread > 99 ? '99+' : unread}
          </span>
        )}
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-[22rem] max-w-[calc(100vw-2rem)] card shadow-pop overflow-hidden animate-rise z-50">
          <div className="flex items-center justify-between px-4 py-3 border-b border-line">
            <p className="text-sm font-semibold">Notifications</p>
            {unread > 0 && (
              <button className="text-xs text-brand hover:underline" onClick={markAllRead}>Mark all read</button>
            )}
          </div>
          <div className="max-h-80 overflow-y-auto">
            {latest.length === 0 && (
              <p className="px-4 py-8 text-center text-sm text-mute">You're all caught up.</p>
            )}
            {latest.map(n => (
              <button key={n.id} onClick={() => { markRead(n.id); setOpen(false); if (n.link) nav(n.link) }}
                className="w-full text-left px-4 py-3 border-b border-line/60 hover:bg-panel2 transition flex gap-3">
                <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${n.severity === 'critical' ? 'bg-bad' : n.severity === 'warning' ? 'bg-warn' : n.severity === 'success' ? 'bg-ok' : 'bg-brand'}`} />
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-2">
                    <span className="text-sm font-medium truncate">{n.title}</span>
                    {!n.read && <Badge tone="brand" dot>new</Badge>}
                  </span>
                  <span className="block text-xs text-mute line-clamp-2 mt-0.5">{n.message}</span>
                  <span className="block text-[10px] text-faint num mt-1">{new Date(n.ts).toLocaleString()}</span>
                </span>
              </button>
            ))}
          </div>
          <button onClick={() => { setOpen(false); nav('/notifications') }}
            className="block w-full px-4 py-3 text-center text-xs font-semibold text-brand hover:bg-panel2 transition border-t border-line">
            View all notifications
          </button>
        </div>
      )}
    </div>
  )
}

/* ---------- profile menu ------------------------------------------------ */
function ProfileMenu() {
  const { user, logout } = useAuth()
  const { setTheme, theme } = useTheme()
  const nav = useNavigate()
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  useEffect(() => {
    const h = (e) => { if (!ref.current?.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [])
  const initial = (user?.name || user?.email || '?').trim()[0].toUpperCase()
  return (
    <div className="relative" ref={ref}>
      <button onClick={() => setOpen(o => !o)} className="flex items-center gap-2.5 group">
        <span className="h-9 w-9 rounded-xl bg-brand/15 text-brand border border-brand/25 grid place-items-center
                         font-bold text-sm">{initial}</span>
        <span className="hidden lg:block text-left leading-tight">
          <span className="block text-sm font-semibold">{user?.name || 'Analyst'}</span>
          <span className="block text-[11px] text-mute">{user?.role === 'admin' ? 'Administrator' : 'Analyst'}</span>
        </span>
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-60 card shadow-pop overflow-hidden animate-rise z-50">
          <div className="px-4 py-3 border-b border-line">
            <p className="text-sm font-semibold truncate">{user?.email}</p>
            <p className="text-[11px] text-mute mt-0.5 num">Member since {new Date(user?.joined || Date.now()).toLocaleDateString()}</p>
          </div>
          <div className="p-1.5">
            <button onClick={() => { setOpen(false); nav(user?.role === 'admin' ? '/settings' : '/profile') }}
              className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm hover:bg-panel2 transition">
              <Icon d={P.settings} /> Account settings
            </button>
            <button onClick={() => { setTheme(theme === 'dark' ? 'light' : 'dark'); setOpen(false) }}
              className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm hover:bg-panel2 transition">
              <Icon d={theme === 'dark' ? P.sun : P.moon} /> {theme === 'dark' ? 'Light' : 'Dark'} theme
            </button>
            <button onClick={logout}
              className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-bad hover:bg-bad/10 transition">
              <Icon d={P.logout} /> Sign out
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

/* ---------- role-aware app layout (sidebar + topbar + outlet) ---------- */
function AppLayout({ role }) {
  const { user } = useAuth()
  const [navOpen, setNavOpen] = useState(false)
  const loc = useLocation()
  useEffect(() => setNavOpen(false), [loc.pathname])

  const nav = user?.role === 'admin' ? [
    { to: '/overview', label: 'Overview', icon: P.chart },
    { to: '/users', label: 'Users', icon: P.users },
    { to: '/model', label: 'Model monitoring', icon: P.cpu },
    { to: '/audit', label: 'Audit logs', icon: P.file },
    { to: '/notifications', label: 'Notifications', icon: P.bell },
    { to: '/settings', label: 'Settings', icon: P.settings },
  ] : [
    { to: '/dashboard', label: 'Dashboard', icon: P.home },
    { to: '/analyze', label: 'New analysis', icon: P.scan },
    { to: '/history', label: 'History', icon: P.history },
    { to: '/notifications', label: 'Notifications', icon: P.bell },
    { to: '/profile', label: 'Profile', icon: P.user },
  ]

  const title = nav.find(n => loc.pathname.startsWith(n.to))?.label || 'Dashboard'

  return (
    <div className="min-h-screen bg-bg flex">
      {/* Sidebar */}
      <aside className={`fixed lg:sticky top-0 left-0 z-40 h-screen w-64 shrink-0 bg-panel border-r border-line
        flex flex-col transition-transform duration-300 ${navOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}`}>
        <div className="h-16 flex items-center gap-2.5 px-5 border-b border-line shrink-0">
          <span className="h-8 w-8 rounded-xl bg-brand grid place-items-center text-white"><Icon d={P.shield} className="h-5 w-5" /></span>
          <div>
            <p className="font-bold leading-none tracking-tight">DeepGuard</p>
            <p className="text-[10px] text-mute mt-0.5 num">{user?.role === 'admin' ? 'ADMIN CONSOLE' : 'ANALYST WORKSPACE'}</p>
          </div>
        </div>
        <nav className="flex-1 overflow-y-auto p-3 space-y-1">
          {nav.map(n => (
            <NavLink key={n.to} to={n.to}
              className={({ isActive }) => `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition
                ${isActive ? 'bg-brand/12 text-brand border border-brand/25' : 'text-mute hover:text-ink hover:bg-panel2 border border-transparent'}`}>
              <Icon d={n.icon} /> {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="p-4 border-t border-line">
          <div className="card !bg-panel2 p-3.5">
            <div className="flex items-center gap-2 text-[11px] text-mute">
              <Icon d={P.cpu} className="h-3.5 w-3.5 text-brand" />
              <span className="num">AdaptiveFusion · ONNX Runtime</span>
            </div>
            <p className="text-[11px] text-faint mt-1.5 leading-relaxed">
              {user?.role === 'admin'
                ? 'The model card, health and metrics on these pages come live from the serving API.'
                : 'Every analysis runs the real dual-stream fusion model on the Django backend.'}
            </p>
          </div>
        </div>
      </aside>

      {navOpen && <div className="fixed inset-0 z-30 bg-black/50 lg:hidden" onClick={() => setNavOpen(false)} />}

      {/* Main column */}
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="sticky top-0 z-20 h-16 glass border-b border-line flex items-center gap-3 px-4 sm:px-6">
          <button className="lg:hidden h-9 w-9 rounded-xl border border-line grid place-items-center text-mute"
            onClick={() => setNavOpen(o => !o)} aria-label="Menu">
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M4 7h16M4 12h16M4 17h16" />
            </svg>
          </button>
          <h1 className="font-semibold text-lg truncate flex-1">{title}</h1>
          <HealthPill />
          <BellMenu />
          <ProfileMenu />
        </header>
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-[1400px] w-full mx-auto animate-fade-in" key={loc.pathname}>
          <Routes>
            {role === 'admin' ? (
              <>
                <Route path="overview" element={<AdminOverview />} />
                <Route path="users" element={<AdminUsers />} />
                <Route path="model" element={<AdminModel />} />
                <Route path="audit" element={<AdminAudit />} />
                <Route path="notifications" element={<UserNotifications scope="admin" />} />
                <Route path="settings" element={<AdminSettings />} />
                <Route path="*" element={<Navigate to="/overview" replace />} />
              </>
            ) : (
              <>
                <Route path="dashboard" element={<UserDashboard />} />
                <Route path="history" element={<History />} />
                <Route path="notifications" element={<UserNotifications scope="user" />} />
                <Route path="profile" element={<Profile />} />
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
              </>
            )}
          </Routes>
        </main>
      </div>
    </div>
  )
}

/* ---------- guards ------------------------------------------------------- */
function RequireRole({ role }) {
  const { user, booting } = useAuth()
  if (booting) {
    return <div className="min-h-screen bg-bg grid place-items-center">
      <div className="flex flex-col items-center gap-3 text-mute">
        <Spinner className="h-7 w-7 text-brand" />
        <p className="text-sm num">Restoring session…</p>
      </div>
    </div>
  }
  if (!user) return <Navigate to="/login" replace />
  if (user.role !== role) {
    return (
      <div className="min-h-screen bg-bg grid place-items-center p-6">
        <div className="card max-w-md p-8 text-center">
          <div className="h-12 w-12 mx-auto rounded-2xl bg-bad/10 text-bad grid place-items-center mb-4">
            <Icon d={P.shield} className="h-6 w-6" />
          </div>
          <h2 className="font-bold text-lg">403 · Role mismatch</h2>
          <p className="text-sm text-mute mt-2 leading-relaxed">
            {user.role === 'admin'
              ? 'Administrators manage the platform but do not run analyses — the analyst workspace is disabled for admin accounts by design.'
              : 'This console is restricted to platform administrators.'}
          </p>
          <a href={user.role === 'admin' ? '/overview' : '/dashboard'}
            className="btn-primary mt-6 inline-flex">Go to my workspace</a>
        </div>
      </div>
    )
  }
  return <AppLayout role={role} />
}

function PublicOnly({ children }) {
  const { user, booting } = useAuth()
  if (booting) return null
  if (user) return <Navigate to={user.role === 'admin' ? '/overview' : '/dashboard'} replace />
  return children
}

/* ---------- root --------------------------------------------------------- */
export default function App() {
  const loc = useLocation()
  useEffect(() => { window.scrollTo(0, 0) }, [loc.pathname])
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<PublicOnly><Login /></PublicOnly>} />
      <Route path="/signup" element={<PublicOnly><Signup /></PublicOnly>} />
      <Route path="/forgot" element={<PublicOnly><Forgot /></PublicOnly>} />
      <Route path="/user/*" element={<RequireRole role="user" />} />
      <Route path="/admin/*" element={<RequireRole role="admin" />} />
      {/* Short routes (in-app links) resolve to the matching workspace. */}
      <Route path="/dashboard" element={<Navigate to="/user/dashboard" replace />} />
      <Route path="/analyze" element={<Analyze />} />
      <Route path="/history" element={<Navigate to="/user/history" replace />} />
      <Route path="/notifications" element={<Navigate to="/user/notifications" replace />} />
      <Route path="/profile" element={<Navigate to="/user/profile" replace />} />
      <Route path="/overview" element={<Navigate to="/admin/overview" replace />} />
      <Route path="/users" element={<Navigate to="/admin/users" replace />} />
      <Route path="/model" element={<Navigate to="/admin/model" replace />} />
      <Route path="/audit" element={<Navigate to="/admin/audit" replace />} />
      <Route path="/settings" element={<Navigate to="/admin/settings" replace />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}

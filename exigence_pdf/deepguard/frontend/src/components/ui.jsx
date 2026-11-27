import { useEffect } from 'react'

/** Shared UI primitives matching the approved DeepGuard design. */

export function Icon({ d, className = 'h-4 w-4', fill = 'none' }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill={fill} stroke="currentColor"
      strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  )
}

export const P = {
  shield: 'M12 3l7 3v5c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z',
  shieldCheck: 'M12 3l7 3v5c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3zM9 12l2 2 4-4',
  home: 'M4 11l8-7 8 7v8a2 2 0 01-2 2h-4v-6h-4v6H6a2 2 0 01-2-2v-8z',
  scan: 'M4 8V6a2 2 0 012-2h2M16 4h2a2 2 0 012 2v2M20 16v2a2 2 0 01-2 2h-2M8 20H6a2 2 0 01-2-2v-2M4 12h16',
  history: 'M12 8v4l3 2M3 12a9 9 0 109-9 9 9 0 00-7.6 4.2M3 4v4h4',
  user: 'M12 12a4 4 0 100-8 4 4 0 000 8zM5 20a7 7 0 0114 0',
  users: 'M9 11a3.5 3.5 0 100-7 3.5 3.5 0 000 7zM2.5 20a6.5 6.5 0 0113 0M16.5 4.2a3.5 3.5 0 010 6.6M21.5 20a6.5 6.5 0 00-4.5-6.2',
  bell: 'M6 8a6 6 0 1112 0c0 6 2 7 2 7H4s2-1 2-7zM10 20a2 2 0 004 0',
  chart: 'M4 20V10M10 20V4M16 20v-7M22 20H2',
  cpu: 'M8 8h8v8H8zM4 9V7a3 3 0 013-3h2M20 9V7a3 3 0 00-3-3h-2M4 15v2a3 3 0 003 3h2M20 15v2a3 3 0 01-3 3h-2',
  settings: 'M12 15a3 3 0 100-6 3 3 0 000 6zM19.4 15a1.7 1.7 0 00.3 1.9l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-1.9-.3 1.7 1.7 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.7 1.7 0 00-1-1.6 1.7 1.7 0 00-1.9.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.7 1.7 0 00.3-1.9 1.7 1.7 0 00-1.5-1H3a2 2 0 110-4h.1a1.7 1.7 0 001.6-1 1.7 1.7 0 00-.3-1.9l-.1-.1a2 2 0 112.8-2.8l.1.1a1.7 1.7 0 001.9.3H10a1.7 1.7 0 001-1.5V3a2 2 0 114 0v.1a1.7 1.7 0 001 1.6 1.7 1.7 0 001.9-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 00-.3 1.9V10a1.7 1.7 0 001.5 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z',
  logout: 'M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4M16 17l5-5-5-5M21 12H9',
  sun: 'M12 17a5 5 0 100-10 5 5 0 000 10zM12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4',
  moon: 'M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z',
  check: 'm5 13 4 4L19 7',
  checkCircle: 'M12 21a9 9 0 100-18 9 9 0 000 18zm-3.5-9 2.5 2.5 5-5',
  x: 'M6 6l12 12M18 6L6 18',
  plus: 'M12 5v14M5 12h14',
  trash: 'M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6M10 11v6M14 11v6',
  search: 'M11 19a8 8 0 100-16 8 8 0 000 16zM21 21l-4.3-4.3',
  download: 'M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3',
  upload: 'M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12',
  zap: 'M13 2 3 14h9l-1 8 10-12h-9l1-8z',
  eye: 'M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8zM12 15a3 3 0 100-6 3 3 0 000 6z',
  eyeOff: 'm10.7 5.1A9.8 9.8 0 0112 5c6.5 0 10 7 10 7a17.4 17.4 0 01-3 3.9M6.6 6.6A17 17 0 002 12s3.5 7 10 7a9.9 9.9 0 004.4-1M9.9 9.9a3 3 0 004.2 4.2M2 2l20 20',
  key: 'M21 2l-2 2m-7.6 7.6a5 5 0 11-7.1 7.1 5 5 0 017.1-7.1zm0 0L15.5 8m0 0 3 3L22 7.5l-3-3M11.5 11.5 15.5 8',
  file: 'M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6zM14 2v6h6M9 15h6M9 11h3',
  logo: 'M12 3l7 3v5c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z',
  camera: 'M14.5 4h-5L7 7H4a2 2 0 00-2 2v9a2 2 0 002 2h16a2 2 0 002-2V9a2 2 0 00-2-2h-3l-2.5-3zM12 16a3 3 0 100-6 3 3 0 000 6z',
  image: 'M3 5a2 2 0 012-2h14a2 2 0 012 2v14a2 2 0 01-2 2H5a2 2 0 01-2-2V5zm2 12 4.7-4.7a1 1 0 011.4 0l4.6 4.6M15 9a1 1 0 100-2 1 1 0 000 2z',
  activity: 'M22 12h-4l-3 9L9 3l-3 9H2',
  server: 'M2 3h20v8H2zM2 13h20v8H2zM6 7h.01M6 17h.01',
  layers: 'm12 2 10 5-10 5L2 7l10-5zM2 12l10 5 10-5M2 17l10 5 10-5',
  wifi: 'M5 12.5a11 11 0 0114 0M8.5 15.5a6.5 6.5 0 017 0M2 9.5a15.5 15.5 0 0120 0M12 19h.01',
  target: 'M12 21a9 9 0 100-18 9 9 0 000 18zm0-4a5 5 0 100-10 5 5 0 000 10zm0-3.5a1.5 1.5 0 100-3 1.5 1.5 0 000 3z',
  refresh: 'M21 12a9 9 0 11-2.6-6.4L21 8M21 3v5h-5',
  clock: 'M12 21a9 9 0 100-18 9 9 0 000 18zm0-13v5l4 2',
  alert: 'm21.7 18.3-8-14a2 2 0 00-3.4 0l-8 14A2 2 0 004 21h16a2 2 0 001.7-2.7zM12 9v4m0 4h.01',
  info: 'M12 21a9 9 0 100-18 9 9 0 000 18zm0-13h.01M12 12v4',
  brain: 'M9.5 2A2.5 2.5 0 0112 4.5v15a2.5 2.5 0 01-4.96.44A2.5 2.5 0 014.5 17a2.5 2.5 0 01-1.98-4A2.5 2.5 0 014 8.6 2.5 2.5 0 015.5 4 2.5 2.5 0 019.5 2zM14.5 2A2.5 2.5 0 0012 4.5v15a2.5 2.5 0 004.96.44A2.5 2.5 0 0019.5 17a2.5 2.5 0 001.98-4A2.5 2.5 0 0020 8.6 2.5 2.5 0 0018.5 4 2.5 2.5 0 0014.5 2z',
  waves: 'M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5c2.6 0 2.4 2 5 2 1.3 0 1.9-.5 2.5-1M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 1.3 0 1.9-.5 2.5-1M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 1.3 0 1.9-.5 2.5-1',
  fingerprint: 'M12 2a10 10 0 00-10 10c0 2 .5 3.9 1.4 5.6M12 2a10 10 0 0110 10c0 1.6-.3 3.1-1 4.5M8 12a4 4 0 018 0c0 2.5-.5 5-1.6 7M6.5 15.5A4 4 0 018 12m-2.4 6.5A11.5 11.5 0 0112 12m.5 0c0 2.7-.7 5.3-2 7.5',
  arrowRight: 'M5 12h14m-6-7 7 7-7 7',
  copy: 'M9 9h11a2 2 0 012 2v9a2 2 0 01-2 2H9a2 2 0 01-2-2v-9a2 2 0 012-2zM5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1',
  edit: 'M17 3a2.8 2.8 0 114 4L7.5 20.5 2 22l1.5-5.5L17 3z',
  mail: 'M2 6a2 2 0 012-2h16a2 2 0 012 2v12a2 2 0 01-2 2H4a2 2 0 01-2-2V6zm20 1-10 6L2 7',
  filter: 'M22 3H2l8 9.5V19l4 2v-8.5L22 3z',
}

export function Card({ className = '', children, ...rest }) {
  return <div className={`card ${className}`} {...rest}>{children}</div>
}

export function Spinner({ className = 'h-5 w-5' }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity=".2" strokeWidth="3" />
      <path d="M22 12a10 10 0 00-10-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}

export function Badge({ tone = 'brand', children, dot = false }) {
  const tones = {
    brand: 'text-brand bg-brand/10 border-brand/25',
    ok: 'text-ok bg-ok/10 border-ok/25',
    bad: 'text-bad bg-bad/10 border-bad/25',
    warn: 'text-warn bg-warn/10 border-warn/25',
    violet: 'text-violet bg-violet/10 border-violet/25',
    mute: 'text-mute bg-panel2 border-line',
  }
  return (
    <span className={`chip ${tones[tone] || tones.brand}`}>
      {dot && <span className="h-1.5 w-1.5 rounded-full bg-current" />}
      {children}
    </span>
  )
}

export function VerdictBadge({ verdict }) {
  if (verdict === 'FAKE' || verdict === 'fake') return <Badge tone="bad" dot>MANIPULATED</Badge>
  if (verdict === 'REAL' || verdict === 'real') return <Badge tone="ok" dot>AUTHENTIC</Badge>
  return <Badge tone="warn" dot>UNCLEAR</Badge>
}

export function Modal({ open, onClose, title, children, wide = false, footer }) {
  useEffect(() => {
    if (!open) return
    const h = (e) => e.key === 'Escape' && onClose?.()
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [open, onClose])
  if (!open) return null
  return (
    <div className="fixed inset-0 z-[80] flex items-end sm:items-center justify-center p-0 sm:p-6"
      role="dialog" aria-modal="true">
      <div className="absolute inset-0 bg-black/60 animate-fade-in" onClick={onClose} />
      <div className={`relative card w-full ${wide ? 'sm:max-w-3xl' : 'sm:max-w-lg'} rounded-b-none sm:rounded-2xl
                      max-h-[92vh] overflow-y-auto animate-rise shadow-pop`}>
        <div className="flex items-center justify-between gap-4 px-5 py-4 border-b border-line sticky top-0 bg-panel z-10">
          <h3 className="font-semibold">{title}</h3>
          <button onClick={onClose} className="btn-ghost !p-2 !rounded-lg" aria-label="Close">
            <Icon d={P.x} className="h-4 w-4" />
          </button>
        </div>
        <div className="p-5">{children}</div>
        {footer && <div className="px-5 py-4 border-t border-line flex justify-end gap-2 sticky bottom-0 bg-panel">{footer}</div>}
      </div>
    </div>
  )
}

export function ConfirmDialog({ open, onClose, onConfirm, title, message, danger = true, busy = false, confirmLabel = 'Confirm' }) {
  return (
    <Modal open={open} onClose={onClose} title={title}
      footer={<>
        <button className="btn-ghost" onClick={onClose} disabled={busy}>Cancel</button>
        <button className={danger ? 'btn-danger' : 'btn-primary'} onClick={onConfirm} disabled={busy}>
          {busy ? <Spinner className="h-4 w-4" /> : confirmLabel}
        </button>
      </>}>
      <p className="text-sm text-mute leading-relaxed">{message}</p>
    </Modal>
  )
}

export function EmptyState({ icon = P.search, title, message, action }) {
  return (
    <div className="flex flex-col items-center justify-center py-14 px-6 text-center">
      <div className="h-12 w-12 rounded-2xl bg-brand/10 text-brand flex items-center justify-center mb-4">
        <Icon d={icon} className="h-6 w-6" />
      </div>
      <p className="font-semibold">{title}</p>
      {message && <p className="mt-1 text-sm text-mute max-w-sm">{message}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}

export function Skeleton({ className = 'h-4' }) {
  return <div className={`animate-pulse-soft rounded-lg bg-line/60 ${className}`} />
}

export function Pagination({ page, pages, onPage, total, label = 'items' }) {
  if (pages <= 1) return null
  const items = []
  const push = (p, t, dis) => items.push(
    <button key={p + t} disabled={dis} onClick={() => !dis && onPage(p)}
      className={`min-w-[2.1rem] h-8 px-2 rounded-lg text-xs font-semibold num transition
        ${p === page ? 'bg-brand text-white' : 'text-mute hover:text-ink border border-line'} disabled:opacity-40`}>
      {t}
    </button>)
  push(1, '«', page === 1)
  push(page - 1, '‹', page === 1)
  for (let i = Math.max(1, page - 2); i <= Math.min(pages, page + 2); i++) push(i, String(i), i === page)
  push(page + 1, '›', page === pages)
  push(pages, '»', page === pages)
  return (
    <div className="flex items-center justify-between gap-3 flex-wrap px-4 py-3 border-t border-line/70">
      <span className="text-xs text-mute num">{total} {label}</span>
      <div className="flex gap-1.5">{items}</div>
    </div>
  )
}

export function Toast() { return null } // toasts live in ToastProvider

export function Input({ label, error, hint, className = '', ...rest }) {
  return (
    <label className="block">
      {label && <span className="block mb-1.5 text-xs font-semibold text-mute">{label}</span>}
      <input className={`field ${error ? '!border-bad/60' : ''} ${className}`} {...rest} />
      {error ? <span className="block mt-1.5 text-xs text-bad">{error}</span>
        : hint ? <span className="block mt-1.5 text-xs text-faint">{hint}</span> : null}
    </label>
  )
}

export function Select({ label, children, className = '', ...rest }) {
  return (
    <label className="block">
      {label && <span className="block mb-1.5 text-xs font-semibold text-mute">{label}</span>}
      <select className={`field ${className} cursor-pointer`} {...rest}>{children}</select>
    </label>
  )
}

export function Toggle({ checked, onChange, label, description, disabled }) {
  return (
    <div className="flex items-start justify-between gap-4 py-3.5">
      <div className="min-w-0">
        <p className="text-sm font-medium">{label}</p>
        {description && <p className="text-xs text-mute mt-0.5 leading-relaxed">{description}</p>}
      </div>
      <button role="switch" aria-checked={checked} disabled={disabled}
        onClick={() => onChange?.(!checked)}
        className={`relative h-6 w-11 shrink-0 rounded-full transition cursor-pointer disabled:opacity-50
          ${checked ? 'bg-brand' : 'bg-line'}`}>
        <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all
          ${checked ? 'left-[1.4rem]' : 'left-0.5'}`} />
      </button>
    </div>
  )
}

import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getDB, saveDB, timeAgo } from '../../api/client.js'
import { useAuth } from '../../context/AuthContext.jsx'
import { useNotifications } from '../../context/NotificationsContext.jsx'
import { useToast } from '../../context/ToastContext.jsx'
import { Icon, P, Card, Badge, EmptyState, Pagination } from '../../components/ui.jsx'

const TYPE_META = {
  analysis: { icon: P.scan, tone: 'brand', label: 'Analysis' },
  model: { icon: P.cpu, tone: 'violet', label: 'Model' },
  user: { icon: P.users, tone: 'warn', label: 'User' },
  account: { icon: P.user, tone: 'ok', label: 'Account' },
  system: { icon: P.settings, tone: 'mute', label: 'System' },
}

export default function UserNotifications({ scope = 'user' }) {
  const { user } = useAuth()
  const { markRead, markAllRead, clearRead, unread } = useNotifications()
  const pushToast = useToast()
  const nav = useNavigate()
  const [page, setPage] = useState(1)
  const [filter, setFilter] = useState('')
  const [busy, setBusy] = useState(false)
  const [version, setVersion] = useState(0)

  const rows = getDB().notifs.filter(n => n.userId === user?.id)
  const filtered = rows.filter(n => filter === '' || (filter === 'unread' ? !n.read : n.read))

  useEffect(() => { setPage(1) }, [filter])

  const per = 12
  const pages = Math.max(1, Math.ceil(filtered.length / per))
  const safePage = Math.min(page, pages)
  const view = filtered.slice((safePage - 1) * per, safePage * per)

  const open = (n) => {
    if (!n.read) {
      markRead(n.id)
      setVersion(v => v + 1)
    }
    if (n.link) nav(n.link)
  }

  const doMarkAll = async () => {
    setBusy(true)
    try { markAllRead(); setVersion(v => v + 1); pushToast('All caught up', 'Every notification was marked as read.', 'ok') }
    finally { setBusy(false) }
  }

  const doClear = async () => {
    setBusy(true)
    try { clearRead(); setVersion(v => v + 1); pushToast('Cleaned up', 'Read notifications were removed.', 'brand') }
    finally { setBusy(false) }
  }

  return (
    <div className="space-y-5 max-w-3xl">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold tracking-tight">Notifications</h2>
          <p className="text-sm text-mute mt-1">
            {unread > 0 ? <><span className="num text-brand font-bold">{unread}</span> unread — events appear here as they happen.</>
              : 'All caught up — new events appear here live.'}
          </p>
        </div>
        <div className="flex gap-2.5">
          <button onClick={doMarkAll} disabled={busy || unread === 0} className="btn-ghost">Mark all read</button>
          <button onClick={doClear} disabled={busy} className="btn-soft">Clear read</button>
        </div>
      </div>

      <div className="flex gap-1.5">
        {['', 'unread', 'read'].map(f => (
          <button key={f || 'all'} onClick={() => setFilter(f)}
            className={`px-3.5 py-2 rounded-xl text-xs font-semibold border transition
              ${filter === f ? 'border-brand/60 bg-brand/12 text-brand' : 'border-line text-mute hover:text-ink'}`}>
            {f === '' ? 'All' : f === 'unread' ? 'Unread' : 'Read'}
          </button>
        ))}
      </div>

      <Card>
        {view.length === 0 ? (
          <EmptyState icon={P.bell} title="Nothing here yet"
            message="Analysis results, model health events and account updates will show up in this feed." />
        ) : (
          <div className="divide-y divide-line/60">
            {view.map(n => {
              const meta = TYPE_META[n.type] || TYPE_META.system
              const sev = { critical: 'bad', warning: 'warn', success: 'ok', info: 'brand' }[n.severity] || 'brand'
              return (
                <button key={n.id + version} onClick={() => open(n)}
                  className={`w-full text-left px-5 py-4 hover:bg-panel2 transition flex gap-4 ${!n.read ? 'bg-brand/[.04]' : ''}`}>
                  <span className={`h-10 w-10 rounded-xl grid place-items-center shrink-0
                    ${sev === 'bad' ? 'bg-bad/10 text-bad' : sev === 'warn' ? 'bg-warn/10 text-warn'
                      : sev === 'ok' ? 'bg-ok/10 text-ok' : 'bg-brand/10 text-brand'}`}>
                    <Icon d={meta.icon} className="h-5 w-5" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-semibold">{n.title}</span>
                      {!n.read && <Badge tone="brand" dot>new</Badge>}
                      <Badge tone={meta.tone === 'mute' ? 'mute' : meta.tone}>{meta.label}</Badge>
                    </span>
                    <span className="block text-xs text-mute mt-1 leading-relaxed">{n.message}</span>
                    <span className="block text-[10px] text-faint num mt-1.5">
                      {new Date(n.ts).toLocaleString()} · {timeAgo(n.ts)}
                      {n.link && <span className="text-brand ml-2">open →</span>}
                    </span>
                  </span>
                  {!n.read && <span className="h-2 w-2 rounded-full bg-brand shrink-0 mt-2" />}
                </button>
              )
            })}
          </div>
        )}
        <Pagination page={safePage} pages={pages} onPage={setPage} total={filtered.length} label="notifications" />
      </Card>
    </div>
  )
}

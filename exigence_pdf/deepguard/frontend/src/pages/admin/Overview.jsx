import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Api, getDB, timeAgo } from '../../api/client.js'
import { Icon, P, Card, Badge, Skeleton, EmptyState, VerdictBadge } from '../../components/ui.jsx'
import { LineArea, Donut, Bars } from '../../components/charts.jsx'

const DAY = 86400000

function Kpi({ icon, label, value, sub, tone = 'brand' }) {
  return (
    <Card className="p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[11px] text-mute font-semibold uppercase tracking-wider">{label}</p>
          <p className="text-2xl font-extrabold num mt-1.5 truncate">{value}</p>
          {sub && <p className="text-xs text-faint mt-1 truncate">{sub}</p>}
        </div>
        <div className={`h-10 w-10 rounded-xl grid place-items-center shrink-0
          ${tone === 'ok' ? 'bg-ok/10 text-ok' : tone === 'bad' ? 'bg-bad/10 text-bad'
            : tone === 'violet' ? 'bg-violet/10 text-violet' : 'bg-brand/10 text-brand'}`}>
          <Icon d={icon} className="h-5 w-5" />
        </div>
      </div>
    </Card>
  )
}

export default function AdminOverview() {
  const [health, setHealth] = useState(null)
  const [meta, setMeta] = useState(null)
  const [error, setError] = useState('')

  const db = getDB()
  const now = Date.now()

  useEffect(() => {
    Promise.all([Api.health(), Api.metadata()])
      .then(([h, m]) => { setHealth(h); setMeta(m) })
      .catch(e => setError(e.message))
  }, [])

  const stats = useMemo(() => {
    const history = db.history
    const d14 = history.filter(h => now - h.ts <= 14 * DAY)
    const fakes14 = d14.filter(h => h.verdict === 'fake').length
    const latencies = history.map(h => h.latency).filter(v => v > 0)
    const users = db.users
    return {
      totalUsers: users.length,
      analysts: users.filter(u => u.role === 'user').length,
      admins: users.filter(u => u.role === 'admin').length,
      activeUsers: users.filter(u => u.status === 'active').length,
      detections14: d14.length,
      totalDetections: history.length,
      fakes14,
      fakeRate: d14.length ? Math.round(fakes14 / d14.length * 100) : 0,
      avgLatency: latencies.length ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length) : 0,
      history,
      days: Array.from({ length: 14 }, (_, i) => {
        const key = new Date(now - (13 - i) * DAY).toISOString().slice(0, 10)
        const rows = history.filter(h => new Date(h.ts).toISOString().slice(0, 10) === key)
        return { date: key, total: rows.length, fakes: rows.filter(h => h.verdict === 'fake').length }
      }),
      signups: Array.from({ length: 14 }, (_, i) => {
        const key = new Date(now - (13 - i) * DAY).toISOString().slice(0, 10)
        return db.users.filter(u => new Date(u.joined).toISOString().slice(0, 10) === key).length
      }),
    }
  }, [db, now])

  if (error && !health) return <EmptyState icon={P.chart} title="Overview unavailable" message={error} />
  if (!health || !meta) return (
    <div className="space-y-5">
      <div className="grid sm:grid-cols-2 xl:grid-cols-4 gap-5">
        {[0, 1, 2, 3].map(i => <Card key={i} className="p-5"><Skeleton className="h-4 w-24" /><Skeleton className="h-8 w-20 mt-3" /></Card>)}
      </div>
      <Card className="p-5"><Skeleton className="h-56 w-full" /></Card>
    </div>
  )

  const ok = health.status === 'ok'
  const healthRows = [
    ['API', ok ? 'operational' : 'offline', ok ? 'ok' : 'bad'],
    ['Model serving', `${meta.architecture} v${meta.model_version}`, ok ? 'ok' : 'bad'],
    ['ONNX providers', (health.onnxruntime_providers || []).join(', '), 'mute'],
    ['Grad-CAM', health.gradcam_available ? 'available' : 'unavailable', health.gradcam_available ? 'ok' : 'warn'],
    ['Uptime', `${Math.round(health.uptime_s)} s`, 'mute'],
  ]
  const recentUsers = [...db.users].sort((a, b) => b.joined - a.joined).slice(0, 4)
  const recentAudit = db.logs.slice(0, 5)
  const labels = stats.days.map(d => d.date.slice(5))

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold tracking-tight">Platform overview</h2>
          <p className="text-sm text-mute mt-1">Users, detections and live model health.</p>
        </div>
        <div className="flex gap-2.5">
          <Badge tone={ok ? 'ok' : 'bad'} dot>{ok ? 'ALL SYSTEMS GO' : 'API OFFLINE'}</Badge>
          <Link to="/admin/model" className="btn-ghost"><Icon d={P.cpu} className="h-4 w-4" /> Model monitoring</Link>
        </div>
      </div>

      {/* Production model banner — live metadata */}
      <Card className="p-5 relative overflow-hidden">
        <div className="absolute -top-20 right-0 h-48 w-72 rounded-full bg-violet/15 blur-3xl pointer-events-none" />
        <div className="relative grid lg:grid-cols-[auto_1fr_auto] gap-6 items-center">
          <div className="h-14 w-14 rounded-2xl bg-violet/12 text-violet border border-violet/25 grid place-items-center">
            <Icon d={P.cpu} className="h-7 w-7" />
          </div>
          <div>
            <div className="flex items-center gap-2.5 flex-wrap">
              <h3 className="font-bold text-lg">Serving model · {meta.name}</h3>
              <Badge tone="violet">v{meta.model_version}</Badge>
            </div>
            <p className="text-xs text-mute mt-1.5">
              {meta.architecture} on {meta.backbone} · dual-stream ({meta.mode}) · input {meta.image_size}×{meta.image_size} ·
              threshold {meta.threshold} — <Link to="/admin/model" className="text-brand hover:underline">full card →</Link>
            </p>
          </div>
          <div className="grid grid-cols-2 gap-6">
            {[
              ['Verdicts served', stats.totalDetections],
              ['Avg latency', `${stats.avgLatency} ms`],
            ].map(([k, v]) => (
              <div key={k}>
                <p className="text-[10px] text-mute font-semibold uppercase tracking-wider">{k}</p>
                <p className="text-xl font-extrabold num text-violet mt-1">{v}</p>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* KPIs */}
      <div className="grid sm:grid-cols-2 xl:grid-cols-4 gap-5">
        <Kpi icon={P.users} label="Analysts" value={stats.analysts}
          sub={`${stats.activeUsers} active · ${stats.admins} admin`} />
        <Kpi icon={P.scan} label="Detections · 14d" value={stats.detections14}
          sub={`${stats.totalDetections} all time`} tone="violet" />
        <Kpi icon={P.shield} label="Fake rate · 14d" value={`${stats.fakeRate}%`}
          sub={`${stats.fakes14} manipulated found`} tone="bad" />
        <Kpi icon={P.zap} label="Avg latency" value={`${stats.avgLatency} ms`}
          sub="server-side inference" tone="ok" />
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-3 gap-5">
        <Card className="lg:col-span-2 p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="font-semibold">Detection volume</h3>
              <p className="text-xs text-mute mt-0.5">Last 14 days · all analysts</p>
            </div>
            <Badge tone="brand">14 days</Badge>
          </div>
          <LineArea height={220} labels={labels}
            series={[
              { name: 'Total', values: stats.days.map(d => d.total), color: '#3b6bff' },
              { name: 'Manipulated', values: stats.days.map(d => d.fakes), color: '#f43f5e' },
            ]} />
        </Card>

        <Card className="p-5">
          <h3 className="font-semibold">Verdict split</h3>
          <p className="text-xs text-mute mt-0.5 mb-4">All-time platform-wide</p>
          <Donut size={140} data={[
            { label: 'Authentic', value: stats.history.filter(h => h.verdict === 'real').length, color: '#10b981' },
            { label: 'Manipulated', value: stats.history.filter(h => h.verdict === 'fake').length, color: '#f43f5e' },
          ]} center={<>
            <p className="text-lg font-extrabold num">{stats.totalDetections}</p>
            <p className="text-[10px] text-mute">total</p>
          </>} />
        </Card>
      </div>

      <div className="grid lg:grid-cols-3 gap-5">
        <Card className="p-5">
          <h3 className="font-semibold">Sign-ups</h3>
          <p className="text-xs text-mute mt-0.5 mb-4">New accounts · 14 days</p>
          <Bars height={170} data={stats.signups} labels={labels} color="#10b981" />
        </Card>

        <Card className="p-5">
          <h3 className="font-semibold">System health</h3>
          <p className="text-xs text-mute mt-0.5 mb-4">From GET /health</p>
          <div className="space-y-3">
            {healthRows.map(([name, val, tone]) => (
              <div key={name} className="flex items-center justify-between gap-3 rounded-xl bg-panel2 border border-line px-4 py-2.5">
                <span className="text-sm">{name}</span>
                <span className={`chip ${tone === 'ok' ? 'text-ok bg-ok/10 border-ok/25' : tone === 'warn' ? 'text-warn bg-warn/10 border-warn/25'
                  : tone === 'bad' ? 'text-bad bg-bad/10 border-bad/25' : 'text-mute bg-panel border-line'} num`}>
                  <span className="h-1.5 w-1.5 rounded-full bg-current" />{val}
                </span>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between px-5 py-4 border-b border-line">
            <h3 className="font-semibold">Latest audit events</h3>
            <Link to="/admin/audit" className="text-xs text-brand hover:underline font-semibold">Full trail →</Link>
          </div>
          <div className="divide-y divide-line/60">
            {recentAudit.map(a => (
              <div key={a.id} className="px-5 py-3">
                <div className="flex items-center gap-2">
                  <Badge tone={a.severity === 'warn' ? 'warn' : a.severity === 'bad' ? 'bad' : 'mute'}>{a.action}</Badge>
                  <span className="text-[11px] text-faint num ml-auto">{timeAgo(a.ts)}</span>
                </div>
                <p className="text-xs text-mute mt-1.5 leading-relaxed line-clamp-2">{a.detail} · {a.userName}</p>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Newest members */}
      <Card>
        <div className="flex items-center justify-between px-5 py-4 border-b border-line">
          <h3 className="font-semibold">Newest members</h3>
          <Link to="/admin/users" className="text-xs text-brand hover:underline font-semibold">Manage →</Link>
        </div>
        <div className="divide-y divide-line/60">
          {recentUsers.map(u => (
            <div key={u.id} className="flex items-center gap-3 px-5 py-3">
              <span className="h-9 w-9 rounded-xl bg-brand/10 text-brand grid place-items-center font-bold text-sm shrink-0">
                {(u.name || u.email)[0]?.toUpperCase()}
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium truncate">{u.name}</p>
                <p className="text-[11px] text-faint num truncate">{u.email}</p>
              </div>
              <Badge tone={u.status === 'active' ? 'ok' : u.status === 'suspended' ? 'bad' : 'warn'}>{u.status.toUpperCase()}</Badge>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

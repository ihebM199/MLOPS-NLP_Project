import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Api, myHistory } from '../../api/client.js'
import { useAuth } from '../../context/AuthContext.jsx'
import { subscribeStore } from '../../api/client.js'
import { Icon, P, Card, Badge, VerdictBadge, EmptyState } from '../../components/ui.jsx'
import { Sparkline, LineArea, Donut } from '../../components/charts.jsx'

const DAY = 86400000

function Kpi({ icon, label, value, sub, data, tone = 'brand' }) {
  const colors = { brand: '#3b6bff', ok: '#10b981', bad: '#f43f5e', violet: '#8b5cf6' }
  return (
    <Card className="p-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[11px] text-mute font-semibold uppercase tracking-wider">{label}</p>
          <p className="text-2xl font-extrabold num mt-1.5">{value}</p>
          {sub && <p className="text-xs text-faint mt-1">{sub}</p>}
        </div>
        <div className={`h-10 w-10 rounded-xl grid place-items-center
          ${tone === 'ok' ? 'bg-ok/10 text-ok' : tone === 'bad' ? 'bg-bad/10 text-bad'
            : tone === 'violet' ? 'bg-violet/10 text-violet' : 'bg-brand/10 text-brand'}`}>
          <Icon d={icon} className="h-5 w-5" />
        </div>
      </div>
      {data && <div className="mt-3"><Sparkline data={data} w={220} h={30} stroke={colors[tone]} /></div>}
    </Card>
  )
}

export default function UserDashboard() {
  const { user } = useAuth()
  const [meta, setMeta] = useState(null)

  const history = myHistory()
  useEffect(() => subscribeStore(() => { /* store mutations re-render via parent state */ }), [])
  useEffect(() => { Api.metadata().then(setMeta).catch(() => {}) }, [])

  const now = Date.now()
  const d14 = history.filter(h => now - h.ts <= 14 * DAY)
  const fakes14 = d14.filter(h => h.verdict === 'fake').length
  const latencies = history.map(h => h.latency).filter(v => v > 0)
  const avgLatency = latencies.length ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length) : 0
  const avgPFake = history.length ? history.reduce((a, h) => a + (h.pFake ?? 0), 0) / history.length : 0

  // last 14 days series
  const days = Array.from({ length: 14 }, (_, i) => {
    const d = new Date(now - (13 - i) * DAY)
    const key = d.toISOString().slice(0, 10)
    const rows = history.filter(h => new Date(h.ts).toISOString().slice(0, 10) === key)
    return { date: key, total: rows.length, fakes: rows.filter(h => h.verdict === 'fake').length }
  })
  const streak = new Set(d14.map(h => new Date(h.ts).toDateString())).size

  const labels = days.map(d => d.date.slice(5))
  const totalSeries = days.map(d => d.total)
  const fakeSeries = days.map(d => d.fakes)
  const recent = history.slice(0, 5)
  const realCount = history.filter(h => h.verdict === 'real').length
  const fakeCount = history.filter(h => h.verdict === 'fake').length

  return (
    <div className="space-y-6">
      {/* Greeting */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold tracking-tight">
            Welcome back, {user?.name?.split(' ')[0] || 'analyst'} 👋
          </h2>
          <p className="text-sm text-mute mt-1">
            {streak > 0
              ? `You've analysed media on ${streak} of the last 14 days.`
              : 'Start a new analysis to build your streak.'}
          </p>
        </div>
        <Link to="/user/analyze" className="btn-primary"><Icon d={P.scan} className="h-4 w-4" /> New analysis</Link>
      </div>

      {/* KPIs */}
      <div className="grid sm:grid-cols-2 xl:grid-cols-4 gap-5">
        <Kpi icon={P.scan} label="Analyses · 14 days" value={d14.length}
          sub={`${history.length} all time`} data={totalSeries} />
        <Kpi icon={P.shield} label="Manipulated found" value={fakes14}
          sub={d14.length ? `${Math.round(fakes14 / d14.length * 100)}% of recent runs` : '—'} data={fakeSeries} tone="bad" />
        <Kpi icon={P.zap} label="Avg latency" value={`${avgLatency} ms`} sub="server-side inference" tone="violet" />
        <Kpi icon={P.chart} label="Avg P(manipulated)" value={`${(avgPFake * 100).toFixed(0)}%`}
          sub="across your history" tone="ok" />
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-3 gap-5">
        <Card className="lg:col-span-2 p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="font-semibold">Detection activity</h3>
              <p className="text-xs text-mute mt-0.5">Last 14 days · total vs manipulated</p>
            </div>
            <Badge tone="brand">14 days</Badge>
          </div>
          <LineArea height={230} labels={labels}
            series={[
              { name: 'Total', values: totalSeries, color: '#3b6bff' },
              { name: 'Manipulated', values: fakeSeries, color: '#f43f5e' },
            ]} />
        </Card>
        <Card className="p-5">
          <h3 className="font-semibold">Verdict split</h3>
          <p className="text-xs text-mute mt-0.5 mb-4">Your full history</p>
          <Donut size={150} data={[
            { label: 'Authentic', value: realCount, color: '#10b981' },
            { label: 'Manipulated', value: fakeCount, color: '#f43f5e' },
          ]} center={<>
            <p className="text-xl font-extrabold num">{history.length}</p>
            <p className="text-[10px] text-mute">total</p>
          </>} />
        </Card>
      </div>

      {/* Engine strip */}
      {meta && (
        <Card className="p-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-mute">
          <span className="flex items-center gap-2 font-semibold text-ink">
            <Icon d={P.cpu} className="h-4 w-4 text-violet" /> {meta.architecture}
          </span>
          <span className="num">backbone {meta.backbone}</span>
          <span className="num">mode {meta.mode}</span>
          <span className="num">input {meta.image_size}×{meta.image_size}</span>
          <span className="num">threshold {meta.threshold}</span>
          <span className="num">v{meta.model_version}</span>
        </Card>
      )}

      {/* Recent */}
      <Card>
        <div className="flex items-center justify-between px-5 py-4 border-b border-line">
          <h3 className="font-semibold">Recent analyses</h3>
          <Link to="/user/history" className="text-xs text-brand hover:underline font-semibold">View all →</Link>
        </div>
        {recent.length === 0 ? (
          <EmptyState icon={P.scan} title="No analyses yet"
            message="Run your first analysis — it takes about a second."
            action={<Link to="/user/analyze" className="btn-primary">Start analysing</Link>} />
        ) : (
          <div className="divide-y divide-line/60">
            {recent.map(r => (
              <Link key={r.id} to="/user/history" className="flex items-center gap-4 px-5 py-3.5 hover:bg-panel2 transition group">
                <div className="h-10 w-10 rounded-xl bg-panel2 border border-line grid place-items-center shrink-0 overflow-hidden">
                  {r.thumb
                    ? <img src={r.thumb} alt="" className="h-full w-full object-cover" />
                    : <Icon d={P.file} className="h-4 w-4 text-mute" />}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium truncate group-hover:text-brand transition">{r.name}</p>
                  <p className="text-[11px] text-faint num mt-0.5">
                    {new Date(r.ts).toLocaleString()} · {r.latency} ms · v{r.modelVersion || '1.0.0'}
                  </p>
                </div>
                <span className="num text-sm font-bold hidden sm:block">{((r.pFake ?? 0) * 100).toFixed(0)}%</span>
                <VerdictBadge verdict={r.verdict} />
              </Link>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}

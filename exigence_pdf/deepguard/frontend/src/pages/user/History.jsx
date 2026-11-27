import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { downloadCsv, fmtDateTime, myHistory, deleteHistory, logEvent } from '../../api/client.js'
import { useToast } from '../../context/ToastContext.jsx'
import { Icon, P, Card, VerdictBadge, Input, Select, EmptyState, Pagination, Modal, ConfirmDialog } from '../../components/ui.jsx'

export default function History() {
  const pushToast = useToast()
  const [version, setVersion] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [verdict, setVerdict] = useState('')
  const [source, setSource] = useState('')
  const [detail, setDetail] = useState(null)
  const [delTarget, setDelTarget] = useState(null)
  const [busy, setBusy] = useState(false)

  const history = useMemo(() => myHistory(), [version])

  useEffect(() => { setPage(1) }, [search, verdict, source])

  const filtered = history.filter(h =>
    (!search || h.name.toLowerCase().includes(search.toLowerCase())) &&
    (!verdict || h.verdict === verdict) &&
    (!source || h.source === source))

  const per = 8
  const pages = Math.max(1, Math.ceil(filtered.length / per))
  const safePage = Math.min(page, pages)
  const view = filtered.slice((safePage - 1) * per, safePage * per)

  const del = async () => {
    setBusy(true)
    try {
      deleteHistory(delTarget.id)
      logEvent('Analysis deleted', 'warn', delTarget.name)
      pushToast('Record deleted', `${delTarget.name} removed from your history.`, 'brand')
      setDelTarget(null)
      setDetail(null)
      setVersion(v => v + 1)
    } finally { setBusy(false) }
  }

  const exportCsv = () => {
    downloadCsv('deepguard_history.csv',
      ['filename', 'verdict', 'p_fake', 'confidence', 'spatial_weight', 'latency_ms', 'source', 'model_version', 'timestamp'],
      filtered.map(h => [h.name, h.verdict.toUpperCase(), (h.pFake ?? 0).toFixed(4), h.confidence.toFixed(4),
        (h.wS ?? 0).toFixed(3), h.latency, h.source, h.modelVersion || '1.0.0', new Date(h.ts).toISOString()]))
    pushToast('Export ready', `${filtered.length} records written to deepguard_history.csv.`, 'brand')
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold tracking-tight">Analysis history</h2>
          <p className="text-sm text-mute mt-1">
            {filtered.length} analyses · {filtered.filter(h => h.verdict === 'fake').length} flagged as fake
          </p>
        </div>
        <div className="flex gap-2.5">
          <button onClick={exportCsv} className="btn-ghost"><Icon d={P.download} className="h-4 w-4" /> Export CSV</button>
          <Link to="/user/analyze" className="btn-primary"><Icon d={P.scan} className="h-4 w-4" /> New analysis</Link>
        </div>
      </div>

      <Card>
        <div className="p-4 grid sm:grid-cols-[1fr_auto_auto] gap-3 border-b border-line">
          <Input placeholder="Search by filename…" value={search}
            onChange={e => setSearch(e.target.value)} />
          <Select value={verdict} onChange={e => setVerdict(e.target.value)}>
            <option value="">All verdicts</option>
            <option value="real">Authentic</option>
            <option value="fake">Manipulated</option>
          </Select>
          <Select value={source} onChange={e => setSource(e.target.value)}>
            <option value="">All sources</option>
            <option value="upload">Uploads</option>
            <option value="sample">Samples</option>
            <option value="webcam">Webcam</option>
          </Select>
        </div>

        {view.length === 0 ? (
          <EmptyState icon={P.history} title="No matching records"
            message={search || verdict || source ? 'Try clearing the filters.' : 'Run your first analysis to populate history.'}
            action={<Link to="/user/analyze" className="btn-primary">Start analysing</Link>} />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[680px]">
              <thead><tr>
                <th className="th">Media</th><th className="th">Verdict</th><th className="th">P(fake)</th>
                <th className="th">Fusion</th><th className="th">Latency</th><th className="th">When</th>
                <th className="th w-20"></th>
              </tr></thead>
              <tbody>
                {view.map(r => (
                  <tr key={r.id} className="hover:bg-panel2 transition cursor-pointer" onClick={() => setDetail(r)}>
                    <td className="td">
                      <div className="flex items-center gap-3 min-w-0">
                        <span className="h-9 w-9 rounded-lg bg-panel2 border border-line grid place-items-center shrink-0 overflow-hidden">
                          {r.thumb
                            ? <img src={r.thumb} alt="" className="h-full w-full object-cover" />
                            : <Icon d={P.file} className="h-4 w-4 text-mute" />}
                        </span>
                        <span className="min-w-0">
                          <span className="block font-medium truncate max-w-[220px]">{r.name}</span>
                          <span className="block text-[11px] text-faint capitalize">{r.source}</span>
                        </span>
                      </div>
                    </td>
                    <td className="td"><VerdictBadge verdict={r.verdict} /></td>
                    <td className="td">
                      <span className={`num font-bold ${r.verdict === 'fake' ? 'text-bad' : 'text-ok'}`}>
                        {((r.pFake ?? 0) * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="td">
                      <div className="w-[76px] h-2 rounded-full overflow-hidden flex bg-line">
                        <span className="h-full bg-brand" style={{ width: `${(r.wS ?? 0.5) * 100}%` }} />
                        <span className="h-full bg-violet" style={{ width: `${(1 - (r.wS ?? 0.5)) * 100}%` }} />
                      </div>
                    </td>
                    <td className="td num text-mute">{r.latency} ms</td>
                    <td className="td text-mute text-xs num">{fmtDateTime(r.ts)}</td>
                    <td className="td">
                      <button onClick={(e) => { e.stopPropagation(); setDelTarget(r) }}
                        className="h-8 w-8 rounded-lg grid place-items-center text-faint hover:text-bad hover:bg-bad/10 transition"
                        aria-label="Delete"><Icon d={P.trash} className="h-4 w-4" /></button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <Pagination page={safePage} pages={pages} onPage={setPage} total={filtered.length} label="records" />
      </Card>

      {/* Detail modal */}
      <Modal open={!!detail} onClose={() => setDetail(null)} title="Analysis detail" wide>
        {detail && (
          <div className="grid sm:grid-cols-[1.2fr_1fr] gap-5">
            <div>
              <div className="relative rounded-xl border border-line overflow-hidden bg-panel2">
                {detail.thumb
                  ? <img src={detail.thumb} alt={detail.name} className="w-full object-contain max-h-80 bg-bg" />
                  : <div className="h-64 grid place-items-center text-faint"><Icon d={P.file} className="h-8 w-8" /></div>}
              </div>
              <p className="text-[11px] text-faint mt-2 leading-relaxed">
                Thumbnails are stored locally with the history entry; re-run the analysis on the workbench for the full-size image and Grad-CAM.
              </p>
            </div>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between">
                  <VerdictBadge verdict={detail.verdict} />
                  <span className="num text-xs text-faint">v{detail.modelVersion || '1.0.0'}</span>
                </div>
                <p className="text-xs text-mute mt-2.5 break-all">{detail.name}</p>
              </div>
              <div className="rounded-xl border border-line divide-y divide-line/60 text-sm">
                {[
                  ['P(manipulated)', `${((detail.pFake ?? 0) * 100).toFixed(1)}%`],
                  ['Confidence', `${(detail.confidence * 100).toFixed(0)}%`],
                  ['Spatial weight', (detail.wS ?? 0).toFixed(3)],
                  ['Frequency weight', (1 - (detail.wS ?? 0.5)).toFixed(3)],
                  ['Latency', `${detail.latency} ms`],
                  ['Source', detail.source],
                  ['Timestamp', new Date(detail.ts).toLocaleString()],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between px-3.5 py-2.5">
                    <span className="text-mute">{k}</span><span className="num font-semibold">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </Modal>

      <ConfirmDialog open={!!delTarget} onClose={() => setDelTarget(null)} onConfirm={del}
        busy={busy} title="Delete record" confirmLabel="Delete"
        message={`Remove “${delTarget?.name}” from your history? This cannot be undone.`} />
    </div>
  )
}

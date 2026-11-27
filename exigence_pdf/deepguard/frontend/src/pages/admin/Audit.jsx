import { useEffect, useMemo, useState } from 'react'
import { downloadCsv, getDB, timeAgo } from '../../api/client.js'
import { useToast } from '../../context/ToastContext.jsx'
import { Icon, P, Card, Badge, Input, Select, EmptyState, Pagination } from '../../components/ui.jsx'

export default function AdminAudit() {
  const pushToast = useToast()
  const [version, setVersion] = useState(0)
  const [page, setPage] = useState(1)
  const [severity, setSeverity] = useState('')
  const [search, setSearch] = useState('')

  const logs = useMemo(() => getDB().logs, [version])

  useEffect(() => { setPage(1) }, [severity, search])

  const filtered = logs.filter(l =>
    (!severity || l.severity === severity) &&
    (!search || l.action.toLowerCase().includes(search.toLowerCase()) ||
      l.userName.toLowerCase().includes(search.toLowerCase()) ||
      (l.detail || '').toLowerCase().includes(search.toLowerCase())))

  const per = 10
  const pages = Math.max(1, Math.ceil(filtered.length / per))
  const safePage = Math.min(page, pages)
  const view = filtered.slice((safePage - 1) * per, safePage * per)

  const exportCsv = () => {
    downloadCsv('deepguard_audit.csv',
      ['timestamp', 'actor', 'action', 'severity', 'ip', 'detail'],
      filtered.map(l => [new Date(l.ts).toISOString(), l.userName, l.action, l.severity, l.ip, l.detail]))
    pushToast('Export ready', `${filtered.length} events written to deepguard_audit.csv.`, 'brand')
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold tracking-tight">Audit logs</h2>
          <p className="text-sm text-mute mt-1">
            Every privileged action, with actor and target — {logs.length} events retained.
          </p>
        </div>
        <button onClick={exportCsv} className="btn-ghost"><Icon d={P.download} className="h-4 w-4" /> Export CSV</button>
      </div>

      <Card>
        <div className="p-4 grid sm:grid-cols-[1fr_auto] gap-3 border-b border-line">
          <Input placeholder="Search by action, actor or detail…" value={search} onChange={e => setSearch(e.target.value)} />
          <Select value={severity} onChange={e => setSeverity(e.target.value)}>
            <option value="">All severities</option>
            <option value="info">Info</option>
            <option value="ok">Success</option>
            <option value="warn">Warning</option>
            <option value="bad">Critical</option>
          </Select>
        </div>

        {view.length === 0 ? (
          <EmptyState icon={P.file} title="No matching events" message="Try clearing the filters." />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[680px]">
              <thead><tr>
                <th className="th">When</th><th className="th">Actor</th><th className="th">Action</th>
                <th className="th">Severity</th><th className="th">IP</th><th className="th">Detail</th>
              </tr></thead>
              <tbody>
                {view.map(l => (
                  <tr key={l.id} className="hover:bg-panel2 transition">
                    <td className="td text-xs text-mute num whitespace-nowrap">
                      {new Date(l.ts).toLocaleString()}
                      <span className="block text-faint">{timeAgo(l.ts)}</span>
                    </td>
                    <td className="td text-sm font-medium">{l.userName}</td>
                    <td className="td text-sm">{l.action}</td>
                    <td className="td">
                      <Badge tone={l.severity === 'bad' ? 'bad' : l.severity === 'warn' ? 'warn' : l.severity === 'ok' ? 'ok' : 'mute'} dot>
                        {l.severity.toUpperCase()}
                      </Badge>
                    </td>
                    <td className="td num text-xs text-mute">{l.ip}</td>
                    <td className="td text-xs text-mute">{l.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <Pagination page={safePage} pages={pages} onPage={setPage} total={filtered.length} label="events" />
      </Card>
    </div>
  )
}

import { useEffect, useMemo, useState } from 'react'
import { downloadCsv, getDB, hashString, logEvent, saveDB, uid } from '../../api/client.js'
import { useToast } from '../../context/ToastContext.jsx'
import { useAuth } from '../../context/AuthContext.jsx'
import { Icon, P, Card, Badge, Input, Select, EmptyState, Pagination, Modal, ConfirmDialog } from '../../components/ui.jsx'

export default function AdminUsers() {
  const pushToast = useToast()
  const { user: me } = useAuth()
  const [version, setVersion] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [role, setRole] = useState('')
  const [status, setStatus] = useState('')
  const [detail, setDetail] = useState(null)
  const [createOpen, setCreateOpen] = useState(false)
  const [inviteOpen, setInviteOpen] = useState(false)
  const [confirm, setConfirm] = useState(null)
  const [busy, setBusy] = useState(false)

  const users = useMemo(() => getDB().users, [version])

  useEffect(() => { setPage(1) }, [search, role, status])

  const filtered = users.filter(u =>
    (!search || u.name.toLowerCase().includes(search.toLowerCase()) || u.email.toLowerCase().includes(search.toLowerCase())) &&
    (!role || u.role === role) &&
    (!status || u.status === status))

  const per = 10
  const pages = Math.max(1, Math.ceil(filtered.length / per))
  const safePage = Math.min(page, pages)
  const view = filtered.slice((safePage - 1) * per, safePage * per)

  const refresh = () => setVersion(v => v + 1)

  const setStatusFor = (u, next) => {
    const db = getDB()
    const target = db.users.find(x => x.id === u.id)
    target.status = next
    saveDB()
    logEvent(next === 'suspended' ? 'User suspended' : 'User reactivated', next === 'suspended' ? 'bad' : 'ok', u.email)
    pushToast(next === 'suspended' ? 'Account suspended' : 'Account reactivated', `${u.name} (${u.email}).`, next === 'suspended' ? 'bad' : 'ok')
    refresh(); setDetail(d => d?.id === u.id ? target : d)
  }

  const doDelete = async () => {
    setBusy(true)
    try {
      const db = getDB()
      const target = confirm
      db.users = db.users.filter(x => x.id !== target.id)
      db.history = db.history.filter(h => h.userId !== target.id)
      db.notifs = db.notifs.filter(n => n.userId !== target.id)
      saveDB()
      logEvent('Account deleted', 'bad', target.email)
      pushToast('Account deleted', `${target.name} was removed with their history.`, 'bad')
      setConfirm(null); setDetail(null)
      refresh()
    } finally { setBusy(false) }
  }

  const exportCsv = () => {
    downloadCsv('deepguard_users.csv',
      ['name', 'email', 'role', 'status', 'joined', 'last_active'],
      filtered.map(u => [u.name, u.email, u.role, u.status, new Date(u.joined).toISOString(), new Date(u.lastActive).toISOString()]))
    pushToast('Export ready', `${filtered.length} accounts written to CSV.`, 'brand')
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold tracking-tight">User management</h2>
          <p className="text-sm text-mute mt-1">
            {users.length} accounts · {users.filter(u => u.status === 'active').length} active ·
            demo store (this browser).
          </p>
        </div>
        <div className="flex gap-2.5">
          <button onClick={exportCsv} className="btn-ghost"><Icon d={P.download} className="h-4 w-4" /> Export CSV</button>
          <button onClick={() => setInviteOpen(true)} className="btn-ghost"><Icon d={P.mail} className="h-4 w-4" /> Invite</button>
          <button onClick={() => setCreateOpen(true)} className="btn-primary"><Icon d={P.plus} className="h-4 w-4" /> Create user</button>
        </div>
      </div>

      <Card>
        <div className="p-4 grid sm:grid-cols-[1fr_auto_auto] gap-3 border-b border-line">
          <Input placeholder="Search by name or email…" value={search} onChange={e => setSearch(e.target.value)} />
          <Select value={role} onChange={e => setRole(e.target.value)}>
            <option value="">All roles</option>
            <option value="user">Analysts</option>
            <option value="admin">Admins</option>
          </Select>
          <Select value={status} onChange={e => setStatus(e.target.value)}>
            <option value="">All statuses</option>
            <option value="active">Active</option>
            <option value="invited">Invited</option>
            <option value="suspended">Suspended</option>
          </Select>
        </div>

        {view.length === 0 ? (
          <EmptyState icon={P.users} title="No matching accounts" message="Try clearing the filters." />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px]">
              <thead><tr>
                <th className="th">Member</th><th className="th">Role</th><th className="th">Status</th>
                <th className="th">Joined</th><th className="th">Last active</th><th className="th w-24"></th>
              </tr></thead>
              <tbody>
                {view.map(u => (
                  <tr key={u.id} className="hover:bg-panel2 transition cursor-pointer" onClick={() => setDetail(u)}>
                    <td className="td">
                      <div className="flex items-center gap-3 min-w-0">
                        <span className="h-9 w-9 rounded-xl grid place-items-center shrink-0 font-bold text-sm text-white" style={{ background: u.color }}>
                          {(u.name || u.email)[0]?.toUpperCase()}
                        </span>
                        <span className="min-w-0">
                          <span className="block font-medium truncate max-w-[200px]">{u.name}</span>
                          <span className="block text-[11px] text-faint num truncate max-w-[200px]">{u.email}</span>
                        </span>
                      </div>
                    </td>
                    <td className="td">
                      <Badge tone={u.role === 'admin' ? 'violet' : 'brand'}>{u.role === 'admin' ? 'ADMIN' : 'ANALYST'}</Badge>
                    </td>
                    <td className="td">
                      <Badge tone={u.status === 'active' ? 'ok' : u.status === 'suspended' ? 'bad' : 'warn'} dot>
                        {u.status.toUpperCase()}
                      </Badge>
                    </td>
                    <td className="td text-xs text-mute num">{new Date(u.joined).toLocaleDateString()}</td>
                    <td className="td text-xs text-mute num">{new Date(u.lastActive).toLocaleDateString()}</td>
                    <td className="td">
                      <div className="flex gap-1.5">
                        {u.id !== me?.id && u.status !== 'suspended' && (
                          <button onClick={(e) => { e.stopPropagation(); setStatusFor(u, 'suspended') }}
                            className="btn-soft !py-1.5 !px-3 !text-xs">Suspend</button>
                        )}
                        {u.id !== me?.id && u.status === 'suspended' && (
                          <button onClick={(e) => { e.stopPropagation(); setStatusFor(u, 'active') }}
                            className="btn-soft !py-1.5 !px-3 !text-xs">Reactivate</button>
                        )}
                        {u.id !== me?.id && (
                          <button onClick={(e) => { e.stopPropagation(); setConfirm(u) }}
                            className="h-8 w-8 rounded-lg grid place-items-center text-faint hover:text-bad hover:bg-bad/10 transition"
                            aria-label="Delete"><Icon d={P.trash} className="h-4 w-4" /></button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <Pagination page={safePage} pages={pages} onPage={setPage} total={filtered.length} label="accounts" />
      </Card>

      {/* Create user modal */}
      <CreateUserModal open={createOpen} onClose={() => setCreateOpen(false)} onDone={refresh} pushToast={pushToast} />

      {/* Invite modal */}
      <InviteModal open={inviteOpen} onClose={() => setInviteOpen(false)} onDone={refresh} pushToast={pushToast} />

      {/* Detail modal */}
      <Modal open={!!detail} onClose={() => setDetail(null)} title="Account detail">
        {detail && (
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <span className="h-12 w-12 rounded-xl grid place-items-center font-bold text-white text-lg" style={{ background: detail.color }}>
                {(detail.name || detail.email)[0]?.toUpperCase()}
              </span>
              <div className="min-w-0">
                <p className="font-bold truncate">{detail.name}</p>
                <p className="text-sm text-mute num truncate">{detail.email}</p>
              </div>
            </div>
            <div className="rounded-xl border border-line divide-y divide-line/60 text-sm">
              {[
                ['Role', detail.role],
                ['Status', detail.status],
                ['Joined', new Date(detail.joined).toLocaleString()],
                ['Last active', new Date(detail.lastActive).toLocaleString()],
                ['Analyses', getDB().history.filter(h => h.userId === detail.id).length],
                ['API key', detail.apiKey ? detail.apiKey.slice(0, 14) + '…' : '—'],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between px-3.5 py-2.5">
                  <span className="text-mute">{k}</span><span className="num font-semibold">{v}</span>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-faint leading-relaxed">
              Accounts live in this browser's demo store. To plug real accounts into the Django
              backend, the auth layer would move server-side — the model API is already live.
            </p>
          </div>
        )}
      </Modal>

      <ConfirmDialog open={!!confirm} onClose={() => setConfirm(null)} onConfirm={doDelete}
        busy={busy} title="Delete account" confirmLabel="Delete"
        message={`Delete “${confirm?.name}” (${confirm?.email}) and their analysis history? This cannot be undone.`} />
    </div>
  )
}

function CreateUserModal({ open, onClose, onDone, pushToast }) {
  const [form, setForm] = useState({ name: '', email: '', password: '', role: 'user' })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setError('')
    if (!form.name.trim() || !/^\S+@\S+\.\S+$/.test(form.email)) { setError('Name and a valid email are required.'); return }
    if (form.password.length < 8) { setError('Password must be at least 8 characters.'); return }
    if (getDB().users.some(u => u.email === form.email.trim().toLowerCase())) { setError('An account with this email already exists.'); return }
    setBusy(true)
    try {
      const db = getDB()
      db.users.push({
        id: uid('u'), name: form.name.trim(), email: form.email.trim().toLowerCase(), org: '',
        pass: hashString(form.password + '|dg'), role: form.role, status: 'active',
        joined: Date.now(), lastActive: Date.now(), color: '#3b6bff',
        apiKey: 'dg_live_' + hashString(form.email + Date.now()).toString(36).padStart(10, '0'),
      })
      saveDB()
      logEvent('User created', 'ok', form.email.trim().toLowerCase())
      pushToast('Account created', `${form.name} can sign in immediately.`, 'ok')
      setForm({ name: '', email: '', password: '', role: 'user' })
      onDone()
      onClose()
    } finally { setBusy(false) }
  }

  return (
    <Modal open={open} onClose={onClose} title="Create user"
      footer={<>
        <button className="btn-ghost" onClick={onClose}>Cancel</button>
        <button className="btn-primary" onClick={submit} disabled={busy}>
          {busy ? <Icon d={P.check} className="h-4 w-4" /> : 'Create account'}
        </button>
      </>}>
      <form onSubmit={submit} className="space-y-4">
        <Input label="Full name" value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} placeholder="Nadia Yamamoto" />
        <Input label="Email" type="email" value={form.email} onChange={e => setForm(f => ({ ...f, email: e.target.value }))} placeholder="nadia@company.com" />
        <Input label="Password" type="text" value={form.password} onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
          placeholder="min. 8 characters" className="num" hint="Shown in clear — demo store, no SMTP." />
        <Select label="Role" value={form.role} onChange={e => setForm(f => ({ ...f, role: e.target.value }))}>
          <option value="user">Analyst</option>
          <option value="admin">Administrator</option>
        </Select>
        {error && <p className="text-sm text-bad bg-bad/10 border border-bad/25 rounded-xl px-3.5 py-2.5">{error}</p>}
      </form>
    </Modal>
  )
}

function InviteModal({ open, onClose, onDone, pushToast }) {
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setError('')
    if (!/^\S+@\S+\.\S+$/.test(email)) { setError('Enter a valid email.'); return }
    if (getDB().users.some(u => u.email === email.trim().toLowerCase())) { setError('An account with this email already exists.'); return }
    setBusy(true)
    try {
      const db = getDB()
      db.users.push({
        id: uid('u'), name: email.split('@')[0].replace(/[._]/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        email: email.trim().toLowerCase(), org: '', pass: 0, role: 'user', status: 'invited',
        joined: Date.now(), lastActive: Date.now(), color: '#8b5cf6',
        apiKey: 'dg_live_' + hashString(email + Date.now()).toString(36).padStart(10, '0'),
      })
      saveDB()
      logEvent('User invited', 'ok', email.trim().toLowerCase())
      pushToast('Invitation created', `${email} appears as invited (demo store — no SMTP).`, 'brand')
      setEmail('')
      onDone()
      onClose()
    } finally { setBusy(false) }
  }

  return (
    <Modal open={open} onClose={onClose} title="Invite an analyst"
      footer={<>
        <button className="btn-ghost" onClick={onClose}>Cancel</button>
        <button className="btn-primary" onClick={submit} disabled={busy}>Create invitation</button>
      </>}>
      <form onSubmit={submit} className="space-y-4">
        <Input label="Email" type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="colleague@company.com" />
        {error && <p className="text-sm text-bad bg-bad/10 border border-bad/25 rounded-xl px-3.5 py-2.5">{error}</p>}
        <p className="text-xs text-faint leading-relaxed">
          This demo runs without a mail server: the invited account is created locally with the
          <span className="num"> invited</span> status; an administrator sets the password via “Create user” when they accept.
        </p>
      </form>
    </Modal>
  )
}

import { useState } from 'react'
import { changePassword, regenerateApiKey, saveDB } from '../../api/client.js'
import { useAuth } from '../../context/AuthContext.jsx'
import { useToast } from '../../context/ToastContext.jsx'
import { Icon, P, Card, Badge, Input, Spinner } from '../../components/ui.jsx'

export default function Profile() {
  const { user, patchUser } = useAuth()
  const pushToast = useToast()
  const [form, setForm] = useState({ name: user?.name || '', org: user?.org || '' })
  const [busy, setBusy] = useState(false)

  const [pw, setPw] = useState({ current: '', next: '', confirm: '' })
  const [pwBusy, setPwBusy] = useState(false)

  const [apiKey, setApiKey] = useState(user?.apiKey || '')
  const [revealed, setRevealed] = useState(false)
  const [keyBusy, setKeyBusy] = useState(false)

  const save = async (e) => {
    e.preventDefault()
    setBusy(true)
    try {
      patchUser({ name: form.name.trim(), org: form.org.trim() })
      pushToast('Profile saved', 'Your identity details were updated.', 'ok')
    } catch (err) { pushToast('Save failed', err.message, 'bad') }
    finally { setBusy(false) }
  }

  const doChangePw = async (e) => {
    e.preventDefault()
    if (pw.next !== pw.confirm) { pushToast('Passwords do not match', 'Re-type the new password.', 'bad'); return }
    setPwBusy(true)
    try {
      changePassword(pw.current, pw.next)
      setPw({ current: '', next: '', confirm: '' })
      pushToast('Password changed', 'Use the new password next time you sign in.', 'ok')
    } catch (err) { pushToast('Change failed', err.message, 'bad') }
    finally { setPwBusy(false) }
  }

  const regenKey = async () => {
    setKeyBusy(true)
    try {
      const k = regenerateApiKey()
      setApiKey(k)
      setRevealed(true)
      pushToast('API key rotated', 'Integrations using the old key must be updated.', 'brand')
    } catch (err) { pushToast('Rotation failed', err.message, 'bad') }
    finally { setKeyBusy(false) }
  }

  const copyKey = async () => {
    try { await navigator.clipboard.writeText(apiKey); pushToast('Copied', 'API key copied to clipboard.', 'brand') }
    catch { pushToast('Copy failed', 'Select and copy the key manually.', 'bad') }
  }

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h2 className="text-xl font-bold tracking-tight">Profile</h2>
        <p className="text-sm text-mute mt-1">Identity, credentials and API access.</p>
      </div>

      <Card className="p-5">
        <div className="flex items-center gap-4 mb-5">
          <span className="h-14 w-14 rounded-2xl bg-brand/15 text-brand border border-brand/25 grid place-items-center text-xl font-bold">
            {(user?.name || user?.email)[0]?.toUpperCase()}
          </span>
          <div className="min-w-0">
            <p className="font-bold truncate">{user?.name || 'Analyst'}</p>
            <p className="text-sm text-mute truncate num">{user?.email}</p>
            <div className="flex gap-2 mt-1.5">
              <Badge tone="brand">ANALYST</Badge>
              <Badge tone={user?.status === 'active' ? 'ok' : 'bad'} dot>{(user?.status || 'ACTIVE').toUpperCase()}</Badge>
            </div>
          </div>
        </div>
        <form onSubmit={save} className="space-y-4">
          <div className="grid sm:grid-cols-2 gap-4">
            <Input label="Full name" value={form.name}
              onChange={e => setForm(f => ({ ...f, name: e.target.value }))} />
            <Input label="Organization" value={form.org}
              onChange={e => setForm(f => ({ ...f, org: e.target.value }))} />
          </div>
          <Input label="Email (read-only)" value={user?.email || ''} readOnly disabled
            hint="Demo accounts live in this browser's storage — the email cannot be changed." />
          <button disabled={busy} className="btn-primary">{busy ? <Spinner className="h-4 w-4" /> : 'Save changes'}</button>
        </form>
      </Card>

      <Card className="p-5">
        <h3 className="font-semibold flex items-center gap-2"><Icon d={P.key} className="h-4 w-4 text-brand" /> Change password</h3>
        <form onSubmit={doChangePw} className="mt-4 space-y-4">
          <Input label="Current password" type="password" value={pw.current} autoComplete="current-password"
            onChange={e => setPw(p => ({ ...p, current: e.target.value }))} />
          <div className="grid sm:grid-cols-2 gap-4">
            <Input label="New password" type="password" value={pw.next} autoComplete="new-password"
              onChange={e => setPw(p => ({ ...p, next: e.target.value }))} />
            <Input label="Confirm new password" type="password" value={pw.confirm} autoComplete="new-password"
              onChange={e => setPw(p => ({ ...p, confirm: e.target.value }))} />
          </div>
          <button disabled={pwBusy} className="btn-primary">{pwBusy ? <Spinner className="h-4 w-4" /> : 'Update password'}</button>
        </form>
      </Card>

      <Card className="p-5">
        <h3 className="font-semibold flex items-center gap-2"><Icon d={P.cpu} className="h-4 w-4 text-brand" /> API access</h3>
        <p className="text-sm text-mute mt-1.5 leading-relaxed">
          The model API itself is open on localhost (the exact contract of the original
          service) — this key identifies your demo integrations and rotates locally.
        </p>
        <div className="mt-4 flex gap-2.5">
          <div className="field flex-1 num text-xs flex items-center !cursor-default">
            <span className="truncate">{revealed ? apiKey : 'dg_live_' + '•'.repeat(24)}</span>
          </div>
          <button type="button" onClick={() => setRevealed(r => !r)} className="btn-ghost !px-3"
            aria-label={revealed ? 'Hide' : 'Reveal'}>
            <Icon d={P.eye} className="h-4 w-4" />
          </button>
          <button type="button" onClick={copyKey} className="btn-ghost !px-3" aria-label="Copy">
            <Icon d={P.copy} className="h-4 w-4" />
          </button>
        </div>
        <div className="mt-4 flex items-center justify-between gap-4 flex-wrap">
          <p className="text-[11px] text-faint num">
            Created {new Date(user?.joined || Date.now()).toLocaleDateString()}
          </p>
          <button onClick={regenKey} disabled={keyBusy} className="btn-soft">
            {keyBusy ? <Spinner className="h-4 w-4" /> : 'Regenerate key'}
          </button>
        </div>
        <div className="mt-4 rounded-xl bg-panel2 border border-line p-3.5 text-xs text-mute">
          <p className="font-semibold text-ink mb-1.5 num">Example call (real endpoint)</p>
          <pre className="overflow-x-auto leading-relaxed">{`curl -X POST http://127.0.0.1:8000/predict \\
  -F "file=@image.jpg"`}</pre>
          <p className="mt-2 text-[11px] text-faint leading-relaxed">
            Response: label, prob_real, prob_fake, confidence, logit, threshold,
            fusion_weights, latency, model_version, image_size — the exact original contract.
          </p>
        </div>
      </Card>
    </div>
  )
}

import { useState } from 'react'
import { resetDemoData, updateSettings, getDB } from '../../api/client.js'
import { useToast } from '../../context/ToastContext.jsx'
import { useAuth } from '../../context/AuthContext.jsx'
import { useNotifications } from '../../context/NotificationsContext.jsx'
import { Icon, P, Card, Badge, Input, Select, Toggle, ConfirmDialog } from '../../components/ui.jsx'

export default function AdminSettings() {
  const pushToast = useToast()
  const { user, setUser } = useAuth()
  const { notify } = useNotifications()
  const [s, setS] = useState(() => ({ ...getDB().settings }))
  const [thresholdBusy, setThresholdBusy] = useState(false)
  const [resetOpen, setResetOpen] = useState(false)

  const save = (patch, toastMsg) => {
    const next = updateSettings(patch)
    setS({ ...next })
    if (toastMsg) pushToast(toastMsg[0], toastMsg[1], 'ok')
  }

  const saveThreshold = async () => {
    setThresholdBusy(true)
    try {
      await new Promise(r => setTimeout(r, 250))
      save({ threshold: s.threshold }, ['Threshold updated', `The verdict line now uses P(fake) ≥ ${s.threshold.toFixed(2)}.`])
      notify('system', 'info', 'Decision threshold updated',
        `New platform threshold: ${s.threshold.toFixed(2)} — verdicts re-evaluate from the next analysis.`, '/settings')
    } finally { setThresholdBusy(false) }
  }

  const doReset = () => {
    const db = resetDemoData()
    setS({ ...db.settings })
    const fresh = db.users.find(u => u.email === 'admin@deepguard.io')
    setUser(fresh)
    setResetOpen(false)
    pushToast('Demo data reset', 'Accounts, history, logs and settings were reseeded.', 'brand')
  }

  return (
    <div className="space-y-5 max-w-3xl">
      <div>
        <h2 className="text-xl font-bold tracking-tight">Platform settings</h2>
        <p className="text-sm text-mute mt-1">
          Applies to this browser's demo store; the model's own threshold stays in
          <span className="num"> backend/.env</span> (DECISION_THRESHOLD) — the verdict line uses the local value.
        </p>
      </div>

      {/* Decision threshold */}
      <Card className="p-5">
        <div className="flex items-center gap-2.5 mb-4">
          <span className="h-9 w-9 rounded-xl bg-brand/12 text-brand border border-brand/25 grid place-items-center">
            <Icon d={P.target} className="h-4.5 w-4.5" />
          </span>
          <div>
            <h3 className="font-semibold">Decision threshold</h3>
            <p className="text-[11px] text-mute">P(fake) ≥ threshold ⇒ FAKE (default 0.50 = the server default)</p>
          </div>
          <Badge tone="brand" >{s.threshold.toFixed(2)}</Badge>
        </div>
        <div className="flex items-center gap-4">
          <input type="range" min="0.05" max="0.95" step="0.05" value={s.threshold}
            onChange={e => setS(v => ({ ...v, threshold: +e.target.value }))}
            className="flex-1 accent-[#3b6bff]" aria-label="Decision threshold" />
          <span className="num text-sm font-bold w-12 text-right">{s.threshold.toFixed(2)}</span>
          <button onClick={saveThreshold} disabled={thresholdBusy} className="btn-primary !py-2">Save</button>
        </div>
        <div className="grid grid-cols-3 gap-2 mt-4 text-center text-[11px] text-mute">
          <div className="rounded-lg bg-panel2 border border-line py-2 num">0.30 · aggressive</div>
          <div className="rounded-lg bg-panel2 border border-line py-2 num">0.50 · balanced</div>
          <div className="rounded-lg bg-panel2 border border-line py-2 num">0.80 · conservative</div>
        </div>
      </Card>

      {/* Grad-CAM */}
      <Card className="p-5">
        <div className="flex items-center gap-2.5 mb-4">
          <span className="h-9 w-9 rounded-xl bg-violet/12 text-violet border border-violet/25 grid place-items-center">
            <Icon d={P.eye} className="h-4.5 w-4.5" />
          </span>
          <div>
            <h3 className="font-semibold">Explainability</h3>
            <p className="text-[11px] text-mute">Default Grad-CAM overlay opacity on the workbench</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <input type="range" min="0.1" max="0.9" step="0.05" value={s.gradcamOpacity}
            onChange={e => save({ gradcamOpacity: +e.target.value })}
            className="flex-1 accent-[#8b5cf6]" aria-label="Grad-CAM opacity" />
          <span className="num text-sm font-bold w-12 text-right">{Math.round(s.gradcamOpacity * 100)}%</span>
        </div>
      </Card>

      {/* Platform toggles */}
      <Card className="p-5">
        <h3 className="font-semibold mb-1">Platform</h3>
        <div className="divide-y divide-line/60">
          <Toggle checked={s.maintenance} onChange={v => save({ maintenance: v },
            ['Maintenance mode', v ? 'Sign-ups and analysis toasts paused for non-admins.' : 'Maintenance mode off.'])}
            label="Maintenance mode" description="Blocks new sign-ups and warns analysts in the workspace." />
          <Toggle checked={s.signupOpen} onChange={v => save({ signupOpen: v })}
            label="Open sign-ups" description="Allow visitors to create analyst accounts." />
          <Toggle checked={s.twoFA} onChange={v => save({ twoFA: v })}
            label="Two-factor authentication (advisory)" description="Marked on the login card — demo store only." />
        </div>
        <div className="grid sm:grid-cols-2 gap-4 mt-4">
          <Input label="Minimum password length" type="number" min={6} max={32} value={s.minPwd}
            onChange={e => save({ minPwd: clampInt(e.target.value, 6, 32) })} className="num" />
          <Input label="Session timeout (minutes)" type="number" min={5} max={480} value={s.sessionTimeout}
            onChange={e => save({ sessionTimeout: clampInt(e.target.value, 5, 480) })} className="num" />
        </div>
      </Card>

      {/* Data */}
      <Card className="p-5">
        <h3 className="font-semibold mb-1">Demo data</h3>
        <p className="text-sm text-mute leading-relaxed">
          Reseed the demo store: accounts (admin@deepguard.io / demo@deepguard.io), history,
          notifications and audit logs return to their initial state. The model API is untouched.
        </p>
        <button onClick={() => setResetOpen(true)} className="btn-danger mt-4">
          <Icon d={P.refresh} className="h-4 w-4" /> Reset demo data
        </button>
      </Card>

      <ConfirmDialog open={resetOpen} onClose={() => setResetOpen(false)} onConfirm={doReset}
        title="Reset demo data" confirmLabel="Reset"
        message="All accounts, analyses, notifications and logs in this browser will be reseeded. You stay signed in as the admin. This cannot be undone." />
    </div>
  )
}

function clampInt(v, lo, hi) {
  const n = parseInt(v, 10)
  if (Number.isNaN(n)) return lo
  return Math.min(hi, Math.max(lo, n))
}

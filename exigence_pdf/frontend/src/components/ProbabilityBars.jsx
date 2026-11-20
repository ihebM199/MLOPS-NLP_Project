/**
 * Animated P(real) / P(fake) bars + verdict.
 * Colours are doubled by text so the verdict never relies on colour alone.
 */

const pct = (value) => `${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)} %`

export default function ProbabilityBars({ result, threshold = 0.5 }) {
  const probReal = result?.prob_real ?? 0
  const probFake = result?.prob_fake ?? 0
  const isFake = (result?.label || 'real') === 'fake'
  const confidence = result?.confidence ?? Math.max(probReal, probFake)
  const ready = Boolean(result)

  return (
    <section className="card p-5" aria-labelledby="probabilities-title">
      <header className="mb-4 flex items-center justify-between">
        <h2 id="probabilities-title" className="card-title">
          Probabilités
        </h2>
        {ready && (
          <span className="chip border-slate-300/60 bg-slate-100 text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-300">
            confiance {pct(confidence)}
          </span>
        )}
      </header>

      <div
        className={`mb-5 flex items-center justify-between rounded-xl border px-4 py-3 transition-colors ${
          !ready
            ? 'border-slate-200 bg-slate-50 text-slate-400 dark:border-white/10 dark:bg-white/5'
            : isFake
              ? 'border-rose-400/40 bg-rose-400/10 text-rose-600 dark:text-rose-300'
              : 'border-emerald-400/40 bg-emerald-400/10 text-emerald-600 dark:text-emerald-300'
        }`}
        aria-live="polite"
      >
        <span className="text-xs font-semibold uppercase tracking-[0.2em]">Verdict</span>
        <span className="text-2xl font-black tracking-tight">
          {ready ? (isFake ? 'FAKE' : 'REAL') : '—'}
        </span>
      </div>

      <div className="space-y-4">
        <Bar
          label="Réelle"
          value={probReal}
          tone="emerald"
          testId="bar-real"
          active={ready && !isFake}
        />
        <div>
          <Bar label="Générée (fake)" value={probFake} tone="rose" testId="bar-fake" active={ready && isFake}>
            <span
              className="pointer-events-none absolute -top-1 bottom-0 w-px bg-slate-500/70 dark:bg-slate-300/70"
              style={{ left: `${Math.max(0, Math.min(1, threshold)) * 100}%` }}
              title={`Seuil de décision : ${pct(threshold)}`}
              aria-hidden="true"
            />
          </Bar>
          <p className="mt-1 text-right text-[11px] text-slate-400">
            seuil P(fake) ≥ {pct(threshold)}
          </p>
        </div>
      </div>
    </section>
  )
}

function Bar({ label, value, tone, testId, active, children }) {
  const colors =
    tone === 'rose'
      ? 'from-rose-500 to-rose-400'
      : 'from-emerald-500 to-emerald-400'
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between text-sm">
        <span className={`font-medium ${active ? 'text-slate-800 dark:text-white' : 'text-slate-500 dark:text-slate-400'}`}>
          {label}
        </span>
        <span className="font-mono text-xs tabular-nums text-slate-500 dark:text-slate-400">
          {pct(value)}
        </span>
      </div>
      <div
        className="relative h-2.5 overflow-hidden rounded-full bg-slate-200 dark:bg-ink-700"
        role="progressbar"
        aria-label={`Probabilité ${label}`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(value * 1000) / 10}
        data-testid={testId}
      >
        <div
          className={`h-full rounded-full bg-gradient-to-r ${colors} transition-[width] duration-700 ease-out`}
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        />
        {children}
      </div>
    </div>
  )
}

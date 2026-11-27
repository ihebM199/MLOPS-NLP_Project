/** Client round-trip vs server timings, with the derived network delta. */

const ms = (value) => `${Number(value ?? 0).toFixed(1)} ms`

function tone(value) {
  if (value < 150) return 'text-emerald-600 dark:text-emerald-300'
  if (value < 400) return 'text-amber-600 dark:text-amber-300'
  return 'text-rose-600 dark:text-rose-300'
}

export default function LatencyBadge({ result, clientMs }) {
  const server = result?.latency?.total_server_ms ?? 0
  const inference = result?.latency?.inference_ms ?? 0
  const preprocess = result?.latency?.preprocess_ms ?? 0
  const network = Math.max(0, Number(clientMs ?? 0) - server)

  return (
    <section className="card p-5" aria-labelledby="latency-title">
      <header className="mb-3 flex items-center justify-between">
        <h2 id="latency-title" className="card-title">
          Latences
        </h2>
        <span
          className={`chip border-current/30 bg-white/40 dark:bg-white/5 ${tone(clientMs ?? 0)}`}
        >
          aller-retour {ms(clientMs)}
        </span>
      </header>

      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric label="Client (RTT)" value={clientMs} hint="performance.now()" />
        <Metric label="Serveur total" value={server} hint="preprocess + inférence" />
        <Metric label="Inférence" value={inference} hint="ONNX Runtime" />
        <Metric label="Réseau" value={network} hint="RTT − serveur" />
      </dl>

      <p className="mt-3 text-xs text-slate-400">
        Prétraitement serveur : {ms(preprocess)}. Les valeurs sont mesurées, jamais simulées.
      </p>
    </section>
  )
}

function Metric({ label, value, hint }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 dark:border-white/10 dark:bg-white/5">
      <dt className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">{label}</dt>
      <dd className={`font-mono text-sm font-semibold tabular-nums ${tone(value ?? 0)}`}>{ms(value)}</dd>
      <p className="text-[10px] text-slate-400">{hint}</p>
    </div>
  )
}

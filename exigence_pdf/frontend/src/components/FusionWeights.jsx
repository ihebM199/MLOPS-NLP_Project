/** Stacked bar showing the adaptive gate output (spatial vs frequency). */

const pct = (value) => `${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)} %`

export default function FusionWeights({ weights }) {
  const spatial = weights?.spatial ?? 0
  const frequency = weights?.frequency ?? 0
  const total = spatial + frequency || 1

  return (
    <section className="card p-5" aria-labelledby="fusion-title">
      <header className="mb-3 flex items-center justify-between">
        <h2 id="fusion-title" className="card-title">
          Poids de fusion
        </h2>
        <span className="text-xs text-slate-400">sortie de la porte</span>
      </header>

      <div
        className="flex h-3 overflow-hidden rounded-full bg-slate-200 dark:bg-ink-700"
        role="img"
        aria-label={`Flux spatial ${pct(spatial / total)}, flux fréquentiel ${pct(frequency / total)}`}
      >
        <div
          className="h-full bg-gradient-to-r from-brand-500 to-brand-400 transition-[width] duration-700"
          style={{ width: `${(spatial / total) * 100}%` }}
        />
        <div
          className="h-full bg-gradient-to-r from-violet-500 to-fuchsia-400 transition-[width] duration-700"
          style={{ width: `${(frequency / total) * 100}%` }}
        />
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-2 text-sm">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-brand-500" aria-hidden="true" />
          <dt className="text-slate-500 dark:text-slate-400">Spatial</dt>
          <dd className="ml-auto font-mono text-xs tabular-nums font-semibold">{pct(spatial / total)}</dd>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-violet-500" aria-hidden="true" />
          <dt className="text-slate-500 dark:text-slate-400">Fréquentiel</dt>
          <dd className="ml-auto font-mono text-xs tabular-nums font-semibold">{pct(frequency / total)}</dd>
        </div>
      </dl>

      <p className="mt-3 text-xs leading-relaxed text-slate-400">
        Le flux fréquentiel (FFT) porte l&apos;analyse des artefacts de compression et des hautes fréquences ;
        la porte adaptative choisit sa contribution image par image.
      </p>
    </section>
  )
}

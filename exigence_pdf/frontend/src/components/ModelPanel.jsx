/** "Model" panel: /model/metadata + /health, no metric is ever invented. */

const fmtPercent = (value) =>
  typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)} %` : null

const fmtNumber = (value) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('fr-FR') : null

export default function ModelPanel({ metadata, health, loading }) {
  const test = metadata?.metrics?.test || {}
  const crossDataset = metadata?.metrics?.cross_dataset || null
  const rows = [
    ['Accuracy', fmtPercent(test.accuracy ?? test.acc)],
    ['F1 (fake)', fmtPercent(test.f1 ?? test.f1_fake)],
    ['AUC', test.auc != null ? Number(test.auc).toFixed(4) : null],
    ['EER', test.eer != null ? fmtPercent(test.eer) : null],
  ]

  return (
    <section className="card p-5" aria-labelledby="model-title">
      <header className="mb-3 flex items-center justify-between">
        <h2 id="model-title" className="card-title">
          Modèle
        </h2>
        {loading ? (
          <span className="text-xs text-slate-400">chargement…</span>
        ) : (
          <span className="font-mono text-xs text-slate-400">{metadata?.model_version || '—'}</span>
        )}
      </header>

      <dl className="space-y-1.5 text-sm">
        <Row label="Architecture" value={metadata?.architecture || 'AdaptiveFusionDetector'} />
        <Row label="Backbone" value={metadata?.backbone || '—'} mono />
        <Row
          label="Paramètres"
          value={fmtNumber(metadata?.parameters) || '—'}
          mono
        />
        <Row
          label="Classes"
          value={metadata?.classes ? `${metadata.classes['0']} = 0 · ${metadata.classes['1']} = 1` : 'real = 0 · fake = 1'}
        />
        <Row label="Règle" value={metadata?.decision_rule || 'logit > 0 ⇒ FAKE'} mono />
        <Row label="Seuil" value={metadata?.threshold != null ? metadata.threshold.toFixed(2) : '—'} mono />
        <Row
          label="ONNX Runtime"
          value={(health?.onnxruntime_providers || []).join(', ') || '—'}
          mono
        />
        <Row
          label="Grad-CAM (PyTorch)"
          value={health ? (health.gradcam_available ? 'disponible' : 'indisponible') : '—'}
        />
      </dl>

      <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50/70 p-3 dark:border-white/10 dark:bg-white/5">
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
          Métriques de test <span className="normal-case tracking-normal text-slate-400">(test 140k)</span>
        </h3>
        {rows.every(([, value]) => value === null) ? (
          <p className="text-xs text-slate-400">
            Aucune métrique disponible ici — lancez <code className="font-mono">python -m src.models.evaluate</code>{' '}
            sur Colab puis redémarrez l&apos;API (les valeurs ne sont jamais simulées).
          </p>
        ) : (
          <dl className="grid grid-cols-2 gap-2 text-sm">
            {rows.map(([label, value]) => (
              <div key={label} className="flex items-baseline justify-between">
                <dt className="text-slate-500 dark:text-slate-400">{label}</dt>
                <dd className="font-mono text-xs font-semibold tabular-nums">{value || '—'}</dd>
              </div>
            ))}
          </dl>
        )}
        {crossDataset && Object.keys(crossDataset).length > 0 ? (
          <p className="mt-2 text-[11px] text-slate-400">
            Cross-dataset : AUC {crossDataset.auc != null ? Number(crossDataset.auc).toFixed(4) : '—'} ·
            accuracy {fmtPercent(crossDataset.accuracy) || '—'}
          </p>
        ) : (
          <p className="mt-2 text-[11px] text-slate-400">
            Cross-dataset : non évalué (remplir <code className="font-mono">data/external</code> puis{' '}
            <code className="font-mono">eval.cross_dataset: true</code>).
          </p>
        )}
      </div>

      {metadata?.limitations?.length > 0 && (
        <div className="mt-4 rounded-xl border border-amber-400/40 bg-amber-400/10 p-3">
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-300">
            Limites connues
          </h3>
          <ul className="list-disc space-y-1 pl-4 text-xs leading-relaxed text-amber-700 dark:text-amber-200/90">
            {metadata.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}

function Row({ label, value, mono = false }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="shrink-0 text-slate-500 dark:text-slate-400">{label}</dt>
      <dd className={`text-right text-xs font-semibold ${mono ? 'font-mono' : ''}`}>{value}</dd>
    </div>
  )
}

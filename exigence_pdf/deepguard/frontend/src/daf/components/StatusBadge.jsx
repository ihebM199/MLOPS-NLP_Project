const VARIANTS = {
  ok: {
    label: 'API en ligne',
    className: 'border-emerald-400/40 bg-emerald-400/10 text-emerald-600 dark:text-emerald-300',
    dot: 'bg-emerald-500',
  },
  degraded: {
    label: 'Modèle indisponible',
    className: 'border-amber-400/40 bg-amber-400/10 text-amber-600 dark:text-amber-300',
    dot: 'bg-amber-500',
  },
  offline: {
    label: 'API hors ligne',
    className: 'border-rose-400/40 bg-rose-400/10 text-rose-600 dark:text-rose-300',
    dot: 'bg-rose-500',
  },
  loading: {
    label: 'Connexion…',
    className: 'border-slate-400/40 bg-slate-400/10 text-slate-500 dark:text-slate-300',
    dot: 'bg-slate-400',
  },
}

export default function StatusBadge({ status = 'loading', detail }) {
  const variant = VARIANTS[status] || VARIANTS.loading
  return (
    <span
      className={`chip ${variant.className}`}
      title={detail || variant.label}
      aria-label={`État du service : ${variant.label}`}
    >
      <span
        className={`h-2 w-2 rounded-full ${variant.dot} ${
          status === 'ok' ? 'motion-safe:animate-pulse' : ''
        }`}
        aria-hidden="true"
      />
      {variant.label}
    </span>
  )
}

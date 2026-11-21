import { useState } from 'react'

import { EXAMPLES } from '../data/examples.js'

/**
 * Pre-loaded 1-click examples.
 *
 * The 8 images (4 real / 4 fake) are extracted from the official *test* split
 * of the 140k dataset by `scripts/export_examples.py` (they are therefore
 * never seen during training).  Until that script has been run on a machine
 * with the dataset, tiles gracefully degrade to a disabled placeholder.
 */

export default function ExampleGallery({ onSelect, onError, disabled = false }) {
  const [missing, setMissing] = useState({})
  const [loadingId, setLoadingId] = useState(null)

  const pick = async (example) => {
    if (disabled || loadingId) return
    setLoadingId(example.id)
    try {
      const response = await fetch(example.src)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const blob = await response.blob()
      const file = new File([blob], `${example.id}.jpg`, { type: blob.type || 'image/jpeg' })
      onError?.(null)
      onSelect?.(file)
    } catch {
      setMissing((current) => ({ ...current, [example.id]: true }))
      onError?.(
        "Exemple indisponible : générez la galerie avec `python scripts/export_examples.py` (nécessite le dataset).",
      )
    } finally {
      setLoadingId(null)
    }
  }

  return (
    <div>
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4" aria-label="Galerie d'exemples">
        {EXAMPLES.map((example) => {
          const unavailable = Boolean(missing[example.id])
          const isFake = example.label === 'Générée'
          return (
            <li key={example.id}>
              <button
                type="button"
                onClick={() => pick(example)}
                disabled={disabled || unavailable || loadingId !== null}
                className={`group relative block w-full overflow-hidden rounded-xl border transition ${
                  unavailable
                    ? 'cursor-not-allowed border-slate-200 bg-slate-100 dark:border-white/5 dark:bg-ink-800'
                    : 'border-slate-200 hover:border-brand-400 hover:shadow-glow dark:border-white/10 dark:bg-ink-800'
                }`}
                title={unavailable ? 'Fichier manquant — lancez scripts/export_examples.py' : `Tester l'exemple ${example.label.toLowerCase()}`}
              >
                {unavailable ? (
                  <span className="flex aspect-square flex-col items-center justify-center gap-1 p-2 text-center text-[10px] text-slate-400">
                    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.5-6 3.5 4.5L15 10l5 6M4 20h16a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1Z" />
                    </svg>
                    exemple à générer
                  </span>
                ) : (
                  <img
                    src={example.src}
                    alt={`Exemple ${example.label.toLowerCase()}`}
                    loading="lazy"
                    className="aspect-square w-full object-cover transition duration-300 group-hover:scale-[1.04]"
                    onError={() => setMissing((current) => ({ ...current, [example.id]: true }))}
                  />
                )}
                <span
                  className={`absolute left-1.5 top-1.5 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
                    isFake ? 'bg-rose-600/90 text-white' : 'bg-emerald-600/90 text-white'
                  }`}
                >
                  {example.label}
                </span>
                {loadingId === example.id && (
                  <span className="absolute inset-0 flex items-center justify-center bg-slate-900/50 text-xs font-semibold text-white">
                    envoi…
                  </span>
                )}
              </button>
            </li>
          )
        })}
      </ul>
      <p className="mt-3 text-[11px] leading-relaxed text-slate-400">
        Exemples issus du split <strong>test</strong> du dataset 140k (jamais vus à l&apos;entraînement),
        exportés par <code className="font-mono">scripts/export_examples.py</code>.
      </p>
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'

/**
 * Grad-CAM viewer: draws the original image on a <canvas>, then blends the
 * server-side heatmap overlay (base64 PNG) with a client-side opacity slider.
 * The overlay is requested with alpha = 1 so the slider controls everything.
 */
export default function GradCamCanvas({
  sourceUrl,
  overlayBase64,
  target,
  alpha,
  loading,
  error,
  disabled = false,
  onTargetChange,
  onAlphaChange,
  onGenerate,
}) {
  const canvasRef = useRef(null)
  const [view, setView] = useState('overlay')

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    context.clearRect(0, 0, canvas.width, canvas.height)

    const original = new Image()
    if (!sourceUrl) {
      context.fillStyle = 'rgb(15, 23, 42)'
      context.fillRect(0, 0, canvas.width, canvas.height)
      return
    }
    let cancelled = false
    original.onload = () => {
      if (cancelled) return
      context.drawImage(original, 0, 0, canvas.width, canvas.height)
      if (view === 'overlay' && overlayBase64) {
        const overlay = new Image()
        overlay.onload = () => {
          if (cancelled) return
          context.save()
          context.globalAlpha = alpha
          context.drawImage(overlay, 0, 0, canvas.width, canvas.height)
          context.restore()
        }
        overlay.src = `data:image/png;base64,${overlayBase64}`
      }
    }
    original.src = sourceUrl
    return () => {
      cancelled = true
    }
  }, [sourceUrl, overlayBase64, alpha, view])

  return (
    <section className="card p-5" aria-labelledby="gradcam-title">
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 id="gradcam-title" className="card-title">
          Grad-CAM <span className="normal-case tracking-normal text-slate-400">(flux spatial)</span>
        </h2>
        <div className="flex items-center gap-2">
          <div className="flex rounded-lg border border-slate-200 p-0.5 dark:border-white/10" role="group" aria-label="Classe expliquée">
            {['fake', 'real'].map((value) => (
              <button
                key={value}
                type="button"
                disabled={disabled || loading}
                onClick={() => onTargetChange?.(value)}
                aria-pressed={target === value}
                className={`rounded-md px-3 py-1 text-xs font-semibold transition ${
                  target === value
                    ? 'bg-brand-600 text-white'
                    : 'text-slate-500 hover:text-brand-600 dark:text-slate-400'
                }`}
              >
                {value === 'fake' ? 'FAKE' : 'REAL'}
              </button>
            ))}
          </div>
          <div className="flex rounded-lg border border-slate-200 p-0.5 dark:border-white/10" role="group" aria-label="Vue">
            {[
              ['overlay', 'Overlay'],
              ['original', 'Original'],
            ].map(([value, label]) => (
              <button
                key={value}
                type="button"
                disabled={!overlayBase64}
                onClick={() => setView(value)}
                aria-pressed={view === value}
                className={`rounded-md px-3 py-1 text-xs font-semibold transition disabled:opacity-40 ${
                  view === value
                    ? 'bg-brand-600 text-white'
                    : 'text-slate-500 hover:text-brand-600 dark:text-slate-400'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className="flex justify-center">
        <canvas
          ref={canvasRef}
          width={384}
          height={384}
          className="max-h-96 w-full max-w-sm rounded-xl border border-slate-200 bg-slate-900 object-contain dark:border-white/10"
          aria-label="Carte Grad-CAM superposée à l'image"
        />
      </div>

      <div className="mt-4 space-y-3">
        <label className="flex items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
          <span className="w-14 font-semibold uppercase tracking-wider">Opacité</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={alpha}
            disabled={!overlayBase64}
            onChange={(event) => onAlphaChange?.(Number(event.target.value))}
            className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-slate-200 accent-brand-500 dark:bg-ink-700"
            aria-label="Opacité de la heatmap"
          />
          <span className="w-10 text-right font-mono tabular-nums">{alpha.toFixed(2)}</span>
        </label>

        <button
          type="button"
          className="btn-primary w-full"
          disabled={disabled || loading}
          onClick={() => onGenerate?.()}
        >
          {loading ? 'Calcul en cours…' : overlayBase64 ? 'Recalculer la carte' : 'Générer la carte Grad-CAM'}
        </button>

        {loading && (
          <p className="text-center text-xs text-slate-400">
            Grad-CAM tourne en PyTorch (CPU, chargé paresseusement) — quelques secondes au premier appel.
          </p>
        )}
        {error && (
          <p role="alert" className="rounded-lg bg-rose-500/10 px-3 py-2 text-xs text-rose-600 dark:text-rose-300">
            {error}
          </p>
        )}
      </div>
    </section>
  )
}

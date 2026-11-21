import { useCallback, useEffect, useRef, useState } from 'react'
import Dropzone from './components/Dropzone.jsx'
import ExampleGallery from './components/ExampleGallery.jsx'
import FusionWeights from './components/FusionWeights.jsx'
import GradCamCanvas from './components/GradCamCanvas.jsx'
import LatencyBadge from './components/LatencyBadge.jsx'
import ModelPanel from './components/ModelPanel.jsx'
import ProbabilityBars from './components/ProbabilityBars.jsx'
import Spinner from './components/Spinner.jsx'
import StatusBadge from './components/StatusBadge.jsx'
import Webcam from './components/Webcam.jsx'
import { getHealth, getMetadata, predict, predictVisualize } from './services/api.js'

const INGESTION_MODES = [
  { id: 'upload', label: 'Importer' },
  { id: 'examples', label: 'Exemples' },
  { id: 'webcam', label: 'Webcam' },
]

function useStoredTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('daf-theme') || 'dark'
    } catch {
      return 'dark'
    }
  })

  useEffect(() => {
    const dark = theme === 'dark'
    document.documentElement.classList.toggle('dark', dark)
    try {
      localStorage.setItem('daf-theme', theme)
    } catch {
      /* private mode: the theme simply is not persisted */
    }
  }, [theme])

  return [theme, setTheme]
}

export default function App() {
  const [theme, setTheme] = useStoredTheme()
  const [health, setHealth] = useState(null)
  const [healthStatus, setHealthStatus] = useState('loading')
  const [metadata, setMetadata] = useState(null)
  const [metadataLoading, setMetadataLoading] = useState(true)

  const [ingestion, setIngestion] = useState('upload')
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [clientMs, setClientMs] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const [gradcamTarget, setGradcamTarget] = useState('fake')
  const [gradcamAlpha, setGradcamAlpha] = useState(0.45)
  const [overlay, setOverlay] = useState(null)
  const [gradcamLoading, setGradcamLoading] = useState(false)
  const [gradcamError, setGradcamError] = useState(null)

  const previewRef = useRef(null)
  const predictRequestRef = useRef(0)
  const gradcamRequestRef = useRef(0)

  useEffect(
    () => () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    },
    [],
  )

  const refreshHealth = useCallback(async () => {
    try {
      const { data, degraded } = await getHealth()
      setHealth(data)
      setHealthStatus(degraded ? 'degraded' : 'ok')
    } catch {
      setHealth(null)
      setHealthStatus('offline')
    }
  }, [])

  useEffect(() => {
    refreshHealth()
    const timer = setInterval(refreshHealth, 20000)
    return () => clearInterval(timer)
  }, [refreshHealth])

  useEffect(() => {
    let active = true
    getMetadata()
      .then(({ data }) => {
        if (active) setMetadata(data)
      })
      .catch(() => {
        if (active) setMetadata(null)
      })
      .finally(() => {
        if (active) setMetadataLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  const updatePreview = useCallback((nextFile) => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    const url = URL.createObjectURL(nextFile)
    previewRef.current = url
    setPreviewUrl(url)
  }, [])

  const handleFile = useCallback(
    async (nextFile) => {
      if (!nextFile) return
      const requestId = predictRequestRef.current + 1
      predictRequestRef.current = requestId
      gradcamRequestRef.current += 1

      setFile(nextFile)
      updatePreview(nextFile)
      setResult(null)
      setClientMs(null)
      setOverlay(null)
      setGradcamError(null)
      setError(null)
      setBusy(true)
      try {
        const { data, clientMs: roundTripMs } = await predict(nextFile)
        if (predictRequestRef.current !== requestId) return
        setResult(data)
        setClientMs(roundTripMs)
      } catch (cause) {
        if (predictRequestRef.current !== requestId) return
        setError(cause.message || 'La prédiction a échoué.')
      } finally {
        if (predictRequestRef.current === requestId) setBusy(false)
      }
    },
    [updatePreview],
  )

  const generateGradcam = useCallback(async () => {
    if (!file) {
      setGradcamError('Importez une image avant de générer la carte Grad-CAM.')
      return
    }
    const requestId = gradcamRequestRef.current + 1
    gradcamRequestRef.current = requestId
    setGradcamLoading(true)
    setGradcamError(null)
    try {
      const { data } = await predictVisualize(file, {
        target: gradcamTarget,
        alpha: 1,
        format: 'json',
      })
      if (gradcamRequestRef.current !== requestId) return
      setOverlay(data.image_base64)
    } catch (cause) {
      if (gradcamRequestRef.current !== requestId) return
      setGradcamError(cause.message || 'Grad-CAM indisponible.')
    } finally {
      if (gradcamRequestRef.current === requestId) setGradcamLoading(false)
    }
  }, [file, gradcamTarget])

  const threshold = metadata?.threshold ?? result?.threshold ?? 0.5

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
      <header className="animate-fade-up">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-brand-600 text-white shadow-glow">
              <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 3l7 3v6c0 4.2-2.9 7.5-7 9-4.1-1.5-7-4.8-7-9V6l7-3Z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="m9 12 2 2 4-4" />
              </svg>
            </span>
            <div>
              <h1 className="text-lg font-black tracking-tight sm:text-xl">Deepfake Adaptive Fusion</h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Fusion spatiale (CNN) + fréquentielle (FFT) · Grad-CAM · ONNX Runtime
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <StatusBadge status={healthStatus} detail={health?.detail || undefined} />
            <button
              type="button"
              className="btn-ghost px-3 py-1.5 text-xs"
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              aria-label={theme === 'dark' ? 'Activer le thème clair' : 'Activer le thème sombre'}
            >
              {theme === 'dark' ? 'Thème clair' : 'Thème sombre'}
            </button>
          </div>
        </div>

        <p className="mt-3 rounded-xl border border-amber-400/40 bg-amber-400/10 px-4 py-2.5 text-xs leading-relaxed text-amber-700 dark:text-amber-200/90">
          Outil de recherche : la généralisation à des générateurs inconnus est limitée. Ne jamais
          utiliser ce résultat comme unique preuve de l&apos;authenticité d&apos;un contenu.
        </p>
      </header>

      {error && (
        <div role="alert" className="animate-fade-up rounded-xl border border-rose-400/40 bg-rose-400/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-200">
          {error}
        </div>
      )}

      <div className="grid flex-1 gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <div className="space-y-6">
          <section className="card animate-fade-up p-5" aria-labelledby="ingestion-title">
            <header className="mb-4 flex flex-wrap items-center justify-between gap-2">
              <h2 id="ingestion-title" className="card-title">
                Image à analyser
              </h2>
              <div className="flex rounded-lg border border-slate-200 p-0.5 dark:border-white/10" role="tablist" aria-label="Mode d'import">
                {INGESTION_MODES.map((mode) => (
                  <button
                    key={mode.id}
                    type="button"
                    role="tab"
                    aria-selected={ingestion === mode.id}
                    onClick={() => setIngestion(mode.id)}
                    className={`rounded-md px-3 py-1 text-xs font-semibold transition ${
                      ingestion === mode.id
                        ? 'bg-brand-600 text-white'
                        : 'text-slate-500 hover:text-brand-600 dark:text-slate-400'
                    }`}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>
            </header>

            {ingestion === 'upload' && (
              <Dropzone onFile={handleFile} onError={setError} previewUrl={previewUrl} disabled={busy} />
            )}
            {ingestion === 'examples' && (
              <ExampleGallery onSelect={handleFile} onError={setError} disabled={busy} />
            )}
            {ingestion === 'webcam' && <Webcam onCapture={handleFile} busy={busy} />}

            {busy && (
              <div className="mt-4 flex items-center justify-between rounded-xl bg-slate-100 px-4 py-3 dark:bg-white/5">
                <Spinner label="Analyse de l'image…" />
                <span className="text-xs text-slate-400">ONNX Runtime</span>
              </div>
            )}
          </section>

          <div className="grid gap-6 sm:grid-cols-2">
            <ProbabilityBars result={result} threshold={threshold} />
            <FusionWeights weights={result?.fusion_weights} />
          </div>

          <LatencyBadge result={result} clientMs={clientMs} />

          <GradCamCanvas
            sourceUrl={previewUrl}
            overlayBase64={overlay}
            target={gradcamTarget}
            alpha={gradcamAlpha}
            loading={gradcamLoading}
            error={gradcamError}
            disabled={!file || busy}
            onTargetChange={setGradcamTarget}
            onAlphaChange={setGradcamAlpha}
            onGenerate={generateGradcam}
          />
        </div>

        <aside className="space-y-6">
          {metadataLoading ? (
            <section className="card p-5">
              <Spinner label="Chargement de la fiche modèle…" />
            </section>
          ) : (
            <ModelPanel metadata={metadata} health={health} loading={metadataLoading} />
          )}

          <section className="card p-5" aria-labelledby="about-title">
            <h2 id="about-title" className="card-title mb-3">
              Comment lire le résultat
            </h2>
            <ul className="space-y-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
              <li>
                <strong className="text-slate-700 dark:text-slate-200">Verdict</strong> : REAL si
                P(fake) &lt; seuil, FAKE sinon. Le seuil est calibré sur la validation.
              </li>
              <li>
                <strong className="text-slate-700 dark:text-slate-200">Poids de fusion</strong> : la
                porte adaptative indique la contribution de chaque flux pour cette image.
              </li>
              <li>
                <strong className="text-slate-700 dark:text-slate-200">Grad-CAM</strong> : calculé en
                PyTorch (chargé à la demande), donc plus lent que <code className="font-mono">/predict</code>.
              </li>
              <li>
                <strong className="text-slate-700 dark:text-slate-200">Latences</strong> : mesurées
                côté client et côté serveur, jamais simulées.
              </li>
            </ul>
          </section>
        </aside>
      </div>

      <footer className="pb-4 text-center text-[11px] text-slate-400">
        Projet MLOps — Axe 1 (CNN avancé, transfer learning, Grad-CAM). Données :
        140k real-and-fake-faces · API FastAPI · ONNX Runtime · DVC · MLflow · Docker.
      </footer>
    </div>
  )
}

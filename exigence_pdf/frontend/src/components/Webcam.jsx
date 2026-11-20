import { useEffect, useRef, useState } from 'react'

/**
 * Webcam capture + optional continuous inference (~1 img/s).
 * - one request in flight at a time (guarded by a ref, not React state);
 * - camera tracks always stopped on Stop and on unmount;
 * - permission denial handled explicitly.
 */
export default function Webcam({ onCapture, busy = false }) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const inFlightRef = useRef(false)
  const busyRef = useRef(busy)
  const onCaptureRef = useRef(onCapture)
  const [streaming, setStreaming] = useState(false)
  const [continuous, setContinuous] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    busyRef.current = busy
  }, [busy])

  useEffect(() => {
    onCaptureRef.current = onCapture
  }, [onCapture])

  const stopTracks = () => {
    const stream = streamRef.current
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) videoRef.current.srcObject = null
  }

  const stop = () => {
    setContinuous(false)
    setStreaming(false)
    stopTracks()
  }

  useEffect(() => stopTracks, [])

  const start = async () => {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Ce navigateur ne supporte pas l'accès à la caméra (contexte HTTPS requis).")
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play().catch(() => undefined)
      }
      setStreaming(true)
    } catch (cause) {
      const denied = cause?.name === 'NotAllowedError' || cause?.name === 'SecurityError'
      setError(
        denied
          ? "Accès à la caméra refusé. Autorisez la caméra dans les réglages du navigateur puis réessayez."
          : "Impossible d'ouvrir la caméra : " + (cause?.message || 'erreur inconnue'),
      )
      stopTracks()
    }
  }

  const grabFrame = () => {
    const video = videoRef.current
    if (!video || !video.videoWidth) return null
    const size = Math.min(video.videoWidth, video.videoHeight)
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const context = canvas.getContext('2d')
    context.drawImage(
      video,
      (video.videoWidth - size) / 2,
      (video.videoHeight - size) / 2,
      size,
      size,
      0,
      0,
      size,
      size,
    )
    return new Promise((resolve) => {
      canvas.toBlob(
        (blob) => {
          resolve(blob ? new File([blob], `webcam-${Date.now()}.jpg`, { type: 'image/jpeg' }) : null)
        },
        'image/jpeg',
        0.92,
      )
    })
  }

  const capture = async () => {
    if (inFlightRef.current || busyRef.current) return
    const file = await grabFrame()
    if (!file) return
    inFlightRef.current = true
    try {
      await onCaptureRef.current?.(file)
    } finally {
      inFlightRef.current = false
    }
  }

  useEffect(() => {
    if (!streaming || !continuous) return undefined
    const timer = setInterval(() => {
      if (!inFlightRef.current && !busyRef.current) capture()
    }, 1100)
    return () => clearInterval(timer)
  }, [streaming, continuous])

  return (
    <div className="space-y-3">
      <div className="relative overflow-hidden rounded-2xl border border-slate-200 bg-slate-950/90 dark:border-white/10">
        <video
          ref={videoRef}
          muted
          playsInline
          className={`aspect-video w-full object-cover ${streaming ? '' : 'opacity-40'}`}
          aria-label="Flux de la caméra"
        />
        {!streaming && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center text-slate-300">
            <svg viewBox="0 0 24 24" className="h-8 w-8" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.55-2.28A1 1 0 0 1 21 8.62v6.76a1 1 0 0 1-1.45.9L15 14M4 6h9a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2Z" />
            </svg>
            <p className="text-xs">Caméra éteinte</p>
          </div>
        )}
        {streaming && continuous && (
          <span className="absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-full bg-rose-600/90 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-white">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white" aria-hidden="true" />
            analyse continue
          </span>
        )}
      </div>

      {error && (
        <p role="alert" className="rounded-lg bg-rose-500/10 px-3 py-2 text-xs text-rose-600 dark:text-rose-300">
          {error}
        </p>
      )}

      <div className="flex flex-wrap gap-2">
        {!streaming ? (
          <button type="button" className="btn-primary" onClick={start}>
            Démarrer la caméra
          </button>
        ) : (
          <>
            <button type="button" className="btn-primary" onClick={capture} disabled={busy}>
              Capturer une photo
            </button>
            <button
              type="button"
              className="btn-ghost"
              onClick={() => setContinuous((value) => !value)}
              aria-pressed={continuous}
            >
              {continuous ? 'Arrêter le continu' : 'Inférence continue'}
            </button>
            <button type="button" className="btn-ghost" onClick={stop}>
              Couper la caméra
            </button>
          </>
        )}
      </div>
      <p className="text-[11px] leading-relaxed text-slate-400">
        Une seule requête est envoyée à la fois (~1 image/s en mode continu). Le flux caméra est coupé
        automatiquement quand vous quittez l&apos;onglet ou arrêtez l&apos;analyse.
      </p>
    </div>
  )
}

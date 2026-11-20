import { useRef, useState } from 'react'
import { MAX_UPLOAD_BYTES, validateFile } from '../services/api.js'

/** Drag & drop + file picker with client-side validation and preview. */
export default function Dropzone({ onFile, onError, previewUrl, disabled = false }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const accept = (file) => {
    try {
      validateFile(file)
      onError?.(null)
      onFile?.(file)
    } catch (error) {
      onError?.(error.message)
    }
  }

  const onDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    if (disabled) return
    const file = event.dataTransfer?.files?.[0]
    if (file) accept(file)
  }

  return (
    <div>
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault()
          if (!disabled) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`group relative flex w-full flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-8 text-center transition ${
          dragging
            ? 'border-brand-400 bg-brand-50/70 dark:bg-brand-500/10'
            : 'border-slate-300 bg-slate-50/60 hover:border-brand-400 hover:bg-brand-50/40 dark:border-white/15 dark:bg-white/5 dark:hover:border-brand-400 dark:hover:bg-brand-500/10'
        } ${disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}
        aria-label="Glisser-déposer une image ou ouvrir le sélecteur de fichier"
      >
        <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-brand-500/10 text-brand-600 transition group-hover:scale-105 dark:text-brand-300">
          <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0L8 8m4-4 4 4" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
          </svg>
        </span>
        <span className="text-sm font-semibold">
          Déposez une image ici
          <span className="block text-xs font-normal text-slate-500 dark:text-slate-400">
            ou cliquez pour parcourir — JPEG, PNG, WebP, BMP (10 Mo max)
          </span>
        </span>
        <span className="text-[11px] text-slate-400">
          {(MAX_UPLOAD_BYTES / 1024 / 1024).toFixed(0)} Mo maximum — analyse locale par le modèle ONNX
        </span>
      </button>

      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/bmp"
        className="sr-only"
        data-testid="file-input"
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) accept(file)
          event.target.value = ''
        }}
      />

      {previewUrl && (
        <figure className="mt-4 overflow-hidden rounded-xl border border-slate-200 dark:border-white/10">
          <img src={previewUrl} alt="Aperçu de l'image sélectionnée" className="max-h-64 w-full object-contain" />
        </figure>
      )}
    </div>
  )
}

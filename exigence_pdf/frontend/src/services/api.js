/**
 * Thin client for the Deepfake Adaptive Fusion API.
 *
 * Every call returns `{ data, clientMs }` where `clientMs` is the
 * browser-side round-trip time (`performance.now()`), so the UI can compare
 * it with the server timings and display the network delta.
 *
 * `VITE_API_URL` empty (default) means "same origin": on Colab the FastAPI
 * app serves both the API and the built SPA from one URL.
 */

export const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '')

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024
export const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']

const FALLBACK_MESSAGES = {
  400: "Image invalide, corrompue ou illisible.",
  413: 'Fichier trop volumineux (10 Mo maximum).',
  415: 'Format non supporté (JPEG, PNG, WebP ou BMP).',
  422: 'Requête invalide : aucun fichier reçu.',
  500: 'Erreur interne du serveur.',
  503: "Le modèle n'est pas chargé : réessayez dans quelques secondes.",
}

/** Error carrying the HTTP status and the machine-readable API code. */
export class ApiError extends Error {
  constructor(message, { status = 0, code = 'ERROR', detail = '' } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.detail = detail
  }
}

/** Validate a file before uploading (fast feedback, the API re-validates). */
export function validateFile(file) {
  if (!file) {
    throw new ApiError('Aucun fichier sélectionné.', { status: 0, code: 'NO_FILE' })
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new ApiError(
      `Fichier trop volumineux : ${(file.size / 1024 / 1024).toFixed(1)} Mo (10 Mo max).`,
      { status: 413, code: 'FILE_TOO_LARGE' },
    )
  }
  const byMime = file.type && ACCEPTED_TYPES.includes(file.type)
  const byName = /\.(jpe?g|png|webp|bmp)$/i.test(file.name || '')
  if (!byMime && !byName) {
    throw new ApiError('Format non supporté : utilisez JPEG, PNG, WebP ou BMP.', {
      status: 415,
      code: 'UNSUPPORTED_TYPE',
    })
  }
  return true
}

async function readJson(response) {
  try {
    return await response.json()
  } catch {
    return null
  }
}

async function timedFetch(input, init) {
  const started = performance.now()
  let response
  try {
    response = await fetch(input, init)
  } catch (cause) {
    throw new ApiError(
      "API injoignable. Vérifiez que le backend est démarré (port 8000) et l'URL VITE_API_URL.",
      { status: 0, code: 'NETWORK_ERROR', detail: String(cause) },
    )
  }
  const clientMs = Math.round((performance.now() - started) * 10) / 10
  return { response, clientMs }
}

async function toResult(response, clientMs) {
  const payload = await readJson(response)
  if (!response.ok) {
    const message =
      (payload && (payload.detail || payload.error)) ||
      FALLBACK_MESSAGES[response.status] ||
      `Erreur HTTP ${response.status}`
    const error = new ApiError(message, {
      status: response.status,
      code: (payload && payload.error) || `HTTP_${response.status}`,
      detail: (payload && payload.detail) || '',
    })
    error.payload = payload
    throw error
  }
  return { data: payload, clientMs }
}

function toFormData(file) {
  const form = new FormData()
  form.append('file', file, file.name || 'capture.jpg')
  return form
}

/** `GET /health` -- a 503 is returned as `degraded: true` instead of throwing. */
export async function getHealth() {
  const { response, clientMs } = await timedFetch(`${API_BASE}/health`, {
    headers: { Accept: 'application/json' },
  })
  if (response.status === 503) {
    return { data: await readJson(response), clientMs, degraded: true }
  }
  const result = await toResult(response, clientMs)
  return { ...result, degraded: false }
}

/** `GET /model/metadata` */
export async function getMetadata() {
  const { response, clientMs } = await timedFetch(`${API_BASE}/model/metadata`, {
    headers: { Accept: 'application/json' },
  })
  return toResult(response, clientMs)
}

/** `POST /predict` -> PredictionResponse */
export async function predict(file) {
  validateFile(file)
  const { response, clientMs } = await timedFetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: toFormData(file),
    headers: { Accept: 'application/json' },
  })
  return toResult(response, clientMs)
}

/** `POST /predict/visualize` -> JSON with a base64 PNG Grad-CAM overlay. */
export async function predictVisualize(file, { target = 'fake', alpha = 1, format = 'json' } = {}) {
  validateFile(file)
  const params = new URLSearchParams({
    target,
    alpha: String(Math.min(1, Math.max(0, alpha))),
    format,
  })
  const { response, clientMs } = await timedFetch(
    `${API_BASE}/predict/visualize?${params.toString()}`,
    {
      method: 'POST',
      body: toFormData(file),
      headers: { Accept: format === 'png' ? 'image/png' : 'application/json' },
    },
  )
  return toResult(response, clientMs)
}

import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, getHealth, predict, predictVisualize, validateFile } from './api.js'

const jsonResponse = (body, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => body,
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('validateFile', () => {
  it('accepts a supported JPEG', () => {
    expect(validateFile({ name: 'face.jpg', type: 'image/jpeg', size: 1024 })).toBe(true)
  })

  it('rejects an unsupported extension', () => {
    expect(() => validateFile({ name: 'doc.pdf', type: 'application/pdf', size: 10 })).toThrow(ApiError)
  })

  it('rejects a file larger than 10 MB', () => {
    expect(() =>
      validateFile({ name: 'big.png', type: 'image/png', size: 11 * 1024 * 1024 }),
    ).toThrow(/trop volumineux/i)
  })

  it('rejects a missing file', () => {
    expect(() => validateFile(null)).toThrow(ApiError)
  })
})

describe('getHealth', () => {
  it('flags a 503 as degraded instead of throwing', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({ status: 'degraded', model_loaded: false }, 503)))
    const result = await getHealth()
    expect(result.degraded).toBe(true)
    expect(result.data.model_loaded).toBe(false)
    expect(typeof result.clientMs).toBe('number')
  })

  it('returns ok on 200', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({ status: 'ok', model_loaded: true })))
    const result = await getHealth()
    expect(result.degraded).toBe(false)
    expect(result.data.status).toBe('ok')
  })
})

describe('predict', () => {
  it('maps an HTTP error to an ApiError with the API code', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse({ error: 'INVALID_IMAGE', detail: 'corrupted', status_code: 400 }, 400)),
    )
    const file = new File(['x'], 'face.jpg', { type: 'image/jpeg' })
    await expect(predict(file)).rejects.toMatchObject({
      name: 'ApiError',
      status: 400,
      code: 'INVALID_IMAGE',
    })
  })

  it('returns the payload and the client round-trip time', async () => {
    const payload = { label: 'fake', prob_fake: 0.9, prob_real: 0.1 }
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(payload)))
    const file = new File(['x'], 'face.jpg', { type: 'image/jpeg' })
    const { data, clientMs } = await predict(file)
    expect(data).toEqual(payload)
    expect(clientMs).toBeGreaterThanOrEqual(0)
  })

  it('reports an unreachable API as a network error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('fetch failed')
      }),
    )
    const file = new File(['x'], 'face.jpg', { type: 'image/jpeg' })
    await expect(predict(file)).rejects.toMatchObject({ code: 'NETWORK_ERROR', status: 0 })
  })
})

describe('predictVisualize', () => {
  it('sends the target/alpha/format query parameters', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ image_base64: 'AAAA' }))
    vi.stubGlobal('fetch', fetchMock)
    const file = new File(['x'], 'face.jpg', { type: 'image/jpeg' })
    const { data } = await predictVisualize(file, { target: 'real', alpha: 0.3, format: 'json' })
    const [url] = fetchMock.mock.calls[0]
    expect(url).toContain('target=real')
    expect(url).toContain('alpha=0.3')
    expect(url).toContain('format=json')
    expect(data.image_base64).toBe('AAAA')
  })
})

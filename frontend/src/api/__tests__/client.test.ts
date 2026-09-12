import { afterEach, describe, expect, it, vi } from 'vitest'
import * as api from '../client'
import { ApiError } from '../types'
import { jobResultFixture } from '../../test/fixtures'

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    blob: async () => new Blob([JSON.stringify(body)]),
  } as unknown as Response
}

describe('api client', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('returns parsed JSON on a successful GET', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ status: 'ok', api_version: '0.1.0', model_name: 'x', frame_version: null })))
    const health = await api.getHealth()
    expect(health.status).toBe('ok')
  })

  it('throws ApiError with the backend detail on a 4xx response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ error: 'upload_not_found', code: 'upload_not_found', detail: "No such upload (upload_id='x')" }, 404)),
    )
    await expect(api.getSrResult('unknown-job')).rejects.toMatchObject({
      status: 404,
      code: 'upload_not_found',
      detail: "No such upload (upload_id='x')",
    })
  })

  it('never surfaces a raw traceback -- only the backend detail string reaches ApiError', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ error: 'internal_error', code: 'internal_error', detail: 'An internal error occurred.' }, 500)))
    try {
      await api.getSrResult('job-1')
      throw new Error('expected rejection')
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError)
      const apiError = error as ApiError
      expect(apiError.detail).not.toMatch(/Traceback|File "|line \d+, in /)
      expect(apiError.status).toBe(500)
    }
  })

  it('wraps a network failure (fetch rejecting) as an ApiError with a clear message', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await expect(api.getHealth()).rejects.toMatchObject({ code: 'network_error' })
  })

  it('uploadScene posts a FormData body with the file and input_scale', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ upload_id: 'u1', filename: 'a.tif', valid: true, band_names: [], width: 1, height: 1, crs: null, resolution_m: null, input_scale: 'reflectance', validation_messages: [] }))
    vi.stubGlobal('fetch', fetchMock)
    const file = new File(['data'], 'a.tif')
    await api.uploadScene(file, 'reflectance')

    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).toMatch(/\/upload$/)
    expect(init.method).toBe('POST')
    const form = init.body as FormData
    expect(form.get('file')).toBe(file)
    expect(form.get('input_scale')).toBe('reflectance')
  })

  it('runSr posts the upload_id and optional seed as JSON', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(jobResultFixture))
    vi.stubGlobal('fetch', fetchMock)
    await api.runSr('upload-1', 7)
    const [, init] = fetchMock.mock.calls[0]
    expect(JSON.parse(init.body as string)).toEqual({ upload_id: 'upload-1', seed: 7 })
  })

  it('builds download URLs against the configured API base', () => {
    expect(api.srDownloadUrl('job-1')).toBe(`${api.API_BASE_URL}/sr/download/job-1`)
    expect(api.uncertaintyDownloadUrl('job-1')).toBe(`${api.API_BASE_URL}/uncertainty/download/job-1`)
    expect(api.ndviDownloadUrl('analysis-1', 'native-ndvi')).toBe(`${api.API_BASE_URL}/analysis/download/analysis-1/native-ndvi`)
  })
})

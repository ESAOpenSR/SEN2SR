/**
 * Small, explicit HTTP client for the FRAME API (frame/api/). Every backend
 * URL flows through here so the base URL is configurable in one place
 * (VITE_FRAME_API_URL, see frontend/README.md) -- no other module
 * constructs a fetch URL by hand, and nothing imports Python FRAME code
 * into the browser.
 */

import { ApiError, type ApiErrorBody, type HealthResponse, type NDVIAnalysisResponse, type SRResultResponse, type UploadResponse } from './types'

export const API_BASE_URL: string = (import.meta.env.VITE_FRAME_API_URL as string | undefined)?.replace(/\/$/, '') || 'http://127.0.0.1:8000'

async function parseErrorBody(response: Response): Promise<ApiErrorBody> {
  try {
    const body = await response.json()
    if (body && typeof body.detail === 'string') {
      return body as ApiErrorBody
    }
  } catch {
    // response wasn't JSON (e.g. a proxy/network error page) -- fall through
  }
  return {
    error: 'unknown_error',
    code: 'unknown_error',
    detail: `The server returned an unexpected response (HTTP ${response.status}).`,
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init)
  } catch {
    throw new ApiError(0, {
      error: 'network_error',
      code: 'network_error',
      detail: `Could not reach the FRAME API at ${API_BASE_URL}. Is the backend running?`,
    })
  }

  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorBody(response))
  }

  return (await response.json()) as T
}

async function requestBlob(path: string): Promise<Blob> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`)
  } catch {
    throw new ApiError(0, {
      error: 'network_error',
      code: 'network_error',
      detail: `Could not reach the FRAME API at ${API_BASE_URL}. Is the backend running?`,
    })
  }
  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorBody(response))
  }
  return response.blob()
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

export function uploadScene(file: File, inputScale: string): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('input_scale', inputScale)
  return request<UploadResponse>('/upload', { method: 'POST', body: form })
}

export function runSr(uploadId: string, seed?: number): Promise<SRResultResponse> {
  return request<SRResultResponse>('/sr/run', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(seed === undefined ? { upload_id: uploadId } : { upload_id: uploadId, seed }),
  })
}

export function getSrResult(jobId: string): Promise<SRResultResponse> {
  return request<SRResultResponse>(`/sr/result/${jobId}`)
}

export function runNdviAnalysis(jobId: string): Promise<NDVIAnalysisResponse> {
  return request<NDVIAnalysisResponse>('/analysis/ndvi', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
  })
}

export function getAnalysis(analysisId: string): Promise<NDVIAnalysisResponse> {
  return request<NDVIAnalysisResponse>(`/analysis/${analysisId}`)
}

export function srDownloadUrl(jobId: string): string {
  return `${API_BASE_URL}/sr/download/${jobId}`
}

export function uncertaintyDownloadUrl(jobId: string): string {
  return `${API_BASE_URL}/uncertainty/download/${jobId}`
}

export function ndviDownloadUrl(analysisId: string, layer: 'native-ndvi' | 'sr-ndvi' | 'ndvi-diff'): string {
  return `${API_BASE_URL}/analysis/download/${analysisId}/${layer}`
}

export function fetchSrGeotiff(jobId: string): Promise<Blob> {
  return requestBlob(`/sr/download/${jobId}`)
}

export function fetchUncertaintyGeotiff(jobId: string): Promise<Blob> {
  return requestBlob(`/uncertainty/download/${jobId}`)
}

export function fetchNdviGeotiff(analysisId: string, layer: 'native-ndvi' | 'sr-ndvi' | 'ndvi-diff'): Promise<Blob> {
  return requestBlob(`/analysis/download/${analysisId}/${layer}`)
}

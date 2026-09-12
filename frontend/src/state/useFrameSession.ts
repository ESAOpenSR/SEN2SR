import { useCallback, useMemo, useRef, useState } from 'react'
import * as api from '../api/client'
import { ApiError } from '../api/types'
import type { NDVIAnalysisResponse, SRResultResponse, UploadResponse } from '../api/types'

export type AsyncStatus = 'idle' | 'pending' | 'success' | 'error'

export type ResultTab = 'overview' | 'uncertainty' | 'ndvi' | 'metadata'

/** The two input scales frame.preprocessing accepts (see POST /upload's
 * `input_scale` field). Most real Sentinel-2 L2A distributions are raw
 * digital numbers (uint16, thousands-scale) -- that stays the default --
 * but a file that's already been scaled to 0-1 reflectance needs the
 * other option, or preprocessing silently divides it again and produces a
 * badly-scaled (visually broken) result with no error to explain why. */
export type InputScale = 'raw_digital_number' | 'reflectance'

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.detail
  if (error instanceof Error) return error.message
  return 'Something went wrong.'
}

export interface FrameSession {
  file: File | null
  selectFile: (file: File | null) => void

  inputScale: InputScale
  setInputScale: (scale: InputScale) => void

  uploadStatus: AsyncStatus
  uploadResult: UploadResponse | null
  uploadError: string | null

  jobStatus: AsyncStatus
  jobResult: SRResultResponse | null
  jobError: string | null

  analysisStatus: AsyncStatus
  analysisResult: NDVIAnalysisResponse | null
  analysisError: string | null

  activeTab: ResultTab
  setActiveTab: (tab: ResultTab) => void

  runFrame: () => Promise<void>
  runNdviAnalysis: () => Promise<void>
  reset: () => void

  hasResult: boolean
}

/** Owns the whole demo workflow's state. Deliberately plain React state
 * (useState/useCallback) rather than a reducer/store library -- the state
 * shape is small and the transitions are simple sequential API calls, so a
 * heavier state-management dependency isn't warranted for this prototype.
 *
 * Sequence: selecting a file immediately uploads it (validation only, no
 * SR -- POST /upload) so the input-preview metadata appears right away;
 * "Run FRAME" is a separate, explicit action that fires POST /sr/run
 * against the resulting upload_id. */
export function useFrameSession(): FrameSession {
  const [file, setFile] = useState<File | null>(null)
  const [inputScale, setInputScaleState] = useState<InputScale>('raw_digital_number')

  const [uploadStatus, setUploadStatus] = useState<AsyncStatus>('idle')
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const [jobStatus, setJobStatus] = useState<AsyncStatus>('idle')
  const [jobResult, setJobResult] = useState<SRResultResponse | null>(null)
  const [jobError, setJobError] = useState<string | null>(null)

  const [analysisStatus, setAnalysisStatus] = useState<AsyncStatus>('idle')
  const [analysisResult, setAnalysisResult] = useState<NDVIAnalysisResponse | null>(null)
  const [analysisError, setAnalysisError] = useState<string | null>(null)

  const [activeTab, setActiveTab] = useState<ResultTab>('overview')

  // Guards against a stale upload response landing after the user has
  // already picked a different file (or changed the input scale, which
  // re-uploads under the hood).
  const requestIdRef = useRef(0)

  const resetDownstream = useCallback(() => {
    setUploadStatus('idle')
    setUploadResult(null)
    setUploadError(null)
    setJobStatus('idle')
    setJobResult(null)
    setJobError(null)
    setAnalysisStatus('idle')
    setAnalysisResult(null)
    setAnalysisError(null)
    setActiveTab('overview')
  }, [])

  const uploadWithScale = useCallback((targetFile: File, scale: InputScale) => {
    requestIdRef.current += 1
    const requestId = requestIdRef.current
    setUploadStatus('pending')
    setUploadResult(null)
    setUploadError(null)
    api
      .uploadScene(targetFile, scale)
      .then((result) => {
        if (requestIdRef.current !== requestId) return
        setUploadResult(result)
        setUploadStatus('success')
      })
      .catch((error: unknown) => {
        if (requestIdRef.current !== requestId) return
        setUploadStatus('error')
        setUploadError(messageFor(error))
      })
  }, [])

  const selectFile = useCallback(
    (next: File | null) => {
      resetDownstream()
      setFile(next)
      if (!next) {
        requestIdRef.current += 1 // invalidate any in-flight request
        return
      }
      uploadWithScale(next, inputScale)
    },
    [inputScale, resetDownstream, uploadWithScale],
  )

  const setInputScale = useCallback(
    (scale: InputScale) => {
      setInputScaleState(scale)
      if (file) {
        resetDownstream()
        uploadWithScale(file, scale)
      }
    },
    [file, resetDownstream, uploadWithScale],
  )

  const runFrame = useCallback(async () => {
    if (!uploadResult) return
    setJobStatus('pending')
    setJobError(null)
    try {
      const job = await api.runSr(uploadResult.upload_id)
      setJobResult(job)
      setJobStatus('success')
    } catch (error) {
      setJobStatus('error')
      setJobError(messageFor(error))
    }
  }, [uploadResult])

  const runNdviAnalysis = useCallback(async () => {
    if (!jobResult) return
    setAnalysisStatus('pending')
    setAnalysisError(null)
    try {
      const analysis = await api.runNdviAnalysis(jobResult.job_id)
      setAnalysisResult(analysis)
      setAnalysisStatus('success')
    } catch (error) {
      setAnalysisStatus('error')
      setAnalysisError(messageFor(error))
    }
  }, [jobResult])

  const reset = useCallback(() => {
    selectFile(null)
  }, [selectFile])

  const hasResult = jobStatus === 'success' && jobResult !== null

  return useMemo(
    () => ({
      file,
      selectFile,
      inputScale,
      setInputScale,
      uploadStatus,
      uploadResult,
      uploadError,
      jobStatus,
      jobResult,
      jobError,
      analysisStatus,
      analysisResult,
      analysisError,
      activeTab,
      setActiveTab,
      runFrame,
      runNdviAnalysis,
      reset,
      hasResult,
    }),
    [file, selectFile, inputScale, setInputScale, uploadStatus, uploadResult, uploadError, jobStatus, jobResult, jobError, analysisStatus, analysisResult, analysisError, activeTab, runFrame, runNdviAnalysis, reset, hasResult],
  )
}

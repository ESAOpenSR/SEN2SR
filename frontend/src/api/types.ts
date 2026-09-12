/**
 * Types mirroring frame/api/schemas.py exactly. Kept hand-written and
 * minimal rather than generated -- the API surface is small and stable
 * for this prototype (see frame/api/README.md).
 */

export interface HealthResponse {
  status: string
  frame_version: string | null
  api_version: string
  model_name: string
}

export interface UploadResponse {
  upload_id: string
  filename: string
  valid: boolean
  band_names: string[]
  width: number
  height: number
  crs: string | null
  resolution_m: number | null
  input_scale: string
  validation_messages: string[]
}

export interface ResolutionDescription {
  native_resolution_m: number
  sr_resolution_m: number
  scale_factor: number
  description: string
}

export interface UncertaintyDistribution {
  mean: number
  median: number
  std: number
  p90: number
  p95: number
  min: number
  max: number
  n_pixels?: number
}

export interface UncertaintySummary {
  label: string
  scalar_summary: number
  scalar_summary_definition: string
  overall_distribution: UncertaintyDistribution
  n: number
  seed: number
  transform_names: string[]
  disclaimer: string
}

export interface SelfConsistencySummary {
  downsample_rmse: number | null
  ndvi_discrepancy_mean_abs: number | null
  b08_b04_ratio_discrepancy_mean_abs: number | null
  note: string
}

export interface SRResultResponse {
  job_id: string
  status: string
  upload_id: string
  model_name: string
  input_shape: number[]
  output_shape: number[]
  resolution: ResolutionDescription
  bands: string[]
  crs: string | null
  uncertainty: UncertaintySummary
  self_consistency: SelfConsistencySummary
  metadata: Record<string, unknown>
  scientific_caveats: string[]
  artifacts: Record<string, string>
  created_at: string
}

export interface NDVIComparisonSummary {
  status: string
  valid_pixel_count: number
  mean_abs_difference: number | null
  rmse: number | null
  max_abs_difference: number | null
  resampling_method: string
}

export interface UncertaintyWeightedSummary {
  status: string
  correlation_uncertainty_vs_abs_diff: number | null
  uncertainty_weighted_mean_abs_diff: number | null
  unweighted_mean_abs_diff: number | null
}

export interface NDVIAnalysisResponse {
  analysis_id: string
  job_id: string
  status: string
  ndvi_formula: string
  native_resolution_m: number
  sr_resolution_m: number
  comparison: NDVIComparisonSummary
  uncertainty_weighted_summary: UncertaintyWeightedSummary
  scientific_caveats: string[]
  metadata: Record<string, unknown>
  artifacts: Record<string, string>
  created_at: string
}

export interface ApiErrorBody {
  error: string
  code: string
  detail: string
}

/** Thrown by the API client for any non-2xx response. Carries the parsed
 * error body when the server returned one (it always should -- see
 * frame/api/app.py's generic exception handler) so callers can show
 * `detail` without ever touching a stack trace. */
export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly detail: string

  constructor(status: number, body: ApiErrorBody) {
    super(body.detail || body.error || `Request failed with status ${status}`)
    this.name = 'ApiError'
    this.status = status
    this.code = body.code
    this.detail = body.detail
  }
}

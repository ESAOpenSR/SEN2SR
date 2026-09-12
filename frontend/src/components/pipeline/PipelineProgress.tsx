import type { AsyncStatus } from '../../state/useFrameSession'
import type { SRResultResponse } from '../../api/types'
import './PipelineProgress.css'

type StepStatus = 'pending' | 'active' | 'done' | 'error'

interface Step {
  key: string
  label: string
}

const UPLOAD_STEP: Step = { key: 'upload', label: 'Upload' }
const RUN_STEPS: Step[] = [
  { key: 'preprocessing', label: 'Preprocessing' },
  { key: 'super-resolution', label: 'Super-resolution' },
  { key: 'uncertainty', label: 'Uncertainty' },
  { key: 'validation', label: 'Validation' },
]
const READY_STEP: Step = { key: 'ready', label: 'Ready' }

function statusFor(kind: 'upload' | 'run' | 'ready', uploadStatus: AsyncStatus, jobStatus: AsyncStatus): StepStatus {
  if (kind === 'upload') {
    if (uploadStatus === 'pending') return 'active'
    if (uploadStatus === 'success') return 'done'
    if (uploadStatus === 'error') return 'error'
    return 'pending'
  }
  if (kind === 'run') {
    if (jobStatus === 'pending') return 'active'
    if (jobStatus === 'success') return 'done'
    if (jobStatus === 'error') return 'error'
    return 'pending'
  }
  // ready
  return jobStatus === 'success' ? 'done' : 'pending'
}

interface StepPipProps {
  step: Step
  status: StepStatus
}

function StepPip({ step, status }: StepPipProps) {
  return (
    <li className={`pipeline-progress__step pipeline-progress__step--${status}`}>
      <span className="pipeline-progress__pip" aria-hidden="true" />
      <span className="pipeline-progress__step-label">{step.label}</span>
    </li>
  )
}

interface PipelineProgressProps {
  uploadStatus: AsyncStatus
  jobStatus: AsyncStatus
  jobResult: SRResultResponse | null
  canRun: boolean
  onRun: () => void
}

/** The real pipeline's own named stages (read GeoTIFF -> preprocess ->
 * frozen SEN2SR + uncertainty ensemble -> geospatial export -> consistency
 * diagnostics -- see frame/api/README.md), reflecting only the two actual
 * API request states this prototype has: /upload and the single
 * synchronous /sr/run call. The four run-stages activate and complete
 * together because /sr/run genuinely doesn't report sub-step progress --
 * showing anything more granular would be a fabricated percentage. */
export function PipelineProgress({ uploadStatus, jobStatus, jobResult, canRun, onRun }: PipelineProgressProps) {
  const uploadPipStatus = statusFor('upload', uploadStatus, jobStatus)
  const runPipStatus = statusFor('run', uploadStatus, jobStatus)
  const readyPipStatus = statusFor('ready', uploadStatus, jobStatus)

  return (
    <div className="pipeline-progress">
      <ol className="pipeline-progress__steps">
        <StepPip step={UPLOAD_STEP} status={uploadPipStatus} />
        {RUN_STEPS.map((step) => (
          <StepPip key={step.key} step={step} status={runPipStatus} />
        ))}
        <StepPip step={READY_STEP} status={readyPipStatus} />
      </ol>

      <div className="pipeline-progress__footer">
        <p className="pipeline-progress__note">
          {jobStatus === 'pending'
            ? 'Executing the FRAME pipeline (single synchronous request) — preprocessing, the frozen SR model, and the uncertainty ensemble run inside this one call.'
            : jobStatus === 'success' && jobResult
              ? `Completed in ${Number(jobResult.metadata.inference_seconds ?? 0).toFixed(3)}s on ${String(jobResult.metadata.device ?? 'unknown device')}.`
              : 'Preprocessing, super-resolution, uncertainty, and validation all execute inside one synchronous request — sub-step timing isn’t reported individually.'}
        </p>
        <button type="button" className="pipeline-progress__run-button" disabled={!canRun || jobStatus === 'pending'} onClick={onRun}>
          {jobStatus === 'pending' ? 'Running…' : jobStatus === 'success' ? 'Run FRAME again' : 'Run FRAME'}
        </button>
      </div>
    </div>
  )
}

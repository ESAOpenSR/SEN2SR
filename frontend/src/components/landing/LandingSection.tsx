import { SatelliteField } from '../atmosphere/SatelliteField'
import { PipelineDiagram } from './PipelineDiagram'
import { UploadPanel } from './UploadPanel'
import { InputPreview } from './InputPreview'
import { PipelineProgress } from '../pipeline/PipelineProgress'
import { ErrorNotice } from '../ui/ErrorNotice'
import { ScientificNotes } from '../notes/ScientificNotes'
import type { FrameSession } from '../../state/useFrameSession'
import './LandingSection.css'

interface LandingSectionProps {
  session: FrameSession
}

export function LandingSection({ session }: LandingSectionProps) {
  const { file, selectFile, inputScale, setInputScale, uploadStatus, uploadResult, uploadError, jobStatus, jobResult, jobError, runFrame } = session

  return (
    <div className="landing-section">
      <div className="landing-section__hero">
        <SatelliteField variant="hero" />
        <div className="landing-section__hero-content">
          <p className="label">SIH 26142 · Earth observation prototype</p>
          <h2 className="landing-section__headline">Sharper Sentinel-2, with the uncertainty shown alongside it.</h2>
          <p className="landing-section__tagline">FRAME turns medium-resolution Sentinel-2 imagery into a sharper SR-derived product while exposing model-stability uncertainty and lightweight downstream analysis.</p>
          <PipelineDiagram />
        </div>
      </div>

      <div className="landing-section__body panel">
        <div className="landing-section__body-inner">
          <UploadPanel file={file} inputScale={inputScale} onChangeInputScale={setInputScale} uploadStatus={uploadStatus} uploadError={uploadError} onSelectFile={selectFile} />

          {uploadStatus === 'success' && uploadResult && (
            <div className="landing-section__preview">
              <p className="label">Input preview</p>
              <InputPreview upload={uploadResult} />
            </div>
          )}

          {uploadStatus === 'success' && uploadResult && (
            <PipelineProgress uploadStatus={uploadStatus} jobStatus={jobStatus} jobResult={jobResult} canRun={uploadStatus === 'success'} onRun={runFrame} />
          )}

          {jobStatus === 'error' && jobError && <ErrorNotice title="FRAME run failed" detail={jobError} onRetry={runFrame} />}
        </div>
        <ScientificNotes />
      </div>
    </div>
  )
}

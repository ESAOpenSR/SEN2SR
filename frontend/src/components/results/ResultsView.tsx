import { Panel } from '../ui/Panel'
import { Tabs } from './Tabs'
import { OverviewTab } from './OverviewTab'
import { UncertaintyTab } from './UncertaintyTab'
import { NdviTab } from './NdviTab'
import { MetadataTab } from './MetadataTab'
import { DownloadPanel } from '../downloads/DownloadPanel'
import { ScientificNotes } from '../notes/ScientificNotes'
import { MetadataGrid } from '../ui/MetadataGrid'
import type { FrameSession } from '../../state/useFrameSession'
import './results.css'

interface ResultsViewProps {
  session: FrameSession
}

export function ResultsView({ session }: ResultsViewProps) {
  const { jobResult, file, activeTab, setActiveTab, analysisStatus, analysisResult, analysisError, runNdviAnalysis } = session
  if (!jobResult || !file) return null

  return (
    <div className="results-view">
      <div className="results-view__main">
        <Panel
          eyebrow="Results"
          title={`Job ${jobResult.job_id.slice(0, 8)}`}
          action={
            <MetadataGrid
              columns={2}
              items={[
                { label: 'Bands', value: jobResult.bands.join(' · ') },
                { label: 'CRS', value: jobResult.crs ?? '—' },
              ]}
            />
          }
        >
          <Tabs active={activeTab} onChange={setActiveTab} />
          {activeTab === 'overview' && <OverviewTab file={file} jobResult={jobResult} />}
          {activeTab === 'uncertainty' && <UncertaintyTab jobResult={jobResult} />}
          {activeTab === 'ndvi' && <NdviTab analysisStatus={analysisStatus} analysisResult={analysisResult} analysisError={analysisError} onRunAnalysis={runNdviAnalysis} />}
          {activeTab === 'metadata' && <MetadataTab jobResult={jobResult} />}
        </Panel>
      </div>

      <aside className="results-view__side panel">
        <div className="results-view__summary">
          <p className="label">Pipeline</p>
          <MetadataGrid
            columns={1}
            items={[
              { label: 'Model', value: jobResult.model_name },
              { label: 'Output grid', value: jobResult.resolution.description },
              { label: 'Inference time', value: `${Number(jobResult.metadata.inference_seconds ?? 0).toFixed(3)} s` },
              { label: 'Device', value: String(jobResult.metadata.device ?? '—') },
            ]}
          />
        </div>
        <DownloadPanel jobResult={jobResult} analysisResult={analysisResult} />
        <ScientificNotes caveats={jobResult.scientific_caveats} />
      </aside>
    </div>
  )
}

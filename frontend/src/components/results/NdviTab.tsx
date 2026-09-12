import { useEffect, useRef } from 'react'
import * as api from '../../api/client'
import { useRemoteRaster } from '../../hooks/useRaster'
import { renderNdvi, renderSequentialHeatmap } from '../../lib/raster'
import { MetadataGrid } from '../ui/MetadataGrid'
import { ErrorNotice } from '../ui/ErrorNotice'
import { RasterLoadingState } from './RasterLoadingState'
import type { AsyncStatus } from '../../state/useFrameSession'
import type { NDVIAnalysisResponse } from '../../api/types'
import './NdviTab.css'

interface NdviTabProps {
  analysisStatus: AsyncStatus
  analysisResult: NDVIAnalysisResponse | null
  analysisError: string | null
  onRunAnalysis: () => void
}

function NdviPanel({ title, cacheKey, layer, analysisId, render }: { title: string; cacheKey: string; layer: 'native-ndvi' | 'sr-ndvi' | 'ndvi-diff'; analysisId: string; render: (canvas: HTMLCanvasElement, width: number, height: number, band: Float32Array) => void }) {
  const raster = useRemoteRaster(() => api.fetchNdviGeotiff(analysisId, layer), cacheKey)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    if (!canvasRef.current || !raster.raster) return
    render(canvasRef.current, raster.raster.width, raster.raster.height, raster.raster.bands[0])
  }, [raster.raster, render])

  return (
    <div className="ndvi-tab__panel">
      <p className="label">{title}</p>
      <RasterLoadingState states={[raster]} label={title.toLowerCase()}>
        <canvas ref={canvasRef} className="ndvi-tab__canvas" />
      </RasterLoadingState>
    </div>
  )
}

export function NdviTab({ analysisStatus, analysisResult, analysisError, onRunAnalysis }: NdviTabProps) {
  if (!analysisResult) {
    return (
      <div className="tab-content ndvi-tab__empty">
        <p className="ndvi-tab__empty-text">Run the Phase 6 NDVI comparison against this SR result — native 10 m NDVI vs. the SR-derived NDVI, reduced to a common grid before comparison.</p>
        <button type="button" className="ndvi-tab__run-button" disabled={analysisStatus === 'pending'} onClick={onRunAnalysis}>
          {analysisStatus === 'pending' ? 'Running NDVI analysis…' : 'Run NDVI analysis'}
        </button>
        {analysisStatus === 'error' && analysisError && <ErrorNotice title="NDVI analysis failed" detail={analysisError} onRetry={onRunAnalysis} />}
      </div>
    )
  }

  const { comparison, uncertainty_weighted_summary: weighted } = analysisResult

  return (
    <div className="tab-content">
      <div className="ndvi-tab__grid">
        <NdviPanel title="Native 10 m NDVI" cacheKey={analysisResult.analysis_id} layer="native-ndvi" analysisId={analysisResult.analysis_id} render={renderNdvi} />
        <NdviPanel title="SR-derived NDVI — 2.5 m pixel grid" cacheKey={analysisResult.analysis_id} layer="sr-ndvi" analysisId={analysisResult.analysis_id} render={renderNdvi} />
        <NdviPanel
          title="Absolute difference"
          cacheKey={analysisResult.analysis_id}
          layer="ndvi-diff"
          analysisId={analysisResult.analysis_id}
          render={(canvas, w, h, band) => renderSequentialHeatmap(canvas, w, h, band, 'ndvi-diff')}
        />
      </div>

      <MetadataGrid
        columns={4}
        items={[
          { label: 'Valid pixels', value: comparison.valid_pixel_count.toLocaleString() },
          { label: 'Mean abs. difference', value: comparison.mean_abs_difference?.toFixed(4) ?? '—' },
          { label: 'RMSE', value: comparison.rmse?.toFixed(4) ?? '—' },
          { label: 'Max abs. difference', value: comparison.max_abs_difference?.toFixed(4) ?? '—' },
          { label: 'Uncertainty ↔ disagreement corr.', value: weighted.correlation_uncertainty_vs_abs_diff?.toFixed(3) ?? 'undefined (zero variance)' },
          { label: 'Uncertainty-weighted mean abs. diff.', value: weighted.uncertainty_weighted_mean_abs_diff?.toFixed(4) ?? '—' },
          { label: 'Unweighted mean abs. diff.', value: weighted.unweighted_mean_abs_diff?.toFixed(4) ?? '—' },
          { label: 'Resampling', value: comparison.resampling_method },
        ]}
      />

      <ul className="ndvi-tab__caveats">
        {analysisResult.scientific_caveats.map((caveat) => (
          <li key={caveat}>{caveat}</li>
        ))}
      </ul>
    </div>
  )
}

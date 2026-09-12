import { useEffect, useRef, useState } from 'react'
import * as api from '../../api/client'
import { useRemoteRaster } from '../../hooks/useRaster'
import { renderSequentialHeatmap, renderUncertaintyOverlay } from '../../lib/raster'
import { BAND_INDEX } from '../../lib/geotiff'
import { MetadataGrid } from '../ui/MetadataGrid'
import { RasterLoadingState } from './RasterLoadingState'
import { LAM_DISTINCTION, UNCERTAINTY_EXPLAINER, UNCERTAINTY_LABEL } from '../../constants/terminology'
import type { SRResultResponse } from '../../api/types'
import './UncertaintyTab.css'

interface UncertaintyTabProps {
  jobResult: SRResultResponse
}

// The uncertainty GeoTIFF's band order (frame/api/services/pipeline.py):
// [B04_std, B03_std, B02_std, B08_std, overall_std].
const OVERALL_STD_BAND_INDEX = 4

type ViewMode = 'map' | 'overlay'

export function UncertaintyTab({ jobResult }: UncertaintyTabProps) {
  const sr = useRemoteRaster(() => api.fetchSrGeotiff(jobResult.job_id), jobResult.job_id)
  const uncertainty = useRemoteRaster(() => api.fetchUncertaintyGeotiff(jobResult.job_id), jobResult.job_id)
  const [mode, setMode] = useState<ViewMode>('map')
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [range, setRange] = useState<[number, number] | null>(null)

  useEffect(() => {
    if (!canvasRef.current || !uncertainty.raster) return
    const band = uncertainty.raster.bands[OVERALL_STD_BAND_INDEX]
    if (mode === 'map') {
      const [lo, hi] = renderSequentialHeatmap(canvasRef.current, uncertainty.raster.width, uncertainty.raster.height, band, 'uncertainty')
      setRange([lo, hi])
    } else if (mode === 'overlay' && sr.raster) {
      const [lo, hi] = renderUncertaintyOverlay(canvasRef.current, sr.raster.width, sr.raster.height, {
        red: sr.raster.bands[BAND_INDEX.RED],
        green: sr.raster.bands[BAND_INDEX.GREEN],
        blue: sr.raster.bands[BAND_INDEX.BLUE],
      }, band)
      setRange([lo, hi])
    }
  }, [mode, uncertainty.raster, sr.raster])

  const distribution = jobResult.uncertainty.overall_distribution

  return (
    <div className="tab-content">
      <div className="uncertainty-tab__intro">
        <p className="uncertainty-tab__label">{UNCERTAINTY_LABEL}</p>
        <p className="uncertainty-tab__explainer">{UNCERTAINTY_EXPLAINER}</p>
      </div>

      <div className="uncertainty-tab__toggle" role="group" aria-label="Uncertainty view mode">
        <button type="button" className={mode === 'map' ? 'is-active' : ''} onClick={() => setMode('map')}>
          Uncertainty map
        </button>
        <button type="button" className={mode === 'overlay' ? 'is-active' : ''} onClick={() => setMode('overlay')}>
          SR image + overlay
        </button>
      </div>

      <RasterLoadingState states={mode === 'overlay' ? [sr, uncertainty] : [uncertainty]} label="the uncertainty raster">
        <div className="uncertainty-tab__canvas-frame">
          <canvas ref={canvasRef} className="uncertainty-tab__canvas" />
          {range && (
            <div className="uncertainty-tab__legend">
              <span>{range[0].toExponential(2)}</span>
              <span className="uncertainty-tab__legend-bar" />
              <span>{range[1].toExponential(2)}</span>
            </div>
          )}
        </div>
      </RasterLoadingState>

      <MetadataGrid
        columns={4}
        items={[
          { label: 'Scalar summary', value: jobResult.uncertainty.scalar_summary.toExponential(3) },
          { label: 'Mean', value: distribution.mean.toExponential(3) },
          { label: 'P90', value: distribution.p90.toExponential(3) },
          { label: 'P95', value: distribution.p95.toExponential(3) },
          { label: 'Median', value: distribution.median.toExponential(3) },
          { label: 'Max', value: distribution.max.toExponential(3) },
          { label: 'TTA ensemble size', value: jobResult.uncertainty.n },
          { label: 'Seed', value: jobResult.uncertainty.seed },
        ]}
      />

      <p className="uncertainty-tab__caption">{jobResult.uncertainty.scalar_summary_definition}</p>
      <p className="uncertainty-tab__caption uncertainty-tab__caption--dim">{LAM_DISTINCTION}</p>
    </div>
  )
}

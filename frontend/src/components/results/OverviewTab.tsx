import * as api from '../../api/client'
import { useDecodedFile, useRemoteRaster } from '../../hooks/useRaster'
import { ComparisonSlider } from './ComparisonSlider'
import { MetadataGrid } from '../ui/MetadataGrid'
import { RasterLoadingState } from './RasterLoadingState'
import type { SRResultResponse } from '../../api/types'

interface OverviewTabProps {
  file: File
  jobResult: SRResultResponse
}

export function OverviewTab({ file, jobResult }: OverviewTabProps) {
  const native = useDecodedFile(file)
  const sr = useRemoteRaster(() => api.fetchSrGeotiff(jobResult.job_id), jobResult.job_id)

  return (
    <div className="tab-content">
      <RasterLoadingState states={[native, sr]} label="imagery">
        {native.raster && sr.raster && <ComparisonSlider nativeRaster={native.raster} srRaster={sr.raster} />}
      </RasterLoadingState>

      <MetadataGrid
        columns={4}
        items={[
          { label: 'Input shape', value: jobResult.input_shape.join(' × ') },
          { label: 'Output shape', value: jobResult.output_shape.join(' × ') },
          { label: 'Output grid', value: jobResult.resolution.description },
          { label: 'Model', value: jobResult.model_name },
        ]}
      />
    </div>
  )
}

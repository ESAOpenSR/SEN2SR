import { MetadataGrid } from '../ui/MetadataGrid'
import type { SRResultResponse } from '../../api/types'

interface MetadataTabProps {
  jobResult: SRResultResponse
}

/** Full reproducibility record for this run -- everything frame/api/README.md
 * promises is recorded (model, input identity, pipeline version proxy via
 * job id, seed, uncertainty configuration, timestamps). */
export function MetadataTab({ jobResult }: MetadataTabProps) {
  const geo = jobResult.metadata.output_geospatial as Record<string, unknown> | undefined

  return (
    <div className="tab-content">
      <MetadataGrid
        columns={3}
        items={[
          { label: 'Job ID', value: jobResult.job_id },
          { label: 'Upload ID', value: jobResult.upload_id },
          { label: 'Status', value: jobResult.status },
          { label: 'Model', value: jobResult.model_name },
          { label: 'Device', value: String(jobResult.metadata.device ?? '—') },
          { label: 'Inference time', value: `${Number(jobResult.metadata.inference_seconds ?? 0).toFixed(4)} s` },
          { label: 'Uncertainty seed', value: jobResult.uncertainty.seed },
          { label: 'TTA ensemble size', value: jobResult.uncertainty.n },
          { label: 'TTA transforms', value: jobResult.uncertainty.transform_names.join(', ') },
          { label: 'Band stack', value: jobResult.bands.join(' · ') },
          { label: 'CRS', value: jobResult.crs ?? '—' },
          { label: 'Created', value: jobResult.created_at },
        ]}
      />

      <hr className="hairline" />

      <p className="label">Self-consistency (vs. own LR input — not ground truth)</p>
      <MetadataGrid
        columns={3}
        items={[
          { label: 'Downsample RMSE', value: jobResult.self_consistency.downsample_rmse?.toFixed(5) ?? '—' },
          { label: 'NDVI discrepancy (mean abs.)', value: jobResult.self_consistency.ndvi_discrepancy_mean_abs?.toFixed(5) ?? '—' },
          { label: 'B08/B04 ratio discrepancy', value: jobResult.self_consistency.b08_b04_ratio_discrepancy_mean_abs?.toFixed(5) ?? '—' },
        ]}
      />

      {geo && (
        <>
          <hr className="hairline" />
          <p className="label">Output geospatial</p>
          <MetadataGrid
            columns={3}
            items={[
              { label: 'CRS', value: String(geo.crs ?? '—') },
              { label: 'Width × height', value: `${geo.width} × ${geo.height} px` },
              { label: 'Transform', value: Array.isArray(geo.transform) ? geo.transform.map((v) => Number(v).toFixed(2)).join(', ') : '—' },
            ]}
          />
        </>
      )}
    </div>
  )
}

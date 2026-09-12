import { MetadataGrid } from '../ui/MetadataGrid'
import type { UploadResponse } from '../../api/types'

interface InputPreviewProps {
  upload: UploadResponse
}

/** Compact technical metadata for the uploaded scene -- exactly the fields
 * POST /upload returns (frame/api/schemas.py::UploadResponse). No
 * acquisition timestamp is shown because /upload doesn't return one. */
export function InputPreview({ upload }: InputPreviewProps) {
  return (
    <MetadataGrid
      columns={5}
      items={[
        { label: 'File', value: upload.filename },
        { label: 'Dimensions', value: `${upload.width} × ${upload.height} px` },
        { label: 'Resolution', value: upload.resolution_m ? `${upload.resolution_m} m` : '—' },
        { label: 'Band stack', value: upload.band_names.join(' · ') },
        { label: 'CRS', value: upload.crs ?? '—' },
      ]}
    />
  )
}

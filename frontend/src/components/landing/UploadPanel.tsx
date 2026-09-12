import { useCallback, useId, useRef, useState } from 'react'
import { ErrorNotice } from '../ui/ErrorNotice'
import type { AsyncStatus, InputScale } from '../../state/useFrameSession'
import './UploadPanel.css'

interface UploadPanelProps {
  file: File | null
  inputScale: InputScale
  onChangeInputScale: (scale: InputScale) => void
  uploadStatus: AsyncStatus
  uploadError: string | null
  onSelectFile: (file: File | null) => void
}

const ACCEPTED_EXTENSIONS = ['.tif', '.tiff']

export function UploadPanel({ file, inputScale, onChangeInputScale, uploadStatus, uploadError, onSelectFile }: UploadPanelProps) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [isDragActive, setDragActive] = useState(false)

  const handleFiles = useCallback(
    (fileList: FileList | null) => {
      const picked = fileList?.[0]
      if (!picked) return
      onSelectFile(picked)
    },
    [onSelectFile],
  )

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault()
      setDragActive(false)
      handleFiles(event.dataTransfer.files)
    },
    [handleFiles],
  )

  return (
    <div className="upload-panel">
      <div className="upload-panel__scale" role="group" aria-label="Input pixel value scale">
        <span className="label">Pixel values</span>
        <div className="upload-panel__scale-options">
          <button type="button" className={inputScale === 'raw_digital_number' ? 'is-active' : ''} onClick={() => onChangeInputScale('raw_digital_number')}>
            Raw digital number
          </button>
          <button type="button" className={inputScale === 'reflectance' ? 'is-active' : ''} onClick={() => onChangeInputScale('reflectance')}>
            Reflectance (0–1)
          </button>
        </div>
      </div>

      <div
        className={`upload-panel__dropzone ${isDragActive ? 'upload-panel__dropzone--active' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(',')}
          className="visually-hidden"
          onChange={(event) => handleFiles(event.target.files)}
          aria-describedby={`${inputId}-hint`}
        />
        <label htmlFor={inputId} className="upload-panel__label">
          <span className="upload-panel__icon" aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <path d="M12 3v12" stroke="var(--accent)" strokeWidth="1.4" />
              <path d="M7 8l5-5 5 5" stroke="var(--accent)" strokeWidth="1.4" fill="none" />
              <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" stroke="var(--text-tertiary)" strokeWidth="1.4" />
            </svg>
          </span>
          <span className="upload-panel__primary-text">{file ? file.name : 'Drop a Sentinel-2 L2A GeoTIFF, or click to browse'}</span>
          <span className="upload-panel__secondary-text" id={`${inputId}-hint`}>
            Supported bands: B04, B03, B02, B08 — this prototype currently supports the RGBN 4-band path only. Most Sentinel-2 L2A distributions are raw digital numbers — only switch to reflectance if your file's pixel values are already scaled to 0–1.
          </span>
        </label>
      </div>

      {uploadStatus === 'pending' && (
        <p className="upload-panel__state mono" role="status">
          Validating upload…
        </p>
      )}
      {uploadStatus === 'error' && uploadError && (
        <ErrorNotice title="Upload rejected" detail={uploadError} onRetry={file ? () => onSelectFile(file) : undefined} />
      )}
    </div>
  )
}

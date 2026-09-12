import { ErrorNotice } from '../ui/ErrorNotice'
import type { AsyncStatus } from '../../state/useFrameSession'
import './RasterLoadingState.css'

interface RasterState {
  status: AsyncStatus
  error: string | null
}

interface RasterLoadingStateProps {
  states: RasterState[]
  label: string
  children: React.ReactNode
}

/** Shared loading/error handling for the raster views (Overview,
 * Uncertainty, NDVI) -- so "decoding a GeoTIFF over the network" always
 * reads as a real, honest loading state rather than a silent gap. */
export function RasterLoadingState({ states, label, children }: RasterLoadingStateProps) {
  const errored = states.find((state) => state.status === 'error')
  if (errored) {
    return <ErrorNotice title={`Could not load ${label}`} detail={errored.error ?? 'Unknown error.'} />
  }

  const pending = states.some((state) => state.status === 'pending' || state.status === 'idle')
  if (pending) {
    return (
      <div className="raster-loading" role="status">
        <span className="raster-loading__pulse" aria-hidden="true" />
        Decoding {label}…
      </div>
    )
  }

  return <>{children}</>
}

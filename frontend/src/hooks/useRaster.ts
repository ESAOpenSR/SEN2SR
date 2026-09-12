import { useEffect, useState } from 'react'
import { decodeGeoTiff, type DecodedRaster } from '../lib/geotiff'
import type { AsyncStatus } from '../state/useFrameSession'

interface RasterState {
  status: AsyncStatus
  raster: DecodedRaster | null
  error: string | null
}

const IDLE: RasterState = { status: 'idle', raster: null, error: null }

/** Decodes a GeoTIFF already held locally (the uploaded File itself, for
 * the native-resolution preview -- no network round trip needed). */
export function useDecodedFile(file: Blob | null): RasterState {
  const [state, setState] = useState<RasterState>(IDLE)

  useEffect(() => {
    if (!file) {
      setState(IDLE)
      return
    }
    let cancelled = false
    setState({ status: 'pending', raster: null, error: null })
    decodeGeoTiff(file)
      .then((raster) => {
        if (!cancelled) setState({ status: 'success', raster, error: null })
      })
      .catch((error: unknown) => {
        if (!cancelled) setState({ status: 'error', raster: null, error: error instanceof Error ? error.message : 'Failed to decode raster.' })
      })
    return () => {
      cancelled = true
    }
  }, [file])

  return state
}

/** Fetches a GeoTIFF from the API (via `loader`) and decodes it.
 * `cacheKey` (typically a job/analysis id) is the effect's real
 * dependency -- it changes exactly when the underlying artifact does. */
export function useRemoteRaster(loader: (() => Promise<Blob>) | null, cacheKey: string | null): RasterState {
  const [state, setState] = useState<RasterState>(IDLE)

  useEffect(() => {
    if (!loader || !cacheKey) {
      setState(IDLE)
      return
    }
    let cancelled = false
    setState({ status: 'pending', raster: null, error: null })
    loader()
      .then((blob) => decodeGeoTiff(blob))
      .then((raster) => {
        if (!cancelled) setState({ status: 'success', raster, error: null })
      })
      .catch((error: unknown) => {
        if (!cancelled) setState({ status: 'error', raster: null, error: error instanceof Error ? error.message : 'Failed to load raster.' })
      })
    return () => {
      cancelled = true
    }
    // cacheKey alone identifies when to refetch -- loader is recreated per
    // render by design (it closes over ids), not a meaningful dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cacheKey])

  return state
}

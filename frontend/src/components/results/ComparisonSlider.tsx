import { useEffect, useId, useRef, useState } from 'react'
import { renderRgbComposite } from '../../lib/raster'
import { BAND_INDEX, type DecodedRaster } from '../../lib/geotiff'
import { SR_PRODUCT_DESCRIPTION } from '../../constants/terminology'
import './ComparisonSlider.css'

interface ComparisonSliderProps {
  nativeRaster: DecodedRaster
  srRaster: DecodedRaster
}

/** A before/after comparison slider between the native-resolution input
 * and the SR-derived output. Both canvases render at their own native
 * pixel resolution (128x128 vs 512x512) and are scaled to the same
 * on-screen box via CSS -- purely a display concern, no resampling for
 * analysis purposes. */
export function ComparisonSlider({ nativeRaster, srRaster }: ComparisonSliderProps) {
  const nativeCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const srCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const [position, setPosition] = useState(50)
  const sliderId = useId()

  useEffect(() => {
    if (!nativeCanvasRef.current) return
    renderRgbComposite(nativeCanvasRef.current, nativeRaster.width, nativeRaster.height, {
      red: nativeRaster.bands[BAND_INDEX.RED],
      green: nativeRaster.bands[BAND_INDEX.GREEN],
      blue: nativeRaster.bands[BAND_INDEX.BLUE],
    })
  }, [nativeRaster])

  useEffect(() => {
    if (!srCanvasRef.current) return
    renderRgbComposite(srCanvasRef.current, srRaster.width, srRaster.height, {
      red: srRaster.bands[BAND_INDEX.RED],
      green: srRaster.bands[BAND_INDEX.GREEN],
      blue: srRaster.bands[BAND_INDEX.BLUE],
    })
  }, [srRaster])

  return (
    <div className="comparison-slider">
      <div className="comparison-slider__frame">
        <canvas ref={nativeCanvasRef} className="comparison-slider__canvas comparison-slider__canvas--native" />
        <canvas
          ref={srCanvasRef}
          className="comparison-slider__canvas comparison-slider__canvas--sr"
          style={{ clipPath: `inset(0 0 0 ${position}%)` }}
        />
        <div className="comparison-slider__divider" style={{ left: `${position}%` }} aria-hidden="true">
          <span className="comparison-slider__grip" />
        </div>
        <span className="comparison-slider__tag comparison-slider__tag--left">10 m Sentinel-2 input</span>
        <span className="comparison-slider__tag comparison-slider__tag--right">{SR_PRODUCT_DESCRIPTION}</span>
      </div>
      <label className="visually-hidden" htmlFor={sliderId}>
        Comparison position between the native input and the SR-derived output
      </label>
      <input
        id={sliderId}
        type="range"
        className="comparison-slider__range"
        min={0}
        max={100}
        value={position}
        onChange={(event) => setPosition(Number(event.target.value))}
      />
    </div>
  )
}

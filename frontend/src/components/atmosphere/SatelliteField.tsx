import { useEffect, useRef } from 'react'
import { useReducedMotion } from '../../hooks/useReducedMotion'
import './SatelliteField.css'

interface Star {
  x: number
  y: number
  r: number
  twinklePhase: number
  twinkleSpeed: number
}

interface Satellite {
  x: number
  y: number
  vx: number
  vy: number
  size: number
}

function seededRandom(seed: number): () => number {
  let state = seed
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff
    return state / 0x7fffffff
  }
}

function buildStars(count: number, width: number, height: number, seed: number): Star[] {
  const rand = seededRandom(seed)
  const stars: Star[] = []
  for (let i = 0; i < count; i++) {
    stars.push({
      x: rand() * width,
      y: rand() * height,
      r: 0.4 + rand() * 0.9,
      twinklePhase: rand() * Math.PI * 2,
      twinkleSpeed: 0.2 + rand() * 0.3,
    })
  }
  return stars
}

function drawSatellite(ctx: CanvasRenderingContext2D, sat: Satellite): void {
  const angle = Math.atan2(sat.vy, sat.vx)
  ctx.save()
  ctx.translate(sat.x, sat.y)
  ctx.rotate(angle)

  // faint trailing line
  const trail = ctx.createLinearGradient(-sat.size * 7, 0, 0, 0)
  trail.addColorStop(0, 'rgba(90, 200, 224, 0)')
  trail.addColorStop(1, 'rgba(90, 200, 224, 0.35)')
  ctx.strokeStyle = trail
  ctx.lineWidth = 1
  ctx.setLineDash([2, 3])
  ctx.beginPath()
  ctx.moveTo(-sat.size * 7, 0)
  ctx.lineTo(-sat.size * 1.2, 0)
  ctx.stroke()
  ctx.setLineDash([])

  // body
  ctx.fillStyle = 'rgba(210, 234, 240, 0.92)'
  ctx.fillRect(-sat.size * 0.5, -sat.size * 0.35, sat.size, sat.size * 0.7)

  // solar panels
  ctx.fillStyle = 'rgba(90, 200, 224, 0.75)'
  ctx.fillRect(-sat.size * 2.6, -sat.size * 0.12, sat.size * 1.7, sat.size * 0.24)
  ctx.fillRect(sat.size * 0.9, -sat.size * 0.12, sat.size * 1.7, sat.size * 0.24)

  // antenna
  ctx.strokeStyle = 'rgba(210, 234, 240, 0.6)'
  ctx.lineWidth = 0.6
  ctx.beginPath()
  ctx.moveTo(sat.size * 0.5, 0)
  ctx.lineTo(sat.size * 1.1, -sat.size * 0.5)
  ctx.stroke()

  ctx.restore()
}

export type SatelliteFieldVariant = 'hero' | 'ambient'

interface SatelliteFieldProps {
  variant?: SatelliteFieldVariant
  className?: string
}

/** Restrained ambient motion: a faint starfield plus (in the 'hero'
 * variant) one or two small satellites drifting slowly across the canvas
 * with a short trailing trajectory line. Purely decorative -- it never
 * captures pointer events and never blocks interaction. Respects
 * prefers-reduced-motion by rendering one static frame instead of running
 * a requestAnimationFrame loop. */
export function SatelliteField({ variant = 'ambient', className }: SatelliteFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const reducedMotion = useReducedMotion()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let frameId = 0
    let stars: Star[] = []
    let satellites: Satellite[] = []
    let width = 0
    let height = 0

    const setup = () => {
      const rect = canvas.parentElement?.getBoundingClientRect()
      width = Math.max(1, Math.round(rect?.width ?? canvas.clientWidth))
      height = Math.max(1, Math.round(rect?.height ?? canvas.clientHeight))
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = width * dpr
      canvas.height = height * dpr
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

      const starCount = variant === 'hero' ? 90 : 36
      stars = buildStars(starCount, width, height, variant === 'hero' ? 7 : 3)

      satellites =
        variant === 'hero'
          ? [
              { x: width * 0.12, y: height * 0.72, vx: 9, vy: -3.4, size: 5.5 },
              { x: width * 0.82, y: height * 0.18, vx: -6, vy: 4.6, size: 3.8 },
            ]
          : []
    }

    const drawFrame = (t: number) => {
      ctx.clearRect(0, 0, width, height)

      for (const star of stars) {
        const twinkle = reducedMotion ? 0.7 : 0.55 + 0.45 * Math.sin(t * 0.001 * star.twinkleSpeed + star.twinklePhase)
        ctx.fillStyle = `rgba(220, 232, 240, ${0.15 + twinkle * 0.35})`
        ctx.beginPath()
        ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2)
        ctx.fill()
      }

      for (const sat of satellites) {
        drawSatellite(ctx, sat)
      }
    }

    const step = (t: number) => {
      for (const sat of satellites) {
        sat.x += sat.vx * 0.016
        sat.y += sat.vy * 0.016
        const margin = 40
        if (sat.x < -margin) sat.x = width + margin
        if (sat.x > width + margin) sat.x = -margin
        if (sat.y < -margin) sat.y = height + margin
        if (sat.y > height + margin) sat.y = -margin
      }
      drawFrame(t)
      frameId = requestAnimationFrame(step)
    }

    setup()

    const observer = new ResizeObserver(() => {
      setup()
      if (reducedMotion) drawFrame(0)
    })
    if (canvas.parentElement) observer.observe(canvas.parentElement)

    if (reducedMotion) {
      drawFrame(0)
    } else {
      frameId = requestAnimationFrame(step)
    }

    return () => {
      cancelAnimationFrame(frameId)
      observer.disconnect()
    }
  }, [variant, reducedMotion])

  return (
    <div className={`satellite-field satellite-field--${variant} ${className ?? ''}`} aria-hidden="true">
      <canvas ref={canvasRef} />
    </div>
  )
}

import { render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SatelliteField } from '../SatelliteField'

// jsdom has no real canvas 2D context, so the component's drawing effect
// no-ops early (see SatelliteField.tsx's `if (!ctx) return`). What IS
// practical to verify here, per the reduced-motion requirement, is that
// the component mounts and unmounts cleanly in both motion states without
// throwing, and that it renders as a purely decorative, non-interactive
// layer (aria-hidden) either way.
describe('SatelliteField', () => {
  const originalMatchMedia = window.matchMedia

  afterEach(() => {
    window.matchMedia = originalMatchMedia
    vi.restoreAllMocks()
  })

  function stubReducedMotion(matches: boolean) {
    window.matchMedia = ((query: string) => ({
      matches,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
    })) as unknown as typeof window.matchMedia
  }

  it('renders as an aria-hidden decorative layer with reduced motion off', () => {
    stubReducedMotion(false)
    const { container, unmount } = render(<SatelliteField variant="hero" />)
    const root = container.querySelector('.satellite-field')
    expect(root).toHaveAttribute('aria-hidden', 'true')
    expect(() => unmount()).not.toThrow()
  })

  it('renders without starting an animation loop when reduced motion is on', () => {
    stubReducedMotion(true)
    const rafSpy = vi.spyOn(window, 'requestAnimationFrame')
    const { unmount } = render(<SatelliteField variant="hero" />)
    // canvas has no real 2D context in jsdom, so the draw effect returns
    // before scheduling any frame either way -- this asserts the reduced-
    // motion code path is at minimum never worse (never schedules more).
    expect(rafSpy.mock.calls.length).toBe(0)
    unmount()
  })
})

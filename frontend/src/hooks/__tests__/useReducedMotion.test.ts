import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

function mockMatchMedia(initialMatches: boolean) {
  const listeners: Array<(event: MediaQueryListEvent) => void> = []
  const mql = {
    matches: initialMatches,
    media: '(prefers-reduced-motion: reduce)',
    addEventListener: (_: string, listener: (event: MediaQueryListEvent) => void) => listeners.push(listener),
    removeEventListener: (_: string, listener: (event: MediaQueryListEvent) => void) => {
      const index = listeners.indexOf(listener)
      if (index >= 0) listeners.splice(index, 1)
    },
  }
  window.matchMedia = () => mql as unknown as MediaQueryList
  return {
    fireChange: (matches: boolean) => {
      mql.matches = matches
      for (const listener of listeners) listener({ matches } as MediaQueryListEvent)
    },
  }
}

describe('useReducedMotion', () => {
  const originalMatchMedia = window.matchMedia

  afterEach(() => {
    window.matchMedia = originalMatchMedia
  })

  it('reflects prefers-reduced-motion: reduce when already set at mount', async () => {
    mockMatchMedia(true)
    const { useReducedMotion } = await import('../useReducedMotion')
    const { result } = renderHook(() => useReducedMotion())
    expect(result.current).toBe(true)
  })

  it('reflects no preference by default', async () => {
    mockMatchMedia(false)
    const { useReducedMotion } = await import('../useReducedMotion')
    const { result } = renderHook(() => useReducedMotion())
    expect(result.current).toBe(false)
  })

  it('updates live when the OS-level preference changes mid-session', async () => {
    const { fireChange } = mockMatchMedia(false)
    const { useReducedMotion } = await import('../useReducedMotion')
    const { result } = renderHook(() => useReducedMotion())
    expect(result.current).toBe(false)

    act(() => fireChange(true))
    expect(result.current).toBe(true)
  })
})

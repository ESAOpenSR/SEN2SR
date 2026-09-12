import { useEffect, useState } from 'react'

const QUERY = '(prefers-reduced-motion: reduce)'

/** Tracks the user's OS-level reduced-motion preference live (not just at
 * mount), so ambient motion (satellite field, transitions) can react if it
 * changes mid-session. */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState<boolean>(() => (typeof window !== 'undefined' && window.matchMedia ? window.matchMedia(QUERY).matches : false))

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return
    const mql = window.matchMedia(QUERY)
    const listener = (event: MediaQueryListEvent) => setReduced(event.matches)
    mql.addEventListener?.('change', listener)
    return () => mql.removeEventListener?.('change', listener)
  }, [])

  return reduced
}

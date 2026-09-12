import { useEffect, useState } from 'react'
import * as api from '../api/client'
import type { HealthResponse } from '../api/types'

export type HealthStatus = 'checking' | 'online' | 'offline'

/** Polls GET /health once on mount and periodically thereafter, so the top
 * bar can honestly report whether the FRAME API is reachable rather than
 * assuming it. */
export function useBackendHealth(pollMs = 20000): { status: HealthStatus; health: HealthResponse | null } {
  const [status, setStatus] = useState<HealthStatus>('checking')
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    let cancelled = false

    const check = async () => {
      try {
        const result = await api.getHealth()
        if (!cancelled) {
          setHealth(result)
          setStatus('online')
        }
      } catch {
        if (!cancelled) {
          setHealth(null)
          setStatus('offline')
        }
      }
    }

    check()
    const interval = window.setInterval(check, pollMs)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [pollMs])

  return { status, health }
}

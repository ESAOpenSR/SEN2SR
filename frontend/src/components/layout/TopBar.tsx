import { SatelliteField } from '../atmosphere/SatelliteField'
import { StatusBadge } from '../ui/StatusBadge'
import { useBackendHealth } from '../../hooks/useBackendHealth'
import './TopBar.css'

export function TopBar() {
  const { status, health } = useBackendHealth()

  return (
    <header className="top-bar">
      <SatelliteField variant="ambient" />
      <div className="top-bar__content">
        <div className="top-bar__identity">
          <span className="top-bar__mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" width="20" height="20">
              <ellipse cx="16" cy="16" rx="12" ry="6" fill="none" stroke="var(--border-strong)" strokeWidth="1" />
              <circle cx="16" cy="16" r="3" fill="none" stroke="var(--accent)" strokeWidth="1.4" />
              <circle cx="27" cy="12.6" r="1.4" fill="var(--accent)" />
            </svg>
          </span>
          <div>
            <h1 className="top-bar__title">FRAME</h1>
            <p className="top-bar__subtitle">Sentinel-2 super-resolution &amp; model-stability analysis console</p>
          </div>
        </div>

        <div className="top-bar__status" role="status">
          {status === 'checking' && <StatusBadge tone="neutral">Connecting</StatusBadge>}
          {status === 'online' && <StatusBadge tone="success">API online · v{health?.api_version}</StatusBadge>}
          {status === 'offline' && <StatusBadge tone="error">API unreachable</StatusBadge>}
        </div>
      </div>
    </header>
  )
}

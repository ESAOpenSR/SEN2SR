import './StatusBadge.css'

export type StatusTone = 'neutral' | 'active' | 'success' | 'error' | 'warning'

interface StatusBadgeProps {
  tone: StatusTone
  children: React.ReactNode
}

export function StatusBadge({ tone, children }: StatusBadgeProps) {
  return (
    <span className={`status-badge status-badge--${tone}`}>
      <span className="status-badge__dot" aria-hidden="true" />
      {children}
    </span>
  )
}

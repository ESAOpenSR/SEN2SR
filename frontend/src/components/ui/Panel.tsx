import './Panel.css'

interface PanelProps {
  title?: string
  eyebrow?: string
  action?: React.ReactNode
  children: React.ReactNode
  className?: string
}

/** The one recurring container shape in this app: thin border, flat
 * surface, small corner radius, an optional micro-label heading. Used
 * instead of ad-hoc cards so every section reads as part of one system. */
export function Panel({ title, eyebrow, action, children, className }: PanelProps) {
  return (
    <section className={`panel ${className ?? ''}`}>
      {(title || eyebrow || action) && (
        <header className="panel__header">
          <div>
            {eyebrow && <p className="label">{eyebrow}</p>}
            {title && <h2 className="panel__title">{title}</h2>}
          </div>
          {action}
        </header>
      )}
      <div className="panel__body">{children}</div>
    </section>
  )
}

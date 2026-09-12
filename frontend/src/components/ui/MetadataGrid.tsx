import './MetadataGrid.css'

export interface MetadataItem {
  label: string
  value: React.ReactNode
}

interface MetadataGridProps {
  items: MetadataItem[]
  columns?: number
}

/** Compact label/value technical metadata presentation -- used instead of
 * large cards, per this phase's design direction ("compact technical
 * metadata presentation rather than large cards"). */
export function MetadataGrid({ items, columns = 4 }: MetadataGridProps) {
  return (
    <dl className="metadata-grid" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
      {items.map((item) => (
        <div className="metadata-grid__item" key={item.label}>
          <dt className="label">{item.label}</dt>
          <dd className="metadata-grid__value mono">{item.value}</dd>
        </div>
      ))}
    </dl>
  )
}

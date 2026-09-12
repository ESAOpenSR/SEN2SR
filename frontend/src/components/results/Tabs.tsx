import type { ResultTab } from '../../state/useFrameSession'
import './Tabs.css'

const TABS: { key: ResultTab; label: string }[] = [
  { key: 'overview', label: 'Overview' },
  { key: 'uncertainty', label: 'Uncertainty' },
  { key: 'ndvi', label: 'NDVI' },
  { key: 'metadata', label: 'Metadata' },
]

interface TabsProps {
  active: ResultTab
  onChange: (tab: ResultTab) => void
}

export function Tabs({ active, onChange }: TabsProps) {
  return (
    <div className="result-tabs" role="tablist" aria-label="Result views">
      {TABS.map((tab) => (
        <button
          key={tab.key}
          type="button"
          role="tab"
          aria-selected={active === tab.key}
          className={`result-tabs__tab ${active === tab.key ? 'result-tabs__tab--active' : ''}`}
          onClick={() => onChange(tab.key)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}

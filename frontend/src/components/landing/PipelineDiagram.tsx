import { SR_PRODUCT_DESCRIPTION } from '../../constants/terminology'
import './PipelineDiagram.css'

const STEPS: { label: string; value: string }[] = [
  { label: 'Input', value: '10 m Sentinel-2' },
  { label: 'Model', value: 'FRAME' },
  { label: 'Output', value: SR_PRODUCT_DESCRIPTION },
  { label: 'Signal', value: 'Model-stability uncertainty' },
  { label: 'Downstream', value: 'NDVI analysis' },
]

/** The one-glance summary of what FRAME does, per the phase brief:
 * "INPUT 10 m Sentinel-2 -> FRAME -> SR-derived 2.5 m pixel grid ->
 * UNCERTAINTY -> ANALYSIS". Deliberately compact -- this communicates the
 * whole story before a single byte is uploaded. */
export function PipelineDiagram() {
  return (
    <ol className="pipeline-diagram" aria-label="FRAME processing pipeline overview">
      {STEPS.map((step, index) => (
        <li className="pipeline-diagram__step" key={step.label}>
          <div className="pipeline-diagram__node">
            <span className="label">{step.label}</span>
            <span className="pipeline-diagram__value">{step.value}</span>
          </div>
          {index < STEPS.length - 1 && (
            <span className="pipeline-diagram__connector" aria-hidden="true">
              <svg width="28" height="10" viewBox="0 0 28 10">
                <line x1="0" y1="5" x2="20" y2="5" stroke="var(--border-strong)" strokeWidth="1" />
                <path d="M18 1 L24 5 L18 9" fill="none" stroke="var(--border-strong)" strokeWidth="1" />
              </svg>
            </span>
          )}
        </li>
      ))}
    </ol>
  )
}

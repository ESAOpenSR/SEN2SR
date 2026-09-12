import { STATIC_SCIENTIFIC_CAVEATS } from '../../constants/terminology'
import './ScientificNotes.css'

interface ScientificNotesProps {
  /** When a result exists, pass its actual `scientific_caveats` from the
   * API so this always reflects what the backend itself states, rather
   * than only a static frontend copy. */
  caveats?: string[]
}

/** A small, deliberately unobtrusive scientific-disclosure area -- present
 * throughout the app, never dominating it. */
export function ScientificNotes({ caveats }: ScientificNotesProps) {
  const items = caveats && caveats.length > 0 ? caveats : STATIC_SCIENTIFIC_CAVEATS
  return (
    <div className="scientific-notes">
      <p className="label">Scientific notes</p>
      <ul className="scientific-notes__list">
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  )
}

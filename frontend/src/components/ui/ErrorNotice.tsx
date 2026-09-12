import './ErrorNotice.css'

interface ErrorNoticeProps {
  title: string
  detail: string
  onRetry?: () => void
}

/** The one place API errors surface as UI. `detail` is always the backend's
 * own `detail` string (frame/api/errors.py guarantees no traceback ever
 * reaches this field) or a client-side message for a network failure --
 * never raw exception text. */
export function ErrorNotice({ title, detail, onRetry }: ErrorNoticeProps) {
  return (
    <div className="error-notice" role="alert">
      <div className="error-notice__icon" aria-hidden="true">
        !
      </div>
      <div className="error-notice__body">
        <p className="error-notice__title">{title}</p>
        <p className="error-notice__detail">{detail}</p>
      </div>
      {onRetry && (
        <button type="button" className="error-notice__retry" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  )
}

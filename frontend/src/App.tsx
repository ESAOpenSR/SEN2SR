import { TopBar } from './components/layout/TopBar'
import { LandingSection } from './components/landing/LandingSection'
import { ResultsView } from './components/results/ResultsView'
import { useFrameSession } from './state/useFrameSession'
import './App.css'

function App() {
  const session = useFrameSession()

  return (
    <div className="app-shell">
      <TopBar />
      <main className="app-shell__main">
        {session.hasResult ? (
          <>
            <div className="app-shell__toolbar">
              <p className="app-shell__toolbar-label mono">
                Analyzing <strong>{session.file?.name}</strong>
              </p>
              <button type="button" className="app-shell__reset-button" onClick={session.reset}>
                New analysis
              </button>
            </div>
            <ResultsView session={session} />
          </>
        ) : (
          <LandingSection session={session} />
        )}
      </main>
    </div>
  )
}

export default App

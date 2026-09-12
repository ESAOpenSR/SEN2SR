import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ResultsView } from '../ResultsView'
import { NdviTab } from '../NdviTab'
import type { FrameSession } from '../../../state/useFrameSession'

function baseSession(overrides: Partial<FrameSession>): FrameSession {
  return {
    file: null,
    selectFile: vi.fn(),
    inputScale: 'raw_digital_number',
    setInputScale: vi.fn(),
    uploadStatus: 'idle',
    uploadResult: null,
    uploadError: null,
    jobStatus: 'idle',
    jobResult: null,
    jobError: null,
    analysisStatus: 'idle',
    analysisResult: null,
    analysisError: null,
    activeTab: 'overview',
    setActiveTab: vi.fn(),
    runFrame: vi.fn(),
    runNdviAnalysis: vi.fn(),
    reset: vi.fn(),
    hasResult: false,
    ...overrides,
  }
}

describe('missing-result handling', () => {
  it('ResultsView renders nothing when there is no completed job result', () => {
    const { container } = render(<ResultsView session={baseSession({})} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('NdviTab prompts to run the analysis rather than rendering empty NDVI panels', () => {
    render(<NdviTab analysisStatus="idle" analysisResult={null} analysisError={null} onRunAnalysis={vi.fn()} />)
    expect(screen.getByRole('button', { name: /run ndvi analysis/i })).toBeInTheDocument()
    expect(screen.queryByText('Native 10 m NDVI')).not.toBeInTheDocument()
  })

  it('NdviTab surfaces a clean error and lets the user retry when the analysis request fails', async () => {
    const onRunAnalysis = vi.fn()
    render(<NdviTab analysisStatus="error" analysisResult={null} analysisError="No such SR job (job_id='x')" onRunAnalysis={onRunAnalysis} />)
    expect(screen.getByRole('alert')).toHaveTextContent("No such SR job (job_id='x')")

    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Retry' }))
    expect(onRunAnalysis).toHaveBeenCalled()
  })
})

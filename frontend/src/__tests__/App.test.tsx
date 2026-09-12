import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import App from '../App'

vi.mock('../api/client', () => ({
  getHealth: vi.fn(() => new Promise(() => {})),
  uploadScene: vi.fn(),
  runSr: vi.fn(),
  runNdviAnalysis: vi.fn(),
}))

describe('App', () => {
  it('renders the landing/input state by default, with the upload control and pipeline story visible', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: 'FRAME' })).toBeInTheDocument()
    expect(screen.getByText(/Drop a Sentinel-2 L2A GeoTIFF/i)).toBeInTheDocument()
    expect(screen.getByText(/B04, B03, B02, B08/)).toBeInTheDocument()
    // no result yet -- the results view (tabs, downloads) must not render
    expect(screen.queryByRole('tablist')).not.toBeInTheDocument()
  })
})

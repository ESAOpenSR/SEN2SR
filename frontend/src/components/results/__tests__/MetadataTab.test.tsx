import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { MetadataTab } from '../MetadataTab'
import { jobResultFixture } from '../../../test/fixtures'

describe('MetadataTab result rendering', () => {
  it('renders the actual job identifiers, model, and reproducibility fields from the API response', () => {
    render(<MetadataTab jobResult={jobResultFixture} />)

    expect(screen.getByText(jobResultFixture.job_id)).toBeInTheDocument()
    expect(screen.getByText(jobResultFixture.upload_id)).toBeInTheDocument()
    expect(screen.getByText(jobResultFixture.model_name)).toBeInTheDocument()
    expect(screen.getByText(String(jobResultFixture.uncertainty.seed))).toBeInTheDocument()
    expect(screen.getByText(jobResultFixture.uncertainty.transform_names.join(', '))).toBeInTheDocument()
    expect(screen.getByText(jobResultFixture.bands.join(' · '))).toBeInTheDocument()
    expect(screen.getAllByText(jobResultFixture.crs as string).length).toBeGreaterThanOrEqual(1)
  })

  it('renders self-consistency numbers with the explicit "not ground truth" framing', () => {
    render(<MetadataTab jobResult={jobResultFixture} />)
    expect(screen.getByText(/vs\. own LR input — not ground truth/i)).toBeInTheDocument()
    expect(screen.getByText(jobResultFixture.self_consistency.downsample_rmse!.toFixed(5))).toBeInTheDocument()
  })
})

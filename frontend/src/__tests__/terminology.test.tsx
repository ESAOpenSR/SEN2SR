import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { GROUND_TRUTH_DISCLAIMER, LAM_DISTINCTION, SR_PRODUCT_DESCRIPTION, STATIC_SCIENTIFIC_CAVEATS, UNCERTAINTY_EXPLAINER, UNCERTAINTY_LABEL } from '../constants/terminology'
import { ScientificNotes } from '../components/notes/ScientificNotes'
import { PipelineDiagram } from '../components/landing/PipelineDiagram'

const FORBIDDEN_PHRASES = ['native 2.5 m sentinel-2', 'true 2.5 m image']

function assertNoForbiddenPhrase(text: string) {
  const lower = text.toLowerCase()
  for (const phrase of FORBIDDEN_PHRASES) {
    expect(lower).not.toContain(phrase)
  }
}

describe('scientific terminology constants', () => {
  it('uses the mandated SR product description exactly', () => {
    expect(SR_PRODUCT_DESCRIPTION).toBe('SR-derived product — 2.5 m pixel grid')
  })

  it('uses the mandated uncertainty label exactly', () => {
    expect(UNCERTAINTY_LABEL).toBe('relative model-stability uncertainty')
  })

  it('never uses a forbidden ground-truth phrase in any terminology constant', () => {
    assertNoForbiddenPhrase(SR_PRODUCT_DESCRIPTION)
    assertNoForbiddenPhrase(GROUND_TRUTH_DISCLAIMER)
    assertNoForbiddenPhrase(UNCERTAINTY_LABEL)
    assertNoForbiddenPhrase(UNCERTAINTY_EXPLAINER)
    for (const caveat of STATIC_SCIENTIFIC_CAVEATS) assertNoForbiddenPhrase(caveat)
  })

  it('states Sentinel-2 has never observed the ground at 2.5 m', () => {
    expect(GROUND_TRUTH_DISCLAIMER.toLowerCase()).toContain('never observed the ground at 2.5 m')
    expect(GROUND_TRUTH_DISCLAIMER).toContain('10 m')
  })

  it('explicitly denies calibrated-probability status for the uncertainty signal', () => {
    expect(UNCERTAINTY_EXPLAINER.toLowerCase()).toContain('not a calibrated probability of error')
  })

  it('distinguishes LAM from uncertainty, and only in its own dedicated sentence', () => {
    expect(LAM_DISTINCTION).toContain('LAM')
    expect(LAM_DISTINCTION.toLowerCase()).toContain('uncertainty')
    // LAM must never be conflated into the primary uncertainty label/explainer
    expect(UNCERTAINTY_LABEL.toLowerCase()).not.toContain('lam')
    expect(UNCERTAINTY_EXPLAINER.toLowerCase()).not.toContain('lam')
  })
})

describe('ScientificNotes component', () => {
  it('renders the static caveats with no forbidden phrases when no result exists yet', () => {
    render(<ScientificNotes />)
    for (const caveat of STATIC_SCIENTIFIC_CAVEATS) {
      expect(screen.getByText(caveat)).toBeInTheDocument()
    }
    assertNoForbiddenPhrase(document.body.textContent ?? '')
  })

  it('renders backend-provided caveats verbatim when a result is available', () => {
    const backendCaveats = ['Backend caveat one.', 'Backend caveat two — mentions the SR-derived product — 2.5 m pixel grid.']
    render(<ScientificNotes caveats={backendCaveats} />)
    for (const caveat of backendCaveats) {
      expect(screen.getByText(caveat)).toBeInTheDocument()
    }
  })
})

describe('PipelineDiagram', () => {
  it('communicates the input -> FRAME -> SR-derived -> uncertainty -> analysis story with correct terminology', () => {
    render(<PipelineDiagram />)
    expect(screen.getByText('10 m Sentinel-2')).toBeInTheDocument()
    expect(screen.getByText('FRAME')).toBeInTheDocument()
    expect(screen.getByText(SR_PRODUCT_DESCRIPTION)).toBeInTheDocument()
    expect(screen.getByText('Model-stability uncertainty')).toBeInTheDocument()
    expect(screen.getByText('NDVI analysis')).toBeInTheDocument()
    assertNoForbiddenPhrase(document.body.textContent ?? '')
  })
})

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ProbabilityBars from './ProbabilityBars.jsx'

describe('ProbabilityBars', () => {
  it('shows the FAKE verdict and the animated bar values', () => {
    render(
      <ProbabilityBars
        result={{ label: 'fake', prob_fake: 0.97, prob_real: 0.03, confidence: 0.97 }}
        threshold={0.5}
      />,
    )
    expect(screen.getByText('FAKE')).toBeInTheDocument()
    expect(screen.getByTestId('bar-fake')).toHaveAttribute('aria-valuenow', '97')
    expect(screen.getByTestId('bar-real')).toHaveAttribute('aria-valuenow', '3')
    expect(screen.getByText(/confiance 97.0 %/i)).toBeInTheDocument()
    expect(screen.getByText(/seuil P\(fake\) ≥ 50.0 %/i)).toBeInTheDocument()
  })

  it('shows the REAL verdict for a real prediction', () => {
    render(
      <ProbabilityBars
        result={{ label: 'real', prob_fake: 0.1, prob_real: 0.9, confidence: 0.9 }}
        threshold={0.5}
      />,
    )
    expect(screen.getByText('REAL')).toBeInTheDocument()
    expect(screen.getByTestId('bar-real')).toHaveAttribute('aria-valuenow', '90')
  })

  it('renders a neutral placeholder before any prediction', () => {
    render(<ProbabilityBars result={null} threshold={0.5} />)
    expect(screen.getByText('—')).toBeInTheDocument()
    expect(screen.getByTestId('bar-fake')).toHaveAttribute('aria-valuenow', '0')
  })
})

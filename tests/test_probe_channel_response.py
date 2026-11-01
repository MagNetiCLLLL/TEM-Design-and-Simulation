import sys
from pathlib import Path

import numpy as np
import pytest


def test_gradient_harmonics_separate_requested_terms_and_seventh_order(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from probe_channel_response import TERMS, geometric_coefficients
    from temsim.optics.crossover_aberrations import reference_pencil

    launch, count = reference_pencil(.03, 0.)
    u = launch[:, 0] + 1j * launch[:, 1]
    angle = np.angle(u)
    response = np.zeros_like(u)
    expected = {}
    for k, (term, degree, symmetry) in enumerate(TERMS):
        c = (k+1) * (3e-12 + (2e-12j if symmetry else 0)) / .03**degree
        expected[term] = [c.real, c.imag]
        a = (1+symmetry/(degree+1))/2
        b = 1-a
        response += abs(u)**degree * (a*c*np.exp(1j*(1-symmetry)*angle)
                                      + b*c.conjugate()*np.exp(1j*(1+symmetry)*angle))
    response += 45 * u.conj()**5 * abs(u)**2
    rays = np.column_stack((response.real, response.imag, u.real, u.imag))
    fit = geometric_coefficients(rays, count)
    for term in expected:
        np.testing.assert_allclose(fit["terms_m"][term], expected[term], atol=1e-12, rtol=1e-8)
    assert fit["holdout_rms_pm"] < 1e-8
    assert geometric_coefficients(rays, count, omit_a5=True)["holdout_rms_pm"] > 1.
    rays[-1, 0] = np.nan
    with pytest.raises(ValueError):
        geometric_coefficients(rays, count)

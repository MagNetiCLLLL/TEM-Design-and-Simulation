"""Finite atomic face membership is a preflight, not a phase-error estimate."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.finite_lobato_fourier import build_periodic_lobato_potential


def _source(depth):
    scene = SimpleNamespace(thickness_nm=.1, lattice_reference_thickness_nm=.1,
                            centre_xy_nm=(0., 0.))
    crystal = (np.ones(3), np.array([[.13, .27, depth]]), np.array([14]), {})
    return scene, crystal


@pytest.mark.parametrize("depth", [.5, .5+2e-10, .5-2e-10])
def test_uncertain_atomic_face_is_rejected_before_any_fourier_integral(depth, monkeypatch):
    scene, crystal = _source(depth)
    monkeypatch.setattr("temsim.physics.finite_lobato_fourier._build",
                        lambda *args, **kwargs: pytest.fail("Ambiguous face reached the builder"))
    with pytest.raises(ValueError, match="face membership ambiguous"):
        build_periodic_lobato_potential(crystal, scene, (2., 2.), (2, 2), 16, 2)


def test_unambiguous_unchanged_sites_reach_the_same_builder(monkeypatch):
    scene, crystal = _source(.25)
    class Reached(Exception):
        pass
    def capture(groups, *args, **kwargs):
        np.testing.assert_array_equal(groups[0].z, [.75])
        raise Reached
    monkeypatch.setattr("temsim.physics.finite_lobato_fourier._build", capture)
    with pytest.raises(Reached):
        build_periodic_lobato_potential(crystal, scene, (2., 2.), (2, 2), 16, 2)

"""Geometric similarity of independent perturbation ODEs; no finite source."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.annular_coil_potential import AnnularCoilPotential
from temsim.physics.round_lens_aberration_benchmark import third_order_reference


def test_fixed_ampere_turns_geometric_scaling_preserves_dimensionless_aberration():
    # Scaling every length by s with NI fixed gives B(s*r,s*z)=B(r,z)/s.
    # At fixed electron energy the Lorentz equations therefore have the same
    # dimensionless solution, while image position and Cs scale by s.
    lens = AnnularCoilPotential("similarity", .006, .010, .010, 3000., quadrature_order=8)
    larger = replace(lens, inner_radius_m=.012, outer_radius_m=.020, length_m=.020)
    references = []
    for field, scale in ((lens, 1.), (larger, 2.)):
        references.append(third_order_reference(field, energy_ev=300000.,
            z_start_m=-.12 * scale, focus_search_end_m=1. * scale,
            maximum_step_m=.0015 * scale))
    a, b = references
    assert a["cs_mm"] > 0
    np.testing.assert_allclose(b["image_z_m"], 2 * a["image_z_m"], rtol=3e-9, atol=1e-12)
    np.testing.assert_allclose(b["cs_mm"], 2 * a["cs_mm"], rtol=3e-8)
    np.testing.assert_allclose(b["cc_mm"], 2 * a["cc_mm"], rtol=3e-8)


def test_zero_current_coil_has_no_fictitious_paraxial_focus():
    lens = AnnularCoilPotential("off", .006, .010, .010, 0., quadrature_order=4)
    with pytest.raises(ValueError, match="No first paraxial focus"):
        third_order_reference(lens, energy_ev=300000., z_start_m=-.02,
            focus_search_end_m=.02, maximum_step_m=.005)

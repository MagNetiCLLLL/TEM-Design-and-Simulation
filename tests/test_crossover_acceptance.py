"""Small analytical checks for the incident-column acceptance inventory."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from temsim.optics.crossover_acceptance import evaluate_crossover_chain


def drift(*, count=16, root=2., z=None):
    z = np.arange(5.) if z is None else np.asarray(z, float)
    phi = np.arange(count)*2*np.pi/count
    tx = np.broadcast_to(.002*np.cos(phi), (len(z), count)).copy()
    ty = np.broadcast_to(.002*np.sin(phi), (len(z), count)).copy()
    dz = (z-root)[:, None]*1e-3
    cp = SimpleNamespace(z_mm=z, x_m=dz*tx+3e-6, y_m=dz*ty-4e-6,
                         tx_rad=tx, ty_rad=ty)
    return cp, np.ones(tx.shape, bool), np.ones(count), np.arange(count)+100


def evaluate(cp, mask, weights, identities, **kwargs):
    return evaluate_crossover_chain(cp, mask, weights, identities, source_current_a=7e-9,
        regions={"test_interval": dict(bounds_mm=[0., 4.], requirement="required_2d_waist")},
        maximum_spacing_mm=1., **kwargs)


def test_executed_waist_keeps_chief_and_current_and_refuses_rank_two_map():
    cp, masks, weight, identities = drift()
    originals = {key: getattr(cp, key).copy() for key in ("x_m", "y_m", "tx_rad", "ty_rad")}
    result = evaluate(cp, masks, weight, identities)
    assert result["status"] == "OBSERVED_AT_BOUNDED_SAMPLING"
    assert result["intervals"]["test_interval"]["status"] == "PASS_AT_BOUNDED_SAMPLING"
    waist, = result["executed_waists"]
    assert waist["z_mm"] == 2.
    assert waist["actual"]["statistics"]["mean_x_m"] == pytest.approx(3e-6)
    assert waist["actual"]["current_a"] == pytest.approx(7e-9)
    assert waist["actual"]["alpha95_mrad"] == pytest.approx(1e3*np.arctan(.002))
    assert waist["actual"]["nominal_pupil_edge_mrad"] is None
    assert result["angular_relationships"][0]["common"]["alpha95_ratio"] == pytest.approx(1.)
    assert result["angular_relationships"][0]["empirical_phase_fit"]["status"] == "UNDERDETERMINED"
    assert result["specimen"]["status"] == "ENDPOINT_OBSERVATION_NOT_TWO_SIDED_WAIST"
    json.dumps(result, allow_nan=False)
    for key, expected in originals.items():
        np.testing.assert_array_equal(getattr(cp, key), expected)
    np.testing.assert_array_equal(weight, np.ones(16))


def test_downstream_loss_does_not_erase_upstream_actual_waist():
    cp, masks, weights, identities = drift(root=1.)
    phi = np.arange(8)*2*np.pi/8
    cp.x_m[:, 8:] = 3e-6+1e-7*np.cos(phi)
    cp.y_m[:, 8:] = -4e-6+1e-7*np.sin(phi)
    cp.tx_rad[:, 8:] = 0.
    cp.ty_rad[:, 8:] = 0.
    masks[3:, :8] = False
    result = evaluate(cp, masks, weights, identities)
    waist, = result["executed_waists"]
    assert waist["z_mm"] == 1.
    assert waist["actual"]["current_a"] == pytest.approx(7e-9)
    assert waist["common"]["current_a"] == pytest.approx(3.5e-9)
    assert not waist["common_2d_waist_at_same_bracket"]
    assert result["common_cohort"]["source_ids"] == identities[8:].tolist()
    assert result["coverage"]["population_change_intervals_mm"] == [[2., 3.]]
    assert result["status"] == "UNRESOLVED"


def test_sampling_and_line_focus_cannot_pass_as_a_resolved_2d_waist():
    cp, masks, weights, identities = drift(z=[0., .5, 1., 3., 4.])
    result = evaluate(cp, masks, weights, identities)
    assert result["status"] == "UNRESOLVED"
    assert not result["coverage"]["spacing_satisfied"]
    assert result["unresolved_candidates"]
    cp, masks, weights, identities = drift()
    cp.y_m[:] = np.linspace(-1e-6, 1e-6, len(weights))
    cp.ty_rad[:] = 0.
    result = evaluate(cp, masks, weights, identities)
    assert result["executed_waists"] == []
    assert result["unresolved_candidates"]


@pytest.mark.parametrize("invalid", ["resurrection", "duplicate_id", "nonfinite_live"])
def test_invalid_capture_is_rejected(invalid):
    cp, masks, weights, identities = drift()
    if invalid == "resurrection":
        masks[1, 0] = False
    elif invalid == "duplicate_id":
        identities[0] = identities[1]
    else:
        cp.tx_rad[1, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate(cp, masks, weights, identities)

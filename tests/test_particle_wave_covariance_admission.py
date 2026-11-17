"""Pure-array uncertainty admission; no emitter or wave propagation runs."""
import numpy as np
import pytest

from temsim.physics import particle_wave_source as source_model


VACUUM_UNCERTAINTY = 1./(4.*np.pi)

# A retained covariance from the completed 3000-particle ideal-coil boundary.
# Replaying this matrix is a numerical regression, not a new source trial.
EXECUTED_COVARIANCE = np.array((
    (126269.60095370244, 403.53104447373875, 1.2779256529233975, .003985464407390889),
    (403.531044473739, 126116.54233366922, .004190703402055836, 1.276371127835235),
    (1.2779256529233975, .004190703402055835, 1.2933397969184464e-5, 4.141551439938649e-8),
    (.003985464407390885, 1.276371127835235, 4.1415514399386446e-8, 1.2917609894197029e-5),
))


@pytest.mark.parametrize("reciprocal_scale", (1e-3, 1., 1e3))
def test_completed_source_covariance_is_rejected_under_reciprocal_unit_scaling(reciprocal_scale):
    scale = np.array((reciprocal_scale, reciprocal_scale, 1/reciprocal_scale, 1/reciprocal_scale))
    covariance = EXECUTED_COVARIANCE*scale[:, None]*scale[None, :]
    with pytest.raises(ValueError, match="symplectic eigenemittances") as captured:
        source_model._require_positive_gaussian_residual(covariance, VACUUM_UNCERTAINTY)
    message = str(captured.value)
    emittances = np.fromstring(message.split("[")[1].split("]")[0], sep=",")
    # Independent 80-digit arithmetic on the saved matrix gives these values.
    np.testing.assert_allclose(emittances, (.001119070044, .000913246895), rtol=2e-8, atol=0)
    assert "particle data and tolerances were not modified" in message


@pytest.mark.parametrize("emittances,admitted", (((.02, .04), False), ((.1, .2), True)))
def test_known_williamson_spectrum_survives_canonical_shear_and_scaling(emittances, admitted):
    base = np.diag((*emittances, *emittances))
    zero, identity = np.zeros((2, 2)), np.eye(2)
    symmetric_chirp = np.array(((3., .2), (.2, 2.)))
    shear = np.block([[identity, zero], [symmetric_chirp, identity]])
    reciprocal_units = np.diag((1000., 1000., .001, .001))
    canonical = reciprocal_units @ shear
    covariance = canonical @ base @ canonical.T
    if admitted:
        assert source_model._require_positive_gaussian_residual(covariance, VACUUM_UNCERTAINTY) is None
    else:
        with pytest.raises(ValueError, match="residual covariance would not be positive definite"):
            source_model._require_positive_gaussian_residual(covariance, VACUUM_UNCERTAINTY)


def test_sub_uncertainty_lift_rejects_before_packet_matrix_products(monkeypatch):
    rng = np.random.default_rng(8561)
    count, wavelength = 256, .019687
    positions = rng.normal(size=(count, 2))*.2
    frequency = rng.normal(size=(count, 2))*.001
    slopes = frequency*wavelength/np.sqrt(1.-np.sum((frequency*wavelength)**2, axis=1))[:, None]
    phase = {"x_m": positions[:, 0]*1e-10, "y_m": positions[:, 1]*1e-10,
             "tx_rad": slopes[:, 0], "ty_rad": slopes[:, 1], "ray_id": np.arange(count)}
    monkeypatch.setattr(source_model, "angular_phase_model",
        lambda *args: (np.zeros_like(positions), lambda q: np.zeros(len(q)), {}))
    monkeypatch.setattr(source_model, "square_root", lambda *args: pytest.fail("Packet construction was reached"))
    with pytest.raises(ValueError, match="different quantum-broadening/coherence model must be declared"):
        source_model.lifted_centers(phase, wavelength, np.full(count, 1/count))

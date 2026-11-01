"""Pure operator checks for the two virtual detector windows (no source trial)."""
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scan_joint_probe_si110 import RANGES, cap_single, event_responses
from temsim.specimen.elastic_transport import screened_rutherford_parameter


def test_two_collision_transfer_matches_independent_direction_integral():
    # Integrate over the first physical scattering rotation; no Legendre series
    # or angular moment powers enter the reference calculation.
    delta = screened_rutherford_parameter(14, 300000.)
    mu = np.cos(np.array([0., .031, .048]))
    table = event_responses(mu, np.full_like(mu,delta), 2, 768, 1536)
    az = np.cos(2.*np.pi*(np.arange(128)+.5)/128)
    for k,(lo,hi) in enumerate(RANGES[:2]):
        for j,inc in enumerate(mu):
            def rotated_ring(q):
                s2 = q*delta/(1.+delta-q)
                c = 1.-2.*s2
                mu_out = c*inc + np.sqrt((1.-inc*inc)*4.*s2*(1.-s2))*az
                return np.mean(cap_single(hi*.001,mu_out,delta)-cap_single(lo*.001,mu_out,delta))
            expected, err = quad(rotated_ring,0,1,epsrel=1e-10,epsabs=1e-11)
            assert err < 1e-9
            assert abs(table[k,j,2]-expected) < 1e-8

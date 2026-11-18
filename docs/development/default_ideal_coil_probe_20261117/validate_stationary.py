"""Check sign, units, clocks and containment against analytic particle-optics limits."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.constants import c, e, m_e

from fit_stationary import metrics, write
from observe_stationary import spatial_widths
from stationary_pupil import REST_EV, _transport, electric_mesh, half_nodes, kinematics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    rng = np.random.default_rng(7930)
    n, length = 1024, 0.016
    initial = np.zeros((n, 5))
    initial[:, :2] = rng.normal(0.0, 2.0e-6, (n, 2))
    initial[:, 2:4] = rng.normal(0.0, 0.006, (n, 2))
    energy = np.linspace(299997.0, 300003.0, n)
    kin = kinematics(energy)
    pc = np.sqrt(energy * (energy + 2 * REST_EV))
    norm = np.sqrt(1.0 + np.sum(initial[:, 2:4] ** 2, axis=1))
    evidence = {}
    for name, bz in (("vacuum_drift", 0.0), ("uniform_axial_B", 0.16)):
        exact = initial.copy()
        tx, ty = initial[:, 2], initial[:, 3]
        if bz:
            omega = c * bz * norm / pc
            angle = omega * length
            sine, cosine = np.sin(angle), np.cos(angle)
            exact[:, 0] += (tx * sine - ty * (1 - cosine)) / omega
            exact[:, 1] += (tx * (1 - cosine) + ty * sine) / omega
            exact[:, 2] = tx * cosine - ty * sine
            exact[:, 3] = tx * sine + ty * cosine
        else:
            exact[:, :2] += length * initial[:, 2:4]
        exact[:, 4] += length * norm * kin[1]
        rows = []
        for steps in (40, 80):
            z = np.linspace(0.0, length, steps + 1)
            field = np.zeros((2 * steps + 1, 20))
            field[:, 0] = bz
            actual = _transport(initial, field, np.diff(z), *kin, np.array([steps]), False)[0][0]
            err = np.max(abs(actual - exact), axis=0)
            rows.append(dict(step_m=length / steps, max_absolute_x_y_tx_ty_time=err.tolist()))
        if np.any(err > np.array([1e-11, 1e-11, 3e-10, 3e-10, 1e-22])) or (
            bz and rows[0]["max_absolute_x_y_tx_ty_time"][2] / err[2] < 14
        ):
            raise ValueError(f"Independent analytic reference failed for {name}")
        evidence[name] = rows

    gradient = 2.1e6
    z = np.linspace(0.0, length, 81)
    nodes = half_nodes(z)
    ps = SimpleNamespace(
        metadata={"basis": "axisymmetric-r2-z-bilinear-v1"},
        arrays={"r": np.array([0.0, 0.01]), "z": z, "voltage": np.tile(gradient * z, (2, 1))},
    )
    axial = np.zeros_like(initial)
    actual = _transport(
        axial,
        np.zeros((len(nodes), 20)),
        np.diff(z),
        *kin,
        np.array([len(z) - 1]),
        False,
        electric_mesh(ps, nodes),
        energy,
    )[0][0]
    final_energy = energy + gradient * length
    rest = m_e * c * c / e
    exact_time = (
        np.sqrt(final_energy * (final_energy + 2 * rest)) - np.sqrt(energy * (energy + 2 * rest))
    ) / (c * gradient)
    error = float(np.max(abs(actual[:, 4] - exact_time)))
    if error > 5e-22 or np.any(actual[:, :4] != 0):
        raise ValueError("Independent axial acceleration or flight time failed")
    evidence["uniform_axial_E"] = dict(
        gradient_v_per_m=gradient, kinetic_gain_ev=gradient * length, max_time_error_s=error
    )
    for name, weights in (("equal", np.ones(n)), ("unequal", rng.uniform(0.01, 1.0, n))):
        ref = metrics(initial, weights, 1.0)
        measured = spatial_widths(initial[:, :2], weights)
        if not np.allclose(measured, [ref["D95_nm"], ref["rms_radius_nm"]], atol=1e-10, rtol=1e-13):
            raise ValueError("Focus widths disagree with production current quantiles")
        evidence["widths_" + name] = dict(D95_nm=measured[0], RMS_radius_nm=measured[1])
    write(
        out,
        dict(
            scope="Analytic zero-current reference trajectories and software checks",
            references=n,
            length_m=length,
            checks=evidence,
        ),
    )


if __name__ == "__main__":
    main()

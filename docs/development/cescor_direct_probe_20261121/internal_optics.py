"""Measure CESCOR relay blocks, physical source images and pupil harmonics."""

import argparse
import json

import numpy as np

from column_current import FACTOR
from observe import axis_crossings
from relay_transfer import FOIL, HP1, HP2, RADIUS, SURFACE, TL21, TL22, Native, advance, save


def relay_blocks(model, field, planes):
    """Mechanical slopes are converted to rotating-frame slopes at both ends."""
    grid, iz, frac = model.t.tables
    mid = grid.shape[1] // 2
    bz = grid[:, mid, mid, 2]
    axis = field[:, 0] + (1 - frac) * bz[iz] + frac * bz[iz + 1]
    energy = np.median(model.en) - np.median(model.phi0) + model.t.ep[:, 0, 0]
    k = (FACTOR * model.t.pc * axis / (energy * (energy + 2 * model.t.rest)) ** 0.5).get()
    theta = np.r_[0.0, np.cumsum(np.diff(model.m.z) * (k[:-2:2] + 4 * k[1::2] + k[2::2]) / 6)]
    idx = model.indices(planes)
    eye = np.eye(2)
    zero = np.zeros((2, 2))
    skew = np.array([[0.0, -1.0], [1.0, 0.0]])
    result = []
    for start, end in zip(idx, idx[1:]):
        delta = theta[end] - theta[start]
        rotation = eye * np.cos(delta) + skew * np.sin(delta)
        to_lab = np.block([[eye, zero], [k[2 * start] * skew, eye]])
        from_lab = np.block([[rotation.T, zero], [-k[2 * end] * rotation.T @ skew, rotation.T]])
        trials = []
        for scale in (1.0, 2.0):
            d = scale * np.array([1e-7, 1e-7, 1e-6, 1e-6])
            phase = np.zeros((8, 5))
            phase[:4, :4] = np.diag(d)
            phase[4:, :4] = -np.diag(d)
            ((out, loss),) = advance(
                model.t, field, phase, np.full(8, np.median(model.en)),
                np.full(8, np.median(model.phi0)), np.full(8, -1, np.int32), start, [end]
            )
            if (loss != -1).any():
                raise ValueError("Paraxial diagnostic intercepted")
            lab = ((out[:4, :4] - out[4:, :4]) / (2 * d[:, None])).T
            matrix = from_lab @ lab @ to_lab
            trials.append(dict(
                mechanical_lab_SI=lab.tolist(), larmor_SI=matrix.tolist(),
                A=float(np.trace(matrix[:2, :2]) / 2), B_mm=float(np.trace(matrix[:2, 2:]) * 500),
                C_per_m=float(np.trace(matrix[2:, :2]) / 2), D=float(np.trace(matrix[2:, 2:]) / 2),
                B_max_mm=float(1000 * np.linalg.norm(lab[:2, 2:], 2)),
                A_singular=np.linalg.svd(lab[:2, :2], compute_uv=False).tolist(),
                lab_image_azimuth_deg=float(np.rad2deg(np.arctan2(lab[1, 0], lab[0, 0]))),
                larmor_matrix_determinant=float(np.linalg.det(matrix)),
            ))
        result.append(dict(z_mm=[model.m.z[start] * 1000, model.m.z[end] * 1000],
                           larmor_rotation_deg=float(np.rad2deg(delta)), perturbation_checks=trials))
    return result


def physical_pupil(front, surface, weights):
    """Predict unseen original landings from foil position only. Harmonics refer to laboratory azimuth."""
    foil = front[:, 0] + 1j * front[:, 1]
    radius = np.abs(foil) / RADIUS
    az = np.angle(foil)
    landing = (surface[:, 0] + 1j * surface[:, 1]) * 1e6
    order = np.random.default_rng(68432).permutation(len(foil))
    train, test = order[::2], order[1::2]
    # A continuous piecewise-linear radial profile shared by all harmonic families.
    knots = np.unique(np.r_[0, np.quantile(radius[train], np.linspace(0, 1, 17)[1:-1]), 1])
    ix = np.searchsorted(knots, radius, side="right").clip(1, len(knots) - 1)
    fraction = (radius - knots[ix - 1]) / (knots[ix] - knots[ix - 1])
    linear = np.zeros((len(radius), len(knots)))
    linear[np.arange(len(radius)), ix - 1] = 1 - fraction
    linear[np.arange(len(radius)), ix] = fraction
    center = np.average(landing[train], weights=weights[train])
    target = landing - center
    power = np.average(abs(target[test]) ** 2, weights=weights[test])
    models = []
    for modes in [(1,), (1, -2, 4), (1, -2, 4, -5, 7)]:
        basis = np.column_stack([linear * np.exp(1j * m * az[:, None]) for m in modes])
        root_w = np.sqrt(weights[train])
        coefficients, _, rank, _ = np.linalg.lstsq(
            root_w[:, None] * basis[train], root_w * target[train], rcond=1e-10
        )
        residual = target[test] - basis[test] @ coefficients
        mse = np.average(abs(residual) ** 2, weights=weights[test])
        models.append(dict(modes=modes, rank=int(rank), test_residual_rms_um=float(np.sqrt(mse)),
                           fraction_of_test_variance_explained=float(1 - mse / power)))
    return dict(train=len(train), test=len(test), centered_test_rms_um=float(np.sqrt(power)), models=models)


def source_images(axial):
    """Bracket two-axis stationary RMS sizes of the same particles throughout the column."""
    with np.load(axial) as data:
        z = data["z_m"]
        moments = data["common_moments"] if "common_moments" in data else data["moments"]
        return {name: axis_crossings(z, moments, a, b) for name, (a, b) in {
            "HP2_to_HP1": (HP2, HP1), "TL22_to_TL21": (TL22, TL21),
            "between_scan_coils": (1.3875, 1.4125), "whole_core_and_scan": (1.10, 1.425)
        }.items()}


def inspect(case, settings, table, pupil_mm, axial=None, step=.075):
    rec = json.loads(settings.read_text())
    q, ni = rec["currents_A"], rec["objective_At"]
    planes = [HP2, HP1, pupil_mm * .001]
    waists = source_images(axial) if axial is not None else {}
    scan = waists.get("between_scan_coils", [])
    source_pair = []
    if len(scan) == 2 and all(len(brackets) == 1 for brackets in scan):
        source_pair = [(TL21 + TL22) / 2, float(np.mean(scan)) * .001]
    model = Native(table, step, 32768, ni, planes + source_pair)
    f, (front, (sample, lost)) = model.replay(q, [FOIL[0], SURFACE])
    live = lost == -1
    report = dict(relays=relay_blocks(model, f, planes), emitted=len(live), survived=int(sum(live)),
                  physical_pupil=physical_pupil(front[0][live], sample[live], model.t.weights[live]))
    if waists:
        report["source_image_waists_mm"] = waists
    if source_pair:
        report["TL_center_to_scan_waist"] = model.relay(q, source_pair)["0-1"]
    from adjust_cs import coefficients
    from fit_threefold import angular_sample
    from publish_cs import independent_fits
    from temsim.physics.beam_statistics import transverse_beam_statistics

    p, w = sample[live], model.t.weights[live]
    angular = angular_sample(p, model.t.ids[live], 1e9 * model.t.current * w,
                             transverse_beam_statistics(*p[:, :4].T, weights=w))
    report["specimen_angular_fit"] = dict(coefficients=coefficients(angular),
                                         held_out_RMS_um=independent_fits(angular, omit=("C3", "C5", "A5")))
    report["specimen_angular_fit"]["angular_range_checks"] = {}
    for maximum in (20, 25):
        mask = angular.alpha_mrad <= maximum
        fits = coefficients(angular, mask)
        report["specimen_angular_fit"]["angular_range_checks"][str(maximum)] = dict(
            packets=int(mask.sum()), C3_mm=fits["C3"]["signed_mm"], C5_mm=fits["C5"]["signed_mm"]
        )
    save(case, report)


if __name__ == "__main__":
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case")
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--axial", type=Path)
    p.add_argument("--pupil-mm", type=float, default=1408.05425036)
    p.add_argument("--step-mm", type=float, default=.075)
    a = p.parse_args()
    inspect(a.case, a.settings, a.table, a.pupil_mm, a.axial, a.step_mm)

"""Propose a C2/C3 Microprobe zoom from captured axial fields."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from numba import njit
from scipy.optimize import least_squares

from temsim.physics.beam_statistics import transverse_beam_statistics
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.physics.round_lens_aberration_benchmark import _charge_over_momentum
from validate_physical_hybrid import prepare

OUT = Path(__file__).resolve().parent
PROFILE = ROOT / "docs/development/fege_focus_20261110/normal-nanoprobe-si110.toml"
ARCHIVED_SOURCE = ROOT / "docs/development/fege_focus_20261110/extractor/new-n2048/gun_exit.npz"


def micro_state(count=1024, step=.2):
    args = SimpleNamespace(profile=PROFILE, step_mm=step, source_particles=count,
        optical_control_seed=None, optical_controls=None, static_acquisition=True,
        coherence_length_nm=None, camera_acquisition=True, detector="camera",
        silicon_110=True, thickness_nm=.4, sample_pixels=2048, sample_width_nm=64.)
    state, selection, changes = prepare(args)
    state.mini_condenser.set_probe_mode("micro_probe")
    state.illumination_mode = "TEM"
    state.acceleration_backend = "CPU"
    state.acceleration_enabled = False
    return state, selection, changes


@njit(cache=True)
def transfer(bz, lengths, rigidity, marks):
    result = np.empty((len(marks), 4))
    a, b, c, d = 1., 0., 0., 1.
    j = 0
    for i in range(len(bz)):
        k = abs(rigidity * bz[i])
        h = lengths[i]
        sine = np.sin(k*h)/k if k else h
        cosine = np.cos(k*h)
        kick = -(k*k)*sine
        a, b, c, d = cosine*a+sine*c, cosine*b+sine*d, kick*a+cosine*c, kick*b+cosine*d
        if j < len(marks) and i+1 == marks[j]:
            result[j] = (a, b, c, d)
            j += 1
    return result


def main():
    state = micro_state()[0]
    gun = state.electron_gun
    source_path = OUT / "extractor/new-n1024/source.npz"
    source_path = source_path if source_path.is_file() else ARCHIVED_SOURCE
    with np.load(source_path, allow_pickle=False) as data:
        xy = np.c_[data["gun_exit_x_m"], data["gun_exit_y_m"]]
        slopes = np.c_[data["gun_exit_tx_rad"], data["gun_exit_ty_rad"]]
        weight = data["gun_exit_weight"]
        live = data["gun_exit_alive"]
    centered_xy = xy-np.average(xy[live], axis=0, weights=weight[live])
    centered_slopes = slopes-np.average(slopes[live], axis=0, weights=weight[live])
    curvature = np.sum(weight[live, None]*centered_xy[live]*centered_slopes[live])/np.sum(weight[live, None]*centered_xy[live]**2)
    aperture = state.condenser_aperture_2
    surface = state.sample.upper_surface_z_mm
    landmarks = np.array([aperture.z_mm, surface])
    z = np.unique(np.r_[np.linspace(gun.exit_plane_z_mm, surface, 16000), landmarks])*1e-3
    marks = np.searchsorted(z, landmarks*1e-3)
    points = np.c_[np.zeros((len(z)-1, 2)), (z[:-1]+z[1:])*.5]
    keys = [part.key for part in state.lenses if part.enabled and part.z_mm <= state.sample.z_mm]
    fields = {}
    for key in keys:
        field = capture_round_lens_potential(state, key).potential.field
        local = field.registration.positions_global_to_local_m(points)
        low, high = np.asarray(field.native_support_mm)*1e-3
        inside = (local[:, 2] > low) & (local[:, 2] < high)
        values = np.zeros(len(points))
        values[inside] = field.field_at_global_positions_t(points[inside])[:, 2]
        fields[key] = values
    lenses = {part.key: part for part in state.lenses}
    c2, c3 = "condenser_lens_2", "condenser_lens_3"
    nominal = np.array([lenses[key].percent for key in (c2, c3)])
    maximum = np.array([lenses[key].max_percent for key in (c2, c3)])
    a2, a3 = fields[c2]/nominal[0], fields[c3]/nominal[1]
    fixed = sum(row for key, row in fields.items() if key not in (c2, c3))
    rigid = float(_charge_over_momentum(gun.nominal_exit_energy_ev))*.5
    polar = rigid*(fixed[-1]+nominal[0]*a2[-1]+nominal[1]*a3[-1])
    lengths = np.diff(z)

    def matrices(values):
        return transfer(fixed+values[0]*a2+values[1]*a3, lengths, rigid, marks).reshape(2, 2, 2)

    def observation(values):
        upper, lower = matrices(values)
        pupil = upper[0, 0]*xy+upper[0, 1]*slopes
        offset = np.array((aperture.offset_x_mm, aperture.offset_y_mm))*1e-3
        keep = live & (np.linalg.norm(pupil-offset, axis=1) < aperture.radius_mm*1e-3)
        if not keep.any():
            return dict(alpha95_mrad=1e6, diameter95_um=0., current_fraction=0., count=0)
        position = lower[0, 0]*xy+lower[0, 1]*slopes
        slope = lower[1, 0]*xy+lower[1, 1]*slopes+polar*np.c_[-position[:, 1], position[:, 0]]
        measured = transverse_beam_statistics(position[:, 0], position[:, 1], slope[:, 0], slope[:, 1],
                                               alive=keep, weights=weight)
        return dict(alpha95_mrad=measured.convergence_95_mrad,
                    alpha_rms_mrad=1e3*measured.convergence_rms_rad,
                    diameter95_um=2e6*measured.radius_95_m, rms_radius_um=1e6*measured.radius_rms_m,
                    current_fraction=float(weight[keep].sum()), count=int(keep.sum()))

    pencil = np.array((1., curvature))
    candidates = []
    starts = [nominal, *[np.array((x, y)) for x in (4., 8., 10.5, 15., 25., 40., 70.)
                         for y in (2., 10., 20., 35., 55., 75.)]]
    for target in (.75, 1., 1.5, 2., 2.2, 3., 5., 10.):
        hits = []
        # The signed sample height is allowed on both image-parity branches.
        # Aperture filtering remains an approximate proposal calculation.
        for parity in (1., -1.):
            for initial in starts:
                def residual(values):
                    m = matrices(values)
                    slope = (m[-1]@pencil)[1]
                    row = observation(values)
                    return np.array(((row["diameter95_um"]-target)/max(target, .5),
                                     slope*parity/100.))
                solved = least_squares(residual, initial, bounds=(np.array((.01, .01)), maximum),
                                       diff_step=1e-5, max_nfev=80, ftol=1e-9, xtol=1e-10, gtol=1e-8)
                measured = observation(solved.x)
                if np.linalg.norm(solved.fun) > .03 or any(np.linalg.norm(solved.x-old) < 1e-4 for old in hits):
                    continue
                hits.append(solved.x)
                candidates.append(dict(target_diameter95_um=target, c2_percent=float(solved.x[0]),
                    c3_percent=float(solved.x[1]), residual=solved.fun.tolist(), **measured))
    candidates.sort(key=lambda row: (row["target_diameter95_um"], row["alpha95_mrad"]))
    result = dict(scope="Zero-current axial-field first-order proposal only",
        mini_signed_excitation_percent=state.mini_condenser.signed_excitation_percent,
        objective_percent=state.objective_lens.percent, source_path=str(source_path),
        source_curvature_per_m=float(curvature), polarity_only_baseline=observation(nominal),
        axial_bz_at_surface_t=float(polar/rigid), candidates=candidates,
        limitations="Full finite-source Lorentz trajectories, nonlinear corrector fields and all physical stops require validation")
    (OUT / "paraxial-proposals.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

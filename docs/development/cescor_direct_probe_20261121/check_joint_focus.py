"""Replay the objective-only setting at independent integration and sampling resolutions."""

import argparse

import numpy as np

from joint_pivot_tuning import stations
from joint_relay_search import model_at, mono_roots
from native_objective_search import Profile
from normal_working_point import EVIDENCE
from relay_transfer import SURFACE, advance
from scan_coil_waist_origin import nearest
from temsim.physics.beam_statistics import transverse_beam_statistics
from trace_normal_joint_field import beam_moments
from verify_joint_alignment import keep_json


def verify(amps, ni):
    rows = []
    for count, step, half in (
        (32768, 0.0375, 24),
        (32768, 0.01875, 24),
        (32768, 0.0375, 8),
        (16384, 0.0375, 24),
    ):
        model, drives = model_at(amps, ni, count, step, half)
        field = model.field(drives)
        focus = SURFACE + np.array([-300000.0, 0.0, 300000.0]) * 1e-9
        grid = stations(model, np.r_[np.arange(1.34, SURFACE, 0.0005), focus])
        phase, contact, first = model.t.input, model.loss, 0
        moments, states, counts = [], [], []
        stop = model.indices([SURFACE])[0]
        focus_ids = model.indices(focus)
        for ids in np.array_split(model.indices(grid), (len(grid) + 15) // 16):
            data = advance(model.t, field, phase, model.en, model.phi0, contact, first, ids)
            for i, (phase, contact) in zip(ids, data, strict=True):
                live = contact == -1
                counts.append(int(live.sum()))
                moments.append(beam_moments(phase, live, model.t.weights))
                if i in focus_ids:
                    states.append(phase[live, :4] - np.mean(phase[live, :4], axis=0))
                if i == stop:
                    statistic = transverse_beam_statistics(
                        *phase[:, :4].T, alive=live, weights=model.t.weights
                    )
                    idx = model.t.ids[live].copy()
            first = ids[-1]
        if min(counts) != max(counts):
            raise ValueError("Changing cohort on the final approach")
        if not rows:
            reference_ids = idx
        moments = np.asarray(moments)
        z_mm = grid * 1000
        radial = mono_roots(z_mm, moments[:, 1] + moments[:, 2])
        window = (z_mm >= 1380) & (z_mm <= 1420)
        profile = Profile(np.asarray(states)).minimum()
        np.testing.assert_allclose(
            profile["surface_um"], statistic.illumination_diameter_95_um, rtol=2e-13
        )
        row = dict(
            emitted=count,
            step_mm=step,
            transverse_pitch_mm=3.6 / half,
            specimen_survivors=len(idx),
            same_full_source_IDs=bool(np.array_equal(reference_ids, idx)) if count == 32768 else None,
            independent_D95_um=statistic.illumination_diameter_95_um,
            surface_rms_um=statistic.radius_rms_m * 1e6,
            incident_three_plane_profile=profile,
            radial_waists_from_1340_mm=radial.tolist(),
            ac_radial_waists_mm=radial[(radial >= 1380) & (radial <= 1420)].tolist(),
            scan_largest_principal_growth_um_mrad=float(moments[window, 4].max()),
            radial_rms_at_AC_um=moments[nearest(z_mm, np.array([1387.5, 1412.5])), 0].tolist(),
        )
        rows.append(row)
        keep_json(EVIDENCE / "joint-objective-focus-sensitivity.json", rows)
        print(count, step, half, row["independent_D95_um"], profile["offset_nm"], radial, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amps", type=float, required=True)
    p.add_argument("--ni", type=float, required=True)
    args = p.parse_args()
    verify(args.amps, args.ni)

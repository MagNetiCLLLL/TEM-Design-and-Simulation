"""Verify the original finite source, the three planes, and the objective-only finish."""

import argparse
import json

import numpy as np

from column_current import START, SURFACE, current_ledger, source_digest
from condenser_normal import spans_mm
from illuminated_focus import focus_offsets_nm
from joint_pivot_tuning import OUT, focal_planes, scan_response, stations
from joint_relay_search import NI, REFERENCE, model_at, mono_roots, surface_metrics
from normal_working_point import EVIDENCE, file_hash
from relay_transfer import FOIL, advance
from scan_waist_audit import check_planes
from temsim.corrector_field_cli import _save
from trace_normal_joint_field import beam_moments, display_rays


def keep_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def scan(args):
    pre = OUT / "aligned-scan.json"
    mix = np.asarray(
        REFERENCE["after"]["downstream_calibration"]["station_command_matrix"]
        if args.fit
        else json.loads(pre.read_text())["scan"]["mix"]
    )
    m, q = model_at(args.amps, args.ni, args.count, args.step, 24)
    front = focal_planes(m, q)
    roots = {k: np.array(v) * 0.001 for k, v in front.items()}
    target = min(front["mini_and_objective"], key=lambda x: abs(x - 1490.0))
    del m
    m, q = model_at(
        args.amps, args.ni, args.count, args.step, 24, extra=np.concatenate(list(roots.values()))
    )
    check = check_planes(m, q, roots, [])
    measured = scan_response(m, q, mix, target, calibrate=args.fit)
    keep_json(
        pre if args.fit else OUT / "focused-scan.json",
        dict(
            tl12_A=args.amps,
            objective_At=args.ni,
            previous_mix=mix.tolist(),
            front_mm=front,
            direct_focal_checks=check,
            scan=measured,
            source_digest=source_digest(m.t),
        ),
    )
    print("Response:", args.amps, args.ni, target, measured["candidates"], flush=True)


def capture(args):
    m, q = model_at(args.amps, args.ni, args.count, args.step, args.half)
    t = m.t
    if args.count == 32768 and source_digest(t) != REFERENCE["source_digest"]:
        raise ValueError("Gun boundary changed")
    f = m.field(q)
    first = advance(t, f, t.input, m.en, m.phi0, m.loss, 0, m.indices([*FOIL, SURFACE]))
    surface, loss = first[-1]
    chosen, bands = display_rays(first[0], first[1], loss)
    # The through-focus experiment continues the same incident trajectories immediately
    # beyond the surface in the incident column field; there is no specimen interaction.
    axial = stations(
        m,
        np.r_[
            START,
            np.arange(0.45, SURFACE, 0.0005),
            SURFACE,
            SURFACE + focus_offsets_nm() * 1e-9,
            np.arange(1.343, 1.524, 0.000125),
            FOIL,
            spans_mm().ravel() * 0.001,
        ],
    )
    ids = m.indices(axial)
    stop = m.indices([SURFACE])[0]
    j = 0
    phase, contact = t.input, m.loss
    moments, profiles, tracks = [], [], []
    for indices in np.array_split(ids, (len(ids) + 15) // 16):
        batch = advance(t, f, phase, m.en, m.phi0, contact, j, indices)
        for index, (phase, contact) in zip(indices, batch, strict=True):
            live = contact == -1
            w = t.weights[live]
            moments.append(beam_moments(phase, live, t.weights))
            xy = phase[live, :2]
            r = np.linalg.norm(xy - np.average(xy, weights=w, axis=0), axis=1)
            rank = int(np.ceil(len(r) * 0.95)) - 1
            d95 = 2e6 * np.partition(r, rank)[rank]
            profiles.append(
                [d95, live.sum(), t.current * 1e9 * w[(xy * xy).sum(axis=1) <= (50e-9) ** 2].sum()]
            )
            if index <= stop:
                tracks.append(np.where(live[chosen, None], phase[chosen, :4], np.nan))
            if index == stop:
                np.testing.assert_array_equal(phase, surface)
                np.testing.assert_array_equal(contact, loss)
        j = indices[-1]
    rows = np.asarray(moments)
    profile = np.asarray(profiles)
    s = surface_metrics(m, surface, loss)
    edge = m.edge(f, m.entrance(f, first[0]), azimuths=96)
    z = axial * 1000
    ends = spans_mm()
    splits = {
        "C2_C3_gap": (ends[0, 1], ends[1, 0]),
        "C3_ADL_gap": (ends[1, 1], ends[2, 0]),
        "whole_column": (z[0], z[-1]),
        "TL12_to_scan_exit": (1345.0, 1420.0),
        "downstream_of_lower_scan": (1420.0, z[-1]),
        "corrector_transfer_doublet": (1166.35, 1249.95),
    }
    growth = {
        "radial": rows[:, 1] + rows[:, 2],
        "x": rows[:, 1],
        "y": rows[:, 2],
        "principal_low": rows[:, 3],
        "principal_high": rows[:, 4],
    }
    topology = {}
    for name, (lo, hi) in splits.items():
        sel = (z >= lo) & (z <= hi)
        topology[name] = {k: mono_roots(z[sel], v[sel]).tolist() for k, v in growth.items()}
    marker = 0.00030
    near = np.flatnonzero(abs(axial - SURFACE) <= marker + 1e-14)
    k = int(near[np.argmin(profile[near, 0])])
    offsets = (axial - SURFACE) * 1e9
    if k in (near[0], near[-1]):
        bracket = None
    else:
        bracket = offsets[[k - 1, k + 1]].tolist()
    tag = args.tag
    raw = EVIDENCE / f"joint-{tag}-particles.npz"
    _save(
        raw,
        source_id=t.ids,
        weight=t.weights,
        energy_offset_eV=t.de,
        gun_exit_phase=t.input,
        surface_phase=surface,
        surface_loss=loss,
        z_mm=z,
        moments=rows,
        profiles=profile,
        track_z_mm=z[ids <= stop],
        tracks=np.asarray(tracks),
        track_source_id=t.ids[chosen],
        track_band=bands,
    )
    front = focal_planes(m, q)
    s.update(
        native_D95_min_offset_nm=float(offsets[k]),
        native_D95_min_um=float(profile[k, 0]),
        native_D95_bracket_nm=bracket,
        near_focus_search_mm=[-0.3, 0.3],
        minimum_rms_offset_nm=float(offsets[near[np.argmin(rows[near, 0])]]),
        core_peak_within_scan_nA=float(profile[near, 2].max()),
        core_peak_offsets_nm=offsets[near[profile[near, 2] == profile[near, 2].max()]][
            [0, -1]
        ].tolist(),
    )
    keep_json(
        EVIDENCE / f"joint-{tag}.json",
        dict(
            TL12_A=args.amps,
            objective_At=args.ni,
            objective_A=args.ni / 1160,
            pupil_um=100.0,
            source_digest=source_digest(t),
            emitted=args.count,
            gun_exit_mm=START * 1000,
            specimen_top_mm=SURFACE * 1000,
            emitted_A=t.current,
            step_mm=args.step,
            field_sha256=file_hash(m.path),
            particles_sha256=file_hash(raw),
            surface=s,
            current_ledger=current_ledger(m.m, t, loss),
            focal_mm=front,
            waists_mm=topology,
            edge_mrad=edge["angle_mrad_min_median_max"],
            edge_survivors=edge["reached_sample"],
            edge_probes=edge["azimuths"],
            fixed_vacuum_currents_A=q.tolist(),
            moment_columns=[
                "rms_um",
                "cov_x_um_mrad",
                "cov_y_um_mrad",
                "min_principal_um_mrad",
                "max_principal_um_mrad",
            ],
            profile_columns=["D95_um", "survivors", "axis_50nm_nA"],
        ),
    )
    print("Captured", tag, s, topology["downstream_of_lower_scan"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["scan", "capture"])
    p.add_argument("--amps", type=float, required=True)
    p.add_argument("--ni", type=float, default=NI)
    p.add_argument("--fit", action="store_true")
    p.add_argument("--count", type=int, default=32768)
    p.add_argument("--step", type=float, default=0.0375)
    p.add_argument("--half", type=int, default=24)
    p.add_argument("--tag", default="objective-focus")
    args = p.parse_args()
    (scan if args.stage == "scan" else capture)(args)

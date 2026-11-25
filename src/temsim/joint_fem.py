"""Run the saved, zero-scan Nanoprobe material-field working point from its physical tip."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from temsim.paths import project_root


def load_profile(path=None):
    if path is None:
        path = project_root() / "configs/operating_modes/fei_50v_hpol_a2/profile.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))


def input_file(profile, name):
    entry = profile["assets"][name]
    path = project_root() / entry["path"]
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
            raise ValueError(f"Field checkpoint checksum mismatch: {path.name}")
    return path


def gun_inputs(profile, electro, gun):
    """Build the actual gun fields and material stops for physical and differential rays."""
    keys = (
        "r",
        "z",
        "voltage",
        "cut_index",
        "lookup",
        "origin",
        "inverse",
        "grad",
        "phi0",
    )
    arrays = tuple(electro[k] for k in keys)
    r, z, phi = arrays[:3]
    support = 0.01 * np.minimum(np.maximum(0.0, z), profile["minimum_ring_bore_m"])
    idx = np.maximum(0, np.searchsorted(r, support, side="right") - 1)
    idx[idx < 2] = 0
    ss = r[idx] ** 2
    slope = np.divide(
        phi[idx, np.arange(len(z))] - phi[0], ss, out=np.zeros(len(z)), where=ss > 0
    )
    metal = np.array(
        [
            [row[k] for k in ("start_m", "stop_m", "inner_m", "outer_m")]
            for row in profile["gun_stops"]
        ]
    )
    mag = np.ascontiguousarray(
        CubicSpline(gun["z_m"], gun["coeff"], axis=0).c.transpose(1, 2, 3, 0)
    )
    return (*arrays, ss, slope), mag, metal


def run(output, *, profile=None):
    from numba import get_num_threads, set_num_threads
    from temsim.physics.condenser_relay import (
        beam_widths,
        gun_object_and_fan,
        source_images,
    )
    from temsim.physics.fei_tip_transport import transport
    from temsim.physics.joint_fem_column import propagate

    if profile is None:
        profile = load_profile()
    measured_relay = profile.get("measured_relay")
    if measured_relay and measured_relay.get("field_sha256") != profile["solution_sha256"]:
        raise ValueError("The saved corrector measurements refer to a different magnetic field")
    inputs = {key: input_file(profile, key) for key in profile["assets"]}
    set_num_threads(min(12, get_num_threads()))
    with (
        np.load(inputs["gun_electric"]) as electro,
        np.load(inputs["gun_magnetic"]) as gun,
        np.load(inputs["column_magnetic"]) as column,
        np.load(inputs["tracking"]) as mesh,
    ):
        receipts = [
            json.loads(str(gun["receipt"])),
            json.loads(str(column["receipt"])),
            *json.loads(str(column["layer_receipts"])),
        ]
        for data in receipts:
            if (
                data["solution_sha256"] != profile["solution_sha256"]
                or data["bias_A"] != profile["excitation_A"]
            ):
                raise ValueError(
                    "Field samples and winding currents belong to different FEM solutions"
                )
        if str(electro["sha"]) != profile["electric_sha"]:
            raise ValueError("Gun electrostatic geometry mismatch")
        ef, mag, metal = gun_inputs(profile, electro, gun)
        phase, weights = mesh["phase"], mesh["weight_uA"]
        if (
            not isinstance(profile["source_count"], int)
            or profile["source_count"] < 1000
            or phase.shape != (profile["source_count"], 6)
            or weights.shape != (len(phase),)
            or not np.isfinite(phase).all()
            or not np.isfinite(weights).all() or not np.all(weights > 0)
            or not np.isclose(weights.sum(), profile["source_current_uA"])
        ):
            raise ValueError("Invalid physical tip population")
        gun_z = np.unique(
            np.r_[np.linspace(0, 0.040, 161), np.linspace(0.040, 0.45, 165)]
        )
        events, ends, gun_paths = transport(
            phase, ef, mag, metal, gun_z, len(phase), 0.5
        )
        if np.any(events < -2):
            raise ValueError("Unresolved first hit in the gun")
        ids = np.flatnonzero(events == -1)
        object_z, fan_at_gun_exit = gun_object_and_fan(ef, mag, metal)
        states = np.concatenate((ends[ids], fan_at_gun_exit))
        endpoint, loss, tracks, column_z = propagate(mesh, column, states)
        # The independent optical probes never enter current, landing or waist statistics.
        fan_paths = tracks[len(ids) :]
        endpoint, loss, column_paths = (
            endpoint[: len(ids)],
            loss[: len(ids)],
            tracks[: len(ids)],
        )
        upper = column_z <= 1.070000001
        relay_z = column_z[upper]
        owners = np.flatnonzero(mesh["keys"] == "condenser_aperture_2")
        foil_radius = profile["c2_aperture_um"] * 0.5e-6
        foil_spans = np.isin(mesh["owner"], owners) & np.isclose(
            mesh["radius"], foil_radius, rtol=1e-8, atol=1e-12
        )
        if not np.any(foil_spans):
            raise ValueError("The tracking geometry has no matching C2 aperture foil")
        foil_last = np.flatnonzero(foil_spans)[-1]
        after_foil = (loss == -1) | (loss > foil_last)
        accepted = column_paths[after_foil][:, upper]
        accepted_weights = weights[ids[after_foil]]
        live_width, column_waists = beam_widths(column_z, column_paths, weights[ids])
        accepted_width, accepted_waists = beam_widths(
            relay_z, accepted, accepted_weights
        )
        full_images = source_images(column_z, fan_paths)
        images = [image for image in full_images if image["z_mm"] <= 1000 * relay_z[-1]]
        selected_accepted = np.linspace(
            0, len(accepted) - 1, min(len(accepted), 40), dtype=int
        )
        relay_summary = dict(
            interlens_interval_mm=profile.get("c1_interlens_interval_mm"),
            gun_object_z_mm=None if object_z is None else float(1000 * object_z),
            angular_step_urad=1.0,
            position_step_um=0.1,
            source_images=images,
            C2_foil_exit_z_mm=float(1000 * mesh["z"][foil_last + 1]),
            C2_transmitted=int(after_foil.sum()),
            C2_transmitted_nA=float(1000 * accepted_weights.sum()),
            C2_bundle_waists_mm=accepted_waists,
            all_live_waists_mm={
                key: [z for z in roots if z <= relay_z[-1] * 1000]
                for key, roots in column_waists.items()
            },
        )
        surface = ids[loss == -1]
        if not len(surface):
            raise ValueError("No physical primaries reached the specimen")
        xy = endpoint[loss == -1, :2] * 1e6
        w = weights[surface]
        centre = np.average(xy, axis=0, weights=w)
        dist = np.linalg.norm(xy - centre, axis=1)
        order = np.argsort(dist)
        d95 = 2 * dist[order[np.searchsorted(np.cumsum(w[order]), 0.95 * w.sum())]]
        clipped = np.flatnonzero(loss >= 0)
        background = clipped[
            np.linspace(0, len(clipped) - 1, min(36, len(clipped)), dtype=int)
        ]
        selected = np.r_[np.flatnonzero(loss == -1), background]
        shown = ids[selected]
        z_mm = np.r_[gun_z, column_z[1:]] * 1000
        paths = (
            np.concatenate((gun_paths[shown], column_paths[selected, 1:]), axis=1)
            * 1000
        )
        summary = dict(
            profile=profile["name"],
            field_sha256=profile["solution_sha256"],
            mesh_id=profile["mesh_id"],
            field_tables={k: v["sha256"] for k, v in profile["assets"].items()},
            matched_field_receipts=len(receipts),
            materials=profile["materials"],
            material_regions=profile["material_regions"],
            currents_A=profile["excitation_A"],
            tetrahedra=profile["tetrahedra"],
            gun_lens_V=profile["gun_lens_V"],
            c2_aperture_um=profile["c2_aperture_um"],
            emitted=len(phase),
            gun_exit=len(ids),
            sample=len(surface),
            effective_count=float(w.sum() ** 2 / (w @ w)),
            gun_exit_nA=float(weights[ids].sum() * 1000),
            sample_nA=float(w.sum() * 1000),
            specimen_z_mm=float(z_mm[-1]),
            D95_um=float(d95),
            condenser_relay=relay_summary,
            full_column_source_images=full_images,
            column_live_waists_mm=column_waists,
        )
        if measured_relay:
            downstream = column_z >= 1.0
            _, specimen_waists = beam_widths(
                column_z[downstream], column_paths[loss == -1][:, downstream], w
            )
            summary["measured_relay"] = measured_relay
            summary["specimen_cohort_waists_mm"] = specimen_waists
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "trajectories.npz",
        z_mm=z_mm,
        paths_mm=paths,
        selected_ids=shown,
        emitter_weights_uA=weights,
        gun_events=events,
        gun_exit_ids=ids,
        column_loss=loss,
        gun_exit_phase=ends[ids],
        sample_slopes=endpoint[loss == -1, 2:],
        sample_ids=surface,
        sample_xy_um=xy,
        sample_weights_uA=w,
        relay_z_mm=relay_z * 1000,
        relay_live_rms_um=live_width[upper] * 1e6,
        relay_C2_rms_um=accepted_width * 1e6,
        relay_display_xy_um=accepted[selected_accepted] * 1e6,
        relay_display_ids=ids[after_foil][selected_accepted],
        C2_transmitted_ids=ids[after_foil],
    )
    from temsim.gui.joint_fem_figures import render

    render(output, summary)
    (output / "profile.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    (output / "result.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--profile", type=Path, help="Replay this saved joint FEM operating point."
    )
    args = parser.parse_args()
    run(args.output, profile=load_profile(args.profile))

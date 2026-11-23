"""Compare the historical field graph to a fresh, owner-resolved winding and FEM assembly."""

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from column_current import BASIS, OWNER, POINT, WORK, desktop_state
from condenser_source_fields import (
    EV,
    FEM,
    LENSES,
    MAT,
    OBJ,
    OUT,
    PROJECTION,
    Fields,
    sha,
    write,
)
from condenser_source_search import axial_nodes, electrostatic, transfer
from condenser_material_binding import check_materials
from temsim.physics.corrector_vacuum_field import vacuum_field
from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field


def peak(z, values):
    index = np.argmax(abs(values))
    return dict(Bz_T=float(values[index]), z_mm=float(1000 * z[index]))


def check_gun_replay(fields, drives):
    from condenser_source_trace import packet_input

    current = [float(drives[0]), float(drives[4])]
    with TemporaryDirectory() as temp_name:
        folder = Path(temp_name)
        proposal = folder / "proposal.json"
        packet = folder / "gun-after-32768.npz"
        shutil.copyfile(OUT / packet.name, packet)
        with patch("condenser_source_trace.OUT", folder):
            write(proposal, {"series_A": current})
            selected, source = packet_input(fields, "after", 32768)
            np.testing.assert_array_equal(selected, drives)
            assert (source[1][:32768] > 0).all()

            def rejected():
                try:
                    packet_input(fields, "after", 32768)
                except ValueError as exc:
                    if "gun exit was produced" in str(exc):
                        return True
                    raise
                raise AssertionError("An inconsistent gun handoff was accepted")

            write(proposal, {"series_A": [current[0] + 0.01, current[1]]})
            changed_current = rejected()
            write(proposal, {"series_A": current})
            shutil.copyfile(POINT / "gun.npz", packet)
            legacy_exit = rejected()
    return dict(
        validated_positive_source=32768,
        rejected_C1_change_A=0.01,
        changed_current_rejected=changed_current,
        unrelated_archive_rejected=legacy_exit,
    )


def audit():
    fields = Fields()
    after = fields.current(*json.loads((OUT / "proposal.json").read_text())["series_A"])
    with np.load(MAT) as s:
        sol = {k: s[k] for k in ("recipe", "report", "corners_m", "moments_Am2")}
    check_materials(fields.state)
    with np.load(WORK / "column/electric.npz") as old:
        for k in ("r", "z", "voltage"):
            np.testing.assert_array_equal(old[k], getattr(fields.electric.base_field, k))
    with np.load(BASIS) as b:
        z = b["z"]
        rd = b["round"][:, :, 0]
        np.testing.assert_array_equal(b["owners"], OWNER)
        unit = fields.round_axis(z)
        errors = np.max(np.abs(unit - rd), axis=1)
        worst = float(errors.max())
        assert worst < 1e-9
    with np.load(WORK / "column/columns.npz") as c:
        np.testing.assert_array_equal(c["z"], z)
        remaining = c["groups"][0, :, 0] - c["transfers"][:, :, 0].sum(axis=0)
    post = np.isin(OWNER, PROJECTION)
    np.testing.assert_allclose(remaining, fields.before[post] @ unit[post], rtol=1e-7, atol=1e-10)
    residual = float(abs(remaining - fields.before[post] @ unit[post]).max())
    mask = (z >= 0.45) & (z <= 1.07)
    oldround = remaining + fields.before[~post] @ rd[~post]
    newround = fields.before @ unit
    diff = float(np.max(abs((oldround - newround)[mask])))
    nominal = capture_instrument_magnetic_field(desktop_state(32768)[1])
    nominal_at_exit = nominal.field_at_global_positions_t([[0.0, 0.0, 0.45]])[0]
    candidates = {
        "before": fields.before,
        "after": after,
        "C1_off": fields.before.copy(),
        "both_off": fields.before.copy(),
    }
    candidates["C1_off"][:4] = 0.0
    candidates["both_off"][:8] = 0.0
    z = np.linspace(0, 1.07, 10701)
    unit = fields.round_axis(z)
    reg = (z >= 0.45) & (z <= 0.55)
    own_obj = fields.objective(z)[:, 0]
    joint = np.interp(z, fields.zjoint, fields.bjoint[:, 2])
    gun = z < 0.45
    joint[gun] = fields.far_corrector(z[gun])[:, 0, 2]
    profiles = {}
    summaries = {}
    for k, q in candidates.items():
        bz = np.array([q[OWNER == owner] @ unit[OWNER == owner] for owner in LENSES])
        total = q @ unit + own_obj + joint
        profiles[k] = np.vstack([bz, total])
        summaries[k] = dict(
            lens_peaks=[peak(z, curve) for curve in bz],
            C1_housing_peaks=[peak(z[reg], curve[reg]) for curve in (*bz, total)],
            Bz_at_gun_exit_T=float(np.interp(0.45, z, total)),
            net_At=[
                float(
                    q[OWNER == name]
                    @ np.asarray([c.ampere_turns for c in fields.units])[OWNER == name]
                )
                for name in LENSES
            ],
            total_housing_peak=peak(z[reg], total[reg]),
            current_A=q.tolist(),
        )
    ci = np.argmax(abs(profiles["before"][0]))
    peak_terms = (fields.before[:4] * unit[:4, ci]).tolist()
    # Check the gun approximant at off-grid XYZ holdouts against the same raw solution, independently
    # upgrading the induced-field volume quadrature to four nodes per tetrahedron.
    rng = np.random.default_rng(7970)
    xyz = rng.uniform(-1.0, 1.0, (45, 3))
    xyz[:, :2] *= 25e-6
    xyz[:, 2] = 0.225 * (1 + xyz[:, 2])
    xyz[-1] = [0, 0, 0.45]
    joint_xyz = xyz.copy()
    joint_xyz[:, 2] -= 0.45
    reference = vacuum_field(joint_xyz, sol, quadrature=4)

    def winding_vector(q):
        result = np.zeros_like(xyz)
        for coil, current in zip(fields.units, q):
            if current:
                result += current * coil.field_at_global_positions_t(xyz)
        dz = fields.objective(xyz[:, 2])
        s = np.sum(xyz[:, :2] ** 2, axis=1)
        radius = -dz[:, 1] / 2 + s * dz[:, 3] / 16
        result[:, :2] += xyz[:, :2] * radius[:, None]
        result[:, 2] += dz[:, 0] - s * dz[:, 2] / 4
        return result

    gun_error = {
        k: float(
            np.max(
                np.linalg.norm(
                    fields.gun_field(q).field_at_global_positions_t(xyz)
                    - winding_vector(q)
                    - reference,
                    axis=1,
                )
            )
        )
        for k, q in candidates.items()
    }
    # Retain a first-order check over the *entire* gun as well as the column section.
    zz = axial_nodes(0.0002)
    mid = (zz[1:] + zz[:-1]) / 2
    ep, ef = electrostatic(zz, 0.3)

    def source_zeros(total):
        coeff = np.interp(mid, z, total)
        _, _, _, _, tr = transfer(zz, ep, ef, coeff, True)
        flips = np.flatnonzero(tr[:-1, 1] * tr[1:, 1] < 0)
        return zz[flips] - tr[flips, 1] * np.diff(zz)[flips] / np.diff(tr[:, 1])[flips]

    zeros = source_zeros(profiles["after"][3])
    after_controls = {}
    for label, removed in [("C1_off", [0]), ("C2_off", [1]), ("both_off", [0, 1])]:
        after_controls[label] = (
            1000 * source_zeros(profiles["after"][3] - profiles["after"][removed].sum(axis=0))
        ).tolist()
    sig = fields.gun_key(after, 32768)
    altered = after.copy()
    altered[0] += 0.01
    assert sig != fields.gun_key(altered, 32768) and sig != fields.gun_key(after, 4096)
    provenance = dict(
        unit_max_difference_T_per_A=worst,
        unit_max_by_winding_T_per_A=errors.tolist(),
        upstream_total_round_Bz_difference_T=diff,
        historical_background_minus_named_projection_max_T=residual,
        historical_background_projection_owners=list(PROJECTION),
        winding_count=len(fields.units),
        comparison_grid_points=len(rd[0]),
        historical_basis_sha256=sha(BASIS),
        objective_file_sha256=sha(OBJ),
        active_material_field_sha256=sha(FEM),
        raw_joint_sha256=sha(MAT),
        live_joint_recipe_matches=True,
        live_objective_geometry_and_material_match=True,
        live_electric_grid_matches=True,
        physical_gun_input_key=fields.physical_key,
        after_gun_key=sig,
        gun_replay_tests=check_gun_replay(fields, after),
        current_desktop_factory_B_at_exit_T=nominal_at_exit.tolist(),
        archived_gun_table_sha256=json.loads((POINT / "gun.json").read_text())["magnetic_table"][
            "coefficient_sha256"
        ],
        prior_archive_reconstruction=json.loads((EV / "illuminated-normal.json").read_text())[
            "gun_boundary"
        ]["archived_nominal_field_reconstruction"],
        gun_vector_holdout_max_T=gun_error,
        raw_exterior_B_at_exit_T=reference[-1].tolist(),
        joint_table_B_at_exit_T=fields.bjoint[0].tolist(),
        after_central_energy_all_tip_image_zeros_mm=(1000 * zeros).tolist(),
        after_first_order_off_controls_tip_image_zeros_mm=after_controls,
        before_C1_groups_T_at_peak=peak_terms,
        group_turns=[int(c.ampere_turns) for c in fields.units],
        windings=[
            dict(
                source=c.source_id,
                center_mm=c.center_z_m * 1000,
                length_mm=c.length_m * 1000,
                inner_radius_mm=c.inner_radius_m * 1000,
                outer_radius_mm=c.outer_radius_m * 1000,
                turns=c.ampere_turns,
            )
            for c in fields.units
        ],
        summaries=summaries,
    )
    write(OUT / "field-provenance.json", provenance)
    np.savez_compressed(OUT / "fields.npz", z_m=z, objective_T=own_obj, joint_T=joint, **profiles)
    print(
        "audit",
        worst,
        diff,
        residual,
        gun_error,
        "image zeros",
        zeros * 1000,
        "interface",
        reference[-1],
        fields.bjoint[0],
        flush=True,
    )


if __name__ == "__main__":
    audit()

"""Reconstruct saved emitter samples and inspect archives; no propagation.

Run from the repository with .venv/Scripts/python.exe and redirect stdout to a
NEW JSON file. Original archives are only read. This does not invoke a gun
trace, field solve, column propagation, optimizer, or coherent calculation.
"""
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from temsim.optics.electron_gun.emitter import ColdFieldEmitter, _halton_dimensions
from temsim.physics.ray_identity import emission_reference

BASE = ROOT / "outputs/joint-probe-100um-30mrad-20261028"
CASES = {
    193: "seed193-h02-v2/candidates/trial-000001",
    769: "seed769-h02/candidates/trial-000001",
    3073: "joint22-trial47-n3073-h02/candidates/trial-000001",
}
CODE = (
    "src/temsim/optics/electron_gun/emitter.py",
    "src/temsim/physics/chromatic.py",
    "src/temsim/physics/ray_identity.py",
    "src/temsim/optics/electron_gun/tip_curvature.py",
)


def file_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = sha256(str((value.dtype.str, value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def quantiles(value):
    return dict(zip(("min", "p01", "p05", "p50", "p90", "p95", "p99", "max"),
                    np.quantile(value, [0, .01, .05, .5, .9, .95, .99, 1]).tolist()))


def run():
    reconstructed, cases = {}, {}
    for count, relative in CASES.items():
        folder = BASE / relative
        snapshot = json.loads((folder / "inputs.json").read_text())
        candidate = json.loads((folder / "candidate.json").read_text())
        node = next(node for node in snapshot["graph"]["nodes"]
                    if node["type"].endswith(":ColdFieldEmitter"))
        attrs = node["attributes"]
        excluded_models = ("_surface_model", "_tip_curvature_nm_inv", "_tip_coherence",
                           "_emission_sampling", "_emission_quadrature", "_emission_cdf_quadrature",
                           "_tuning_boundary_probes", "_tuning_surface_probes")
        if any(key in attrs for key in excluded_models):
            raise ValueError("This audit is restricted to the captured legacy flat source")
        emitter = ColdFieldEmitter(**{field.name: attrs[field.name] for field in fields(ColdFieldEmitter)})
        # Source construction only. This function never invokes propagation.
        bundle = emitter.emit()
        reconstructed[count] = bundle
        reference = emission_reference(bundle)
        kinetic = emitter.emission_energy_ev + bundle.energy_offset_ev
        radial = np.hypot(bundle.x_m, bundle.y_m) * 1e9
        angular = np.hypot(bundle.tx_rad, bundle.ty_rad) * 1e3
        coordinates = np.asarray(_halton_dimensions(count, (2, 3, 5, 7, 11, 13))).T
        with np.load(folder / "particles.npz", allow_pickle=False) as data:
            checks = {key: bool(np.array_equal(reference[key], data["emission_" + key]))
                      for key in ("position_m", "direction", "ray_id", "normal")}
            checks.update(
                source_ids_equal_emission_ids=bool(np.array_equal(data["source_ray_id"], reference["ray_id"])),
                weights_exact_one_over_count=bool(np.array_equal(data["weights"], np.full(count, 1 / count))),
                all_source_ids_unique=len(np.unique(data["source_ray_id"])) == count,
            )
            if not all(checks.values()):
                raise ValueError(f"Source reconstruction mismatch at N={count}")
            surface_x, surface_y = data["x_m"][-1], data["y_m"][-1]
            live = data["alive"] & (data["weights"] > 0)
            center_x = np.average(surface_x[live], weights=data["weights"][live])
            center_y = np.average(surface_y[live], weights=data["weights"][live])
            surface_r = np.hypot(surface_x - center_x, surface_y - center_y) * 1e9
            exit_r = np.hypot(data["gun_exit_x_m"], data["gun_exit_y_m"]) * 1e6
            c2_r = np.hypot(data["c2_x_m"], data["c2_y_m"]) * 1e6
            energy_error = data["gun_exit_kinetic_energy_ev"] - 300000. - kinetic
            neighbors = []
            for ray_id in range(303, 310) if count == 769 else ([306] if count > 306 else []):
                i = int(np.flatnonzero(data["source_ray_id"] == ray_id)[0])
                neighbors.append(dict(
                    source_id=ray_id, halton_sequence_index=ray_id + 18,
                    cdf_coordinates_2_3_5_7_11_13=coordinates[i].tolist(),
                    launch_xy_nm=[float(bundle.x_m[i] * 1e9), float(bundle.y_m[i] * 1e9)],
                    launch_radius_nm=float(radial[i]),
                    launch_slopes_mrad=[float(bundle.tx_rad[i] * 1e3), float(bundle.ty_rad[i] * 1e3)],
                    launch_slope_norm_mrad=float(angular[i]), launch_kinetic_ev=float(kinetic[i]),
                    launch_radius_empirical_percentile=float(np.mean(radial <= radial[i]) * 100),
                    launch_slope_norm_empirical_percentile=float(np.mean(angular <= angular[i]) * 100),
                    launch_kinetic_empirical_percentile=float(np.mean(kinetic <= kinetic[i]) * 100),
                    gun_exit_kinetic_ev=float(data["gun_exit_kinetic_energy_ev"][i]),
                    gun_exit_radius_um=float(exit_r[i]),
                    gun_exit_radius_empirical_percentile=float(np.mean(exit_r <= exit_r[i]) * 100),
                    c2_reference_radius_um=float(c2_r[i]),
                    surface_centered_radius_nm=float(surface_r[i]),
                    surface_radius_empirical_percentile=float(np.mean(surface_r[live] <= surface_r[i]) * 100),
                    archived_absolute_weight=float(data["weights"][i]),
                ))
            hashes = {key: array_hash(data[key]) for key in (
                "emission_position_m", "emission_direction", "emission_ray_id", "emission_normal",
                "source_ray_id", "weights", "gun_exit_kinetic_energy_ev", "gun_exit_energy_offset_ev")}
            correlation = np.corrcoef(np.asarray([bundle.x_m, bundle.y_m, bundle.tx_rad, bundle.ty_rad, kinetic]))
            radial_correlation = np.corrcoef(np.asarray([radial, angular, kinetic]))
            cases[str(count)] = dict(
                archive_directory=str(folder.relative_to(ROOT)),
                input_digest=snapshot["digest"], implementation=snapshot["implementation"],
                raw_files_sha256={name: file_hash(folder / name) for name in ("inputs.json", "candidate.json", "particles.npz")},
                selected_arrays_sha256=hashes, source_reconstruction_checks=checks,
                emitter_parameters={field.name: attrs[field.name] for field in fields(ColdFieldEmitter)},
                source_policy="Legacy flat-tip six-dimensional Halton; no selected surface/Sobol/product/explicit-CDF/curvature model",
                launch_energy_evidence="Reconstructed by emitter.emit from saved emitter parameters; initial kinetic energy is not a saved emission_reference array",
                launch_radius_nm=quantiles(radial), launch_slope_norm_mrad=quantiles(angular),
                launch_kinetic_ev=quantiles(kinetic),
                launch_kinetic_mean_ev=float(kinetic.mean()), launch_kinetic_std_ev=float(kinetic.std()),
                kinetic_pearson_x_y_tx_ty=correlation[-1, :-1].tolist(),
                kinetic_pearson_radius_slope_norm=radial_correlation[-1, :-1].tolist(),
                maximum_abs_gun_exit_minus_300keV_minus_reconstructed_launch_ev=float(np.max(np.abs(energy_error))),
                inspected_nodes=neighbors,
                archived_surface_metrics={key: candidate["observation"].get(key) for key in (
                    "alpha95_mrad", "d90_nm", "d95_nm", "rms_radius_nm", "local_waist_nm", "current_a")},
            )
    refinement = []
    for first, second in ((193, 769), (769, 3073)):
        a, b = reconstructed[first], reconstructed[second]
        delta = b.energy_offset_ev[:first] - a.energy_offset_ev
        refinement.append(dict(counts=[first, second],
            prefix_xy_slopes_identical=all(np.array_equal(getattr(a, key), getattr(b, key)[:first])
                                         for key in ("x_m", "y_m", "tx_rad", "ty_rad")),
            prefix_energies_identical=bool(np.array_equal(a.energy_offset_ev, b.energy_offset_ev[:first])),
            maximum_existing_node_energy_change_ev=float(np.max(np.abs(delta))),
            mean_existing_node_energy_change_ev=float(delta.mean())))
    return dict(
        schema="temsim-source-emission-readonly-audit-v1", execution_scope="Source emit reconstruction and archived-array statistics only; no propagation",
        code_files_sha256={name: file_hash(ROOT / name) for name in CODE},
        audit_script_sha256=file_hash(Path(__file__)), cases=cases, source_count_refinement=refinement,
        conclusions=[
            "Projected positions and slopes use radial-cutoff Gaussian disks; the six-dimensional joint source is not an all-Gaussian or uniform distribution.",
            "Energy combines an exponential term and logistic-like quantiles, then sample-dependent centering, RMS scaling and a bounded sigmoid map.",
            "Changing N preserves old position/direction nodes but changes their kinetic energies through sample-dependent normalization.",
            "No source ID permutation or weight alteration was found; saved emission arrays exactly match reconstruction.",
            "Seed769 ID306 is at ordinary initial position radius but high initial slope norm and energy; its gun-exit radius is already the largest in that archive.",
            "These observations do not establish whether the later tail is a correct physical field response, a numerical error, or both.",
        ], limitations=[
            "The 3073 archive has trial47 optical controls, while 193/769 are original seeds. Downstream radii across these archives are not a fixed-optics convergence comparison.",
            "Source-law parameters are identical apart from N; pure emitter reconstruction isolates source sampling from downstream controls.",
            "Initial kinetic energy was reconstructed, not read from a separately saved launch-energy array. Agreement with archived gun-exit energy is supporting evidence.",
            "Small pairwise sample correlations do not certify full joint-distribution independence or quadrature convergence.",
            "No field-step, independent-source-rule, experimental-source or aberration attribution validation was performed.",
        ])


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))

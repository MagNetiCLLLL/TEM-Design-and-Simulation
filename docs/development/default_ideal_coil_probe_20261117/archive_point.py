"""Archive a measured stationary probe, its executable inputs and numerical checks."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np

from export_profile import controls as desktop_controls
from fit_stationary import metrics, write
from stationary_pupil import CHANNELS, current_vector
from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.working_point import WorkingPointCheckpoint
from temsim.working_point_export import make_portable_inputs


def archive(recipe_path, trial, observation, out):
    out.mkdir(parents=True, exist_ok=True)
    snapshot = InstrumentSnapshot.from_dict(json.loads((trial / "inputs.json").read_text()))
    state = snapshot.restore()
    recipe = json.loads(recipe_path.read_text())
    gun = json.loads((trial / "gun.json").read_text())
    report = json.loads((observation / "optics.json").read_text())
    if (
        report["input_digest"] != snapshot.digest
        or gun["input_digest"] != snapshot.digest
        or gun["sha256"] != report["gun_sha256"]
        or sha256((trial / "gun.npz").read_bytes()).hexdigest() != gun["sha256"]
    ):
        raise ValueError("The gun, currents and observation must describe the same executed setting")
    components = {c.key: c for c in state.corrector_elements}
    if any(
        state.lens_field_map_descriptors[key]["ideal_current_segments"] != spec
        for key, spec in recipe["recipes"].items()
    ) or any(
        getattr(components[key], attr) != recipe["multipole_commands"][key][attr] for key, attr in CHANNELS
    ):
        raise ValueError("Recipe differs from the executed physical controls")
    with np.load(observation / "surface.npz") as rays, np.load(trial / "gun.npz") as source:
        live = source["alive"] & (source["weight"] > 0)
        if (
            str(rays["input_digest"]) != snapshot.digest
            or float(rays["z_mm"]) != state.sample.upper_surface_z_mm
            or not np.array_equal(rays["ray_id"], source["ray_id"][live])
            or not np.array_equal(rays["weight"], source["weight"][live])
        ):
            raise ValueError("The transmitted IDs or absolute weights differ from this unobstructed gun")
        check = metrics(rays["phase"], rays["weight"], float(rays["emitted_current_a"]))
        for key, val in check.items():
            if not np.isclose(val, report["full"]["surface"][key], rtol=2e-12, atol=1e-21):
                raise ValueError(f"Surface particles do not reproduce {key}")
    controls = {key: recipe[key] for key in ("owner_per_group", "current_a", "recipes", "multipole_commands")}
    controls["status"] = "Executed parked ideal-coil controls at the specimen upper surface"
    write(out / "currents.json", controls)
    for source, name in (
        (trial / "gun.npz", "gun.npz"),
        (observation / "surface.npz", "surface.npz"),
        (trial / "gun.json", "gun.json"),
    ):
        shutil.copyfile(source, out / name)
    for base, name in ((trial, "inputs"), (observation, "optics")):
        (out / (name + ".json.gz")).write_bytes(
            gzip.compress((base / (name + ".json")).read_bytes(), compresslevel=9, mtime=0)
        )
    archived = WorkingPointCheckpoint(
        snapshot,
        {},
        state.sample.upper_surface_z_mm,
        snapshot.physical_digest,
        {"package_kind": "INSTRUMENT_INPUTS_ONLY"},
    )
    portable = make_portable_inputs(archived)
    package = out / "portable-input.temwp"
    portable.write_package(package, overwrite=True)
    restored = WorkingPointCheckpoint.read_package(package)
    if restored.digest != portable.digest or desktop_controls(
        restored.compatible_state()
    ) != desktop_controls(state):
        raise ValueError("Portable input package did not restore the measured optical controls and field")

    def positions(interval):
        return [dict(z_mm=row["z_mm"], bracket_mm=row["bracket_mm"]) for row in interval["regions"]]

    condenser = {
        key: dict(
            observed=row["observed"],
            required=row["required"],
            passed=row["passed"],
            regions=positions(row["evidence"]),
        )
        for key, row in report["condensers"].items()
    }
    corrector = {key: positions(row) for key, row in report["corrector"]["intervals"].items()}
    width, angle = check["D95_nm"], check["alpha95_mrad"]
    focus = report["full"]["D95"]
    # Compare in the original mm coordinate. Subtracting two coordinates near
    # 1599 mm before converting to nm can round an endpoint just beyond 1 nm.
    surface_mm = state.sample.upper_surface_z_mm
    waist_pass = focus["status"] == "BOUNDED_OBSERVED_MINIMUM" and all(
        surface_mm - 1e-6 <= v <= surface_mm + 1e-6 for v in focus["neighbour_bracket_mm"]
    )
    summary = dict(
        specimen_upper_surface_mm=state.sample.upper_surface_z_mm,
        model=state.round_lens_field_model,
        integration_step_mm=report["step_mm"],
        focus_spacing_nm=report["focus_spacing_nm"],
        emitted=gun["population"],
        surviving=report["positive_surviving"],
        effective=report["effective"],
        current_a=report["current_a"],
        full=report["full"],
        nested=report["prefixes"],
        acceptance=dict(
            alpha95_within_29p7_to_30p3=29.7 <= angle <= 30.3,
            D95_focus_within_1_nm=waist_pass,
            condensers=all(v["passed"] for v in condenser.values()),
            corrector=report["corrector"]["passed"],
            no_extra_waist=report["direct"]["passed"],
        ),
        condenser=condenser,
        corrector=corrector,
        final_interval_mm=report["direct"]["bounds_mm"],
        closest_aperture=min(report["mechanical_clearances"], key=lambda row: row["clearance_mm"]),
        column_field_comparison=report["field_witness"],
        gun_field_comparison=gun.get("magnetic_table"),
        full_field_local_focus={k: v for k, v in report["native_focus"].items() if k != "full_profile"},
        input_digest=snapshot.digest,
        physical_input_digest=snapshot.physical_digest,
        solver_source_identity=snapshot.implementation,
        portable_input_digest=portable.snapshot.digest,
        ordered_round_winding_groups=len(current_vector(controls)),
        sha256={
            f.name: sha256(f.read_bytes()).hexdigest()
            for f in sorted(out.iterdir())
            if f.is_file()
            and f.name
            in {
                "inputs.json.gz",
                "optics.json.gz",
                "gun.npz",
                "gun.json",
                "surface.npz",
                "currents.json",
                "portable-input.temwp",
            }
        },
    )
    write(out / "measurement.json", summary)
    print(
        f"Archived {len(current_vector(controls))} winding groups, alpha95 {angle:.6f} mrad, D95 {width:.6f} nm"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "trial", "observation", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    archive(args.recipe, args.trial, args.observation, args.output)

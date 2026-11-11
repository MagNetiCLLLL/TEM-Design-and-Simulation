"""Archive completed Microprobe evidence and standalone candidate profiles."""
from __future__ import annotations

from hashlib import sha256
import gzip
import json
from pathlib import Path
import shutil
import sys
import tomllib

import numpy as np
import tomli_w

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / "docs/development/microprobe_20261111"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(path):
    return dict(sha256=sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def relative_change(new, old):
    return {key: dict(old=old[key], new=new[key], relative_change=(new[key] / old[key] - 1.)
                        if old[key] else None)
            for key in ("d95_um", "radial_rms_um", "alpha95_mrad", "alpha_rms_mrad", "current_a")}


def main():
    batches = {name: read(HERE / "column" / name / "report.json") for name in
               ("n1024-step020", "n1024-step010-oneum", "n2048-step010")}
    if any(value["status"] != "COMPLETE" for value in batches.values()):
        raise ValueError("All authorized physical batches must be complete before archival")
    if [len(value["trials"]) for value in batches.values()] != [4, 1, 3]:
        raise ValueError("Incomplete physical family")
    if ARCHIVE.exists():
        raise FileExistsError("Use a new archive rather than overwriting recorded evidence")
    ARCHIVE.mkdir(parents=True)
    manifest = dict(scope="Selected archival evidence; original full trajectories remain at recorded local paths",
                    files={}, original_full_files={})

    def copy(source, target):
        destination = ARCHIVE / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest["files"][target] = dict(original_path=str(source.relative_to(ROOT)), **fingerprint(destination))

    def compress_json(source, target):
        destination = ARCHIVE / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        manifest["files"][target] = dict(original_path=str(source.relative_to(ROOT)),
                                         uncompressed=fingerprint(source), **fingerprint(destination))

    for count in (1024, 2048):
        folder = HERE / "extractor" / f"new-n{count}"
        for name in ("input.toml", "receipt.json", "launch.npz", "prepared.json", "axis_field.npz",
                     "electrostatic_request.json", "independent_validation.json", "source_nesting.json",
                     "snapshot_count_diff.json"):
            if (folder / name).is_file():
                copy(folder / name, f"extractor/new-n{count}/{name}")
        compress_json(folder / "inputs.json", f"extractor/new-n{count}/inputs.json.gz")
        with np.load(folder / "source.npz", allow_pickle=False) as data:
            output = ARCHIVE / f"extractor/new-n{count}/gun_exit_and_emission.npz"
            np.savez_compressed(output, **{key: data[key] for key in data.files
                if key.startswith(("gun_exit_", "emission_")) or key in ("source_id", "weights")})
        manifest["original_full_files"][str((folder / "source.npz").relative_to(ROOT))] = fingerprint(folder / "source.npz")

    result = dict(scope="Basic classical Microprobe illumination; corrector conjugacies not jointly qualified",
        reference_profile="docs/development/fege_focus_20261110/normal-nanoprobe-si110.toml",
        source_law="Unchanged driven Gaussian-Schell classical particle source, 20 nm assumed coherence length",
        source_current_a=1e-5, mini_polarity=1, mini_magnitude_percent=26.678337967960122,
        objective_percent=29.3264011126, c2_aperture_diameter_um=100.,
        initial_user_selected_screen=dict(diameter95_um_range=[.75, 2.2], maximum_alpha95_mrad=.3,
            selected_on="2026-11-11", note="Nominal family labels; actual diameters and sampling shifts reported separately"),
        mechanical_angles="About the weighted chief direction; no canonical substitution",
        numerical_step_comparison_1um=relative_change(batches["n1024-step010-oneum"]["trials"][0]["surface"],
                                                     batches["n1024-step020"]["trials"][1]["surface"]),
        source_count_comparison_1um=relative_change(batches["n2048-step010"]["trials"][0]["surface"],
                                                   batches["n1024-step010-oneum"]["trials"][0]["surface"]),
        count_and_step_family_comparison={}, family=[])
    keys = ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s", "alive", "z_mm")
    for name, batch in batches.items():
        folder = HERE / "column" / name
        copy(folder / "report.json", f"column/{name}/report.json")
        if (folder / "family-audit.json").is_file():
            copy(folder / "family-audit.json", f"column/{name}/family-audit.json")
        for trial in batch["trials"]:
            case = folder / f"trial-{trial['index']:02d}"
            target = f"column/{name}/{case.name}"
            for filename in ("input.toml", "report.json", "rotation-analysis.json"):
                if (case / filename).exists():
                    copy(case / filename, f"{target}/{filename}")
            compress_json(case / "inputs.json", f"{target}/inputs.json.gz")
            manifest["original_full_files"][str((case / "observations.npz").relative_to(ROOT))] = fingerprint(case / "observations.npz")
            with np.load(case / "observations.npz", allow_pickle=False) as data:
                planes = np.array([row["z_mm"] for row in trial["axial_profile"]])
                indices = np.searchsorted(data["z_mm"], planes)
                if not np.array_equal(data["z_mm"][indices], planes):
                    raise ValueError("A retained plane is not an actual observation")
                np.savez_compressed(ARCHIVE / target / "sample-neighborhood.npz",
                    **{key: data[key][indices] if key in keys else data[key] for key in data.files})

    source_doc = tomllib.loads((HERE / "extractor/new-n2048/input.toml").read_text(encoding="utf-8"))
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.profile_io import apply_profile_values, read_profile
    for index, slug, oldindex in ((2, "0p75um", 3), (0, "1p00um", 1), (1, "2p20um", 2)):
        trial = batches["n2048-step010"]["trials"][index]
        folder = HERE / "column/n2048-step010" / f"trial-{index:02d}"
        document = tomllib.loads((folder / "input.toml").read_text(encoding="utf-8"))
        document["devices"]["sample"] = source_doc["devices"]["sample"].copy()
        document["sample_model"] = source_doc["sample_model"].copy()
        destination = ARCHIVE / f"microprobe-{slug}-si110.toml"
        destination.write_text("# Basic parallel-illumination candidate; corrector conjugacies are not jointly qualified.\n"
                               "# Incident optics were measured without specimen scattering; see README.md.\n"
                               "# The original local Si CIF path and specimen settings are restored for use.\n"
                               + tomli_w.dumps(document), encoding="utf-8")
        selection, values = read_profile(destination)
        state = default_state()
        catalog = AssemblyCatalog()
        catalog.apply(state, catalog.normalise_selection(selection))
        apply_profile_values(state, values)
        expected = trial["controls"]
        if (state.mini_condenser.polarity != 1 or state.illumination_mode != "TEM"
                or state.electron_gun.emitter.ray_count != 2048 or state.step_mm != .1
                or state.sample.specimen_mode != "atomic"
                or abs(state.sample.upper_surface_z_mm - trial["surface"]["z_mm"]) > 1e-10):
            raise ValueError("Candidate profile mode, source budget or physical surface did not round-trip")
        for part in (*state.lenses, *state.corrector_elements):
            for field, value in expected.get(part.key, {}).items():
                if not np.isclose(getattr(part, field), value, rtol=0., atol=1e-12):
                    raise ValueError(f"Profile changed physical drive {part.key}.{field}")
        result["family"].append(dict(profile=destination.name, surface=trial["surface"],
            axial_profile=trial["axial_profile"], per_stage_losses=trial["per_stage_losses"],
            topology_summary=trial["topology_summary"], controls=trial["controls"],
            input_digest=trial["execution"]["input_digest"],
            original_positive_count=trial["execution"]["emitted_positive_current_points"],
            profile_round_trip="PASS; original specimen settings restored, no scattering calculation added"))
        result["count_and_step_family_comparison"][slug] = relative_change(trial["surface"],
                                              batches["n1024-step020"]["trials"][oldindex]["surface"])
    for source in (HERE / "model-review.json", HERE / "paraxial-proposals.json"):
        if source.exists():
            copy(source, source.name)
    drivers = [HERE / "extractor/run_micro_gun.py", HERE / "column/run_microprobe.py",
        HERE / "column/audit_family.py", HERE / "analyze_micro_rotation.py", HERE / "propose_microprobe.py",
        ROOT / "outputs/fege-focus-20261110/focus/run_focus.py"]
    for source in drivers:
        copy(source, "drivers/" + source.name)
    write(ARCHIVE / "results.json", result)
    for path in sorted(ARCHIVE.rglob("*")):
        if path.is_file():
            key = str(path.relative_to(ARCHIVE)).replace("\\", "/")
            manifest["files"].setdefault(key, dict(generated_from="Recorded complete physical observations", **fingerprint(path)))
    write(ARCHIVE / "artifact_manifest.json", manifest)
    print(json.dumps(dict(archive=str(ARCHIVE), family=[dict(profile=x["profile"], **x["surface"])
                                                       for x in result["family"]]), indent=2))


if __name__ == "__main__":
    main()

"""Offline independent quality check of the three archived candidate profiles."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / "docs/development/microprobe_20261111"
OUT = HERE / "profile-verification"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.profile_io import apply_profile_values, read_profile, save_profile
from temsim.specimen.cif_io import read_cif_atoms
from temsim.specimen.geometry import quaternion_to_matrix, sample_orientation_quaternion
from verify_candidate_profile import differences


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_doc = tomllib.loads((ARCHIVE / "extractor/new-n2048/input.toml").read_text(encoding="utf-8"))
    source_gun_keys = sorted(key for key in source_doc["devices"]
                             if key.startswith(("feg_", "electron_gun")))
    results = json.loads((ARCHIVE / "results.json").read_text(encoding="utf-8"))
    checked = []
    with np.load(ARCHIVE / "extractor/new-n2048/launch.npz", allow_pickle=False) as data:
        source_arrays = dict(data)
    for entry in results["family"]:
        path = ARCHIVE / entry["profile"]
        before_hash = sha256(path.read_bytes()).hexdigest()
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        selection, values = read_profile(path)
        state = default_state()
        catalog = AssemblyCatalog()
        catalog.apply(state, catalog.normalise_selection(selection))
        apply_profile_values(state, values)
        saved = OUT / (path.stem + "-resaved.toml")
        save_profile(saved, state, selection)
        restored = tomllib.loads(saved.read_text(encoding="utf-8"))
        diff = differences(document, restored)
        if diff:
            raise ValueError("Archived candidate changed on load/save: " + path.name)
        for table in ("gun_source_model", "gun_controls"):
            if restored[table] != source_doc[table]:
                raise ValueError("Source law changed: " + path.name + ":" + table)
        for key in source_gun_keys:
            if restored["devices"][key] != source_doc["devices"][key]:
                raise ValueError("Gun drive changed: " + path.name + ":" + key)
        if restored["devices"]["sample"] != source_doc["devices"]["sample"]:
            raise ValueError("Original specimen controls not restored: " + path.name)
        if restored["sample_model"] != source_doc["sample_model"]:
            raise ValueError("Original specimen model not restored: " + path.name)
        emitted = state.electron_gun.emit(2048)
        for key, expected in source_arrays.items():
            if not np.array_equal(np.asarray(getattr(emitted, key)), expected):
                raise ValueError("Emitted source array differs: " + path.name + ":" + key)
        assert state.electron_gun.emitter.ray_count == 2048
        assert np.count_nonzero(emitted.weight > 0.) == 2048
        assert len(np.unique(emitted.ray_id)) == 2048
        assert state.step_mm == .1
        assert state.simulation_mode == "unified" and state.illumination_mode == "TEM"
        assert state.acceleration_backend == "CPU" and not state.acceleration_enabled
        assert state.mini_condenser.polarity == 1
        assert state.mini_condenser.percent == 26.678337967960122
        assert state.objective_lens.percent == 29.3264011126
        assert state.sample.specimen_mode == "atomic"
        orientation = quaternion_to_matrix(sample_orientation_quaternion(state.sample))
        physical_crystal_beam_direction = orientation.T @ np.array([0., 0., 1.])
        assert np.allclose(physical_crystal_beam_direction, np.array([1., 1., 0.]) / np.sqrt(2.), rtol=0., atol=1e-12)
        assert abs(state.sample.upper_surface_z_mm - entry["surface"]["z_mm"]) <= 1e-10
        assert not state.sample.wave_enabled and not state.sample.stem_wave_enabled
        assert not state.sample.stem_fourdstem_enabled
        parts = {part.key: part for part in (*state.lenses, *state.corrector_elements,
                                           *state.stigmators)}
        controls = []
        for key, fields in entry["controls"].items():
            for field, expected in fields.items():
                if not np.isclose(getattr(parts[key], field), expected, rtol=0., atol=1e-12):
                    raise ValueError("Recorded drive changed: " + path.name + ":" + key + "." + field)
                controls.append(key + "." + field)
        cif = Path(state.sample.cif_path)
        assert cif.is_absolute() and cif.is_file()
        original_cwd = Path.cwd()
        try:
            os.chdir(OUT)
            atoms = read_cif_atoms(str(cif))
        finally:
            os.chdir(original_cwd)
        after_hash = sha256(path.read_bytes()).hexdigest()
        assert after_hash == before_hash
        checked.append(dict(profile=path.name, status="VERIFIED", profile_sha256=before_hash,
            semantic_roundtrip_differences=diff, source_law_preserved=True,
            source_launch_arrays_equal=sorted(source_arrays), original_positive_source_count=2048,
            checked_gun_devices=source_gun_keys, checked_optical_drives=controls,
            step_mm=state.step_mm, simulation_mode=state.simulation_mode,
            illumination_mode=state.illumination_mode,
            mini_signed_excitation_percent=state.mini_condenser.signed_excitation_percent,
            objective_percent=state.objective_lens.percent, sample_mode=state.sample.specimen_mode,
            sample_zone_axis_uvw=list(state.sample.zone_axis_uvw),
            physical_axis="[110]",
            physical_crystal_beam_direction=physical_crystal_beam_direction.tolist(),
            orientation_note="Physical quaternion aligns [110] with the beam; retained zone-axis input metadata is [001], as in the original source profile",
            sample_upper_surface_z_mm=state.sample.upper_surface_z_mm,
            recorded_reference_z_mm=entry["surface"]["z_mm"],
            cif_path=str(cif), cif_atoms_from_nonroot_cwd=len(atoms), profile_unchanged=True))
    if len(checked) != 3:
        raise ValueError("Exactly three archived candidate profiles required")
    report = dict(status="VERIFIED", scope="Offline profile and source-emission roundtrip only; no physical propagation or scattering",
        profiles=checked, archive=str(ARCHIVE.relative_to(ROOT)),
        source_reference="extractor/new-n2048/input.toml", source_array_reference="extractor/new-n2048/launch.npz")
    output = OUT / "archived-family.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    shutil.copyfile(output, ARCHIVE / "profile-family-verification.json")
    print(json.dumps(dict(status=report["status"], profiles=[item["profile"] for item in checked],
                         archived_verification="profile-family-verification.json"), indent=2))


if __name__ == "__main__":
    main()

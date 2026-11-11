"""Independent offline verification of Microprobe candidate profile assembly."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tomllib

import numpy as np
import tomli_w

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "profile-verification"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.profile_io import apply_profile_values, read_profile, save_profile
from temsim.specimen.cif_io import read_cif_atoms


def differences(left, right, path=""):
    if type(left) != type(right):
        return [dict(path=path, before=left, after=right)]
    if isinstance(left, dict):
        return [item for key in left.keys() | right.keys()
                for item in differences(left.get(key), right.get(key), path + "/" + key)]
    if isinstance(left, list):
        if len(left) != len(right):
            return [dict(path=path, before=left, after=right)]
        return [item for index, (a, b) in enumerate(zip(left, right))
                for item in differences(a, b, path + "/" + str(index))]
    return [] if left == right else [dict(path=path, before=left, after=right)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folder = HERE / "column/n2048-step010/trial-00"
    trial = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    source_doc = tomllib.loads((HERE / "extractor/new-n2048/input.toml").read_text(encoding="utf-8"))
    original = tomllib.loads((folder / "input.toml").read_text(encoding="utf-8"))
    document = deepcopy(original)
    document["devices"]["sample"] = source_doc["devices"]["sample"].copy()
    document["sample_model"] = source_doc["sample_model"].copy()
    path = OUT / "trial-00-restored-si110.toml"
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    selection, values = read_profile(path)
    catalog = AssemblyCatalog()
    state = default_state()
    catalog.apply(state, catalog.normalise_selection(selection))
    apply_profile_values(state, values)
    saved = OUT / "trial-00-second-save.toml"
    save_profile(saved, state, selection)
    restored = tomllib.loads(saved.read_text(encoding="utf-8"))
    diff = differences(document, restored)
    controls = []
    parts = {part.key: part for part in (*state.lenses, *state.corrector_elements,
                                       *state.stigmators)}
    for key, fields in trial["controls"].items():
        if fields and key not in parts:
            raise ValueError("Unknown controlled optical part: " + key)
        for field, expected in fields.items():
            actual = getattr(parts[key], field)
            if not np.isclose(actual, expected, rtol=0., atol=1e-12):
                raise ValueError(f"Changed physical drive: {key}.{field}")
            controls.append(f"{key}.{field}")
    assert state.mini_condenser.polarity == 1
    assert state.mini_condenser.percent == 26.678337967960122
    assert state.objective_lens.percent == 29.3264011126
    assert state.illumination_mode == "TEM"
    assert state.simulation_mode == "unified"
    assert state.electron_gun.emitter.ray_count == 2048
    assert state.step_mm == .1
    assert state.sample.specimen_mode == "atomic"
    assert abs(state.sample.upper_surface_z_mm - trial["surface"]["z_mm"]) <= 1e-10
    assert restored["gun_source_model"] == original["gun_source_model"]
    assert restored["gun_controls"] == original["gun_controls"]
    assert restored["sample_model"] == source_doc["sample_model"]
    sample_expected = deepcopy(source_doc["devices"]["sample"])
    assert restored["devices"]["sample"] == sample_expected
    assert not state.sample.wave_enabled and not state.sample.stem_wave_enabled
    assert not state.sample.stem_fourdstem_enabled
    actual_cif = str(state.sample.cif_path)
    original_cwd = Path.cwd()
    cwd_checks = []
    try:
        for directory in (ROOT, OUT):
            os.chdir(directory)
            try:
                atoms = read_cif_atoms(actual_cif)
                cwd_checks.append(dict(cwd=str(directory), status="LOADED", atoms=len(atoms)))
            except (FileNotFoundError, OSError) as error:
                cwd_checks.append(dict(cwd=str(directory), status=type(error).__name__, error=str(error)))
    finally:
        os.chdir(original_cwd)
    report = dict(status=("VERIFIED" if all(row["status"] == "LOADED" for row in cwd_checks)
                          else "VERIFIED_WITH_CWD_LIMITATION") if diff == [] else "ROUNDTRIP_DIFFERENCES",
        scope="Offline profile loading and CIF reading only; no physical tracing or scattering",
        candidate=str(path.relative_to(ROOT)), source_count=state.electron_gun.emitter.ray_count,
        step_mm=state.step_mm, simulation_mode=state.simulation_mode,
        illumination_mode=state.illumination_mode, mini_polarity=state.mini_condenser.polarity,
        mini_magnitude_percent=state.mini_condenser.percent,
        objective_percent=state.objective_lens.percent,
        sample_upper_surface_z_mm=state.sample.upper_surface_z_mm,
        trial_actual_reference_z_mm=trial["surface"]["z_mm"],
        sample_mode=state.sample.specimen_mode, source_model_preserved=True,
        original_sample_model_restored=True, checked_optical_controls=controls,
        profile_second_save_differences=diff, cif_path=actual_cif, actual_cif_reads=cwd_checks)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

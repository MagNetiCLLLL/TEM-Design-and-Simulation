"""Audit stored round-lens current references without solving fields or rays."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from temsim.coil_reference_metadata import ROUND_LENS_PROFILES, current_intervals_mm, validate_coil_references


def audit():
    paths = (*sorted((ROOT/"configs/instruments/column").glob("*.toml")),
        ROOT/"configs/instruments/project_and_recording_system/NoEnergyFilter.toml",
        ROOT/"configs/subassemblies/projector_stack.toml")
    records, files = [], []
    for path in paths:
        raw = path.read_bytes()
        parts = tomllib.loads(raw.decode("utf-8"))["parts"]
        validate_coil_references(parts)
        coils = [part for part in parts if part.get("mechanical_profile") == "magnetic_excitation_coil"]
        owners = [part for part in parts if part.get("mechanical_profile") in ROUND_LENS_PROFILES]
        missing = [part["key"] for part in owners if "air_core_reference_ampere_turns" not in part]
        files.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha256(raw).hexdigest(),
            round_owner_count=len(owners), round_coil_count=len(coils), missing_references=missing))
        for owner in owners:
            associated = [coil for coil in coils if coil.get("field_source_key") == owner["key"]]
            rows = []
            for coil in associated:
                intervals = current_intervals_mm(coil, owner)
                rows.append(dict(key=coil["key"], parent_key=coil.get("parent_key"),
                    field_source_key=coil["field_source_key"],
                    material_class=coil["material_class"],
                    inner_diameter_mm=coil["mechanical_inner_diameter_mm"],
                    outer_diameter_mm=coil["mechanical_outer_diameter_mm"],
                    display_envelope_start_mm=coil["local_start_z_mm"],
                    display_envelope_end_mm=coil["local_end_z_mm"],
                    current_intervals_local_z_mm=intervals,
                    current_bearing_total_length_mm=sum(end-start for start, end in intervals),
                    ampere_turn_fraction=coil["coil_ampere_turn_fraction"]))
            records.append(dict(path=path.relative_to(ROOT).as_posix(), key=owner["key"],
                reference_ampere_turns=owner.get("air_core_reference_ampere_turns"),
                reference_status=owner.get("air_core_reference_status"),
                gaussian_reference_coil_length_mm=owner.get("gaussian_reference_coil_length_mm"),
                coils=rows, coverage_complete=bool(rows) and "air_core_reference_ampere_turns" in owner))
    return dict(schema="round-coil-reference-coverage-v1", source_files=files,
        stored_owner_records=len(records), distinct_round_owner_keys=sorted({row["key"] for row in records}),
        complete=all(row["coverage_complete"] for row in records), records=records,
        scope="Static stored geometry/current-reference audit; no field solve, finite-source trial, operating-preset or thermal qualification",
        nonround_scope="Multipoles, stigmators, scan/alignment deflectors, magnetic prisms and electrostatic devices retain their native non-round field models; no circular solenoid is fabricated for them")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/air-core-copper-coil-20261113/coil-coverage.json")
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"records", "source_files"}}, indent=2))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())

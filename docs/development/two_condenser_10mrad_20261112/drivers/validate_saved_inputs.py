"""Verify saved high-angle inputs without emitting or propagating electrons."""
from pathlib import Path
from hashlib import sha256
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import InstrumentSnapshot, capture_instrument_snapshot, decode_instrument
from temsim.profile_io import read_profile, apply_profile_values
from temsim.optics.column import default_state
from temsim import input_io
from temsim.immutable_json import json_digest, thaw_json
from temsim.paths import CONFIG_ROOT
from temsim.physics.beam_current import effective_source_current_a

HERE = Path(__file__).resolve().parent
primary = HERE / "focused4096-s005"
snapshot = InstrumentSnapshot.from_dict(json.loads((primary / "inputs.json").read_text()))
report = {"source_snapshot_digest": snapshot.digest, "source_implementation": snapshot.implementation,
          "scope": "Read-only restoration and profile checks; no emitter invocation or particle calculation."}
try:
    restored = snapshot.restore()
    report["original_snapshot_restore"] = {"status": "PASS", "digest_equal": capture_instrument_snapshot(restored).digest == snapshot.digest}
except ValueError as exc:
    report["original_snapshot_restore"] = {"status": "HISTORICAL_DEPENDENCIES_CHANGED", "reason": str(exc)}

# Make a distinct portable derivative, overlaying exact recorded dependency bytes
# on a newly captured configuration tree. Unlisted support files are current.
# The original graph, report and digest are never overwritten or relabeled.
root = Path(CONFIG_ROOT).resolve()
rows_by_path = {}
for path in sorted(root.rglob("*")):
    if path.is_file():
        data = path.read_bytes()
        rows_by_path[str(path)] = dict(path=str(path), config_relative=path.relative_to(root).as_posix(),
            content_hex=data.hex(), sha256=sha256(data).hexdigest())
for row in snapshot.external_inputs:
    path = Path(row["path"])
    rows_by_path[str(path)] = dict(path=str(path), config_relative=path.relative_to(root).as_posix() if path.is_relative_to(root) else None,
                     content_hex=row["content_hex"], sha256=row["sha256"])
rows = list(rows_by_path.values())
payload = dict(schema=input_io.ARCHIVE_SCHEMA, config_root=str(root), files=rows,
               runtime=thaw_json(input_io.runtime_identity()))
payload["digest"] = json_digest(payload)
input_io.InputArchive(payload)
state = decode_instrument(snapshot.graph)
input_io.bind_archive(state, payload)
portable = capture_instrument_snapshot(state)
assert portable.external_inputs == snapshot.external_inputs
portable_state = portable.restore()
roundtrip = capture_instrument_snapshot(portable_state)
assert portable.digest == roundtrip.digest
portable_path = HERE / "portable-focused4096-inputs.json"
portable_path.write_text(json.dumps(portable.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
report["dependency_pinned_derivative"] = dict(status="PASS", original_digest=snapshot.digest,
    derivative_digest=portable.digest, derivative_path=portable_path.name,
    inventory_count=len(rows), inventory_scope="Newly captured current configuration tree overlaid with every exact dependency byte from the original snapshot. Extra support files and runtime were captured now, not retroactively attributed to the trial.",
    qualification="INPUTS_ONLY_DERIVATIVE_NO_NEW_PHYSICAL_RESULT", runtime_identity="Recorded at derivative creation; not retroactively attributed to original trial.")

selection, values = read_profile(primary / "input.toml")
loaded = default_state()
AssemblyCatalog().apply(loaded, selection)
apply_profile_values(loaded, values)

def observed(state):
    return dict(lenses={item.key: dict(percent=item.percent, polarity=item.polarity, enabled=item.enabled)
                        for item in state.lenses},
        aperture_diameter_mm=state.condenser_aperture_2.diameter_mm,
        aperture_enabled=state.condenser_aperture_2.enabled,
        hardware=state.layout_c3_hardware, column_mode=state.column_mode,
        upper_surface_z_mm=state.sample.upper_surface_z_mm,
        sample_inserted=state.sample.inserted,
        step_mm=state.step_mm, count=state.electron_gun.emitter.ray_count,
        source_current_a=effective_source_current_a(state),
        receivers={item.key: bool(item.inserted) for item in (state.fluorescent_screen, state.camera)})

expected, actual = observed(portable_state), observed(loaded)
assert expected == actual, (expected, actual)
report["operating_profile_round_trip"] = dict(status="PASS", selection=str(selection), observed=actual)
fine = json.loads((HERE/"focused4096-s005/report.json").read_text())
coarse = json.loads((HERE/"final4096-s010/report.json").read_text())
source = json.loads((HERE/"reference2048-s005/report.json").read_text())
report["relative_surface_d95_difference"] = dict(source_2048_relative_to_4096=source["d95_nm"]/fine["d95_nm"]-1,
    step_010_relative_to_005=coarse["d95_nm"]/fine["d95_nm"]-1)
report["observer_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
(HERE/"saved-input-validation.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
print(json.dumps(report, indent=2))

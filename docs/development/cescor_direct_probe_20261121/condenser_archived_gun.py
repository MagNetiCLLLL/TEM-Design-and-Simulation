"""Reconstruct the archived gun's magnetic evaluator in its original checkout in a separate process."""

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    point = here.parent / "default_ideal_coil_probe_20261117/working-point"
    old = root / "outputs/default-ideal-coil-probe-20261117/published-checkout"
    site = old / "src"
    sys.path[:0] = [str(site), str(old / "docs/development/default_ideal_coil_probe_20261117")]
    from gun_table import GunField
    from temsim.instrument_snapshot import InstrumentSnapshot
    from temsim.physics.gun_field_environment import instrument_gun_field_context
    from temsim.vacuum import bind_gun_environment
    from temsim.calculation_manifest import solver_source_identity
    import temsim

    if not Path(temsim.__file__).resolve().is_relative_to(site):
        raise ValueError("The archived evaluator was not imported from the isolated checkout")
    data = json.loads(gzip.decompress((point / "inputs.json.gz").read_bytes()))
    snapshot = InstrumentSnapshot.from_dict(data)
    expected = json.loads((point / "gun.json").read_text())
    if snapshot.digest != expected["input_digest"]:
        raise ValueError("The original gun was not traced with this instrument")
    state = snapshot.restore()
    bind_gun_environment(state)
    with instrument_gun_field_context(state) as native:
        table = GunField(native, (state.electron_gun.exit_plane_z_mm + 3) * 0.001, 0.5)
        receipt = table.receipt()
        z_mm = np.array([0.0, 400.0, 440.0, 445.0, 450.0])
        points = np.c_[0 * z_mm, 0 * z_mm, z_mm * 0.001]
        result = dict(
            snapshot=snapshot.digest,
            original_implementation=solver_source_identity(),
            evaluator_sha256=receipt["coefficient_sha256"],
            expected_sha256=expected["magnetic_table"]["coefficient_sha256"],
            z_mm=z_mm.tolist(),
            Bz_T=table.field_at_global_positions_t(points)[:, 2].tolist(),
        )
    output = root / "outputs/condenser-source-audit-20261122/archived-gun-field.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result, flush=True)
    if result["evaluator_sha256"] != result["expected_sha256"]:
        raise ValueError("The archived gun field table was not reproduced byte for byte")


if __name__ == "__main__":
    main()

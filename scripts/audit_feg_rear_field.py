"""Static Curved-tip rear electrode response; no particles or coherent waves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from temsim import module_manifest
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tip_assembly import model_from_part
from temsim.physics.grounded_tip_field import field_request, GroundedTipField


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gun = FieldEmissionGun()
    gun.emitter.surface_model = model_from_part(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    # Vacuum points: EEE bore outside the shank, then 10 nm ahead of the apex.
    probes = np.array([[.2e-3, 0., -1.65e-3], [0., 0., 10e-9]])
    rows, fields = [], []
    args.output.mkdir(parents=True, exist_ok=True)
    for label, bias in (("nominal", 0.), ("eee_plus_250v", .25)):
        request = field_request(gun)  # Also bind the standalone assembly.
        gun._gun_rear_parts[0]["bias_kv"] = bias
        request = field_request(gun)
        (args.output/f"{label}-input.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
        field = GroundedTipField(request)
        fields.append(field)
        rows.append(dict(label=label, eee_bias_kv=bias, field=field.report,
            potential_rise_v=field.potential_rise_v_at_global_positions(probes).tolist(),
            electric_field_v_per_m=field.field_at_global_positions_v_per_m(probes).tolist()))
        print(f"{label}: grid {field.report['grid_shape']}, residual {field.report['linear_residual']:.3g}", flush=True)
    np.testing.assert_array_equal(fields[0].r, fields[1].r)
    np.testing.assert_array_equal(fields[0].z, fields[1].z)
    delta = fields[1].rise-fields[0].rise
    assert delta.min() >= -1e-5 and delta.max() <= 250.+1e-5
    assert np.all(np.array(rows[1]["potential_rise_v"]) > rows[0]["potential_rise_v"])
    report = dict(scope="Static electrode response, not an emission or trajectory trial",
        tip_origin_z_mm=0., flat_default_preserved=True, probes_m=probes.tolist(), cases=rows,
        nodal_potential_change_bounds_v=[float(delta.min()), float(delta.max())],
        limitations=["EEE normal-emission wiring and undisclosed rear dimensions are engineering assumptions.",
                     "No dielectric polarization, space charge or cleaning model.",
                     "This checks boundary response and the maximum principle, not mesh convergence or OEM accuracy."])
    (args.output/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"probes_m": probes.tolist(), "cases": [
        {k: v for k, v in row.items() if k != "field"} for row in rows]}, indent=2))


if __name__ == "__main__":
    main()

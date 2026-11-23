"""Gun-only continuation of the archived near-C2, objective-focused working point."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INPUTS = HERE.parent / "c1-zoom/transport-inputs"
CASE = "c1-zoom-near-c2-focused"
BASELINE_KV = 5.2
TARGET_MM = 24.0
sys.path.insert(0, str(HERE.parents[1] / "cescor_direct_probe_20261121"))

from condenser_source_fields import prepared_source
from material_relay_column import Magnet
from temsim.optics.electron_gun.base import EmissionBundle
from temsim.optics.electron_gun.observations import ExactGunPlaneObserver
from temsim.optics.electron_gun.tip_curvature import curve_bundle
from temsim.optics.electron_gun.tracing import _diagnostic_emission, _trace_feg_to_exit
from temsim.physics.grounded_particle_step import gun_step_workers
from temsim.physics.instrument_electric import configure_instrument_electric_domain, instrument_electric_end_mm
from temsim.vacuum import bind_gun_environment


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare():
    record = json.loads((INPUTS / "index.json").read_text())["trials"][CASE]
    for key in ("bore_field", "particle_source"):
        path = INPUTS / record[key]["file"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == record[key]["sha256"]
    state, _ = prepared_source()
    gun = state.electron_gun
    bind_gun_environment(state)
    configure_instrument_electric_domain(gun, instrument_electric_end_mm(state))
    with np.load(INPUTS / record["bore_field"]["file"]) as field:
        gun._instrument_magnetic_field = Magnet(field["z"], field["coef"])
    gun._instrument_magnetic_identity = None
    assert gun.uses_geometry_electric_field and not gun.monochromator_installed
    assert gun.electrostatic_lens.voltage_reference == "tip"
    assert gun.accelerator.voltage_start_reference == "gun_lens"
    assert (gun.electrostatic_lens.voltage_kv, gun.extractor.voltage_kv,
            gun.accelerator.high_tension_kv) == (BASELINE_KV, 4., 300.)
    assert (gun.dpa_aperture.z_mm, gun.dpa_aperture.diameter_um,
            gun.c1_aperture.diameter_um, gun.exit_plane_z_mm) == (TARGET_MM, 4000., 1000., 450.)
    return gun, record


def set_voltage(gun, kv):
    gun.electrostatic_lens.voltage_kv = float(kv)
    gun.validate()
    base = gun.base_electric_field
    request = base.request
    lens, = [r for r in request["rings"] if r["key"] == "electrostatic_lens"]
    assert lens["potential_rise_v"] == kv * 1000
    assert request["extraction_v"] == 4000. and request["high_tension_v"] == 300000.
    return base.report["request_sha256"]


def angular_trace(gun, kv, planes, amplitudes=(1.,)):
    identity = set_voltage(gun, kv)
    directions = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    slopes = np.vstack([a * 1e-6 * directions for a in amplitudes])
    zeros = np.zeros(len(slopes))
    probes = EmissionBundle(zeros, zeros, slopes[:, 0], slopes[:, 1],
                            zeros, zeros, np.arange(len(slopes)))
    probes = _diagnostic_emission(gun, curve_bundle(probes, gun.emitter))
    observer = ExactGunPlaneObserver(planes, gun.exit_plane_z_mm)
    with gun_step_workers():
        result = _trace_feg_to_exit(gun, emitted_bundle=probes, observer=observer)
    obs = observer.result()
    if not (obs.reached.all() and obs.alive.all()):
        raise ValueError(f"Diagnostic ray failed to reach an observation plane at {kv:g} kV")
    position = np.stack((obs.x_m, obs.y_m), axis=-1)
    matrices = []
    for i, a in enumerate(amplitudes):
        xy = position[:, 4*i:4*i+4, :]
        matrices.append(np.stack(((xy[:, 0]-xy[:, 1]), (xy[:, 2]-xy[:, 3])), axis=-1)/(2e-6*a))
    return np.array(matrices), identity, result


def sweep(gun, args):
    if args.mesh:
        gun._gun_field_cells_per_bore = args.mesh
    mesh = gun.base_electric_field.request["numerics"]["cells_per_bore"]
    output = HERE / f"voltage-scan-mesh-{mesh}.json"
    rows = json.loads(output.read_text()) if output.exists() else []
    def evaluate(kv):
        existing = next((r for r in rows if r["lens_kV"] == float(kv)), None)
        if existing is None:
            matrices, identity, _ = angular_trace(gun, float(kv), [TARGET_MM])
            response = matrices[0, 0]*1e3
            existing = dict(lens_kV=float(kv), field_sha256=identity,
                            angular_response_mm_per_rad=response.tolist())
            rows.append(existing)
            write_json(output, rows)
        signed = np.trace(existing["angular_response_mm_per_rad"])*.5
        print(f"{kv:13.8f} kV | signed angular response at {TARGET_MM:g} mm: {signed: .9g} mm/rad", flush=True)
        return signed
    for kv in args.voltages:
        evaluate(kv)
    if args.bracket:
        kv = brentq(evaluate, *args.bracket, xtol=2e-7)
        print(f"Root: {kv:.9f} kV", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voltages", type=float, nargs="*", default=[])
    parser.add_argument("--bracket", type=float, nargs=2)
    parser.add_argument("--mesh", type=int)
    sweep(prepare()[0], parser.parse_args())

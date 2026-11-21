"""Propagate a matched gun exit through the solved material field and tune illumination currents."""

import argparse
from hashlib import sha256
import json

import numpy as np

from solve_fields import CACHE, PREVIOUS
from column import Mesh, POINT, SURFACE, metrics
from particles import Tracker
from tune import fit, waist_proposal


def original_controls():
    return json.loads((PREVIOUS / "results.json").read_text())["cases"]["photo"]["primary"]["controls"]


def field_trial(ni, level, step, hex_scale=0, padding=8):
    mesh = Mesh(step, [SURFACE], CACHE)
    mesh.hex *= hex_scale
    return mesh, Tracker(mesh), mesh.base("armco", ni, level, padding)


def report_surface(mesh, tracker, base, controls, ni, level, step, name, hex_scale=0, padding=8):
    phase, loss, peak, margin = tracker.run(mesh.field(base, controls))
    phase = phase[-1]
    ok = loss == -1
    rows = phase[ok]
    w = tracker.weights[ok]
    values = metrics(rows, w, tracker.current)
    d95, rms = waist_proposal(rows, w)
    gun = json.loads((POINT / "gun.json").read_text())
    field = json.loads((CACHE / f"armco-{abs(ni):g}-{level}-p{padding:g}.json").read_text())
    hashes = {key: sha256((CACHE / (key + ".npz")).read_bytes()).hexdigest()
              for key in ("columns", "electric", "mechanical")}
    recipe = dict(gun_sha256=gun["sha256"], field_digest=field["field_digest"],
                  objective_ampere_turns=ni, hexapole_scale=hex_scale, controls=list(controls), transport_sha256=hashes,
                  max_step_mm=step, specimen_upper_surface_mm=float(SURFACE * 1000))
    digest = sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    names, counts = np.unique(mesh.keys[mesh.owner[loss[~ok]]], return_counts=True)
    report = dict(**recipe, input_digest=digest, population=gun["population"], surface=values,
                  original_count=len(phase), surviving_count=int(ok.sum()),
                  effective_count=float(w.sum()**2 / (w @ w)),
                  losses=dict(zip(names.tolist(), counts.tolist())),
                  D95_drift_proposal_nm=d95*1e9, rms_drift_proposal_nm=rms*1e9,
                  max_live_radius_mm=float(peak[ok].max()*1000), min_live_clearance_mm=float(margin[ok].min()*1000))
    # Restore the local static potential in the mechanical energy passed to the specimen.
    j = int(np.searchsorted(mesh.z, SURFACE)) * 2
    r, v, zi, frac, dz = mesh.electric
    def potential(node, coords):
        radial = np.sum(coords**2, axis=1)
        return np.interp(radial, r**2, v[:, zi[node]] * (1 - frac[node]) + v[:, zi[node]+1] * frac[node])
    energy = 300000 + tracker.de + potential(j, phase[:, :2]) - potential(0, tracker.input[:, :2])
    out = CACHE / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "surface.npz", x_m=phase[:, 0], y_m=phase[:, 1], tx_rad=phase[:, 2],
                        ty_rad=phase[:, 3], flight_time_s=phase[:, 4], kinetic_energy_ev=energy,
                        ray_id=tracker.ids, weight=tracker.weights, alive=ok, loss_step=loss,
                        emitted_current_a=tracker.current, z_mm=SURFACE*1000, input_digest=digest)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(name, values, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("--ni", type=float, default=-3000.)
    p.add_argument("--level", default="base")
    p.add_argument("--step", type=float, default=0.2)
    p.add_argument("--padding", type=float, default=8)
    p.add_argument("--hex-scale", type=float, default=0, help="Scale the saved corrector excitations together")
    p.add_argument("--start", help="Cached previous measurement directory")
    p.add_argument("--iterations", type=int, default=0)
    p.add_argument("--c3-scale", type=float, help="Override the C3 lens drive multiplier before retuning")
    a = p.parse_args()
    q = (json.loads((CACHE / a.start / "metrics.json").read_text())["controls"]
         if a.start else original_controls())
    if a.c3_scale is not None:
        q[1] = a.c3_scale
    mesh, tracker, base = field_trial(a.ni, a.level, a.step, a.hex_scale, a.padding)
    if a.iterations:
        q, _, history = fit(mesh, tracker, "armco", a.ni, a.level, a.padding, q, 0, len(tracker.ids), a.iterations)
        (CACHE / (a.name + "-iterations.json")).write_text(json.dumps(history, indent=2), encoding="utf-8")
    report_surface(mesh, tracker, base, q, a.ni, a.level, a.step, a.name, a.hex_scale, a.padding)

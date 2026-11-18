"""Repeat a proposed current recipe on a finer independent zero-current pupil grid."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.constants import c, e, m_e
from scipy.interpolate import make_interp_spline
from aberrations import trace
from pencil import coefficients
from temsim.physics.beam_statistics import transverse_beam_statistics


def field_for_recipe(r, auxiliary_basis, mesh_mm):
    f, h, names, _ = coefficients(r, mesh_mm)
    stop = 1.5991999975
    start = 0.45
    z = np.linspace(start, stop, len(f))
    with np.load(auxiliary_basis) as d:
        columns = d["fields"]
        src = np.linspace(start, stop, len(columns))
        aux = make_interp_spline(src, columns[:, 16:], k=5)(z)
        channels = d["names"]
        units = d["values"]
    if list(channels[:2, 0]) != names:
        raise ValueError("Primary channel order differs")
    gains = np.zeros(len(channels))
    gains[:2] = 1.0
    for i, ((key, attr), unit) in enumerate(zip(channels, units)):
        commands = r.get("multipole_commands", {}).get(key, {})
        if attr in commands:
            gains[i] = commands[attr] / unit
        elif attr == "strength_m3" and key in r.get("hexapole_strength_m3", {}):
            gains[i] = r["hexapole_strength_m3"][key] / unit
    combined = np.column_stack((f[:, 8:16], aux)).reshape(len(z), 6, 4).transpose(0, 2, 1) @ gains
    field = np.ascontiguousarray(np.column_stack((f[:, :4], combined)))
    return field, h, start, stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("auxiliary_basis", type=Path)
    parser.add_argument("gun", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mesh-mm", type=float, default=0.075)
    args = parser.parse_args()
    r = json.loads(args.recipe.read_text())
    field, h, start, stop = field_for_recipe(r, args.auxiliary_basis, args.mesh_mm)
    with np.load(args.gun) as data:
        s = transverse_beam_statistics(
            *(data[k] for k in ("x_m", "y_m", "tx_rad", "ty_rad")),
            alive=data["alive"],
            weights=data["weight"]
        )
        w = data["weight"][data["alive"]]
        en = data["energy_offset_ev"][data["alive"]]
        mean = np.average(en, weights=w)
        sd = np.sqrt(np.average((en - mean) ** 2, weights=w))
    angles = np.arange(24) * 2 * np.pi / 24
    ratios = np.array([0.08, 0.15, 0.23, 0.28, 0.32, 0.45, 0.6, 0.8, 0.95, 1.0, 1.15])
    xy = (
        s.radius_95_m * ratios[:, None, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    ).reshape(-1, 2)
    rays = np.column_stack((xy, s.radial_wavefront_curvature_per_m * xy))
    rows = []
    for de in (mean - sd, mean, mean + sd):
        energy = 300000.0 + de
        factor = np.sqrt(
            300000.0 * (300000.0 + 2 * m_e * c * c / e) / (energy * (energy + 2 * m_e * c * c / e))
        )
        end = trace(rays, field * factor, h)
        radii = np.hypot(*end[:, :2].T).reshape(len(ratios), -1) * 1e9
        polar = np.arctan(np.hypot(*end[:, 2:].T)).reshape(len(ratios), -1) * 1e3
        rows.append(
            dict(
                energy_offset_ev=de,
                rings=ratios.tolist(),
                rms_nm=np.sqrt(np.mean(radii**2, axis=1)).tolist(),
                mean_mechanical_angle_mrad=polar.mean(axis=1).tolist(),
            )
        )
    from hashlib import sha256

    reply = dict(
        status="Independent source-free verification; no emitted current or physical focus measurement",
        recipe_sha256=sha256(args.recipe.read_bytes()).hexdigest(),
        reference_gun_sha256=sha256(args.gun.read_bytes()).hexdigest(),
        mesh_mm=h * 1000,
        phase_planes=[start, stop],
        field_terms="Native round B0 through B0''' and actual winding hexapole coefficients",
        energy_tests=rows,
    )
    args.output.write_text(json.dumps(reply, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(rows[1], flush=True)


if __name__ == "__main__":
    main()

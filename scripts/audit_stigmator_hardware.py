"""Export assembly-specific stigmator ownership and a cheap CPU field check.

No ray tracing, calibration, GPU work or changes to saved settings are performed.
The diagrams use resolved axial positions, not dimensions inferred from patents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/stigmator-research-20261019"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    from temsim.optics.column import default_state
    from temsim.optics.stigmator_hardware import inspect_stigmator_hardware
    from temsim.physics.core import multipole_focusing_fields, skew_quadrupole_field

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = default_state()
    catalog = AssemblyCatalog()
    reports = []
    states = []
    selected_keys = ("condenser_lens_2", "condenser_deflector", "condenser_lens_3",
                     "beam_deflector", "adapter_lens", "probe_tl12_lens",
                     "ac_deflector", "condenser_stigmator", "mini_condenser",
                     "objective_upper_pole", "sample", "objective_lower_pole",
                     "image_diffraction_deflector", "objective_stigmator",
                     "selected_area_aperture", "diffraction_stigmator", "diffraction_lens")
    for option in catalog.columns:
        state = type(base).from_dict(base.to_dict())
        assembly = catalog.apply(state, AssemblySelection("FEG", option.name, "No Energy Filter"))
        rows = []
        for part in assembly.parts:
            if part.key in selected_keys:
                rows.append(dict(key=part.key, name=part.name, parent=part.parent_key,
                                 start_mm=part.start_z_mm, centre_mm=part.center_z_mm,
                                 end_mm=part.end_z_mm))
        rows.sort(key=lambda row: row["centre_mm"])
        reports.append(dict(column=option.name, hardware=inspect_stigmator_hardware(state).to_dict(),
                            axial_components=rows))
        states.append(state)

    main_index = next(i for i, r in enumerate(reports) if r["column"] == "C3 + Probe Corrector")
    chosen = states[main_index]
    checks = []
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    grid = np.linspace(-1, 1, 9)
    x, y = np.meshgrid(grid, grid)
    for axis, (label, angles, drives) in zip(axes, (
        ("X drive 1% | basis 0 / 45 deg", (0., 45.), (1., 0.)),
        ("Y drive 1% | basis 0 / 45 deg", (0., 45.), (0., 1.)),
        ("X drive 1% | basis 15 / 60 deg", (15., 60.), (1., 0.)),
    )):
        state = type(chosen).from_dict(chosen.to_dict())
        for component in (*state.stigmators, *state.corrector_elements):
            component.enabled = False
        stigma = state.condenser_stigmator
        stigma.enabled = True
        stigma.channel_x_angle_deg, stigma.channel_y_angle_deg = angles
        stigma.strength_x_percent, stigma.strength_y_percent = drives
        z = np.array([stigma.z_mm])
        xx, yy = multipole_focusing_fields(z, state)
        xy = skew_quadrupole_field(z, state)
        tensor = np.array([[xx[0], xy[0]], [xy[0], yy[0]]])
        report_row = next(r for r in inspect_stigmator_hardware(state).stigmators
                          if r.key == stigma.key)
        np.testing.assert_allclose(tensor, report_row.tensor_at_reference_m2, atol=1e-12)
        # This displays transverse slope derivative, not magnetic B arrows.
        response = -tensor @ np.stack([x.ravel(), y.ravel()])
        axis.quiver(x, y, response[0].reshape(x.shape), response[1].reshape(y.shape),
                    color="#146ba3", pivot="mid")
        axis.set(title=label, xlabel="x (arbitrary common scale)", ylabel="y", aspect="equal")
        axis.axhline(0, color="0.8", linewidth=.6)
        axis.axvline(0, color="0.8", linewidth=.6)
        checks.append(dict(label=label, basis_deg=angles, drives_percent=drives,
                           tensor_m2=tensor.tolist(), trace=float(np.trace(tensor)),
                           eigenvalues_m2=np.linalg.eigvalsh(tensor).tolist(),
                           field_fwhm_mm=stigma.length_mm,
                           provider="physics.core multipole_focusing_fields + skew_quadrupole_field"))
    fig.suptitle("Executed column quadrupole response: d(theta)/dz = -K r (isolated stigmator)")
    fig.savefig(output / "stigmator-field-bases.png", dpi=150)
    plt.close(fig)

    rows = reports[main_index]["axial_components"]
    fig, axis = plt.subplots(figsize=(12, 8), layout="constrained")
    for i, row in enumerate(rows):
        key = row["key"]
        color = "#9a4dc1" if "stigmator" in key else "#178d88" if "deflector" in key else "#4077b5"
        axis.plot([row["start_mm"], row["end_mm"]], [i, i], color=color, linewidth=7, solid_capstyle="butt")
        axis.scatter([row["centre_mm"]], [i], color=color, s=18, zorder=3)
        axis.annotate(f'{row["centre_mm"]:.3f}', (row["end_mm"], i), xytext=(7, -3),
                      textcoords="offset points", fontsize=8)
    axis.set_yticks(range(len(rows)), [r["name"] for r in rows])
    axis.invert_yaxis()
    axis.set(xlabel="Configured global axial Z (mm), before rigid pose",
             title="C3 + Probe Corrector | physical sequence, not an OEM dimensional drawing")
    axis.grid(axis="x", alpha=.2)
    axis.margins(x=.12, y=.04)
    fig.savefig(output / "configured-axial-order.png", dpi=140)
    plt.close(fig)

    sources = ["src/temsim/optics/stigmator_field.py", "src/temsim/cad_multipoles.py",
               "src/temsim/optics/stigmator_hardware.py", "configs/instruments/catalog.toml"]
    sources += [f"configs/instruments/{o.file}" for o in catalog.columns]
    sources += ["configs/instruments/gun/FEG.toml",
                "configs/instruments/project_and_recording_system/NoEnergyFilter.toml"]
    manifest = dict(schema="stigmator-hardware-review-v1", scope="configured_geometry_not_oem_validation",
                    assemblies=reports, field_checks=checks,
                    source_sha256={s: hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources})
    (output / "hardware-audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Audited {len(reports)} assemblies and {len(checks)} field bases: {output}")


if __name__ == "__main__":
    main()

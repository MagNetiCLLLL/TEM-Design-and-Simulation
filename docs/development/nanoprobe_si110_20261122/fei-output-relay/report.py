"""Save the final operating point, physical-z plane register, and linked study notes."""

import json
import numpy as np
from relay import HERE, WORK, load_bias, write


def read(path):
    return json.loads(path.read_text())


def gap_front(folder):
    roots = read(folder / "front.json")["all_fronts"]
    return next((s["z_mm"] for s in roots if 1420 < s["z_mm"] < 1512.5), None)


def downstream(row):
    return next((q for q in row["waists"]["live"]["radial_mm"] if 1420 < q < 1512.5), None)


def render():
    old = read(HERE / "before-million/result.json")
    mid = read(HERE / "tl12-set/result.json")
    cur = read(HERE / "final/result.json")
    focus = read(HERE / "final/focus.json")
    checks = read(HERE / "final/checks.json")
    rows = []
    for key, rec in [("before", old), ("tl12-set", mid), ("final", cur)]:
        folder = HERE / key
        p = read(folder / "pivot.json")
        pair = gap_front(folder)
        installed = next(
            (
                s["z_mm"]
                for s in read(folder / "front.json")["installed_field_fronts"]
                if 1420 < s["z_mm"] < 1512.5
            ),
            None,
        )
        waist = downstream(rec)
        b = load_bias(WORK / (rec["field"] + ".npz"))
        rows.append(
            dict(
                stage=key,
                source_count=rec.get("emitted_particles", 524288),
                tl12_A=b["probe_tl12_lens"],
                objective_series_A=b["objective_lens/physical-pack-4"],
                radial_waist_mm=waist,
                pivot_mm=p["z_mm"],
                pivot_XY_mm=p["axis_minima_mm"],
                pair_front_mm=pair,
                installed_field_front_mm=installed,
                pivot_source_count=p["source_count"],
                waist_minus_pivot_mm=None if waist is None else waist - p["z_mm"],
                front_minus_pivot_mm=None if pair is None else pair - p["z_mm"],
                D95_um=rec["surface"]["D95_um"],
                specimen_nA=rec["surface"]["sample_nA"],
                current_weighted_rim_median_mrad=rec["rim_outer_5pct"]["median_mrad"],
            )
        )
    last = rows[-1]
    rad = cur["waists"]["live"]["radial_mm"]
    source = read(HERE / "final/gun-waists.json")
    c1 = read(HERE / "final/c1-image.json")
    ring = focus["aperture_rim"][1]
    edge = focus["outermost_ray"]
    ledger = cur["current_ledger_A"]
    with np.load(WORK / (cur["field"] + ".npz")) as solved:
        magnetic = json.loads(str(solved["report"]))
        solids = json.loads(str(solved["recipe"]))["solids"]
    r = dict(
        mode="Nanoprobe Normal",
        emitter="FEI comparison geometry, literature-parameterized cold source",
        gun_lens_V=50.0,
        extractor_V=4000.0,
        independent_anode_V=4000.0,
        gun_aperture_diameter_mm=4.0,
        gun_aperture_position_mm=24.0,
        source_current_uA=15.0,
        c2_aperture_diameter_um=100.0,
        dc_excitation_A=load_bias(WORK / (cur["field"] + ".npz")),
        field_sha256=cur["common_field_sha256"],
        magnetic_circuit=dict(
            mesh_id=magnetic["mesh_id"],
            nodes=magnetic["nodes"],
            tetrahedra=magnetic["tetrahedra"],
            residual=magnetic["residual"],
            material_grades=sorted({s["material"] for s in solids}),
            modeled_solids=len(solids),
        ),
        field_solution=cur["field"],
        intermediate_checkpoint="checkpoint.json",
        field_table=cur["field_table"],
        gun_packet=cur["gun_packet"],
        z_reference="Absolute z from the emitting tip; the magnetic mesh origin is at z=450 mm.",
        planes=rows,
        source_to_specimen_live_radial_minima_mm=rad,
        gun_paraxial=source["paraxial_half_step"],
        c1_source_image=c1["images"],
        final_focus=focus,
        aperture_rim_deficit_from_30_mrad=30.0 - ring["median_mrad"],
        nearest_sampled_edge_deficit_from_30_mrad=30.0 - edge["angle_mrad"],
        edge_source_prefixes=checks["source_prefixes"],
        current_ledger_A=ledger,
    )
    write(HERE / "operating-point.json", r)

    def fmt(x, p=3):
        return "—" if x is None else f"{x:.{p}f}"

    tab = "\n".join(
        "| "
        + " | ".join(
            [
                v["stage"],
                fmt(v["tl12_A"]),
                fmt(v["objective_series_A"], 5),
                fmt(v["radial_waist_mm"]),
                fmt(v["pivot_mm"]),
                fmt(v["pair_front_mm"]),
                fmt(v["D95_um"], 2),
                fmt(v["current_weighted_rim_median_mrad"], 3),
            ]
        )
        + " |"
        for v in rows
    )
    text = f"""# Nanoprobe Normal output relay with the +50 V gun

The finite-source zero-scan beam has its downstream radial waist at {fmt(last["radial_waist_mm"])} mm. The fixed two-axis scan drive has its least-displacement plane at {fmt(last["pivot_mm"])} mm. The isolated Mini and incident objective focus has its front at {fmt(last["pair_front_mm"])} mm. Distances use the physical tip as the origin. The lower scan winding ends at 1420 mm and the Mini winding begins at 1512.5 mm.

| Stage | TL12 (A) | Series objective (A) | Postscan waist (mm) | AC pivot (mm) | Joint front (mm) | Specimen D95 (µm) | Rim median (mrad) |
|---|---:|---:|---:|---:|---:|---:|---:|
{tab}

All three static rows use the same {cur['emitted_particles']:,} original positive-current primaries. The comparison of nested source counts is in [final numerical checks](final/checks.json). All four scan supplies are zero for the DC trajectories and the probe. The signed scan diagnostic uses the installed current mixing, a new material differential solve at the DC bias, paired ±1 µrad X and Y commands from the tip, and a common surviving population. The two individual final pivot minima are {fmt(last['pivot_XY_mm'][0])} and {fmt(last['pivot_XY_mm'][1])} mm. The front calculation isolates the Mini and objective supplies on the same material mesh and zeros the outgoing canonical angular response referred to the specimen surface. The corresponding front in the entire energized column is {fmt(last['installed_field_front_mm'])} mm. Both objective packs share the same excitation.

The +4 kV extractor, independent +4 kV anode, 4 mm gun aperture, 100 µm C2 aperture, C1 lens (−34 A), C2 lens (+9 A), and weak C3 lens (+6.472 A) are common to the three settings. There are {cur['surface']['count']} positive-weight landings at the specimen. The inherited AC calibration is fixed throughout. A short preliminary supply survey used axial changes about the full 3D field. The reported operating points use converged nonlinear material solutions, independent direct-field interpolation checks and fresh physical-tip transport.

At the specimen surface z = {focus['specimen_z_mm']:.7f} mm, the current-weighted D95 is {last['D95_um']:.3f} µm and the total current is {last['specimen_nA']:.4f} nA. D95 is twice the current-weighted 95th percentile of distance from the beam centroid. The local minimum of D95 occurs at {focus['minimum_D95_z_mm']:.4f} mm. A vacuum continuation measures both sides of the surface with the local electric and magnetic field, using 0.025 mm plane spacing. The RMS envelope minima in this window are {', '.join(fmt(z) for z in focus['rms_waists_mm'])} mm. The final live radial minima from C1 to the specimen are {', '.join(fmt(z) for z in rad)} mm.

The outer five percent of the aperture radius contains {ring['n']} transmitted source IDs between 47.5 and 50 µm. Angles are the 3D direction separations from the current-weighted chief direction at the specimen. That band's median is {ring['median_mrad']:.3f} mrad, with a range of {ring['range_mrad'][0]:.3f}–{ring['range_mrad'][1]:.3f} mrad. Its deficit from 30 mrad is {30.-ring['median_mrad']:.3f} mrad. The nearest actual ray to the geometric rim reaches a radius of {focus['rim_maximum_um']:.4f} µm and arrives at {edge['angle_mrad']:.3f} mrad (deficit {30.-edge['angle_mrad']:.3f} mrad). The fraction of the rim current traveling radially inward relative to the beam centroid at the specimen is {ring['inward_current_fraction']:.1%}. The complete beam has an angular 95th percentile of {cur['surface']['alpha95_mrad']:.3f} mrad.

With the inherited scan mixing and fixed Mini current, the final waist is {last['waist_minus_pivot_mm']:+.3f} mm downstream of the AC pivot in signed coordinates. The isolated front lies {last['front_minus_pivot_mm']:.3f} mm downstream of the pivot. This setting meets the postscan waist location but leaves a separate front focal plane. The 524,288-primary prefix at the same final currents places its local D95 minimum at {focus['first_half_D95_focus']['z_mm']:.3f} mm. The difference from the full-source minimum measures source-sampling sensitivity.

[Full-column trajectories](column-rays.png) · [Specimen current density](specimen-probe.png) · [Pupil angles and through-focus profile](pupil-and-focus.png) · [Operating point](operating-point.json) · [Saved in-progress configuration](checkpoint.json) · [Signed scan response](final/pivot.json) · [Front-plane verification](final/front.json) · [Individual C2 edge rays](final/aperture-edge.json)

## Reproduction

Run [restore_checkpoint.py](restore_checkpoint.py) to restore the published final 3D magnetic volume, electrostatic gun field and vacuum tracking tables. [Archive manifest and checksums](field-replay/manifest.json)

Use the project Python environment and the FEI gun and material mesh caches from the preceding study. In this folder, run `exact.py --count {cur['emitted_particles']} -- {last['tl12_A']:.6f} {last['objective_series_A']:.6f} final` for physical-tip tracking. Use `pivot.py {cur['field']} final` and `front.py {cur['field']} final` for the two independent differential measurements. Run `verification.py final` for a repeated transport step and source-prefix checks, then `figures.py` and `report.py` for the plots and this record. All physical trial inputs have strictly positive represented source currents.
"""
    (HERE / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    render()

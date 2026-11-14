"""Archive completed copper-coil studies and render their scientific figures."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from study_copper_coil_aberrations import make_field, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plans = json.loads((args.input / "requests.json").read_text())
    refinements = json.loads((args.input / "refinement-requests.json").read_text())
    reports = {r["name"]: json.loads((args.input / r["name"] / "report.json").read_text())
               for r in plans + refinements}
    args.output.mkdir(parents=True, exist_ok=False)
    for request in plans + refinements:
        shutil.copytree(args.input / request["name"], args.output / request["name"])
    for filename in ("requests.json", "refinement-requests.json", "initial-observer.py", "corrected-observer.py",
                     "transport-field-source.py", "independent-baseline-array-audit.json", "seven-case-readonly-crosscheck.json"):
        if (args.input / filename).exists():
            shutil.copy2(args.input / filename, args.output / filename)
    drivers = args.output / "drivers"
    drivers.mkdir()
    for name in ("scripts/study_copper_coil_aberrations.py", "scripts/report_copper_coil_aberrations.py",
                 "scripts/ideal_lens_beam.py", "scripts/ideal_lens_wall.py",
                 "src/temsim/physics/round_lens_aberration_benchmark.py", "src/temsim/physics/annular_coil_potential.py"):
        shutil.copy2(ROOT / name, drivers / Path(name).name)
    rows = []
    for request in plans:
        report = reports[request["name"]]
        field = make_field(request, 24)
        peak = float(field.axis_bz_t(0.))
        half = brentq(lambda z: float(field.axis_bz_t(z)) - peak / 2, 0., .5)
        p = report["populations"][-1]
        rows.append(dict(name=request["name"], inner_diameter_mm=request["inner_diameter_mm"],
            outer_diameter_mm=request["outer_diameter_mm"], length_mm=request["length_mm"],
            ampere_turns=request["ampere_turns"], centre_field_t=peak, field_fwhm_mm=half * 2e3,
            cs_mm=report["intrinsic_cs_mm"], image_z_mm=report["request"]["paraxial_image_m"] * 1e3,
            d95_nm=p["at_paraxial_image"].get("d95_nm"), rms_radius_nm=p["at_paraxial_image"].get("rms_radius_nm"),
            alpha95_mrad=p["at_paraxial_image"].get("alpha95_mrad"),
            transmission=p["at_paraxial_image"].get("transmission", 0.),
            original_positive_count=p["original_positive_count"], surviving_positive=p["surviving_at_image"],
            effective=p["effective_at_image"], source_radius_mm=report["request"]["source_radius_m"] * 1e3,
            source_d95_relative_change=report["source_count_comparison"]["relative_d95_change"],
            cs_fit_relative_error=p["vacuum_fit"]["relative_cs_difference"],
            field_quadrature_relative_change=report["quadrature_check"]["max_difference_over_axis_peak"]))
    base, half = (reports[name] for name in ("baseline", "baseline-half-step"))
    b, h = (report["populations"][-1] for report in (base, half))
    checks = dict(baseline_half_step_relative_d95_change=h["at_paraxial_image"]["d95_nm"] / b["at_paraxial_image"]["d95_nm"] - 1,
        baseline_half_step_relative_cs_fit_change=h["vacuum_fit"]["cs_fit_mm"] / b["vacuum_fit"]["cs_fit_mm"] - 1,
        maximum_source_count_relative_d95_change=max(abs(row["source_d95_relative_change"]) for row in rows),
        maximum_finite_pupil_cs_relative_error=max(row["cs_fit_relative_error"] for row in rows),
        maximum_sampled_field_quadrature_relative_change=max(row["field_quadrature_relative_change"] for row in rows),
        total_unique_originals_per_case=2048, newly_transported_partition_count=2 * (len(plans) + len(refinements)),
        finite_source_definition="Eight distinct numerical trials, each 1024 original particles plus a transported 1024-ID append; no zero-current diagnostic counted",
        scope="Numerical sensitivity of this declared air-core model; not a measured instrument or thermal qualification")
    write_json(args.output / "summary.json", dict(cases=rows, checks=checks))
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    z = np.linspace(-.05, .05, 801)
    for name in ("inner-8mm", "baseline", "inner-16mm"):
        request = next(r for r in plans if r["name"] == name)
        field = make_field(request, 24)
        bz = field.axis_bz_t(z)
        label = f"Coil ID {request['inner_diameter_mm']:g} mm"
        axes[0].plot(z * 1e3, bz, label=label)
        axes[1].plot(z * 1e3, bz / float(field.axis_bz_t(0.)), label=label)
    axes[0].set(xlabel="Axial position z (mm)", ylabel="On-axis Bz (T)", title="Absolute field")
    axes[1].set(xlabel="Axial position z (mm)", ylabel="Bz / centre Bz", title="Normalized field shape")
    for ax in axes:
        ax.legend(frameon=False)
    fig.suptitle("Copper winding: fixed OD 20 mm, length 10 mm, NI 3000 A turns")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"field-profiles.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    groups = ((("inner-8mm", "baseline", "inner-16mm"), "inner_diameter_mm", "Coil inner diameter (mm)", "Fixed OD, length, NI"),
              (("ni-1500", "baseline", "ni-6000"), "ampere_turns", "Ampere-turns (A turns)", "Fixed complete coil geometry"),
              (("length-5mm", "baseline", "length-20mm"), "length_mm", "Winding length (mm)", "Fixed ID, OD, NI"))
    for ax, (names, key, label, title) in zip(axes, groups):
        selected = [next(row for row in rows if row["name"] == name) for name in names]
        ax.plot([row[key] for row in selected], [row["cs_mm"] for row in selected], "o-")
        ax.set(xlabel=label, ylabel="Intrinsic Cs (mm)", title=title, yscale="log")
    fig.suptitle("300 keV; fixed declared 10 mrad paraxial edge pupil")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"intrinsic-cs.{extension}", dpi=180)
    plt.close(fig)
    header = "| Case | Coil ID / OD / length (mm) | NI (A turns) | Centre B (T) | Field FWHM (mm) | Cs (mm) | D95 (nm) | RMS radius (nm) | alpha95 (mrad) |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    table = header + "".join(f"| [{r['name']}]({r['name']}/report.json) | {r['inner_diameter_mm']:g} / {r['outer_diameter_mm']:g} / {r['length_mm']:g} | {r['ampere_turns']:g} | {r['centre_field_t']:.8f} | {r['field_fwhm_mm']:.6f} | {r['cs_mm']:.6f} | {r['d95_nm']:.6f} | {r['rms_radius_nm']:.6f} | {r['alpha95_mrad']:.6f} |\n" for r in rows)
    readme = f"""# Geometry-driven copper-coil intrinsic aberration study

The new round-lens field is the Biot-Savart integral of the declared circular copper winding. It is not a three-Gaussian fit. Pole-piece/yoke magnetism, saturation, thermal effects, space charge, specimen scattering and coherent propagation are absent. The [input design and public-source review](../air_core_copper_coil_20261113/README.md) separates engineering assumptions from published evidence.

## Controlled inputs and results

All cases use 300 keV monochromatic, zero-emittance, uniform-disk source current of 1 microampere. The declared **paraxial edge** angle is 10 mrad; this is not alpha95. Each case has its own recorded pupil radius to hold that edge angle fixed. The source plane is z=-200 mm and the first paraxial image is independently determined. Finite axial tails are always evaluated; the plotted interval and FWHM are not field cutoffs.

{table}
![Absolute and normalized axial fields](field-profiles.png)

![Intrinsic spherical aberration sensitivity](intrinsic-cs.png)

At fixed outer diameter, length and NI, increasing coil ID lowers the central field and broadens its axial distribution. The reported Cs increases over these three cases. Increasing NI reduces Cs over the tested interval. Winding length is not a universal monotone control: these three sampled lengths place the lowest Cs near the middle case. These are local results of the declared air-core geometry, not universal lens-design laws.

## Sampling, current and numerical checks

Every finite-source trial transports 1024 original positive-current particles, then **appends and transports another 1024 without changing or rerunning the prefix**. Original IDs, positions, directions, energy, emission time and training masks are retained. Each N-particle publication uses source current/N per original; intercepted current is never restored. Raw first-partition weights remain intact. Emitted, surviving and effective counts are separate in each report.

The primary 12-by-12 radial/axial coil quadrature is compared with 24-by-24. An independent first/third-order perturbation ODE supplies intrinsic Cs; full off-axis Lorentz particle transport supplies the finite-pupil fit. Quadrature and reference-step refinement are separate from finite-source refinement. The baseline additionally repeats complete 1024+1024 transport at half the maximum integration step.

- Largest absolute 1024-to-2048 relative D95 change: {checks['maximum_source_count_relative_d95_change']:.9g}.
- Largest relative finite-pupil Cartesian Cs difference from the independent third-order reference: {checks['maximum_finite_pupil_cs_relative_error']:.9g}.
- Largest sampled B difference for primary versus refined coil quadrature, divided by centre B: {checks['maximum_sampled_field_quadrature_relative_change']:.9g}.
- Baseline finite-source half-step relative D95 change: {checks['baseline_half_step_relative_d95_change']:.9g}.
- Baseline finite-source half-step relative fitted Cs change: {checks['baseline_half_step_relative_cs_fit_change']:.9g}.

These comparisons establish bounded numerical sensitivities, not exact continuum error or real-instrument accuracy. The field comparison cylinder uses the maximum radius at retained observation planes; it is a sampled check, not a certified continuous beam envelope. No installed operating preset is qualified by this study.

## Size and focus conventions

**RMS is a radius**, measured about the current-weighted centroid. **D95 is a diameter** containing 95% of the current actually surviving at the observation plane. Alpha95 is the 95% current angular radius about the mean direction, converted through arctan. The table uses the fixed first paraxial image plane. Each report also retains 65 through-focus planes and their lowest sampled D95; that discrete minimum is not a continuous optimum. Intrinsic Cs is defined by the unblocked lens and is not reduced by discarding intercepted rays.

## Retained evidence and observer correction

Each case directory retains the source, both transported characteristic partitions, wall classifications, field snapshot, frozen request, perturbation references and final report. The source is synthetic and does not qualify the project's physical FEG. No coherent calculation was run.

The first baseline finished both transport partitions before a statistics-only failure: wall survivors deliberately have NaN hit coordinates, and the initial observer incorrectly treated them as lost. The corrected observer maps only surviving comparison coordinates to infinity; raw wall/source/trajectory files remain unchanged. The complete baseline transport was reused. [The independent array audit](independent-baseline-array-audit.json) verifies source prefix/weights, unchanged field source and transport helpers, current balance, RMS, D95, alpha95 and both fit conventions. The original and corrected observers and transport-field source are retained separately. Baseline recovered elapsed time covers statistics only and must not be read as transport runtime. Later script versions additionally bind transport source hashes and NumPy/SciPy versions; older records lacking that binding cannot be automatically resumed.

## Reproduction

The archived drivers identify the implementation used; active production files may subsequently change. To run a new trial against the current implementation, use a **new output directory** and the archived request files:

```text
python scripts/study_copper_coil_aberrations.py --requests docs/development/copper_coil_aberration_20261113/requests.json --output outputs/new-copper-coil-study
python scripts/study_copper_coil_aberrations.py --requests docs/development/copper_coil_aberration_20261113/refinement-requests.json --output outputs/new-copper-coil-study
```

The immutable [artifact manifest](artifact_manifest.json) binds every other file by relative path, byte length and SHA-256. Solver inputs, study drivers and primary/reference numerical results are explicit; plotted curves are not extra particle experiments.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    files = [dict(path=p.relative_to(args.output).as_posix(), bytes=p.stat().st_size,
                  sha256=sha256(p.read_bytes()).hexdigest()) for p in sorted(args.output.rglob("*")) if p.is_file()]
    write_json(args.output / "artifact_manifest.json", dict(schema="copper-coil-study-artifacts-v1", files=files))
    print(json.dumps(dict(output=str(args.output), files=len(files), checks=checks), indent=2))


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()

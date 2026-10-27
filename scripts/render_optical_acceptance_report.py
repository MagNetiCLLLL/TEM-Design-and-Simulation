"""Render an English joint optical acceptance report from measured evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def write_report(report, destination):
    r = report
    pupil = r["physical_pupil"]
    chain = r["crossover_chain"]
    sample = chain["specimen"]["actual"]
    lines = ["# Frozen-preset physical pupil and joint optical acceptance", "",
        "Scope: Finite-Field Optics, STEM Nanoprobe Normal, C3 + Probe Corrector.", "",
        "The optical controls, source settings, dimensions and positions were not changed. "
        "The target is a 30 mrad physical aperture-edge semi-angle; alpha95 is reported independently. "
        "The initial engineering target interval is 29.7-30.3 mrad, not a manufacturer specification.", "",
        "## Outcome", "",
        f"Overall status: **{r['qualification']['overall']}**. Physical-pupil target: **{pupil['status']}**.", "",
        f"The archived real population has alpha95 **{sample['alpha95_mrad']:.6f} mrad**, "
        f"alpha99 **{sample['alpha99_mrad']:.6f} mrad**, and specimen current "
        f"**{sample['current_a']:.9g} A**. No current floor or ceiling was imposed. "
        "These current quantiles and the maximum sampled particle angle do not define a physical pupil edge.", ""]
    if "specimen" in pupil:
        edge = pupil["specimen"]
        lines += [f"The reference-source physical rim maps to **{edge['alpha_edge_min_mrad']:.6f}-"
            f"{edge['alpha_edge_max_mrad']:.6f} mrad** at the specimen. "
            "This is a bounded, accepted-side measurement at the nominal emission energy. "
            "It is not a claim that the source fills that cone or that a crystal CBED disk has been calculated.", ""]
    stops = sorted({row["limiting_component"] for row in r["boundary"]["sectors"] if row["limiting_component"]})
    lines += ["## Actual limiting elements and filling", "",
        "The first mechanical rejection adjacent to the transmitted reference pupil is: "
        + (", ".join(f"`{key}`" for key in stops) or "not resolved")+". "
        "The reported component comes from the rejected-side physical contact, not from an assumed C2 label.", "",
        "| Aperture | Z (mm) | Radius (um) | Actual sampled radius / aperture radius | Current-95 radius / aperture radius | Nominal-energy support witnesses / aperture radius |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in r["aperture_filling"]:
        if "aperture_radius_um" in row:
            def percent(value):
                return "Not measured" if value is None else f"{100*value:.3f}%"
            lines.append(f"| {row['name']} | {row['z_mm']:.4f} | {row['aperture_radius_um']:.4f} | "
                f"{percent(row['actual_sampled_max_radius_fraction'])} | {percent(row['actual_current95_radius_fraction'])} | "
                f"{percent(row['nominal_energy_source_support_sampled_max_fraction'])} |")
        else:
            lines.append(f"| {row['key']} | {row['status']} | - | - | - | - |")
    source = r["source"]
    support = r["source_support"]
    lines += ["", "Filling is measured relative to the actual hole centre at the plate midplane. "
        "It is a radial occupancy metric, not an illuminated area fraction. Midplane survivors cannot reconstruct "
        "incoming rays already lost at a plate's entrance face. Tilted/translated apertures without exact local-plane "
        "sampling are explicitly unmeasured. The JSON records each plate's actual front and rear faces.", "",
        f"Reference emission energy: **{source['emission_mean_energy_ev']} eV**. "
        f"Configured energy-spread FWHM: **{source['energy_spread_fwhm_ev']} eV**. "
        f"Configured source-direction cutoff: **{source['source_cutoff_slope_mrad']} mrad in slope units**. "
        "These are separate physical quantities; the column accelerating voltage is not the emission energy.", "",
        f"The nominal-energy source-support check propagated {support['rays']} zero-current witnesses from "
        f"the source centre and eight spatial-support rim positions. All survived: "
        f"**{support['all_supported_witnesses_transmitted']}**. "
        f"Their measured specimen angle range is {support['sample_angle_min_mrad']} to "
        f"{support['sample_angle_max_mrad']} mrad relative to the retained chief ray (null means no transmitted evidence). "
        "This finite support sampling is not a rigorous mixed-energy phase-space bound.", ""]
    if "mechanical_rim_within_source_cutoff" in support:
        valid = sum(support["mechanical_rim_within_source_cutoff"])
        lines += [f"Only **{valid}/{len(support['mechanical_rim_within_source_cutoff'])}** measured mechanical-rim "
            "directions lie inside the configured source angular support. Geometrically admitted but unilluminated "
            "rim rays retain exactly zero current; no missing intensity has been synthesized.", ""]
    lines += ["## Complete incident crossover chain", "",
        "| Reference plane | Z (mm) | Actual alpha95 (mrad) | Mapped physical rim (mrad) | RMS radius (nm) |",
        "| --- | ---: | ---: | ---: | ---: |"]
    edge_planes = {row["z_mm"]:row for row in pupil.get("planes",[])}
    for row in chain["executed_waists"]+[chain["specimen"]]:
        a = row["actual"]; edge = edge_planes.get(row["z_mm"])
        text = "Not measured" if edge is None else f"{edge['alpha_edge_min_mrad']:.6f}-{edge['alpha_edge_max_mrad']:.6f}"
        lines.append(f"| {row['region']} | {row['z_mm']:.7f} | {a['alpha95_mrad']:.6f} | {text} | {a['rms_radius_nm']:.6f} |")
    lines += ["", "The C2-C3 required crossover and C3-ADL forbidden crossover tests are "
        f"satisfied at the archived sampling: **{chain['required_topology_satisfied_at_sampling']}**. "
        "Each recorded waist requires an executed two-dimensional three-plane bracket with stable positive-current membership. "
        "An interpolated root alone cannot pass. The additional waist near **1585.9 mm** is retained "
        "as an actual Mini Condenser-to-specimen observation and design-review item; it is not silently deleted or made a new target.", "",
        "| Transfer | Same-cohort alpha95 ratio | Same boundary-ray angular ratio range |",
        "| --- | ---: | ---: |"]
    for relation in chain["angular_relationships"]:
        a,b = edge_planes.get(relation["from_z_mm"]),edge_planes.get(relation["to_z_mm"])
        ratio = None if a is None or b is None else np.array(b["alpha_edge_by_source_azimuth_mrad"])/np.array(a["alpha_edge_by_source_azimuth_mrad"])
        text = "Not measured" if ratio is None else f"{ratio.min():.6f}-{ratio.max():.6f}"
        lines.append(f"| {relation['from_region']} -> {relation['to_region']} | "
            f"{relation['common']['alpha95_ratio']:.6f} | {text} |")
    lines += ["", "The same identified boundary rays are propagated through every listed plane. "
        "These are measured angular ratios, not universal equality constraints. The separate finite-population "
        "4-by-4 regressions in JSON are empirical mechanical-slope fits, not canonical maps or proof of corrector "
        "conjugacy. HP2/HP1 conjugacy, nonlinear aberration cancellation and the driven AC pivot remain separate requirements.", "",
        f"At the specimen the current-weighted radial covariance predicts a local drift waist offset of "
        f"**{r['specimen_focus']['radial_covariance_waist_offset_nm']:.6f} nm**, compared with the provisional "
        "1 nm focus budget. This is a local phase-space estimate inside the objective field; "
        "an exact two-sided field-propagated focus and particle-count convergence are still required.", "",
        "## Numerical evidence and limitations", "",
        f"The existing ensemble contains {len(chain['common_cohort']['source_ids'])} common particles and "
        f"{chain['coverage']['checkpoint_count']} exact incident-column observation planes. Maximum Z spacing "
        f"is {chain['coverage']['maximum_gap_mm']:.6f} mm, with 0.1 mm regions near C1, AC and the specimen. "
        "Its inputs and original source arrays were checked against its saved receipt. "
        "It is reused evidence, not a newly rerun high-particle-count beam.", "",
        f"Boundary search: {r['boundary']['azimuth_count']} azimuths, "
        f"{r['boundary']['evaluations']} source-to-column search probes. "
        f"Step-check status: **{r['numerical_checks']['status']}**.", ""]
    checks = r["numerical_checks"]
    if "maximum_change_mrad" in checks:
        lines += [f"Halving the maximum column step from {checks['coarse_step_mm']} to {checks['fine_step_mm']} mm "
            f"changes the specimen rim angle by at most **{checks['maximum_change_mrad']:.9g} mrad**. "
            f"Maximum relative launch-boundary bracket width is **{checks['radial_bracket_relative_width_max']:.6g}**. "
            "The fine-step check follows the same accepted launches; it does not independently re-solve the mechanical boundary.", ""]
    if "azimuth_nested_envelope_check" in checks:
        check = checks["azimuth_nested_envelope_check"]
        lines += [f"The nested {check['coarse_count']}-to-{check['fine_count']}-azimuth comparison changes the "
            f"sampled minimum/maximum edge by {check['minimum_change_mrad']:.9g} / "
            f"{check['maximum_change_mrad']:.9g} mrad. This is a bounded envelope check, not continuous-azimuth certification.", ""]
    lines += ["A source-to-specimen rim image can differ from the outer boundary of the projected angular domain "
        "if the nonlinear mapping folds. The radial interior scan is retained separately. No source current, phase-space "
        "coordinates or tilt is recentered to improve the result. Angles are exact unit-direction separations from "
        "the separately reported chief ray.", "",
        "Not qualified by this run: full finite-source/mixed-energy pupil envelope; continuous azimuth acceptance; "
        "particle-count convergence; exact gun-internal waist brackets; post-specimen optics; HP conjugacy and aberration "
        "performance; dynamic scan pivot. Previous particle-count refinement was not converged. The specimen entrance "
        "is an endpoint observation, not a new two-sided exact focus bracket.", "",
        "## Reproducibility", "",
        f"Input SHA-256: `{r['input_sha256']}`.", "",
        f"Particle archive SHA-256: `{r['particle_archive_sha256']}`.", "",
        "The adjacent JSON contains the unchanged input reference, solver identities, exact physical stop labels, "
        "all directional rim values, chief tilt/position, current statistics, timings and limitations. "
        "The local diagnostic cache is excluded from Git. The command entry point is "
        "`scripts/measure_optical_acceptance.py`; it performs no preset fitting.", ""]
    Path(destination).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_report(json.loads(args.report.read_text(encoding="utf-8")), args.output)

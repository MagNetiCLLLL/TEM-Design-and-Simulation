"""Generate the final ideal-lens narrative from completed primary observations."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in HERE.parents if (path/"agent.md").is_file())
DESTINATION = ROOT/"docs/development/ideal_lens_aberration_20261113/README.md"
reference = json.loads((HERE/"variational-reference/report.json").read_text())
requests = json.loads((HERE/"requests.json").read_text())
cases = [json.loads((HERE/"finite-source"/row["name"]/"report.json").read_text()) for row in requests]
audit = json.loads((HERE/"audit-summary.json").read_text())
primary = next(row for row in cases if row["request"]["name"] == "fixed-b0.3-s6")


def population(case, count=2048):
    return next(row for row in case["populations"] if row["original_positive_count"] == count)


def view(case, diameter=5.76):
    return next(row for row in population(case)["tube_views"] if row["tube_diameter_mm"] == diameter)


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |", *["| "+" | ".join(map(str, row))+" |" for row in rows]])


intrinsic_rows = []
for case in reference["cases"]:
    variant = case["variants"]["refined8"]
    values = variant["coefficients"]
    intrinsic_rows.append((f"{case['peak_t']:g}", f"{case['sigma_mm']:g}", f"{values['cs_mm']:.8g}",
        f"{values['cc_mm']:.8g}", f"{values['image_z_m']*1e3:.8g}"))
intrinsic_table = table(("B peak (T)", "σ (mm)", "Intrinsic Cs (mm)", "Frozen-hardware Cc (mm)", "First paraxial image Z (mm)"), intrinsic_rows)

fixed = [row for row in cases if row["request"]["family"] == "fixed_source"]
fixed_table = table(("B peak (T)", "σ (mm)", "Paraxial edge slope (mrad)", "Actual α95 (mrad)", "Actual D95 at reference (nm)"),
    [(f"{row['request']['peak_t']:g}", f"{row['request']['sigma_mm']:g}", f"{row['paraxial_edge_slope_mrad']:.7g}",
      f"{view(row)['at_paraxial_image']['alpha95_mrad']:.7g}", f"{view(row)['at_paraxial_image']['d95_nm']:.8g}") for row in fixed])
matched = [row for row in cases if row["request"]["family"] == "matched_paraxial_edge"]
matched_table = table(("B peak (T)", "Prepared source radius (mm)", "Intrinsic Cs (mm)", "Actual α95 (mrad)", "Actual D95 at reference (nm)"),
    [(f"{row['request']['peak_t']:g}", f"{row['source_radius_mm']:.8g}", f"{row['intrinsic_cs_mm']:.8g}",
      f"{view(row)['at_paraxial_image']['alpha95_mrad']:.8g}", f"{view(row)['at_paraxial_image']['d95_nm']:.9g}") for row in matched])
native_rows = []
for row in population(primary)["tube_views"]:
    image, minimum = row["at_paraxial_image"], row["native_d95_minimum"]
    rms_minimum = min(row["focus_planes"], key=lambda p: p.get("rms_radius_nm", float("inf")))
    bracket = row["neighbour_offset_bracket_nm"]
    native_rows.append((f"{row['tube_diameter_mm']:g}", f"{100*image['transmission']:.7f}", f"{image['d95_nm']:.7f}",
        f"{minimum['d95_nm']:.7f}", f"{minimum['offset_from_paraxial_nm']*1e-3:.6g}",
        f"[{bracket[0]*1e-3:.6g}, {bracket[1]*1e-3:.6g}]", f"{rms_minimum['offset_from_paraxial_nm']*1e-3:.6g}"))
native_table = table(("Tube ID (mm)", "Current transmission (%)", "D95 at paraxial image (nm)", "Lowest sampled D95 (nm)", "Its offset (µm)", "Neighbour bracket (µm)", "Sampled RMS-minimum offset (µm)"), native_rows)
high_angle = [row for row in cases if row["paraxial_edge_slope_mrad"] > 100.]
high_table = table(("Primary case", "Declared paraxial edge slope (mrad)", "Actual α95 at widest declared tube (mrad)"),
    [(row["request"]["name"], f"{row['paraxial_edge_slope_mrad']:.8g}",
      f"{max(population(row)['tube_views'], key=lambda p:p['tube_diameter_mm'])['at_paraxial_image']['alpha95_mrad']:.8g}") for row in high_angle])
large = next(row for row in cases if row["request"]["name"] == "filled-tube5.76")
large_fit = population(large)["vacuum_fit"]
radial7 = next(row for row in large_fit["radial_geometric_fits"] if max(row["powers"]) == 7)
worst_d95 = max(abs(check["relative_d95_change"]) for row in cases for check in row["source_count_comparison"]["tube_checks"])
worst_cs = max(abs(row["source_count_comparison"]["relative_cs_fit_change"]) for row in cases)
field_difference = max(row["field_order_evidence"]["order7_vs9_relative_B"] for row in cases)
field_curl = max(row["field_order_evidence"]["order9_normalized_curl"] for row in cases)
tests = list(ET.parse(HERE/"preflight-tests.xml").iter("testcase"))

text = f"""# Intrinsic aberrations of an ideal Gaussian magnetic lens, 2026-11-13

A prescribed, round Gaussian magnetic lens has **nonzero intrinsic spherical
aberration** even when no empirical Cs action, iron geometry, winding model or
material imperfection is added. At 300 keV, the B peak = 0.3 T, σ = 6 mm
reference gives **Cs = {primary['intrinsic_cs_mm']:.9f} mm**. The nine declared
B/σ combinations, finite-source tube tests and separate source-boundary
comparisons below establish results for this numerical model. They do not
identify a universal minimum-Cs lens, a TEM working point, or an optimized
diffraction-limited probe.

The experiment and acceptance checks were declared in [the study plan](study-plan.md)
before finite-source execution. There are 14 primary cases plus 12 numerical
refinements, each built from an original 1024-particle transport and a second
1024-particle extension. The independent [variational reference](variational-reference/report.json)
has no emitted finite source. No coherent-wave or material-field solve is
performed and no application optical preset is retuned.

## Field and reference definitions

The axial profile is `B(z) = B_peak exp(-z²/(2 σ²))`, centred at Z = 0 and
explicitly continued to all Z. **σ is the Gaussian standard deviation**:
FWHM is 2.35482 σ, not σ, and neither quantity is a hard field extent. The
declared peaks are 0.15, 0.3 and 0.6 T; σ is 3, 6 or 12 mm. A near-axis
regular vector-potential expansion begins

```
A_phi = r B(z)/2 - r³ B''(z)/16 + r⁵ B''''(z)/384 + ...
B = curl(A)
```

The regular expansion is described by
[Khan and Jagannathan, equations 1-2](https://arxiv.org/html/2009.13943v2).
The new finite-source calculations retain odd radial powers through order 9,
with orders 5 and 7 used in separate refinements. A finite truncation has a
remaining current-free curl residual; it must not be called an exact global
Maxwell solution. [The implementation receipt](implementation-reference/high_order_round_lens.py)
and per-case field dictionaries specify the numerical law.

The expectation of intrinsic aberrations is consistent with
[CEOS's statement of Scherzer's theorem](https://www.ceos-gmbh.de/en/basics/scherzerstheorem)
for round, static lens fields without space charge. These references motivate
the model and interpretation; neither publication predicts the numerical
Gaussian Cs values reported here. No external image is reproduced.

At E0 = 300000 eV, the first paraxial image is obtained from the independent
linear ray equation. The source is at -8 σ with the declared zero-Larmor
entrance matrix. That fixed field, initial ray family and image plane are
retained for Cs/Cc measurements. The displacement convention is

```
delta_r = -Cs * theta * |theta|² + Cc * theta * (delta_E/E0)
```

Here theta is the declared outward image-space **paraxial pupil slope**,
not the actual finite-source α95. Third-order variational Lorentz equations
give Cs independently of the finite-pupil fits; Cc uses frozen-hardware energy
perturbations without refocusing each energy. This is a classical reference,
not a quantum calculation from the cited paper.

{intrinsic_table}

Image Z is measured from the field centre, not a pole face. Several first
images lie inside substantial field; the strongest/widest case even has
negative image Z. These must not be read as focal distances from a hard field
edge. There is no monotonic or global-optimum conclusion beyond the declared
parameter/reference family. Equal B_peak × σ cases satisfy the recorded
dimensionless scaling checks; changing σ at fixed peak also changes lens
strength, rather than merely rescaling an otherwise identical lens.

![Intrinsic variational Cs and Cc over the declared grid](intrinsic-coefficients.png)

## Prepared source, physical tube and current

The synthetic beam is monochromatic at 300 keV and has zero incoherent
emittance. Its declared current is 1 µA. The fixed-source family uses a
uniform-area disk of radius 0.95 mm and the fixed paraxial entrance matrix;
it is not an FEG, a 0.3 eV spread distribution or an experimental brightness
model. Samples use fixed inverse-CDF Halton coordinates with persistent IDs.
Increasing N from 1024 to 2048 appends particles without changing existing
positions, directions, energy, clocks, source boundary or trajectories.

The original 1024 raw partition retains its historical weights 1/1024.
The published 2048 population explicitly uses 1/2048 for every original ID at
the same 1 µA source current. This new population normalization does not
rewrite the earlier raw record and never restores intercepted current.
Each newly transported partition has 1024 original positive-current packets;
larger-population observables reuse the exact completed prefix.

The passive tube IDs are 1.5, 2 and 5.76 mm. Its magnetic feedback is
deliberately absent: changing the tube never changes the prescribed field.
The original vacuum characteristics continue numerically after contact for
the independently declared vacuum aberration fit; **that continuation is
counterfactual**. Only the prefix before first wall contact is a physical
trajectory for a tube view. Contact is classified over each DOP853 dense
polynomial, including exit/re-entry and tangent cases, rather than just saved
endpoints. Complete contact phases, source IDs, bounds and status ledgers are
retained. Unresolved contacts must be rejected, not treated as ordinary losses.

The [real-column inventory](vacuum-tube-inventory.json) supplies context only:
the reconstructed C1 lens/C2 lens-to-Objective continuous pipe is ID 5.76 mm
(radius 2.88 mm), despite 20 mm main lens bores. Some carrier and sample-stage
passages are ID 5 mm. The inserted C2 optical stop is a different boundary.
The ideal experiment does not reconstruct all these actual devices. A tube
diameter does not stand for a ferromagnetic pole bore in a solved material
model, and tube clipping is not evidence that intrinsic Cs has improved.

At B peak = 0.3 T and σ = 6 mm the identical fixed source and field give:

{native_table}

D95 is the diameter containing 95% of the **actual transmitted current**
about its current-weighted centroid; RMS radius uses the full transmitted
second moment. Both are geometric ray statistics, not wave or diffraction
resolution. The 1.5 mm tube reduces transmitted current and D95, while the
independent intrinsic Cs remains {primary['intrinsic_cs_mm']:.8g} mm. The
vacuum Cs fit uses the complete predeclared source, never a wall-selected
population.

![Passive-tube current, geometric size and unchanged intrinsic coefficient](tube-current-size-intrinsic.png)

The native focus scan has **129 executed planes**. Its lowest sampled D95
and neighbouring bracket are secondary observations, not exact continuous
best focus or a precision-qualified optimized probe. The base scan spacing
is 9.375 µm. D95 and RMS minima are separately observed; neither replaces the
fixed paraxial image used for the principal size comparisons.

![Coarse native focus scan with separate D95 and RMS statistics](native-focus.png)

## Comparisons answer different questions

The nine **fixed-source** cases keep radius 0.95 mm and use the 5.76 mm tube
for this table. Actual convergence changes substantially, so D95 does not
have to decrease monotonically with stronger peak field.

{fixed_table}

![Actual fixed-source size and convergence at the paraxial reference](fixed-source-grid.png)

The **matched-edge** family instead changes the prepared source radius to
obtain a 10 mrad nominal paraxial edge slope at σ = 6 mm. This is an explicit
source-boundary change, not the same beam compared at different fields.
An edge slope is not a 95%-current containment angle; actual α95 is reported.

{matched_table}

The approximately 12 nm value is an actual geometric diameter at its
specified paraxial reference plane. It is not a globally optimized probe,
a native continuous-focus minimum or a wave-resolution result.

![Matched nominal edge with independently reported actual α95](matched-paraxial-edge.png)

The **equal tube-filling** family uses source radii 0.7125, 0.95 and 2.736 mm
for tube IDs 1.5, 2 and 5.76 mm respectively: each source fills 95% of the
tube radius at entrance. Field and σ remain fixed, but source boundaries
differ. The middle point reuses the already executed base source, rather
than transporting duplicate particles to inflate a sample count.

For the largest source, the recorded Cartesian fit's C3 recovery error is
**{100*large_fit['relative_cs_difference']:.7f}%**, exceeding the declared
0.1% reference gate. Both its 1024- and 2048-particle failure records remain
failures. An independent radial fit through seventh order gives C3 error
{100*radial7['relative_cs_difference']:.7f}%, supporting higher-order finite-pupil
contamination of the lower-order fit. This does **not** qualify C5 or C7, make
the failed fit pass, or change the independent intrinsic Cs coefficient.

![Equal filling and finite-pupil C3 recovery](filled-pupil-sensitivity.png)

## High-angle scope and numerical checks

The following primary cases exceed the earlier benchmark's 100 mrad nominal
pupil-slope scope. Their new finite-geometric results are explicitly labelled
exploration; the earlier benchmark's range is not silently extended.

{high_table}

All 14 primary cases satisfy the declared 1024-to-2048 source-count comparison
gates. The largest paraxial-image D95 change is {100*worst_d95:.8g}% and
the largest recorded Cartesian C3 change is {100*worst_cs:.8g}%. These are
observed numerical sensitivities, not a proof of continuous-source error or
experimental accuracy. No intercepted or tail packet is deleted from the
declared vacuum fit.

All sampled primary field-grid/path checks pass their stated gates. The
largest order-7 versus order-9 sampled B difference is {field_difference:.8g};
the largest normalized order-9 curl residual is {field_curl:.8g}. These are
sampled estimates, not rigorous global field-error bounds. Twelve separate
refinements cover four representative sources at orders 5 and 7 and with a
halved maximum step. The four actual half-step integrations pass their
declared gates and use genuinely finer accepted meshes. Order-5/7 trajectory
differences are preserved as diagnostics, not relabelled integration-step
passes.

![Declared order and step refinements](numerical-refinements.png)

The independent [array audit](audit-summary.json) checks all 26 raw case
inventories, original IDs/weights, directions, energy and flight clocks,
physical current ledgers, actual D95/RMS/α95, whole-source fit recovery,
12 refinement source/observation matches and the retained failures. The
[validation summary](validation-summary.json) also records {len(tests)} passing
targeted software tests. This is not a claim that the full application suite
was run or that every fit acceptance gate passed.

## Reproduction and retained evidence

The [declared requests](case-inventory.json), original
[primary plan](declared-requests/requests.json) and
[refinement plan](declared-requests/requests-refinement.json) preserve every
input. Each finite-source directory contains original report/request/field
bytes, the complete source, exact complete wall ledgers, and selected original
phase rows from both 1024-particle partitions. All original IDs and historical
weights remain present. Selection includes paraxial image, D95/RMS observed
minima and neighbours, focus endpoints and representative path planes. All
129 focus scalar records and moments at every original checkpoint are kept.

Selected phase arrays are byte-checked against the corresponding original
partition rows. Every reported focus statistic and absolute current balance
is independently recomputed by the curator at relative tolerance 2e-13.
[The verification receipt](selected-phase-verification.json) declares the
selection. Raw full-file origins and SHA256 hashes are retained in the
[scientific manifest](artifact_manifest.json); the roughly 384 MiB raw
characteristic collection is not duplicated. The compact selection is **not
the complete trajectory or a retained dense interpolant**. Original dense
polynomial contact evidence is represented by its exact computed ledger,
receipts and independent tests; recomputing dense trajectories requires a
replay of the original model.

The byte-preserved research drivers are under `drivers/`; implementation
references are under `implementation-reference/`. The three original finite
research helper paths are `scripts/study_ideal_lens_aberrations.py`,
`scripts/ideal_lens_beam.py` and `scripts/ideal_lens_wall.py`. Their path-based
imports and receipts expect the original repository layout: for replay,
restore those exact bytes to those recorded paths in a checkout with the
recorded source identity. The archived copies are provenance evidence, not
an assertion that moving them to an arbitrary directory preserves import
semantics. The independent reference driver originally lives under
`outputs/ideal-lens-aberration-20261113/ideal_lens_reference.py` and expects
the same source tree. The curator and plotting drivers only read completed
data and do not perform a new physical calculation.

All figures are newly generated from the retained numerical observations;
PNG, vector SVG and standalone PDF versions are included. Archiving and
successful integrity checks do not remove the declared high-angle, finite-
pupil, source-count, coarse-focus or material-model limitations.
"""
DESTINATION.write_text(text, encoding="utf-8")
print(DESTINATION)

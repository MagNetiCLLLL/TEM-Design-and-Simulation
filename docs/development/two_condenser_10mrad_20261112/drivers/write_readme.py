"""Write the 70 um high-angle archive narrative from completed observations."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in HERE.parents if (path/"agent.md").is_file())
OUT = ROOT/"docs/development/two_condenser_10mrad_20261112/README.md"
primary = json.loads((HERE/"focused4096-s005/report.json").read_text())
step = json.loads((HERE/"final4096-s010/report.json").read_text())
reference = json.loads((HERE/"reference2048-s005/report.json").read_text())
projection = json.loads((HERE/"projector/weak-c2-final-exact/final-exact-objective.json").read_text())
size_difference = 100*(reference["d95_nm"]/primary["d95_nm"]-1)
step_difference = 100*(step["d95_nm"]/primary["d95_nm"]-1)
validation_path = HERE/"validation-summary.json"
validation_note = "Final default-promotion and configuration/GUI checks are retained in [the validation summary](validation-summary.json)."
if validation_path.exists():
    validation = json.loads(validation_path.read_text())
    count = validation["current_unique_test_counts"]["passed"]
    validation_note = (f"The final de-duplicated summary records **{count} current passing software cases**. "
        "The original final-mode batch and target rerun receipts are preserved separately; "
        "the rerun corrects the stale Microprobe seed expectation that had assumed inherited C3 behavior. "
        "Read [the validation summary](validation-summary.json) for exact case identities and receipts.")

text = f"""# Physical C1+C2 Nanoprobe: 70 um aperture and alpha95 >= 10 mrad

The true two-condenser column has a **fixed 4096-particle calibrated working
point** with geometric D95 = **{primary['d95_nm']:.9f} nm** and alpha95 =
**{primary['alpha95_mrad']:.8f} mrad** at the specimen upper surface. The
observed native D95 minimum is at that surface in both 4096-particle step
checks. **Source-sampling convergence is not established:** the unchanged
2048-particle reference has its D95 minimum 16 nm upstream and fails the
1 nm placement gate. The default records this sampling-sensitive status;
this is not a general or continuous-source focus qualification.

The explicit new requirements are a **70 um physical C2 aperture diameter**,
**alpha95 >= 10 mrad**, and direct final focusing by the Mini Condenser-
Objective combined field. Actual hardware is C1+C2, with no C3 or probe
corrector and no Normal/Large angle branch. It is not the three-condenser
column with C3 switched off. The separate [low-angle study](../two_condenser_nanoprobe_20261112/README.md)
remains unchanged; its approximately 2.2 mrad and 0.04 nm values are not
results of this new high-angle study.

All sizes are **classical geometric ray-bundle statistics**. The incident
system has no probe spherical-aberration corrector. These values do not
establish diffraction-limited resolution, coherent-wave probe size, an OEM
calibration or experimental microscope performance. No new coherent
calculation was performed.

## Controls, hardware and current

| Device | Final excitation (%) |
| --- | ---: |
| C1 | 52.5 |
| C2 | 17.75 |
| Mini Condenser | -10.0 signed Nanoprobe excitation |
| Objective | 29.292665862003528 |
| Diffraction lens | 13.59281212137673 |
| Intermediate lens | 8.51783304453163 |
| Projector 1 | 32.1132572718703 |
| Projector 2 | 9.826874422815507 |

The instrument is FEG / C2 / No Energy Filter at 300 kV. The C2 opening is
captured as an installed, enabled and inserted finite-thickness plate with
diameter 70 um and radius 35 um, centred at 765 mm with faces at 764.9 and
765.1 mm. The component mask, legacy clipping and native mechanical contact
geometry agree; a 34 um radial test point passes and a 36 um point is opaque.
The audit receives an explicit 0.070 mm expected diameter; its historical
100 um default is not silently inferred or changed.

The specimen centre is 1025.2 mm, active thickness is 5 nm, and the actual
upper-surface reference is **1025.1999975 mm**. The formal Nanoprobe/
Diffraction pair retracts the fluorescent screen and camera while retaining
HAADF/DF/BF insertion choices. These are actual saved device states.

At 4096 particles, all original positive-current packets reach the physical
gun exit. C2 intercepts 206 packets and **0.5029296875 uA**; the specimen
receives 3890 packets with **{primary['sample_current_a']*1e6:.10f} uA**,
or **{primary['emitted_current_transmission']*100:.8f}%** of the represented
10 uA emission current. Effective sample count is 3890. Every original ID
and weight 1/4096 remains recorded, including lost packets. No clipped
current is restored or surviving weight rescaled. Other interception losses
are absent in these runs and the absolute current ledger closes.

The early 1024-particle branch comparison retained physical current losses:

| C1 (%) | C2 (%) | alpha95 (mrad) | Surviving count | Specimen current (uA) |
| ---: | ---: | ---: | ---: | ---: |
| 55 | 80 | 10.408752 | 807 | 7.880859 |
| 55 | 20 | 10.135157 | 806 | 7.871094 |
| 52.5 | 17.75 | 10.313303 | 975 | 9.521484 |

Those initial points were defocused and are archived as branch-selection
evidence, not passing final focus observations. The weaker, higher-current
C1/C2 combination was retained for joint Objective focus matching.

## Native D95, RMS and numerical sensitivity

D95 is the diameter about the actual current-weighted transverse centroid
containing 95% of the current reaching a plane. RMS radius is the second-
moment radius of the entire transmitted current. Alpha95 is the semi-angle
containing 95% of the transmitted angular current about its chief direction;
angular RMS is separate. Neither alpha95 nor D95 is a known aperture-edge
quantity. This study introduces no inherited corrected-column 30 mrad or
0.7 nm acceptance target.

| Original positive count | Step (mm) | Surface D95 (nm) | Surface RMS radius (nm) | alpha95 (mrad) | Angular RMS (mrad) | Native D95 offset (nm) | Native RMS offset (nm) | D95 surface gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 0.05 | {primary['d95_nm']:.9f} | {primary['rms_radius_nm']:.9f} | {primary['alpha95_mrad']:.8f} | {primary['angular_rms_mrad']:.8f} | 0 | +72 | PASS for this finite sample |
| 4096 | 0.10 | {step['d95_nm']:.9f} | {step['rms_radius_nm']:.9f} | {step['alpha95_mrad']:.8f} | {step['angular_rms_mrad']:.8f} | 0 | +72 | PASS for this finite sample |
| 2048 | 0.05 | {reference['d95_nm']:.9f} | {reference['rms_radius_nm']:.9f} | {reference['alpha95_mrad']:.8f} | {reference['angular_rms_mrad']:.8f} | -16 | +66 | FAIL |

The calibrated 4096-particle D95 minimum has executed neighbour planes at
-0.25 and +0.25 nm. This is a **discrete observation bracket for that finite
sample**, not a continuous-source +/-0.25 nm error bound. Its native RMS
minimum is separately observed at +72 nm, bracketed by +70 and +74 nm; the
minimum RMS radius there is 1.150311326 nm and D95 is 4.461030562 nm.
The two minima differ by about 72 nm, with a separation bracket of
69.75-74.25 nm that excludes zero. A D95 focus must not be relabelled as an
RMS focus, and no back-projected tangent waist substitutes for either one.

The 0.1 mm 4096-particle D95 is {step_difference:.7f}% larger than the 0.05 mm
value. Their original source arrays are identical. Integration-step
sensitivity is small at this calibrated population. In contrast, the
2048-particle surface D95 is **{size_difference:.5f}% larger** than the
4096-particle value at identical controls, source law, requested planes and
stop. Its D95 minimum has a -18 to -14 nm bracket. It was not independently
refocused to hide this difference. It has 1950 survivors, effective count
1950 and 9.521484375 uA current (95.21484375% transmission).

All controlled trials use checkpoint population budget 4096 and identical
executed checkpoint coordinates. The 2048/4096 fine runs also have identical
integration coordinates. The scene's emitter-count metadata changes the
aggregate potential digest; individual executed provider identity records
remain equal. This does not assert a separate byte comparison of every
field-sampling array. Read the [controlled comparison](final-controlled-comparison.json)
and [scene/count diagnostic](final-scene-count-diagnostic.json). These
results support a fixed-population calibration and explicitly leave
continuous-source focus convergence unresolved.

![Fixed-control native D95 and RMS through focus](native-focus.png)

## Crossover interpretation

The following statements refer to executed finite two-dimensional RMS
source-image regions and their bounded sampling, not ideal point crossings:

| Interval | Observed source-image regions and limits |
| --- | --- |
| Gun exit - C1 | No 2D waist in the observed 450-500 mm interval. |
| C1 - C2 | One region, representative plane 503.0 mm, inside the C1 mechanical body (450-550 mm). |
| C2 - Mini | One region near 687.936455 mm in the fixed sample-surviving cohort; the all-current interval is UNRESOLVED at the physical C2 population-transition singleton near 765 mm. |
| Mini - specimen upper surface | No additional pre-specimen 2D RMS waist. |
| Full final Mini/Objective support through focus | One final 2D region; its RMS representative is 72 nm after the surface, while the independently measured native D95 minimum is at the surface for the calibrated 4096 sample. |

The C2 transition must not be counted as zero waists or silently promoted to
an exact all-current topology count. The current before interception and
the common surface-surviving cohort have different statistics:

| Representative plane | Population | D95 (nm) | RMS radius (nm) | alpha95 (mrad) | Angular RMS (mrad) |
| --- | --- | ---: | ---: | ---: | ---: |
| 503.0 mm | All 4096 original packets | 36.281239 | 10.186664 | 0.135835 | 0.076268 |
| 503.0 mm | 3890 surface survivors | 31.894958 | 9.245731 | 0.119407 | 0.069225 |
| 687.936455 mm | All 4096 original packets | 361.280846 | 101.434873 | 0.456338 | 0.256223 |
| 687.936455 mm | 3890 surface survivors | 317.643597 | 92.070042 | 0.401171 | 0.232562 |
| Specimen upper surface | 3890 surface survivors | 3.009126 | 1.226283 | 10.208349 | 5.917838 |

The complete last-field support is observed from about 848.65 mm through
the native focus window; no separate extra pre-specimen crossover is found.
Absent CEOS corrector crossovers or absent C3 hardware are not inserted to
match another assembly. A scan-deflector conjugacy or finite-angle scanning
image qualification is not supplied by this incident-probe observation.

![Physical column envelope and common transmitted cohort](column-envelope.png)

The largest-radius primary packet has ordinary original weight 1/4096,
centroid radius 3.644717 nm, and a C2-plane radius 34.795900 um inside the
35 um opening. Its radius is only 1.000714 times the next largest; it
contributes about 0.2271% of the radial second moment. This is not a lone
zero-current or artificially low-weight outlier. The tail is retained.
The observed uncorrected geometric distribution has different D95/RMS
minima; this evidence does not separately apportion spherical/chromatic
aberration or prove a single cause. See the [source/tail audit](final-source-tail-audit.json)
and [native crossover statistics](final-crossover-statistics.json).

![Absolute physical current and original surviving count](current-retention.png)

## Independent Diffraction reference

At the exact final upstream controls, nine **zero-current differential
pencils**, with emitter invocation forbidden, give canonical position-
response norm {projection['canonical_position_residual_norm']:.9g} and
effective camera length {projection['effective_camera_length_m']*1e3:.10f} mm
at the retracted main-screen reference Z = 2140.15 mm. All nine reach the
reference. State and maximum step are both 0.05 mm, position increment is
1e-10 m, and mechanical-slope increment is 1e-7 rad. The full vector-potential
canonical basis, matrices, receipts and original hashes are preserved in the
[exact-point record](projector/weak-c2-final-exact/final-exact-objective.json).

The 0.1/0.05 mm and half-increment checks in
[the provisional case](projector/weak-c2-provisional/PROVISIONAL_REPORT.md) belong to
Objective 29.29264%, not the exact final 29.292665862003528%. They remain
separate original records. The observer scripts live in the parent archived
projector directory; an artifact key recorded as a basename retains that
original script-location relationship.

This is a local on-axis projection reference, not a 9-particle finite source
or current measurement. Finite-angle detector acceptance, specimen scattering,
DF scanning images, camera signal and coherent resolution remain unqualified.

## Saved inputs, provenance and compact evidence

[The primary profile](two-condenser-nanoprobe-diffraction.toml) is an exact
copy of the observed 4096-particle, 0.05 mm input. Its specimen is retracted
and raster/wobble and specimen-wave flags are disabled to measure the vacuum
incident probe through the physical upper-surface reference. Loading this
complete input restores those diagnostic choices as well as the optics.

The original snapshot restore and operating-profile round trip pass and
retain the measured hardware, controls, source count, aperture and surface.
The separately labelled [portable input derivative](portable-focused4096-inputs.json)
pins the external configuration inventory and also passes input restoration.
It is an input-only packaging derivative; it is not a new physical execution
or a rewritten original snapshot. Details are in [saved-input validation](saved-input-validation.json).
{validation_note} Promoting defaults and
validating inputs does not convert the source-sampling limitation into a PASS.

Every trial retains its original report, request and profile bytes, a
losslessly compressed complete input snapshot, and embedded manifest bytes
checked against their recorded hashes. All native scalar focus observations
and moments at every original checkpoint are preserved. Complete-population
native phase rows are selected at observed D95/RMS minima, their neighbours,
the exact surface, representative defocus, endpoints and topology evidence.
Each selected row retains all original particles, IDs, weights and survival
states. Exact original indexes, full raw-array paths and SHA256 hashes are
recorded. This selection is **not the complete original trajectory array**.

The [selected-phase verification](selected-phase-verification.json) compares
every selected original array's dtype, shape and bytes and recomputes native
D95/RMS minima, brackets and surface statistics from already executed data.
The scientific [artifact manifest](artifact_manifest.json) covers all archived
files except itself. Initial observer bytes were recovered from the saved
b003 version by removing the later audit-helper hash entry; their SHA256
exactly matches the original ab71 trial receipts. The current and intermediate
observers are separately preserved. No historical trial or low-angle sealed
archive was modified, and archiving does not independently qualify a working
point or extend the scope of the underlying observations.
"""
OUT.write_text(text, encoding="utf-8")
print(OUT)

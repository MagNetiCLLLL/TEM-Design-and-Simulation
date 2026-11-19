# Installed lens-winding inventory and ideal-field equivalence

Audit date: 2026-11-15. This is a read-only audit of the current working tree.
No default optical control, source parameter, boundary, propagation tolerance,
field quadrature, startup preference or production implementation was edited.

## Findings

- The standard default assembly has **13 installed round-lens winding records**.
  Every winding has an explicit annular ID, OD and occupied axial envelope.
- All 13 lack a turn count, bare-wire diameter and independent conductor
  current. Their legacy reference is **3000 A turn at 100% excitation**, with
  one winding share of 1.0 per owner. This does not determine N and I separately.
- At exactly equal per-segment NI, all 13 new/legacy operators give bitwise
  identical sampled fields. The original envelope, pose and quadrature agree.
- The unmodified new/legacy input adapters are **not strictly identical for
  every nominal N/I decomposition**. C1 lens, C3 lens and Diffraction lens fail
  the existing componentwise comparison at near-zero symmetry components due
  to last-bit current-normalization roundoff. This finding is retained, not
  hidden by the controlled equal-coefficient check or by relaxed tolerances.

## Baseline and units

The baseline is the normal desktop startup without a compatible saved-result
restore: **FEG / C3 + Probe Corrector / No Energy Filter**, Finite-Field Optics,
Normal Nanoprobe and Diffraction. The audit reproduces the stored preset
assignments on a detached state, without invoking transport or an optical fit.
It follows `MainWindow` startup, rather than reporting the different factory
component values returned before operating-mode application.

The default round-field selection is still `gaussian`. The NI values below
are the prescribed-current source excitation that would be used at these same
controls, not a calibrated electrical current of the Gaussian lens model.
Only detached comparison states select the compatible `copper_coil` identifier
for **Ideal prescribed-current coils in vacuum**.

The local automatic-result archive was independently inspected through its
index/implementation gate. Its older implementation does not match the current
solver; the existing recovery code would use the standard startup above. No
GUI restore, large particle-array reload, index update or fallback calculation
was executed in this audit. Details are in the independent review receipt.

All geometric lengths below are **mm**. ID and OD describe the winding pack,
not the lens housing, pole bore or individual bare conductor. Dimensions and
NI references are simulator inputs; their saved status is
`engineering_reference_not_oem_not_particle_qualified`, not an OEM measurement.
The full-precision JSON and CSV retain the original floating-point values.

## Installed round-lens windings

`Undefined` in the three electrical/wire columns means the optional input field
is absent; it does not mean zero. Actual conductor current is also undefined.
Current cannot be recovered uniquely from NI without a turn count.

| Lens | ID | OD | Occupied axial length | N | Bare wire diameter | I at 100% (A) | Excitation (%) / polarity | Signed ideal NI (A turn) |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| C1 lens | 96 | 156 | 90 | Undefined | Undefined | Undefined | 51.6666666667 / + | 1550.000000 |
| C2 lens | 96 | 156 | 162 | Undefined | Undefined | Undefined | 13.8990401648 / + | 416.971205 |
| C3 lens | 96 | 156 | 153 | Undefined | Undefined | Undefined | 15.6882368118 / + | 470.647104 |
| Probe Adapter lens | 78.33 | 91.05 | 32.76 | Undefined | Undefined | Undefined | 10 / + | 300.000000 |
| Probe TL22 lens | 78.33 | 93.5109425 | 22.95 | Undefined | Undefined | Undefined | 59.9315250358 / - | -1797.945751 |
| Probe TL21 lens | 78.33 | 93.3164759 | 22.95 | Undefined | Undefined | Undefined | 59.9552790684 / + | 1798.658372 |
| Probe TL12 lens | 78.33 | 93.63 | 22.86 | Undefined | Undefined | Undefined | 36.0045587087 / + | 1080.136761 |
| Mini Condenser lens | 34 | 51 | 18 | Undefined | Undefined | Undefined | 43.8864661722 / - | -1316.593985 |
| Objective lens | 60 | 132 | 145 + 145 = 290 | Undefined | Undefined | Undefined | 68.9781060083 / + | 2069.343180 |
| Diffraction lens | 60 | 132 | 90 | Undefined | Undefined | Undefined | 14.5776171972 / + | 437.328516 |
| Intermediate lens | 60 | 132 | 162 | Undefined | Undefined | Undefined | 0.0350826572594 / + | 1.052480 |
| P1 lens | 60 | 132 | 108 | Undefined | Undefined | Undefined | 32.1132572719 / + | 963.397718 |
| P2 lens | 60 | 132 | 162 | Undefined | Undefined | Undefined | 9.82687442282 / + | 294.806233 |

### Objective source envelope

The Objective child record has a historical display length of 162 mm. The
authoritative current-bearing envelope comes from its parent yoke/inset
definition and has two 145 mm regions:

| Region | Module-local Z | Global Z | NI share |
| --- | --- | --- | ---: |
| Upper | 946.5 to 1091.5 | 1396.5 to 1541.5 | 0.5 |
| Lower | 1206.9 to 1351.9 | 1656.9 to 1801.9 | 0.5 |

The 115.4 mm intervening gap carries no winding current. The ideal field
retains the full ID 60 / OD 132 envelope in each occupied region. Detailed
upper CAD includes a 0.2 mm inner insulation layer, leaving a conductor
allocation with ID 60.4 mm; that passive layer does not shrink the homogenized
ideal-field envelope. The 290 mm total must not be replaced by the child
display length or by an uninterrupted 405.4 mm bounding interval.

## Optional round lenses outside the default assembly

The catalog contains seven additional unique round-lens owners. CETCOR uses
five of them; B-COR adds TL31 and TL32. None is installed in the current default
assembly, and their disabled/dormant state must not be treated as a known
conductor current. All seven also lack N, wire diameter and independent I100;
each has a saved 3000 A turn legacy reference.

| Optional lens | ID (mm) | OD (mm) | Occupied axial length (mm) | N / wire diameter / I100 |
| --- | ---: | ---: | ---: | --- |
| Image TL11 lens | 115.18 | 145.3984515 | 40.5 | All undefined |
| Image TL12 lens | 85.7 | 101.0654192 | 41.4 | All undefined |
| Image TL21 lens | 75 | 99.9595812 | 32.4 | All undefined |
| Image TL22 lens | 75 | 99.7567878 | 32.4 | All undefined |
| Image TL31 lens (B-COR) | 75 | 99.9595812 | 32.4 | All undefined |
| Image TL32 lens (B-COR) | 75 | 99.7567878 | 32.4 | All undefined |
| Image Adapter lens | 75 | 89.5151499 | 32.4 | All undefined |

The independent catalog inventory records module membership. The ideal-field
comparison in this audit covers the **13 installed default owners**, not a
qualification of inactive image-corrector topologies.

## Other magnetic hardware and control channels

Deflectors, stigmators and hexapoles are not annular round-lens sources. Their
pole/coil insert geometry and calibrated multipole drives are separate from
the new round-winding N/d/I contract. The independent classification lists:

- Five conventional deflection hardware groups and five probe steering inserts.
- Two main hexapoles and four stigmator assemblies.
- Four quadrupole channels, two auxiliary hexapole channels, a shared Descan
  channel and the virtual DP12 channel, with optical channels distinguished
  from their physical hosts to avoid double counting.

These records also do not provide per-wire N, diameter or independent current.
For example, a stigmator's `coil_count=8` counts coil bodies/poles, not wire turns.
They are excluded from the axisymmetric ideal-source equivalence claim.

## New/legacy field comparison

The comparison keeps every source's ID, OD, occupied intervals, rigid pose,
12-point radial/axial quadrature order, full field tails and existing numerical
formulas unchanged. The public A/B and B-only sampling APIs are both checked.

The 9,975 unique vacuum points cover the axis, tiny radii, both sides of the
near-axis branch transition, the bore, the exterior, source centres, axial
faces and nearby points, and tails out to ten envelope length scales. The
Objective gap and its two source intervals are retained. No point is a source
particle or a transported electron.

Three test-only decompositions are used per installed lens:

| Case | N | Bare wire diameter (mm) | I100 (A) | Reference NI |
| --- | ---: | ---: | ---: | ---: |
| Explicit | 100 | 0.25 | 30 | 3000 A turn |
| Different N/I | 200 | 0.125 | 15 | 3000 A turn |
| Different wire | 100 | 0.125 | 30 | 3000 A turn |

All synthetic layouts are valid within their original CAD allocation. These
values are test fixtures, not inferred winding data, and are not written into
the default input files.

The archived `equivalence.json` convenience scalar named
`actual_synthetic_current_a` was recomputed as `I100 * percent * 0.01 * polarity`.
Read it as a nominal recomputation, not the exact binary64 provider readback.
The authoritative current is
`captured_source.metadata.provenance.winding_current_definition.current_a`,
which retains the provider's original grouping. For example, the C1 lens
convenience value is 15.5 A and the captured value is 15.499999999999998 A.
The field comparisons use the captured potentials, never this convenience
scalar. Original evidence bytes and their hashes are retained.

### Exact per-segment NI control: all sampled values identical

To isolate the field operator, each newly captured potential is additionally
cloned with every segment's NI explicitly set to the corresponding legacy
captured NI. The unmodified new result remains separately saved. This
controlled operator check has **39 passing cases / 29,925 point-case queries**,
with zero differences in B for both sampling APIs; all per-segment coefficients
and geometry metadata are identical. This establishes consistency of the
homogenized field at exactly the same source distribution and NI.

This coefficient alignment is a diagnostic only. It is not a production code
change, an automatic retuning step, or proof that the unmodified input adapters
always produce identical coefficients.

### Unmodified input adapters: strict comparison has roundoff failures

The existing criterion remains **rtol 4e-14 / absolute tolerance 1e-20 T**,
applied componentwise. No threshold was changed to make the results pass.

| Lens | Captured legacy NI | Captured new NI (N=100, I100=30 A) | Max absolute B difference (T) | Strict component comparison |
| --- | ---: | ---: | ---: | --- |
| C1 lens | 1550.0 | 1549.9999999999998 | 5.204170427930421e-18 | Fails at near-zero symmetry components |
| C3 lens | 470.6471043546408 | 470.6471043546407 | 1.3010426069826053e-18 | Fails at near-zero symmetry components |
| Diffraction lens | 437.3285159150572 | 437.3285159150571 | 1.3010426069826053e-18 | Fails at near-zero symmetry components |
| Other 10 lenses | Identical | Identical | 0 | Passes, bitwise identical |

All three decompositions have the same three failing owners: **9 of 39 raw
adapter cases fail**, with 30 passing. The largest per-segment NI discrepancy
is 2.2737367544323206e-13 A turn. The largest sampled vector relative B
difference is 6.506178975192727e-16. These small vector errors do not override
the recorded strict componentwise failures.

The cause is binary64 arithmetic grouping: the legacy path forms
`(NI100 * percent) * 0.01 * polarity`, while the explicit path forms
`I100 * (percent * 0.01 * polarity)` before multiplying by N. Subsequent
interval-length allocation can also expose a last-bit difference. Near a
symmetry plane, tiny transverse terms arise from cancellation; a last-bit NI
change perturbs the cancellation residue enough to exceed 1e-20 T even though
the complete field vector differs at about machine precision. Source shape,
quadrature and the Biot-Savart kernel are unchanged.

This audit reports the adapter finding without editing its arithmetic or
default inputs. A future compatibility fix should preserve the reviewed
legacy numerical normalization and this strict criterion, then rerun the
all-installed-lens checks. The existing 30 targeted component regressions
passed but did not include this complete set of default percentages and
symmetry points; their green status does not erase the new finding.

## Preservation, evidence and reproduction

The detached baseline state was identical before and after comparison.
SHA-256 hashes of **834 source/input files** were also identical before and
after. The audit emits no source, performs no finite-source trial or coherent
calculation, changes no particle count and performs no physical refocus.

Run from the repository root with the project Python, choosing a new output
directory to preserve earlier evidence:

```powershell
.venv/Scripts/python.exe -u docs/development/coil_inventory_20261115/audit.py --output outputs/coil-inventory-20261115/reproduction
```

Curated evidence is in [validation/manifest.json](validation/manifest.json),
including the inventory, CSV, field metrics, full A/B sample arrays, input
snapshot, protected-file hashes, and independent review receipts. The source
script is [audit.py](audit.py). The two initial strict-failure attempts remain
under `outputs/coil-inventory-20261115/` with their original scripts and logs;
they were not overwritten or relabelled as passing runs.

This checks the two homogenized ideal-source implementations. It does not
measure or qualify a real discrete winding, pole material, lead field,
thermal/electrical rating, coherent probe or microscope performance.

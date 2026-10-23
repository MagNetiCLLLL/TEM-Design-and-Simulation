# CETCOR image corrector: changes and validation in this round

Date: 2026-10-11. Current results are **CEOS optical-topology corrections, first-order position-conjugacy calibration, and CAD with consistent detail**. Passing first-order checks does not establish correction of finite hexapoles or objective spherical aberration. This document corresponds to `outputs/image_corrector_20261011/`; the initial full-library CAD package remains in its old directory, preserving historical inputs.

## Changes following comparison with CEOS sources

Based on CEOS's [original Linck 2025 slides](references/cad-library/CEOS-Linck-2025/original.pdf), [Lu 2025 CETCOR/CESCOR comparison diagram](references/cad-library/CEOS-Lu-2025/page-02.png), and [original sources and itemized evidence](image_corrector_reference.md). Millimeter dimensions were not read from schematic proportions.

- `OL pre/post` are the two halves of the current objective field spanning the specimen. Removed the redundant `image_ol_post_lens` identity and second drive; its former material allocation now serves TL11.
- TL11 and TL12 are separate round lenses, with DP12 between them. Reversed the old treatment placing “DP12 at the same Z as a 1 mm TL12”; TL12 now has pole pieces, a bulk winding pack, yoke, and housing.
- DP21 moves to the TL21 exit side, while DP22 remains on the TL22 entrance side. Their numerical fields can overlap lens fringe fields; field intervals are not separate coil housings.
- DPH1/DPH2 optical references are bound to HP1/HP2, respectively, and follow HP movement; their deflection drives remain independent. Public sources do not yet establish radial winding construction for this instrument, so no invented DPH/QPH mechanical solids are added.
- Image HP1/HP2 use a shared structural family of six pole pins, six bulk winding packs, and one yoke, at the same detail level as the existing probe HP. Original CETCOR mechanical drawings were not obtained; this is explicitly labeled an engineering inference within a structural family, not OEM CAD for a current CETCOR.

Both image-column configurations are updated, preserving total module length, centers of HP1 and subsequent parts, and installation interfaces. Tip parameters and upstream probe-corrector tuning are unchanged.

## Current local geometry

The following gives **module local Z in mm** for `C3_ImageCorrector.toml`; add 324 mm to all corresponding dual-corrector module values. Absolute instrument Z also includes the gun-exit installation offset. Dimensions are explicit project engineering allocations, not CEOS measurements.

| Element | Center | Mechanical or effective-field interval | Description |
|---|---:|---|---|
| HPol / QPol / DP11 | 1037.9 | 6 mm effective field each | Shared entrance-alignment region; mechanical hosts unconfirmed |
| TL11 | 1062.9 | 1040.4–1085.4 | 45 mm long, 180 mm OD |
| DP12 | 1099.65 | 1092.15–1107.15 | Within the 28.5 mm gap between TL11/TL12 |
| TL12 | 1136.9 | 1113.9–1159.9 | 46 mm long, 160 mm OD, 13 mm bore |
| HP1 / DPH1 | 1202.4 | HP outer envelope 1180.4–1224.4 | Same optical reference; no separate DPH material inferred |
| TL21 | 1256.4 | 1238.4–1274.4 | Existing 36 mm round lens |
| DP21 / DP22 | 1277.9 / 1281.4 | Existing finite field lengths | Explicit upstream/downstream positions; fringe-field overlap allowed |
| TL22 | 1306.4 | 1288.4–1324.4 | Original position retained |
| HP2 / DPH2 | 1346.4 | HP outer envelope 1324.4–1368.4 | Same representation rules as HP1 |
| ADL / SAD | 1386.4 / 1476.4 | Original positions retained | SAD is the specimen image plane here |

HP uses the configured 180 mm OD, 8 mm bore, 44 mm outer envelope, and 30 mm pole-pin allocation. The 152/154 mm OD from another example in the paper does not replace current dimensions. The CEOS product page's 250×396×306 mm dimensions lack interface datums and cannot be equated with the full objective-to-SAD optical path, so the entire assembly was not compressed to fit them.

## First-order calibration and actual GPU checks

Magnetic-field amplitudes, widths, Gaussian shapes, normalization conventions, polarities, and default percentages of all five lenses come from the **selected TOML**. New assemblies use that file's defaults; geometry-only refresh retains the user's live excitations. Original field forms are retained: percentages are not OEM current calibration, and `maximum_peak_field_t` for an unnormalized Gaussian should not be directly interpreted as a measured physical peak.

| Lens | New default excitation |
|---|---:|
| TL11 | 47.2226289532% |
| TL12 | 72.2461574042% |
| TL21 | 60.0001347725% |
| TL22 | 60.0001718262% |
| ADL | 48.9773611737% |

The full 4×4 matrix in the same laboratory coordinates `(x,y,theta_x,theta_y)` is used, reporting position, angle, rotation, and canonical/mechanical coordinate transforms separately. For `r₂=A r₁+B θ₁`, **position conjugacy checks the entire 2×2 block B≈0**, not a single-axis slice or a plotted trajectory crossing the axis.

`final-validation/report.json` records actual results for both configurations at 300 kV:

| Relationship | image-only absolute Z | Dual-corrector absolute Z | Maximum ‖B‖₂ across both configurations |
|---|---|---|---:|
| Specimen→I1 between TL11/TL12 | 1275.2→1549.9 mm | 1599.2→1873.9 mm | 2.33×10⁻¹³ m/rad |
| Specimen→I2 between TL21/TL22 | 1275.2→1731.4 mm | 1599.2→2055.4 mm | 4.01×10⁻¹³ m/rad |
| Specimen→SAD | 1275.2→1926.4 mm | 1599.2→2250.4 mm | 3.11×10⁻¹³ m/rad |
| HP1→HP2 | 1652.4→1796.4 mm | 1976.4→2120.4 mm | 2.34×10⁻¹¹ m/rad |

Singular values of HP position magnification differ from 1 by less than 2.8×10⁻¹⁰. These values describe solution residuals in the numerical model, not experimental instrument precision.

Each configuration uses 193 diagnostic trajectories: an axial ray and three rings of 64 rays at 1, 5, and 10 mrad. There are 16 actual CUDA receipt records; with finite HP and Cs enabled, comparison at 0.05/0.025 mm steps gives approximately 1.02×10⁻¹¹ m SAD position RMS difference. Final validation took about 29.3 s, with identical input and implementation SHA values before/after execution. Ray diagrams use trajectories retained from this propagation, without another plotting run. This population is a controlled optical diagnostic, not a full electron population from an actual Tip, and aperture-cutoff qualification was not performed.

![Actual CUDA internal ray paths](../outputs/image_corrector_20261011/final-validation/internal-ray-paths.png)

The upper figure is a first-order view; the lower retains finite HP and the current Cs operator. The lower trajectories do not all meet at one point, illustrating why first-order conjugacy is not final aberration correction.

## Conditions not yet established

- HP1→HP2 rotation in laboratory coordinates is about −175.7914°, still about 4.2086° from −180°; the C block is of order 24.99 m⁻¹. It satisfies unit position conjugacy but is **not a full −I transform**.
- `finite-pair/report.json` also records a bounded second-order response check of finite HP. In a diagnostic copy with scan drives explicitly disabled, adjusting HP2 strength and azimuth still leaves about 11.333% weighted second-order phase-space response. Its step-size difference is about 4.85×10⁻⁹ of the single-HP response and cannot be attributed to integration error. This fit includes an artificial 0.05 m angular weight and **was not written into defaults**; original HP excitations are retained, with no correction claim.
- The existing objective Cs kick is a spatial form at the center plane. For a specimen point emitting at different angles from the objective center, it cannot establish accurate compensation of the objective's angular spherical aberration. The old, physically unqualified assertion that “the image corrector cancels 80% of spherical aberration” was therefore removed, while existing probe assertions remain; no passing image-Cs result is invented in this round.
- A coma-free aperture plane is an aberration condition and cannot be named from a first-order matrix or geometric BFP label alone. A supplementary short-range search gives an image-only HP1 position-conjugate source plane at Z≈1275.717286 mm (0.517286 mm after the specimen), checked by combining existing full-path matrices with one short GPU propagation, with ‖B‖₂≈5.42×10⁻¹⁴ m/rad. The existing objective BFP label is at 1289.452626 mm and gives ‖B‖₂≈0.07727 m/rad to HP1, clearly different. Evidence is in `final-validation/conjugate-source-search.json`; only position conjugacy is established, and neither this source plane nor the BFP label can be called a validated coma-free plane.
- This round did not repeat long coherent propagation, Tip propagation, or actual desktop-interaction acceptance. Existing propagation methods were not replaced by a new material-field model because of CAD details.

## CAD, regression, and reproduction

Updated delivery: [CAD instructions](../outputs/image_corrector_20261011/cad/README.md), [native old/new comparison at the same scale](../outputs/image_corrector_20261011/mechanical-comparison/README.md). Includes 43 STEP files: 2 complete configuration modules and 41 optical units, totaling 324 constructed geometric solids. This combines two alternative configurations; they must not be superposed into one microscope.

All STEP files passed actual CadQuery/OpenCascade export and reimport, with maximum round-trip bound difference 1.22×10⁻¹¹ mm and maximum relative volume difference 2.28×10⁻⁹; these are format errors, not physical dimensional precision. Native 3D and STEP share geometry rules. External STEP edits do not automatically write back to TOML.

Targeted topology, parameter authority, coordinate, DPH binding, material-envelope, CAD, and CI-registration checks passed; the final combined topology/calibration/default-operating-point group had 16 passing cases. The full project suite was not repeated. An independent read-only review checked actual assembly and State round trips for both variants, removal of old OLpost references, and same-coordinate APIs.

Reproduce the final optical validation from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/validate_image_corrector.py --rays --output outputs/image_corrector_recheck
```

This validation requires an actual GPU, and the report retains backend receipts. Use `scripts/solve_image_corrector.py` to resolve the operating point; candidates are written only to reports and must not overwrite defaults without validation.

Restart the running application, open Configure instrument, select a configuration with Image Corrector, and Assemble to load the new default assembly. Saved operating points containing old `image_ol_post_lens` cannot be silently applied to the five-lens system and require reassembly/recalculation; old numerical evidence remains in original snapshots rather than masquerading as new-version calculations.

Original sources, PDFs, figure pages, summaries, SHA values, and provenance are kept together in the [reference library](references/cad-library/README.md). All millimeter geometry, material categories, and unconfirmed parts retain provenance labels; these results are not manufacturing drawings or OEM magnetic-field calibration.

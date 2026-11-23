# Nanoprobe Normal with the refined objective field

The downstream waist moves from **1474.703 mm to 1481.883 mm**, 117.317 mm before the specimen. At the former waist the refined-field beam has a 0.2133 µm RMS radius and is still converging at 0.02925 µm per mm of propagation. The two eigenvalues of the transverse beam-growth matrix cross zero within 0.00005 mm of each other at the new waist. The beam expands throughout the AC assembly (1380–1420 mm), reaches a maximum RMS radius of 1.072 µm at 1420.844 mm and then converges. The scan windings supply zero kick.

| Fixed-current result | Original objective mesh | 0.5 mm local refinement |
| --- | ---: | ---: |
| Last radial waist before the upper surface | 1474.703 mm | 1481.883 mm |
| Specimen D95 | 3.0893 µm | 5.1974 µm |
| Specimen RMS radius | 0.8582 µm | 1.4441 µm |
| Current inside 50 nm of the mechanical axis | 67.29 nA | 23.35 nA |
| Specimen current | 15.000 µA | 15.000 µA |
| Specimen angle containing 95% of the current | 0.01929 mrad | 0.03031 mrad |

The new specimen D95 is 68.2% larger. The specimen current centroid is at (−0.0370, −0.0431) µm. Over the final 0.4 mm, D95 falls from 5.1986 to 5.1974 µm and the RMS-radius slope is still negative at the upper surface. The condenser topology persists. The paraxial source image is at 524.675 mm with 5.07-fold demagnification, the C2–C3 gap has one radial waist at 790.994 mm, and the C3–ADL gap has none.

## Working point

The 300 kV run starts at the effective source, crosses the gun exit at 450 mm and stops at the physical specimen upper surface at 1599.1999975 mm. The C1 and C2 series currents are −30 A and +16 A. The 100 µm C2 aperture, weak C3 setting, Mini Condenser, corrector and objective drives come from the [condenser-pair working point](evidence/condenser-pair/working-point.json). The objective current is −28.22027595 A. The scan, wobble, tilt and DP12 offset commands are zero.

The nonlinear objective calculation includes the installed pole and return geometry and its ARMCO pure-iron B–H curve. Its original domain is retained, with extra axial planes between 1280 and 1440 mm at 0.5 mm spacing. The signed bore field replaces the previous objective solution in both the gun and the remainder of the column. Each magnetic mesh has its own source launch. Initial particle positions, directions, current weights, identities and emission energies match the reference. The gun exit is propagated in each field. The 17 zero-current rays used to measure source imaging are excluded from beam sizes and currents.

At the 1380 mm entrance to the scan assembly, the objective contribution to axial field changes from −138.80 to −42.86 mT. The independent 1 mm axial mesh gives −42.83 mT there, a downstream waist at 1481.8794 mm and a specimen D95 of 5.1987 µm. Relative to the 0.5 mm result, the waist moves 0.0032 mm and D95 changes by 1.27 nm. Halving the maximum column integration step from 0.05 to 0.025 mm preserves the waist and D95 to the precision reported above. The largest displacement of a matched specimen landing in that step test is 0.0034 nm or less.

## Reading the figures

The [full-column trajectory](evidence/refined-objective-normal/full-column-rays.png) follows the same 64 original particle identities in both projections. The [scan-region detail](evidence/refined-objective-normal/tl12-scan-rays.png) compares the radial RMS envelopes of all 32,768 positive-current particles. A waist is located where the derivative of the centered, current-weighted mean-square radius changes from negative to positive. Separate transverse growth measurements are retained in the [working point](evidence/refined-objective-normal/working-point.json).

The [specimen probe](evidence/refined-objective-normal/specimen-probe.png) bins incident current per unit area at the specified upper surface. The orange cross is the mechanical axis. The dashed circle encloses 95% of the arriving current around its centroid. Its diameter is computed from the individual particle coordinates before binning.

## Reproduction

Run these commands in order from the repository root in its Python environment. The tracking step is in millimeters. The second run reuses the matching 0.5 mm gun exit to isolate the effect of the downstream integration step. The third launches the same population in the 1 mm objective mesh.

```text
python docs/development/cescor_direct_probe_20261121/refined_objective_normal.py
python docs/development/cescor_direct_probe_20261121/refined_objective_normal.py --step-mm 0.025 --reuse-emission
python docs/development/cescor_direct_probe_20261121/refined_objective_normal.py --mesh-mm 1
python docs/development/cescor_direct_probe_20261121/render_refined_objective_normal.py
```

The [field construction and local replay](evidence/condenser-scan-region.json) give the field samples at each scan station. The [matched controls](evidence/refined-objective-normal/matched-controls.json) preserve the old, step-halved and independently remeshed results. The working point records the field and artifact checksums.

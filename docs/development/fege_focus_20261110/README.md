# FEG extractor and Normal Nanoprobe validation

The study uses the classical unified finite-field solver at revision `65f1e8c`. Optical focus is observed through vacuum at the Si[110] upper-surface reference, 1599.1999998 mm from the emitter. The 0.4 nm specimen thickness fixes that reference. The result describes the incident probe before specimen scattering.

## Paired extractor comparison

The historical straight 4 mm bore at z = 6–10 mm is compared with the installed 4 mm counterbore at z = 0.30–0.75 mm followed by a 0.50 mm throat through z = 1.30 mm. Both solve the current gun at 300 kV with a 4.0 kV extractor, 5.2 kV gun lens and the same accelerator divider. All original launch arrays, weights and source IDs match exactly. The other electrostatic boundary values match. The old contour is applied to the current gun so that the differences isolate the extractor contour and position together.

| Observable | Historical contour | Installed contour |
| --- | ---: | ---: |
| Axial accelerating field magnitude at the emitting plane (MV/m) | 0.653697 | 5.282432 |
| On-axis potential rise at z = 1.30 mm (kV) | 0.849418 | 3.997695 |
| Emitted / gun exit positive-current particles | 1024 / 1024 | 1024 / 1024 |
| Gun exit current (µA) | 10.000 | 10.000 |
| Gun exit radial RMS (µm) | 4.9171 | 6.9334 |
| Gun exit D95 (µm) | 16.9190 | 23.8929 |
| Gun exit RMS semi-angle (µrad) | 1.56493 | 6.25935 |
| Gun exit alpha95 (µrad) | 2.70025 | 10.77932 |
| Gun exit radial position-slope curvature (m⁻¹) | +0.31821 | +0.90278 |
| Gun exit kinetic energy mean / RMS spread (eV) | 300000.3000 / 0.1273983 | 300000.3000 / 0.1273983 |

The installed extractor accelerates the electrons earlier. The beam is initially narrower at z = 1.30 mm, where its RMS radius is 0.513 µm versus 1.206 µm. Its largest axis radius at the new throat exit is 1.243 µm against a 250 µm clear radius. In the later gun field, the changed phase space produces a larger and more divergent gun-exit beam. The largest centroid radius belongs to ID 607 in both cases. Its initial transverse momentum is the largest in the source and the next outermost arrival is close in radius.

At 2048 emitted particles the installed gun still transmits 10 µA. Exit D95 grows by 1.20% and alpha95 by 1.33%. The field data and electrostatic request are identical across the source-count refinement. The larger deterministic sample keeps the source law and changes the sampled energy and angular population. The 10 µA value is the prescribed source current for this classical planar-cathode model.

[Paired fields and beam figure](extractor/comparison-n1024.png) · [Full extractor comparison](extractor/comparison-n1024.json) · [Gun source-count refinement](extractor/source-refinement-n1024-n2048.json)

## Full-column comparison

The C2/C3 pair controls the convergence-angle branch. The hexapole changes act through their installed finite potentials. Mini Condenser and the shared Objective drive provide joint focus placement. The sample reference and the 100 µm C2 aperture diameter are fixed throughout. Alpha95 is the half-angle that contains 95% of the arriving current about its weighted chief direction. D95 is the diameter that contains 95% of the arriving current about its weighted centroid. The native D95 and RMS axial minima are measured independently.

| Candidate, step 0.20 mm | Arrivals / emitted | Current (µA) | Alpha95 (mrad) | RMS semi-angle (mrad) | Surface D95 (nm) | Surface RMS radius (nm) | Observed D95 minimum offset (nm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C2/C3 zoom with archived corrector | 473 / 1024 | 4.61914 | 32.56307 | 21.65107 | 2230.738 | 488.754 | +5800 |
| HP1/HP2 and HPC correction with joint focus drives | 473 / 1024 | 4.61914 | 31.2469 | 21.2125 | 628.112 | 143.108 | +500 |
| HPC, HP1/HP2 and HPol correction with revised zoom and joint focus drives (`trial-00`) | 459 / 1024 | 4.48242 | 29.19233 | 20.08001 | 518.027 | 115.140 | +400 |
| Historical main-field axes and auxiliary field components | 473 / 1024 | 4.61914 | 30.19816 | 21.03660 | 2.05733 | 0.51322 | −4.25 |
| Historical main-field axes with native D95 focus correction | 473 / 1024 | 4.61914 | 30.19815 | 21.03660 | 1.86009 | 0.51503 | 0.00 |

At these checkpoints the effective sample count equals the surviving count. The full-column losses are at C2 and the original-current balance closes. For the three-drive correction the RMS minimum is 580 nm upstream, while the D95 minimum is 500 nm downstream. The peripheral population drives this separation and the broad D95. In the initial zoom case several dozen surviving particles lie outside a 1 µm centroid radius. Their original IDs cluster near the C2 opening edge, and the axial checkpoints show continuous local position and direction.

The three checked zoom cases each show a finite 2D region between C2 and C3, no C3-to-ADL crossover, one region in TL22–TL21 and one between the AC coils. Each also shows a broad, shallow finite 2D waist near 1300 mm in TL21–TL12, absent in the archived 7.85 mrad comparison. The resolved final Mini-to-specimen region is the only region in that interval. The near-surface RMS waist and the native D95 minimum are separate measurements of this region.

The HPol row above and the HPol curve in the comparison figure represent separate executions. The figure uses the `trial-01` values identified in [its source data](focus-comparison.json).

## Historical main-hexapole field axes

The two-axis HPC and HPol drives were re-expressed from their historical laboratory fields in the installed normal and skew bases. Under those same auxiliary fields, restoring the archived HP1 and HP2 magnetic orientations reduces the nonlinear 5.3 µm diagnostic ring residual from 462.87 to 0.208 nm. The 49 zero-current pencils identify the response direction. The original particles provide the full-current measurements in the table. The production geometry comparison verifies equality of all 104 opaque pieces, aperture selections and domain boundaries across the optical change.

With C2 at 10.561%, C3 at 33.60%, Mini Condenser at 26.678337967960122% and Objective at 29.3264070792%, the 1024-particle, 0.20 mm calculation finds the smallest observed D95 at the exact specimen entrance. The measured neighboring planes bracket it from −0.250 to +0.200 nm. The RMS radius is smallest at +2.50 nm. The absolute current in the fixed-axis 50 pm disk is 0.48828 µA at the entrance. The final Mini-to-specimen interval contains one finite 2D waist region.

[Main-field diagnostic](historical-corrector-zoom-response.json) · [Mechanical comparison](focus/historical-axis-n1024-s020/mechanical-comparison.json) · [1024-particle surface report](focus/historical-axis-n1024-s020/trial-01/report.json)

## Source-count and integration refinement

With the same physical settings and 2048 original particles at a 0.10 mm step, 937 reach the entrance with 4.575195 µA. The surface D95 is 1.882562 nm and alpha95 is 30.065526 mrad. The sampled native minimum moves upstream to −3.50 nm in the [−3.75, −3.25] nm bracket. The nominal surface current in the 50 pm disk is 0.463867 µA. The movement exceeds the engineering placement budget, which is more sensitive to sampling than the surface diameter.

The first 1024 source IDs of the larger population have the same individual arrival decisions as the earlier run. As a diagnostic reanalysis of saved data, this subgroup has an observed D95 minimum at −0.25 nm. The added source IDs alone give −3.50 nm. The larger source draw also changes the launch directions and energies of the matching IDs slightly under the same physical source law. The subgroup is an attribution check and carries only its original subset current.

[2048-particle fixed-control report](focus/historical-axis-n2048-s010/trial-00/report.json)

Reducing the shared Objective drive to 29.3264011126% places the native D95 minimum at the exact upper-surface checkpoint on both 2048-particle integration meshes. The two executions have the same original IDs, weights, source and observation coordinates, all aperture loss labels and all per-plane survival masks. The step is the sole numerical input difference.

| Maximum step (mm) | Surface D95 (nm) | Surface RMS radius (nm) | Alpha95 (mrad) | RMS semi-angle (mrad) | Axial 50 pm disk current (µA) | D95 minimum bracket (nm from surface) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.10 | 1.8380582 | 0.5193098 | 30.0655229 | 20.8610185 | 0.5029297 | [−0.250, +0.200] |
| 0.05 | 1.8385570 | 0.5192683 | 30.0655230 | 20.8610187 | 0.5078125 | [−0.250, +0.200] |

Both retain 937 effective positive-current particles and 4.575195 µA at the specimen. C2 absorbs the other 1111 particles and 5.424805 µA. D95 changes by +0.0271% when halving the step. The RMS waist is +6.50 nm on both meshes. At the smaller step, the next local D95 minima at +0.50 and +1.50 nm are 0.212% and 0.911% above the observed global minimum. The focus claim is bounded by the sampled axial and source populations.

Reanalysis of the 69 saved phase-space planes in each trial finds the same fixed-axis 50 pm disk-current maximum of 0.5175781 µA at −1.00 and −0.25 nm. The specimen entrance attains the observed D95 minimum on both meshes. Its disk current is 2.83% below the maximum at 0.10 mm and 1.89% below at 0.05 mm. Both deficits exceed the 1% allowance in the [joint calibration criteria](../../OPTICAL_CONTROL_ROLES.md#joint-central-current-and-d95-calibration). These observations establish the D95 placement within the reported sampling bracket. Joint core-current and D95 placement remains unqualified.

### Observed Normal branch at 0.05 mm maximum step

| Region | Representative z (mm from emitter) | RMS radius | RMS semi-angle (mrad) | Alpha95 (mrad) |
| --- | ---: | ---: | ---: | ---: |
| C2 to C3 | 827.500 | 41.26 nm | 0.53545 | 0.77068 |
| TL22 to TL21 | 1211.633 | 14.425 µm | 2.05311 | 2.99106 |
| TL21 to TL12 shallow waist | 1298.887 | 77.861 µm | 0.26393 | 0.46245 |
| Between AC coils | 1408.750 | 336.97 nm | 1.85453 | 2.67563 |
| Final 2D RMS waist | 1599.2000063 | 0.50142 nm | 20.86102 | 30.06552 |

The C3-to-ADL interval is free of a resolved crossover. The Mini-to-specimen interval contains the one near-surface waist. The broad TL21–TL12 waist remains a topology difference against the lower-angle reference. The study is a stationary incident-probe validation and leaves driven scan performance unverified.

[Silicon operating profile](normal-nanoprobe-si110.toml) · [Field controls](normal-nanoprobe-si110.controls.json) · [Step comparison](focus/historical-axis-same-source-step-comparison.json) · [Fine-step particle report](focus/historical-axis-n2048-s005/trial-00/report.json) · [Probe comparison figure](focus-comparison.png)

## Loading the candidate and reading the evidence

Launch the application with the repository root as its working directory, then load `normal-nanoprobe-si110.toml` as an operating profile. Its CIF path is relative to the repository root. It restores the installed optics and Si[110] specimen. The through-focus trial propagates the incident beam in vacuum around the same upper-surface coordinate.

The two refined trials retain their complete executed scalar through-focus profiles in their reports and the original particle phase rows within 10 nm of the surface in `surface-phase.npz`. The three gun arms retain their original emitted and gun-exit phase arrays and on-axis field samples. Full trajectory histories and executable restart snapshots remain in the original local run directory. Path labels in some archived reports are relative provenance labels. The artifact manifest records the original file hashes and the archived bytes or canonical LF text hashes.

# STEM Nanoprobe Normal: Descan and Lower Objective

Date: 2026-10-18. Scope: **C3 + Probe Corrector, STEM Nanoprobe Normal, 300 kV, Diffraction projection, 1 µm square scan field**. All checks used background code.

## Adopted changes

1. The complete Image/Diffraction two-stage deflector in `C3_ProbeCorrector.toml` moves upward by 100 mm. Descan remains a dynamic channel of the same hardware, with no second coil set. Positions in other column variants are unchanged.
2. New **Position and angle** calibration jointly solves upper/lower X/Y drives to cancel scan displacement and mechanical incidence-angle increments at the selected plane. Historical **Position at selected plane** remains available.
3. Actual Descan commands, upper/lower coupling and accepted constraints are saved. Old held records still restore their original `-AC` commands; selecting a new constraint does not automatically rewrite an old calibration. Increasing FOV checks static Image/Diffraction alignment plus scan peaks and rolls back if limits are exceeded.
4. Downstream section readouts show calculated scan displacement and mechanical-angle changes. These are scan-induced changes; actual particle records retain absolute centroids and true tilts.

| Item | Original position / value | Updated |
|---|---:|---:|
| Upper-coil centre, global Z / mm | 1774.4 | 1674.4 |
| Lower-coil centre, global Z / mm | 1799.4 | 1699.4 |
| Two-stage body extent, global Z / mm | 1766.9–1806.9 | 1666.9–1706.9 |
| Coil length / centre spacing / mm | 15 / 25 | 15 / 25 |
| Selected-area aperture Z / mm | 1824.4 | 1824.4 |
| Main fluorescent-screen Z / mm | 2714.15 | 2714.15 |
| Peak drive for joint calibration at original position / mrad | 0.559525 | New position 0.242460 |

Drives here are integrated nominal deflection angles in the model, not directly coil currents. Positions are engineering modelling choices, not manufacturer measurements.

## Why matching one position is insufficient

The original scheme fixed the upper Descan command to reversed AC and solved only lower-stage coupling to align centroids at the selected-area aperture, without also constraining incidence angle. Actual finite-beam validation kept scan displacement there near 0.39 nm but left approximately 782 µrad angle change, producing approximately 0.892 mm main-screen motion.

The new local-response solve is `[U L] [Ku; Kl] = -P_ac`, whose four rows are X/Y positions and X/Y mechanical slopes. `U` and `L` come from propagation derivatives through actual finite-length coils and intervening lens fields. Position rows are scaled by coil spacing, and the system is solved after checking its condition number, without using a pseudoinverse to hide uncontrollable axes. Accepted AC drives, lens excitation and physical Image/Diffraction static alignment are retained.

This agrees with JEOL's [de-scan](https://www.jeol.com/words/emterms/20160713.103817.php) description of a two-stage imaging-side deflector: detector position and angle require separate control. Specific installation coordinates and drives come from this project's propagation and assembly calculations.

## Finite-beam results

3000 physical Tip samples were rerun from the current configuration. All schemes reused the same executed upstream particle states, weights, energies and IDs. The scan centre and four corners, Descan on/off and 0.1 / 0.05 mm steps gave 60 GPU propagations plus three CPU replays. The table gives maximum 2D vector changes at the corners relative to each scheme's scan centre, using 0.05 mm data.

| Observable | Original position, position only | Original position, position and angle | New position, position and angle |
|---|---:|---:|---:|
| Selected-area aperture displacement / nm | 0.390513 | 0.390513 | 0.390432 |
| Selected-area aperture mechanical-angle change / µrad | 782.081 | 0.00160476 | 0.00160413 |
| Main-screen displacement / nm | 892006.671 | 0.391971 | 0.391977 |
| Main-screen mechanical-angle change / µrad | 2122.018 | 0.00126705 | 0.00126713 |

![Finite electron-beam comparison](assets/descan-normal-2026-10-18.png)

**The main improvement in screen stability comes from the joint drive solve.** Moving the deflector reduces peak drive from 0.559525 to 0.242460 mrad, approximately 56.7%. The tiny residual difference between positions is not presented as improved centroid accuracy. The moved system's scaled condition number is approximately 62.98, higher than the original 6.85, but passes perturbation and step validation. The higher −102 mm candidate lowers drive only another approximately 2.7%, with less assembly margin, so −100 mm was adopted.

All 60 propagations retain the same 2662 survivors and source-current fraction 0.887333; Descan adjustment adds no clipping. Centroids are not forced to zero, tilt is not removed and current is not normalized. For example, the new scheme's actual centre-beam centroid is approximately `(−0.823789, +0.984893) µm` at the selected-area aperture and `(−2.009288, −1.886446) µm` at the main screen; these offsets remain in the data.

Central-beam RMS radii at the selected-area aperture/screen are approximately 417.447 / 895.893 µm. Small scan-displacement residuals mean the pattern barely moves during scanning, not that these beam spots shrink to subnanometre size.

## Common tilt centres and crossovers

- Current AC-only scan centroids have two shared intersections downstream of the specimen: global **Z≈1605.1935, 1625.7351 mm**, with nonzero scan slopes. These are intersections of different scan-centroid trajectories.
- The same central beam's current-weighted RMS envelope has a post-specimen local waist at **Z≈1612.48 mm, bracketed by ±0.01 mm**. Exact integration observations at that point and ±0.005 / ±0.01 mm, with step refinement from 0.05 to 0.025 mm, give a central RMS radius of approximately 0.22708 nm. Centred `x·tx + y·ty` covariance is negative before and positive after the waist for the same particle population, supporting actual local convergence followed by divergence of the finite spot.
- This waist is distinct from the shared intersections and from the geometric centre of Lower Objective.
- After joint Descan compensation, downstream first-order scan-centroid trajectories coincide. Every zero-response plane therefore cannot be labelled a separate tilt centre, nor can every downstream plane be inferred to be a crossover.

Strong-excitation C3 values in the original study do not directly apply to Normal here: this run uses C3 at 9.2079848% and Objective at 68.9801%. Lower Objective's lower pole shoe spans 1601.9–1663.9 mm and is centred at 1632.9 mm; the lower yoke centre at 1729.4 mm and lower field reference at 1600.55 mm are distinct definitions.

## Assembly and numerical validation

- The new body's upper end is 3 mm from the lower pole-shoe exit, and its lower end is 25 mm from the objective-stigmator allocation entrance.
- The more complete 3D reconstruction in the original research workspace contains 20 solids. All 300 exact geometry pairs against 15 neighbours showed no volume intersection; minimum material clearance was 3.38136 mm. STEP rereading yielded 20 valid solids, and all 40 symmetric-difference volumes between that research configuration, the candidate and STEP were zero. Unmodelled tubing and manufacturer tolerances were excluded from the validated scope.
- This scoped Descan commit includes only the shared-deflector position change, not that auxiliary-support reconstruction; the committed version still uses the original 8 copper-coil solids. The 20-solid clearance/STEP results are historical evidence for the original research model, not CAD validation of this scoped commit.
- At the new position, maximum absolute centroid changes for 0.1→0.05 mm were 0.004762 nm at the selected-area aperture and 0.002714 nm at the screen, distinct from scan-centre difference residuals.
- New-position CPU/GPU differences were at most 7.29×10⁻¹⁰ nm in centroid and 2.75×10⁻¹² µrad in mechanical slope, with identical survivors and current.
- Rerunning the production field plan and actual aperture clipping in the updated installation reproduced the diagnostic candidate at centre and one corner, with centroid differences at most 1.35×10⁻⁹ nm. This step did not substitute a candidate field provider.

These are numerical results for the current simulated field model and bounded operating point. No real-instrument calibration, specimen scattering, detector pixel intensities or optical acceptance for other operating modes was performed. SAA remains classified as mixed here; calibration does not rename it a conjugate image plane.

## Use and reproduction

Select **Position and angle** in scan calibration. With held AC already present, use **Calibrate Descan (keep AC)**; to resolve AC as well, use **Calibrate and hold**. Held drives do not automatically remove physical changes after lens edits; recalibration must be explicit.

Alternatively load `configs/design_candidates/nanoprobe_normal_descan_position_angle_20261018.toml`. Its `.temwp` is a complete input package importable into Working Points, containing updated positions and held drives, not purported saved particle results. Publication packages were regenerated from a clean checkout of this scoped commit and are distinct from old packages from the more complete original research workspace. Restoration validates archived-input integrity and retains original implementation identity; compatible current implementations may read the inputs, but all physical results require recalculation without reusing historical calculations.

Reproduce the historical-position first-order study consistently:

```powershell
.venv/Scripts/python.exe scripts/study_descan_phase_space.py --output outputs/descan_repeat --baseline-pair-center-mm 1786.9 --shifts 0 -102 -100 -90 -80 -76 --pivot-audit
.venv/Scripts/python.exe scripts/validate_descan_transport.py --output outputs/descan_repeat_particles --study outputs/descan_repeat/study.json --shifts -100 0 --backend gpu --step-mm 0.1 --step-refinement --compare-backend cpu --no-chief-diagnostics
.venv/Scripts/python.exe scripts/validate_descan_transport.py --output outputs/descan_repeat_installed --replay-prefix-source outputs/descan_repeat_particles --backend gpu
```

Finite-particle comparisons, installation replay and exact waist bracketing use `scripts/validate_descan_transport.py`.
Historical `legacy0` is explicitly built from original positions/drives in the study file and does not track new default installation positions.
The main batch did not reuse a source from an earlier date. Installation review reused that batch's prefix only after checks of source settings, lenses, AC, fixed observation stations and particle IDs.
Primary raw evidence is retained in:

- `outputs/descan_optimization_20261018/all_candidates.json`: position, step and perturbation studies.
- `outputs/descan_transport_physical_20261018/report.json`: actual particle records, absolute centroids/slopes/current, step and matching-ID comparisons.
- `outputs/descan_transport_physical_20261018/installed/installed-replay.json`: actual installation replay and exact waist bracketing.
- `outputs/descan_optimization_20261018/mechanical-candidate-minus100.json`: solid clearances.
- `outputs/descan_optimization_20261018/production-minus100-identity-validation.json`: consistency of the original research workspace's 20-solid configuration with candidate/STEP, not the CAD structure of this scoped commit.
- `outputs/descan_optimization_20261018/image-diffraction-minus100.step`: updated shared-deflector 3D model.
- `outputs/descan_optimization_20261018/final-geometry-scan-regression.xml`: 225 final position/scan regressions.
- `outputs/descan_optimization_20261018/final-held-cache-regression.xml`: 190 final regressions for held calibration, caches, shared hardware, configuration and dimensions. The disjoint groups total **415 passed**.

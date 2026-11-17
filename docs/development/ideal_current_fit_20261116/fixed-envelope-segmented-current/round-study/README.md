# Fixed-envelope segmented prescribed-current round-lens study

Status: field-only feasibility and candidate-current evidence. No emitter, finite-source particle transport, coherent propagation, resistance or constant-power calculation was run. No production source or configuration file was edited by this study.

## Geometry and source constraints

The reference is the exact archived ideal-coil geometry and controls in mixed-run-v3. The old resized/relocated inverse-design candidates are not used. All 13 enabled round owners retain the original winding inner radius, outer radius, axial interval, pose, physical turn total, and conductor-metal-area records. Objective retains 580 turns in each of its two unchanged axial intervals.

Each original rectangular radial-axial current cross-section is tiled by internal cells. Every cell has a positive integer physical-turn allocation. Radial strip widths are proportional to strip turns; axial lengths within each strip are proportional to cell turns. Thus each cell area is exactly its turn fraction times the parent area, to floating-point reconstruction precision. Equal current gives exactly the original continuous uniform azimuthal current density.

This is electrical segmentation of the declared homogenized ideal-current cross-section. It is not proof that each arbitrary internal cell boundary aligns with the existing individual-wire CAD layout. The original CAD and wire diameter are not changed. Actual turn-addressed wiring would require grouping the existing wire coordinates separately.

A production uniform-current fast path can merge equal-current cells into the original parent field source. This avoids numerical differences caused solely by repartitioning the original q12 quadrature. Those differences are explicitly recorded for the unmerged experimental basis.

## Method

Axial-only splits use 1, 2, 4, 8, 16 and 32 requested cells per original interval; additional radial-axial nested tilings are explored, with no more cells than allocated physical turns. Signed currents are found by weighted SVD on the full real axis. Finite central panels and rational infinite-tail transformations retain all tails. Fits are repeated for SVD cutoffs 1e-6 through 1e-14. These cutoffs are experiments in conditioning, not changes to project tolerances.

Selected per-owner solutions are independently evaluated on a finer whole-axis quadrature and original q12 versus refined q24 source quadrature. Native AnnularCoilPotential constructors and axis evaluations are checked. MP80 spot checks show the error introduced by strong signed-current cancellation. A smaller least-squares residual alone is not admitted as a focused or numerically qualified working point.

## Per-owner results at SVD cutoff 1e-10

| Owner | Original physical turns | Cells | Full-axis q24 relative L2 error | Maximum absolute current (A) | q12 to q24 relative field difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| condenser_lens_1 | 300 | 32 | 9.695729% | 6.1602518e+10 | 4.997e-08 |
| condenser_lens_2 | 540 | 32 | 0.758001% | 4.8680666e+08 | 2.946e-09 |
| condenser_lens_3 | 510 | 32 | 2.719284% | 1.6394966e+09 | 3.668e-08 |
| adapter_lens | 23 | 23 | 7.548830% | 9.7752841e+09 | 6.723e-08 |
| probe_tl22_lens | 19 | 19 | 16.749529% | 4.1020125e+11 | 1.072e-07 |
| probe_tl21_lens | 19 | 19 | 16.735117% | 3.820185e+11 | 1.025e-07 |
| probe_tl12_lens | 19 | 19 | 10.970188% | 2.354626e+11 | 8.417e-08 |
| mini_condenser | 17 | 17 | 0.040260% | 66287692 | 3.192e-11 |
| objective_lens | 1160 | 128 | 0.149652% | 1.6163699e+09 | 5.613e-10 |
| diffraction_lens | 360 | 64 | 0.086776% | 1.7731164e+08 | 5.749e-10 |
| intermediate_lens | 648 | 32 | 0.873765% | 3039.5288 | 1.469e-11 |
| projector_lens_1 | 432 | 64 | 0.001434% | 5566201.4 | 1.153e-11 |
| projector_lens_2 | 648 | 64 | 0.112626% | 27283815 | 2.791e-10 |

## Joint fixed-working-point fit

The joint study uses 545 internal cells chosen from the per-owner designs and permits all signed cell currents to fit the complete Gaussian Bz sum. It is a fixed-point field fit, not a calibration preserving each logical lens control response. Independent C2 lens/C3 lens changes must be re-evaluated rather than inheriting the old Gaussian gain.

| SVD cutoff | Retained rank | Full-axis q24 L2 error | Maximum absolute current (A) | q12 to q24 relative field difference |
| --- | ---: | ---: | ---: | ---: |
| 1e-06 | 200 | 14.067686% | 14363270 | 3.033e-12 |
| 1e-08 | 267 | 8.259107% | 1.1460415e+09 | 2.352e-10 |
| 1e-10 | 320 | 4.602676% | 9.2230593e+10 | 1.924e-08 |
| 1e-12 | 373 | 2.504482% | 8.5852328e+12 | 9.461e-07 |
| 1e-14 | 417 | 1.200748% | 8.1209164e+14 | 4.018e-05 |

Larger retained condition numbers yield smaller fitted Gaussian residuals but much larger currents and cancellation sensitivity. The 1e-14 joint solution needs about 8.12e14 A in its largest ideal segment. Ideal zero resistance does not make floating-point cancellation disappear. No thermal or hardware limit is imposed here; the numerical losses are still reported.

## Independent high-precision checks

MP80 evaluates the same finite cells using exact serialized binary64 geometry/current inputs and high-precision 12-point Gauss-Legendre nodes and weights. This isolates errors that can be hidden by repeated binary64 calculations.

| Owner / cutoff | Maximum native binary64 versus MP80 axis B error (T) |
| --- | ---: |
| condenser_lens_1 / 1e-10 | 2.7887155e-09 |
| probe_tl22_lens / 1e-10 | 2.3839216e-09 |
| probe_tl22_lens / 1e-14 | 1.743002e-05 |
| objective_lens / 1e-10 | 2.3784863e-11 |

## Limits and next use

The fixed envelopes can provide substantially better axial fields by internal opposite-current regions, but the explored finite-dimensional fits do not reconstruct every old narrow Gaussian equally well. The relay lenses are particularly ill-conditioned. Aggressive truncation choices must not be installed merely because the residual is smaller.

No focus, crossover topology, current transmission, D95, convergence angle, source-count convergence or scanning image has been qualified. Field fitting and actual finite-source focus are separate checks. The candidate-library JSON preserves every signed current and physical cell; the joint library additionally preserves its fixed-working-point control limitation.

Files: results.json and summary.csv (per-owner sweep); candidate-library.json (physical cell and current definitions); joint-results.json and joint-candidate-library.json; precision-audit.json; per-owner-field-comparison.png; joint-field-comparison.png; profiles.npz and joint-profiles.npz. The exact first fit script is retained as executed-fit-v1.py; fit.py only corrects its explicit project import root for portability.

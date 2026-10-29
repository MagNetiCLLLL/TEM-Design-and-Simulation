# Shape evolution in saved seed769 h02 trajectories

This diagnostic uses 227 actual gun checkpoints and 433 actual column checkpoints from `seed769-native-d95-waist-h02/particles.npz`. The latter include the column at mostly 5 mm intervals and the specimen window at 1 nm intervals. No positions are interpolated, and no new propagation is performed. The JSON records every plane, original-file/code hashes and exact versus bracketed landmark coverage.

All 769 original particles survive at every saved plane; current remains 10 uA. Thus clipping or particle reweighting does not explain the observed shape change in this dataset.

## What changes, beyond one particle being the largest

The native ratio D95/D50 changes from 2.02412 at emission to 2.24887 at the gun exit. It remains about 2.2504 through much of the condenser path, rises to 5.28351 at the actual corrector checkpoint z=1210 mm, and returns to 2.25042 at z=1500 mm. At the exact specimen surface it becomes 4.41819. A pure uniform magnification would preserve this ratio. These changes directly demonstrate a redistribution between the core and outer-current regions.

At the surface, D50 is 0.581139 nm and D95 is 2.567580 nm. ID306 carries only 1/769 (0.13004%) of the current, but contributes 61.0185% of the full radial second moment. At the gun exit its second-moment contribution was only 0.8840%. This explains why RMS is especially sensitive to the rare outer component, while the changing D95/D50 ratio separately shows a change in the broader distribution.

| Region | Actual z (mm) | D50 (nm) | D95 (nm) | RMS radius (nm) | D95/D50 | ID306 second-moment share |
|---|---:|---:|---:|---:|---:|---:|
| emission | 0.0000000000 | 4.95503 | 10.0296 | 2.91824 | 2.02412 | 0.0195% |
| column | 450.0000000000 | 491.563 | 1105.46 | 307.179 | 2.24887 | 0.8840% |
| column | 650.0000000000 | 11778.4 | 26506 | 7361.73 | 2.25040 | 0.8847% |
| column | 765.0000000000 | 18447.5 | 41514.1 | 11530.1 | 2.25039 | 0.8847% |
| column | 1090.0000000000 | 100063 | 225181 | 62541.5 | 2.25039 | 0.8847% |
| column | 1145.0000000000 | 112739 | 253357 | 70470.6 | 2.24729 | 0.8713% |
| column | 1165.0000000000 | 115287 | 263201 | 72572.4 | 2.28300 | 0.6880% |
| column | 1210.0000000000 | 9233.96 | 48787.7 | 10930.8 | 5.28351 | 1.9483% |
| column | 1250.0000000000 | 97399.2 | 231795 | 62373.9 | 2.37985 | 1.6778% |
| column | 1300.0000000000 | 99755.3 | 224724 | 62193.9 | 2.25275 | 0.9586% |
| column | 1500.0000000000 | 216677 | 487614 | 135579 | 2.25042 | 0.9294% |
| column | 1595.0000000000 | 106779 | 241148 | 66953.1 | 2.25838 | 0.9410% |
| column | 1599.1999975000 | 0.581139 | 2.56758 | 1.32441 | 4.41819 | 61.0185% |

Nominal lens centers that are absent from the archive are not assigned invented observations. The table contains actual saved planes; e.g. 1145, 1165 and 1210 mm are checkpoint coordinates, not assertions of exact element centers or qualified waist coordinates.

## Why the two minima differ in this actual data

Within the saved +/-100 nm window, the smallest observed D95 is 2.189194937 nm at +16.000000 nm from the surface. At that plane the RMS radius is 1.390518361 nm. The smallest observed RMS radius is 1.308177989 nm at -13.000000 nm, where D95 is 2.868755858 nm. These are actual samples, not fitted parabolic roots.

The inner core, 95%-current contour and rare outer component do not shrink in lockstep as z changes. ID306's smallest observed centered radius lies at the upstream window boundary (-100 nm), so this window does not resolve that individual particle's radius minimum. It is therefore incorrect to describe the whole profile as one Gaussian whose width merely changes with z. D95 remains the intended waist observable; RMS is only an auxiliary explanation of the differing minimum.

## Attribution limit

The saved data localize shape changes in the gun, the corrector path and near the final focus. They do not prove whether each change is the correct nonlinear/chromatic field response, residual numerical integration error, or both. In particular, these statistics do not identify a spherical-aberration coefficient or establish manually added aberrations. Fixed-control, fixed-source finer-step runs are required before causal attribution. The separate source audit also records that count refinement changes the legacy energy nodes.

## Reproduce

Run `.venv/Scripts/python.exe outputs/joint-probe-100um-30mrad-20261028/diagnostics/seed769_shape_evolution_h02.py` from the repository root after selecting new output names. Existing diagnostic files are refused. The source archives remain read-only. Output qualification: `NOT_QUALIFIED`.

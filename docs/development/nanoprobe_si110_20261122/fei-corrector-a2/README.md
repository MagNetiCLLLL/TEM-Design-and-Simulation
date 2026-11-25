# Nanoprobe Normal HPol A2 adjustment

The two auxiliary HPol currents were tuned against signed geometric threefold astigmatism. All other winding currents, source conditions, physical stops and the specimen coordinate match the preceding corrector-relay setting. Each trace starts from a 15 µA tip at +50 V gun-lens bias with a 100 µm C2 foil and zero scan drive. The objective carries -14.517475 A.

| Quantity | Relay baseline | A2 adjustment |
| --- | ---: | ---: |
| HPol X current (A) | -10.729281 | -10.180505 |
| HPol Y current (A) | -12.922527 | -12.884576 |
| Fitted A2 cosine / sine (mm) | +25.3352 / -31.4306 | -0.1964 / -0.0828 |
| Fitted A2 magnitude (mm) | 40.3703 | 0.2132 |
| Specimen D95 (nm) | 373.92 | 676.00 |
| Specimen radial RMS radius (nm) | 118.20 | 167.39 |
| Current within 50 nm of the centroid (nA) | 1.06648 | 0.32820 |
| Integrated specimen current (nA) | 6.144104 | 6.144104 |
| D95 waist offset from upper surface (µm) | +8.50 | +33.50 |

## Focal planes

The specimen upper surface is at 1599.1999975 mm from the emitting tip. Positive offsets point downstream. The search follows the same transmitted IDs in the local magnetic and electric fields through the final objective, with a vacuum continuation downstream of the fixed physical surface. The axial samples run from 1534.5 to 1600.8 mm. A 2.5 µm coarse search around the specimen is followed by a 0.25 µm sampling interval around the minima. The hardware is fixed throughout.

| Plane | z (mm) | Offset (µm) | D95 (nm) | Radial RMS radius (nm) |
| --- | ---: | ---: | ---: | ---: |
| Specimen upper surface | 1599.20000 | 0 | 676.0 | 167.4 |
| Minimum D95 | 1599.23350 | +33.50 | 541.9 | 175.9 |
| Minimum radial RMS | 1599.19900 | -1.00 | 680.0 | 167.4 |
| Minimum x RMS | 1599.20575 | +5.75 | 653.0 | 167.7 |
| Minimum y RMS | 1599.19325 | -6.75 | 703.0 | 167.6 |

The four current maps share fixed laboratory axes, 25 nm square bins and a common linear pA scale. The dashed circle encloses 95% of the transmitted current about its current-weighted centroid. Pixel intensities are sums of the original physical weights. [Probe planes](probe-waists.png) · [Axial widths](axial-focus.png) · [Tip-to-specimen trajectories](column-rays.png)

## Measurement and sampling

Each side uses the same 50,000 distinct positive-weight original tip states. The first 10,000 IDs retain their original phase coordinates. Adding 40,000 particles increases each of the eight interleaved strata from 1,250 to 6,250 samples. The per-ID quadrature weight in each stratum is consequently one fifth of its historical value. The baseline reuses the archived prefix, gun events, exit momenta, terminal states and stored paths. Every appended ID is transported until intercepted or received. An independent 50,000-primary baseline replay reproduced all specimen positions and slopes, loss codes, gun-exit states, weights and the 919 full paths common to the two archives. The A2-adjusted case is traced anew from the tip in its own complete material solution. Both cases deliver the same 3,232 primaries at the gun exit and the same 917 IDs at the specimen. The effective specimen count is 140.4.

The fit uses mechanical angles about the current-weighted chief ray, the fixed laboratory specimen plane and the project convention for the negative wave-aberration gradient. A2 and D4 have distinct radial dependence and are fitted simultaneously with translation and the other Cartesian aberration terms through fifth geometric order. A2 values are effective ray-fitted coefficients of this simulated finite source. The baseline A2 azimuth is 102.96° modulo 120°. The residual is smaller than its variation across the ten delete-one-block angular fits. Its azimuth has no resolved physical orientation. No extrapolation to 30 mrad is used. [Müller et al., Microscopy and Microanalysis 12 (2006)](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf)

Restricting the fit to the cone carrying 95% of the current gives A2 41.52 → 1.56 mm. The angular cutoffs are 2.2082 and 2.2088 mrad. The table uses the full transmitted cone. After the final HPol change one original ray switched its first recorded interception from the fourth accelerator electrode to the accelerator bore. This ray is intercepted in both fields. All gun-exit and specimen-arrival IDs match.

The four signed ±0.1 A trials use the local HPol derivatives from the converged nonlinear circuit; both quadratures and response columns are recorded. Each accepted current vector is solved on the full 6,841,872-tetrahedron circuit with 94 magnetic regions. The final replay verifies all 10 gun and column field records against the current vector and solution hash. The production bore uses degree-five transverse polynomials in eight overlapping radial domains and cubic axial interpolation. The starting model averages the field over the three nominal symmetry sectors. An independent near-axis fit over both hexapoles and HPol gives residual A2 0.1716 / 0.1716 mm at 100 / 50 µm radial sampling. Both use the same nonlinear material mesh. Mesh refinement of the material bodies and real-instrument calibration remain outside this measurement.

## Reproduction

The following commands replay the actual installed working point, find the two waists and fit the signed A2/D4 components from the weighted specimen landings.

```powershell
.\.venv\Scripts\python.exe -m temsim.joint_fem outputs/hpol-replay
.\.venv\Scripts\python.exe docs/development/nanoprobe_si110_20261122/fei-corrector-a2/focus.py outputs/hpol-replay
.\.venv\Scripts\python.exe docs/development/nanoprobe_si110_20261122/fei-corrector-a2/metrics.py outputs/hpol-replay
```

[Numeric evidence](evidence.json) · [Landing positions, slopes, source IDs, currents and axial profiles](probe-planes.npz) · [Redraw the figures](render.py) · [Saved HPol working point](../../../../configs/operating_modes/fei_50v_hpol_a2/profile.json)

## Saved checkpoint

The [captured profile](checkpoint/profile.json), [run summary](checkpoint/result.json) and [particle archive](checkpoint/trajectories.npz) preserve the final 50,000-primary run, including source IDs, weights, interceptions, specimen landings and the recorded trajectories. The installed profile includes the field tables needed for replay.

The native material solution `global-f1ae7382f80d48e7.npz` uses `corners_m` from the [shared material mesh](../fei-output-relay/field-replay/material-mesh.npz). The [HPol material state](checkpoint/material-state.npz) holds the remaining arrays, solver report and field recipe. To reconstruct the native archive from the repository root, write the mesh array first and retain the stored order of the other arrays.

```python
from pathlib import Path

import numpy as np

study = Path("docs/development/nanoprobe_si110_20261122")
with (
    np.load(study / "fei-output-relay/field-replay/material-mesh.npz") as mesh,
    np.load(study / "fei-corrector-a2/checkpoint/material-state.npz") as state,
    Path("global-f1ae7382f80d48e7.npz").open("xb") as target,
):
    np.savez_compressed(target, corners_m=mesh["corners_m"], **{key: state[key] for key in state.files})
```

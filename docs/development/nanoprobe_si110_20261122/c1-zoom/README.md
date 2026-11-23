# C2–C3 crossover position and current-bearing specimen semi-angle

This Nanoprobe Normal study compares the same 16,384 positive-current electrons in the same three-condenser column at 300 keV. The axial coordinate starts at the effective source. The downstream face of the C2 lens is at 750 mm, the C2 aperture foil occupies 764.9–765.1 mm, and the specimen upper surface is at 1599.1999975 mm. The C2 aperture diameter is 100 µm.

## Moving the crossover with two supplies

The C1 lens is fixed at −36 A, the Mini Condenser at +226.70155 A and the series objective at −1.23194506 A. Both hexapoles, all corrector relay lenses, and all other supplies are identical across these rows. The static scan currents are zero. For each row the solver recomputed the common nonlinear field of the installed windings and 94 magnetic regions and relaunched the effective source in the new field. The comparison verifies equality of the launch positions, directions, absolute weights and source identities. The current at the specimen is 15.0 µA in each case.

| C2 lens (A) | C3 lens (A) | Crossover z (mm) | Distance below C2 exit (mm) | Specimen α95 (µrad) | Specimen D95 (nm) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| +16.65900 | +6.70787 | 765.436 | 15.436 | 5.6755 | 7.796 |
| +17.57600 | +6.50000 | 752.808 | 2.808 | 6.8701 | 10.664 |
| +17.72155 | +6.47200 | 751.007 | 1.007 | 7.0500 | 11.224 |

Advancing the crossover by 14.429 mm toward the C2 lens increased the measured semi-angle by 24.2%. The enclosed-current diameter at the fixed surface also grew. The downstream TL22–TL21 waist stayed within 1208.08–1208.10 mm and the post-scan waist within 1423.13–1423.15 mm. Each setting has exactly one intermediate waist in the C2–C3 gap and none from the C3 lens to the ADL. The finite population and the common downstream excitation bound this result to the tested branch. [Controlled measurements](c2-c3-distance.json) · [Condenser envelope and measured semi-angle](c2-c3-distance.png) · [Specimen landing distributions on a common scale](c2-c3-specimen-probes.png)

The measured α95 is the current-weighted 95th percentile of `atan(hypot(dx/dz, dy/dz))` relative to the fixed column axis. D95 is twice the 95th-percentile distance from the specimen current centroid. Each probe panel uses absolute landing current per unit area at the actual physical entrance. The dashed circle marks D95. Fine structure on a scale below the particle sampling or the plotted pixel size requires an independent check.

## Surface focus and a larger-angle control

At the 1.007 mm crossover setting, changing only the series objective to −1.243412 A moved the sampled D95 minimum from 128 µm upstream to the specimen upper surface on a 1 µm axial grid. The specimen D95 changed from 11.224 to 11.208 nm and α95 from 7.050 to 7.040 µrad. The refocused RMS radius is 2.655 nm at the specimen and reaches its minimum 68 µm downstream in vacuum continuation. The fixed-axis 50 pm disk received 14.65 nA at the upper surface. [Axial D95 and RMS measurements](../c1-zoom-near-c2-focused/axial-focus-profile.json) · [Refocused ray diagram](../c1-zoom-near-c2-focused/full-column-rays.png) · [Specimen probe](../c1-zoom-near-c2-focused/specimen-probe.png)

An independent control used +17.750 A on the C2 lens and +1.000 A on the C3 lens. All other currents matched the three-row comparison, including the −1.23194506 A objective. All 16,384 source particles reached the specimen. The measured α95 increased to **0.44738 mrad**, or 63.5 times the 7.050 µrad comparison value. The first two waists remained at 507.137 and 751.546 mm. The third moved to 1271.640 mm, downstream of the TL22–TL21 gap. The fourth moved to 1572.374 mm, past the Mini Condenser. The beam was diverging at the specimen and its D95 was 23.989 µm. This control measures angular spread after an upstream focus. The required 30 mrad converging probe remains unresolved. [Control trace](../angle-first-low-c3/tracking.json) · [Full-column rays](../angle-first-low-c3/full-column-rays.png) · [Specimen landings](../angle-first-low-c3/specimen-probe.png)

## C1 source image and downstream registration

Changing the previous C1 lens from −30 A to −42 A on its own moved the first radial waist from 524.72 to 496.66 mm and increased specimen α95 from 4.604 to 45.284 µrad. The second waist moved into the C2 lens at 743.28 mm, and additional downstream crossings displaced the terminal focus. The three-condenser comparison uses the moderate −36 A setting, which places the first waist around 507 mm. The −42 A trial used 8,192 source particles. [C1-only control](../c1-strong-control/tracking.json) · [Original working point](../relay-edge/tracking.json)

For the 2.808 mm setting, independent positive and negative 1 µrad commands in X and Y used the existing scan mixing, four incremental material-field solutions and fresh propagation of 2,048 source electrons for each drive. The least-displacement scan plane was at 1423.029 mm. The coordinate-specific minima were 1422.879 and 1423.181 mm. The residual position singular values were 1.97–2.76 nm/µrad. The zero-drive waist was at 1423.139 mm. An isolated nonlinear solve of the Mini Condenser and the series objective at the same currents placed their incident front focal plane at 1479.682 mm. This leaves a 56.65 mm separation from the driven scan plane. The experiment therefore establishes the local condenser trend while the front-plane registration remains open. [Driven pivot and isolated pair measurements](../c1-zoom-focused/ac-and-front-planes.json)

## Reproduction

The tracked working points and exact bias vectors are in `../c1-zoom-foil`, `../c1-zoom-focused` and `../c1-zoom-near-c2`. Each `tracking.json` records the nonlinear solution file and checksum, emitted source and magnetic sampling pitch. The finite-source trials use the project runtime, a 0.25 mm axial bore-field fit, cubic field interpolation, aperture interception and a 0.05 mm maximum tracking step. `condenser_zoom.py` searches candidate supplies using first-order transfers. All values in the table come from the subsequent three-dimensional finite-source traces. Run `c2_c3_distance.py` from the study directory to check the matched input and produce the comparison, including disjoint source halves and 0.025 mm integration at both endpoints.

The archived transport inputs contain the stationary bore-field coefficients and the original emission and gun-exit particles for each of the three rows, the earlier baseline, the final objective focus and the larger-angle control. The index gives source IDs through the linked arrays, original filenames and SHA-256 checksums. Full-volume nonlinear solver checkpoints used to change excitations are retained in the local `outputs/material-relay-20261122` directory. [Transport input index](transport-inputs/index.json)

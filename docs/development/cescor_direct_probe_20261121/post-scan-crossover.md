# Crossovers inside and downstream of the probe corrector

All distances run downstream from the effective source. The lower scan winding ends at 1420.0 mm, Mini Condenser begins at 1512.5 mm, and the specimen entrance is at 1599.1999975 mm. Static trajectories use zero current in all four AC circuits. The horizontal and vertical signed tilt measurements run separately.

## Inside the corrector

There is one shared transverse crossover in the shaded HP2–TL12 interval (1129.50–1379.00 mm). After the final objective-only adjustment its position is 1209.15335 mm, between TL22 and TL21, 1.053 mm into the nominal QPC interval (1208.10–1222.60 mm). The QPC excitation is zero. At 2 mm before, at the crossover and at 2 mm after, the directly tracked RMS radii are 9.394, 0.794 and 9.394 nm. The two directional roots differ by 0.000131 mm. The same dense count over 1070–1420 mm, from the ADL entrance to the lower scan exit, also gives one. The next static waist is 2.987 mm downstream of that exit. [Both ray projections, component boundaries and beam radius](evidence/material-relay/post-scan-objective-focus/corrector-crossover.png) · [Dense counts and measured radii](evidence/material-relay/post-scan-objective-focus/corrector-crossover-count.json)

The dense check propagated all 16,384 original positive-current particles in the same nonlinear field as the whole-column trace. It recorded 2,806 planes at a nominal spacing of 0.125 mm, with extra observations at the earlier root and on either side. No particle was clipped over the interval. Counting individual ray intersections in a single projection or counting conjugate planes from a tilt test measures a different quantity.

## Lower scan winding to Mini Condenser

C1 lens and C2 lens remain at −30 and +16 A with the 100 µm diameter C2 aperture. C3 lens, TL12 and Mini Condenser carry +7.114073, −118.247040 and +226.701546 A. Each static operating field is a new nonlinear solution for the installed coils and 94 magnetic material volumes. The front focal plane comes from a separate solve on the same geometry with only Mini Condenser and the series objective excited. The AC pivot comes from independent signed 1 µrad commands from the effective source. [Magnetic and electrostatic field provenance](evidence/material-relay/post-scan-objective-focus/field-audit.json)

| Setting | Series objective (A) | Corrector waist (mm) | Post-scan waist (mm) | AC pivot (mm) | Lens-pair front (mm) | Specimen D95 (nm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Relay aligned | −10.828688 | 1209.152 | 1422.9955 | 1423.0024 | 1423.0023 | 574.96 |
| Objective-only adjustment, same AC calibration | −10.854711 | 1209.153 | 1422.9868 | 1423.0038 | 1422.8554 | 575.47 |

Before the objective adjustment the static waist lies 6.85 µm upstream of the measured pivot. After the adjustment the gap waist has an RMS radius of 2.010 nm. It lies 17.0 µm upstream of the pivot and 131.4 µm downstream of the lens-pair front. The front has moved upstream by 146.9 µm while the measured pivot has moved downstream by 1.4 µm. The frozen scan response has a maximum residual position of 0.0364 nm per µrad at its pivot. At the specimen both settings transmit 15.0 µA, and the objective-only setting gives a 95th-percentile populated semi-angle of 0.0000411 mrad. The D95 minimum remains unbracketed throughout the ±0.4 mm specimen focus check. These results establish the internal count and the post-scan waist. Direct specimen focus and the requested working semi-angle remain unresolved. [Before adjustment](evidence/material-relay/post-scan-registered/working-point.json) · [After adjustment](evidence/material-relay/post-scan-objective-focus/working-point.json) · [Specimen through focus](evidence/material-relay/post-scan-objective-focus/surface-focus.json)

[Whole-column trajectories](evidence/material-relay/post-scan-objective-focus/full-column-rays.png) · [TL12 and both scan stations](evidence/material-relay/post-scan-objective-focus/tl12-scan-rays.png) · [Direct specimen-plane current distribution](evidence/material-relay/post-scan-objective-focus/specimen-probe.png)

A second current setting produced a narrower 7.824 nm specimen D95 with 8,192 source particles, one internal waist at 1222.047 mm and a post-scan crossover at 1444.572 mm. Its lens-pair front lay another 13.785 mm downstream. The specimen D95 minimum was 8 µm downstream of the entrance. This case quantifies the trade-off observed in the tested settings. [Independent trajectory](evidence/material-relay/post-scan-compact/tracking.json) · [Specimen through focus](evidence/material-relay/post-scan-compact/surface-focus.json)

## Previous operating point

At the previous 1444.814 mm lens-pair front the zero-scan RMS radius was 224.402 nm. None of the 32,768 positive-current source particles moved to the opposite side of the centroid across the 10 mm interval centred there. One transverse direction was still contracting as the other expanded. That calibration delivered a 5.487 nm specimen D95 and lacked a shared post-scan crossover. [Same-particle comparison](evidence/material-relay/crossover-control-fine/post-scan-check.json) · [Previous specimen and relay measurements](evidence/material-relay/relay-final/working-point.json)

## Independent sharp-crossover control

With C3 lens +6.710114 A, TL12 −65 A, Mini Condenser +239.912653 A and the series objective −31.079874 A, 8,192 freshly launched particles crossed at 1447.4630 mm. The two directional minima were 1447.4629 and 1447.4631 mm. All four disjoint sets of 2,048 original particles reproduced the minimum. Direct observation at the measured waist gave an RMS radius of 0.542 nm, compared with 73.370 and 73.359 nm at 5 mm upstream and downstream. Every tracked particle crossed into the opposite half-plane relative to the current centroid. [Trajectories and RMS radius](evidence/material-relay/crossover-control-fine/post-scan-crossover.png) · [Native particle measurements](evidence/material-relay/crossover-control-fine/post-scan-check.json)

The 1 µrad driven pivot was 1457.83513 mm and the independently solved lens-pair front was 1457.83511 mm. The zero-scan crossover lay 10.372 mm upstream of both. The static on-axis field at the crossover was −142.81 mT. With only Mini Condenser and the series objective powered on the same material mesh it was −141.98 mT. These two nonlinear operating points demonstrate a substantial lens field in this model at the crossover. Absolute fringe fields in the microscope still require measurement. The specimen D95 was 7.344 nm, the RMS minimum was 0.14 µm beyond the entrance and the sampled D95 minimum was 17 µm beyond it. [Signed scan test](evidence/material-relay/crossover-control-scan/scan-calibration.json) · [Specimen through focus](evidence/material-relay/crossover-control-fine/surface-focus.json)

## Measurement

The current-weighted radial variance reaches a minimum at each counted crossover. Both eigenvalues of half the axial derivative of the transverse position covariance pass from negative to positive in the same axial neighborhood. The trajectory comparison keeps the original ray identities. A negative dot product of a ray's two centroid-relative transverse positions puts it in the opposite half-plane in laboratory coordinates. The specimen probe is a histogram of the actual weighted geometrical landings at the upper surface. D95 is the diameter enclosing 95% of the transmitted current about its centroid.

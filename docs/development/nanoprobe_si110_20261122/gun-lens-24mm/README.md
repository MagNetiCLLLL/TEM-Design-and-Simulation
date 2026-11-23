# First source image at the Nanoprobe Normal gun aperture

Set the electrostatic gun lens to **+0.01713 kV relative to the tip** to place the first real image of the effective source inside the nominal 24 mm gun aperture. The parent working point uses +5.2 kV. The extractor stays at +4.0 kV, the gun aperture diameter is 4.0 mm, the C1 aperture diameter is 1.0 mm, and the accelerator terminates at 300 kV. All independent downstream supplies retain the near-C2, objective-focused settings. The original electrical divider starts at the gun lens, so each intermediate accelerator electrode follows its existing divider fraction when the gun-lens setting changes. This voltage applies to the simulator's nominal electrode geometry. [Parent transport inputs](../c1-zoom/transport-inputs/index.json)

| Observable | +5.2 kV | +0.01713 kV |
| --- | ---: | ---: |
| First source image, tip to 450 mm exit | None found | 24.0004 mm |
| Gun-exit current | 15.0 µA | 15.0 µA |
| Original / surviving / effective particle counts | 16,384 / 16,384 / 16,384 | 16,384 / 16,384 / 16,384 |
| Gun-exit current-weighted α95 | 11.33 µrad | 96.93 µrad |
| Gun-exit angular RMS | 6.30 µrad | 53.80 µrad |
| Gun-exit RMS radius | 0.340 µm | 2.960 µm |
| Slope of RMS radius at the exit | −1.79 µrad | −11.27 µrad |

The [electrode geometry and acceptance audit](electrode-audit.md) compares this gun with scaled FEI patent drawings, checks the field directions at two physical electrode voltages, and locates primary-electron losses relative to the lens body and gun aperture.

The angular semi-angle is the cone around the current-weighted mean propagation direction that contains 95% of the transmitted current. The signed slope measures the change in transverse RMS radius with axial position. At 450 mm both envelopes are contracting in the existing column field. At 26 mm the tuned beam is expanding. The axial magnetic flux density at the exit is −74.7 mT. The ray diagram draws the same original IDs at both voltages. The enlarged aperture panels share a vertical scale. [Matched trajectories](gun-ray-diagram.png) · [Angular response and electric potential](source-image-and-field.png)

## Locating the first real image

For every voltage the calculation uses the electrode-geometry Laplace field and the archived common magnetic field. From the central effective-source point at 0.3 eV, opposite launch perturbations in x and y independently give a two-component separation vector at each observation plane. A near-antiparallel reversal with vanishing separation identifies a real source image. The axial survey starts just downstream of emission, covers both faces of the 23.9–24.1 mm foil and continues to the 450 mm gun exit. Orthogonal perturbations agree. The +5.2 kV pairs are separated by 0.519 nm at the foil centre. As voltage decreases through +25 V toward zero, the signed response at the foil centre reverses. [Voltage sweep](voltage-scan-mesh-16.json)

| Electric grid (r × z) | Nominal transport step | Tuned first image, ±1 µrad |
| --- | ---: | ---: |
| 313 × 2737 | 0.20 mm | 24.00044 mm |
| 409 × 3632 | 0.20 mm | 24.00441 mm |
| 409 × 3632 | 0.10 mm | 24.00511 mm |

The ±0.5 and ±2 µrad controls locate the same first image within each discretisation. The interval of about 5 µm across the checks is a numerical sensitivity. It gives no hardware placement tolerance. In the tuned field the on-axis potential between the extractor and gun aperture drops to about +169 V relative to the tip before rising to +3.89 kV at the aperture. The entrance and exit fringe fields of this retarding electrode produce a substantially different trajectory from the +5.2 kV setting.

## Finite-source check and continuation

The tuned full trial transports the parent's 16,384 original particles, compares their launch coordinates, angles, IDs and weights, and retains interception, flight time and exit energy. All pass the gun and C1 apertures. The current-weighted D95 at the **gun aperture** is 3.14 nm. The 8,192-ID prefix gives an exit α95 of 96.946 µrad, compared with 96.931 µrad for all particles. A paired check of the first 1,024 original IDs at the finer grid and half step changes their exit α95 from 94.749 to 94.726 µrad. Their RMS envelope slope changes sign between 24.000 and 24.010 mm. The maximum sampled gun radius is 9.78 µm within the archived magnetic field's 50 µm bore-fit radius. [Numerical results and control settings](operating-point.json) · [Finer-grid finite-source check](matched-refinement.json)

The new gun handoff contains exit coordinates and slopes, source IDs and original weights, actual exit energy offsets, flight times and aperture flags. Use the tuned gun voltage and the recorded field identities when continuing downstream. The specimen was outside the scope of this gun-stage calculation. [Gun-exit particle data](gun-exit-particles.npz)

Run these commands from this directory using the project Python environment and the parent study inputs.

```text
python gun_focus.py --voltages 5.2 8 12 20 32 50 80 4 2 1 0 0.025 0.05 0.1 0.25 0.5 --bracket 0 1
python verify_planes.py 5.2
python verify_planes.py 0.01713
python verify_planes.py 0.01713 --mesh 22
python verify_planes.py 0.01713 --mesh 22 --step-factor 0.5
python finite_source.py 0.01713
python finite_source.py 0.01713 --mesh 22 --paired-check
python report.py
```

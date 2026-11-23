# Condenser poles and returns in Nanoprobe Normal

The [full-source replay in the refined objective field](refined-objective-normal.md) continues this working point at the same currents and aperture. The specimen results below use the original objective mesh.

The installed condenser magnetic circuit moves the second effective-source conjugate upstream at the former vacuum-winding currents. At −41 A in C1 lens and +19.5 A in C2 lens, the second paraxial image moves from 794.75 to 728.95 mm, inside the C2 lens housing. A matched 4,096-particle material-field launch confirms its finite-source waist at 728.94 mm. Reducing the two series currents to **−30 A and +16 A** restores a single crossover in the clear space between C2 lens and C3 lens. A fresh 32,768-particle trace reaches the specimen upper surface at 1599.1999975 mm. The resulting specimen D95 is **3.089 µm** and the measured angular 95th percentile is **0.01929 mrad**. The condenser topology passes. The fixed downstream setting falls short of a 30 mrad, atomically focused probe. [Matched upstream ray diagrams](evidence/condenser-pair/condenser-ray-comparison.png)

The original ray diagram uses a coarse axial grid for the objective near the upstream scan winding. At the 1380 mm scan entrance its interpolated objective contribution is −138.80 mT. At identical excitation, local 1 mm and 0.5 mm finite-element grids give −42.83 and −42.86 mT. The original scan-station turning angles and downstream specimen results therefore depend on the archived objective reconstruction. A matched local replay moves the radial envelope maximum from 1399.70 to 1420.84 mm. Both replays have zero scan and alignment commands. The local comparisons start at 1330 mm. [Scan-region trajectories and fields](evidence/condenser-scan-region.png) · [Field, drive and envelope measurements](evidence/condenser-scan-region.json)

## Magnetic circuit and field comparison

The joint axisymmetric nonlinear vector-potential solution contains the actual C1 lens lower pole (490–540 mm), C2 lens upper pole (560–620 mm) and the two outer return yokes (450–550 and 550–750 mm). The installed opposed faces leave a 20 mm gap. The archived nominal device geometry contains no separately specified radial return end-plates joining the poles to the outer yokes. Both poles have explicit ARMCO pure iron assignments. Both yokes are specified as soft magnetic, so the calculation assigns them the same sourced B–H curve. The ungraded cartridge carrier and its nonmagnetic separator have relative permeability one in this model. These are nominal, uncalibrated magnetic properties. [Supplier B–H data, Figure 7](https://www.aksteel.nl/files/downloads/clf_productdata__armco_pure_iron_pdb_euro_final_072022_92.pdf)

The winding cross-sections, 300 and 540 physical turns and their signs are inherited. All C1 lens groups carry the first series current, and all C2 lens groups carry the second. The nonlinear solve includes the incident C3 lens winding field at its stored Normal setting. Tracking adds the solved magnetization to one copy of the analytic winding field. This free-space recovery continues to the gun and the specimen. The frozen corrector and objective retain their own independently solved material fields. Their back-reaction and that of the other distant parts on the upstream iron are omitted. The maximum on-axis contribution of all omitted drives across the condenser iron's axial span is 0.060 mT.

| C1 lens / C2 lens series currents | Field model | C1-region negative peak | C2-region positive peak |
| --- | --- | --- | --- |
| −41 / +19.5 A | Vacuum condenser windings | −91.57 mT at 495.0 mm | +57.96 mT at 665.5 mm |
| −41 / +19.5 A | Windings, poles and returns | −139.10 mT at 479.0 mm | +77.91 mT at 636.5 mm |
| −30 / +16 A | Windings, poles and returns | −100.64 mT at 479.0 mm | +66.18 mT at 635.5 mm |

At the comparison currents the largest calculated flux density inside the iron is 0.783 T. At the selected currents it is 0.569 T. The tracking mesh has 25,612 nodes and a 1.5 mm maximum local pitch. Refining to 1 mm and enlarging the boundary increases the mesh to 50,295 nodes. The induced bore field changes by at most 0.364 mT at the comparison currents and 0.260 mT at the selected currents. The selected paraxial source image and second crossover move by 0.023 and 0.045 mm. Direct on-axis differentiation of the bounded finite-element potential and the open-bore magnetization recovery differ by up to 1.68 mT at the selected currents. [Signed axial field](evidence/condenser-pair/axial-field.png) · [Field samples, convergence and material assignments](evidence/condenser-pair/axial-comparison.json)

## Effective source and ray topology

The 300 kV effective source emits 15 µA. The launch directions, positions, original particle identities and energies are checked against the original emission. Every source particle is relaunched in the new field, including its upstream fringe. Seventeen zero-current differential trajectories measure source imaging and are excluded from all current, size and angle statistics. The source and column stages preserve particle IDs and incident current weights.

| Interval | Full-packet result |
| --- | --- |
| C1 lens housing, 450–550 mm | One source image at 524.6745 mm. Both transverse magnifications are 0.19719, giving 5.07-fold linear demagnification. The finite-source radial RMS minimum is at 524.6714 mm. |
| C2 lens housing, 550–750 mm | Zero radial waists. |
| C2 lens exit to C3 lens entrance, 750–830 mm | One radial waist at 790.9951 mm. |
| C3 lens housing, 830–1000 mm | Zero radial waists. |
| C3 lens exit to ADL entrance, 1000–1070 mm | Zero radial waists. |
| ADL entrance through the scan system to the specimen | Further radial waists at 1141.7004, 1307.0052 and 1474.7029 mm. The upper and lower scan region has no radial waist around 1408 mm. |

Imaging uses both ±1 and ±0.5 µrad source-angle perturbations and their corresponding source-position perturbations. Waist counting uses the sign of the centred position–slope covariance. Both principal transverse directions independently reproduce the upstream interval counts. All post-C2 lens currents, including the weak-excitation C3 lens, Mini Condenser and objective, match the parent working point. The individual-path plots show 64 source IDs. The projected density plots include all 32,768 source particles in both transverse projections. [Full column, individual paths](evidence/condenser-pair/full-column-rays.png) · [Enlarged sections](evidence/condenser-pair/ray-sections.png) · [Full ensemble, source through specimen](evidence/condenser-pair/all-ray-density.png)

## Specimen entrance

All 32,768 particles pass through the 100 µm C2 aperture and arrive at the actual specimen upper surface. At the foil entrance the current-weighted 95% radius is 0.300 µm and the largest sampled radius is 0.543 µm, compared with a 50 µm physical aperture radius. No measurement of a 30 mrad marginal cone can be inferred from this aperture on this branch.

| Measured at the surface | Value |
| --- | ---: |
| Centroid-centred D95 diameter | 3.0893 µm |
| Centred radial RMS radius | 0.8582 µm |
| Current-weighted 95th percentile of angle to the optical axis | 0.01929 mrad |
| Largest sampled angle to the optical axis | 0.03458 mrad |
| Total incident current | 15.000 µA |
| Current within 50 nm of the mechanical axis | 67.29 nA |
| Current centroid relative to the optical axis | (−27.65, −23.38) nm |

The last upstream radial waist is approximately 124.50 mm before the specimen. Over the last 0.4 mm of the actual path, D95 decreases from 3.0946 to 3.0893 µm. The image shows the physical landing positions and absolute current density on the fixed plane. Its circle encloses 95% of current about the current centroid. The fixed-axis 50 nm current is a separate observable. With 4,096 particles D95 is 3.0545 µm. Relaunching the same 4,096 particles in the finer, wider magnetic solution gives 3.0523 µm. With 32,768 particles halving the column integration step from 0.05 to 0.025 mm and reducing the induced near-axis expansion from seventh to fifth derivative changes D95 by less than 0.00001 µm and preserves every transmitted identity. Both the source-count and material-mesh changes exceed the integration-step effect. The material-grade uncertainty is unmeasured. [Specimen current-density map](evidence/condenser-pair/specimen-probe.png) · [Full operating point and output hashes](evidence/condenser-pair/working-point.json) · [Landings, current weights and matched ray paths](evidence/condenser-pair/column-and-landings.npz)

## Reproduction

Run in the project environment with the inherited Nanoprobe Normal study inputs available. Column tracking uses CUDA. The tuned case is a separate research working point.

```powershell
$study = 'docs/development/cescor_direct_probe_20261121'
python "$study/condenser_pair_study.py" tune
python "$study/condenser_pair_field_report.py"
foreach ($count in 4096,32768) {
    python "$study/condenser_pair_gun.py" --count $count
    python "$study/condenser_pair_column.py" --count $count --step 0.05 --order 7
    python "$study/condenser_pair_column.py" --count $count --step 0.025 --order 5
}
python "$study/condenser_pair_study.py" prepare-controls
foreach ($case in 'matched-current','mesh-check') {
    $env:TEMSIM_CONDENSER_PAIR_OUT = (Resolve-Path "outputs/condenser-pair-fem-20261122/$case").Path
    python "$study/condenser_pair_gun.py" --count 4096
    python "$study/condenser_pair_column.py" --count 4096 --step 0.05
}
Remove-Item Env:TEMSIM_CONDENSER_PAIR_OUT
python "$study/condenser_pair_controls.py"
foreach ($stage in 'gun','column','render') {
    python "$study/condenser_pair_ray_density.py" $stage
}
python "$study/condenser_pair_render.py"
```

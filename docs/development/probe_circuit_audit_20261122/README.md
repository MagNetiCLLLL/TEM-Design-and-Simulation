# Magnetic circuit and axial field of the probe corrector

The instrument is the three-condenser `C3_ProbeCorrector` Nanoprobe Normal column. This study covers ADL through TL12 and samples the field downstream at the upper and lower AC scan windings. Column coordinates increase towards the specimen. The module-local origin is at column z = 450 mm. The observables are the longitudinal magnetic field on the axis and the leading transverse sextupole field in the bore. The results describe a static magnetic operating point.

The Nanoprobe Normal trajectory replay now uses the audited eight-channel solution at its recorded currents. The extra crossover near 1408 mm disappears at fixed objective current. [Working point, whole-column rays and scan-region detail](../cescor_direct_probe_20261121/normal-joint-field.md)

## Where TL12 acts

With the installed poles and return path, the TL12 field peaks in its 14 mm pole gap. At the two scan centers its change is 3.72% and 0.35% of its peak when energized in the assembled corrector. The old ray diagram used a different TL12 source with three signed vacuum winding groups. That source alone gave +471 and +233 mT at the two scan centers, or 94% and 47% of its own peak. Those archived numbers describe a different excitation and cannot measure the effect of adding iron. [Previous field and ray diagram](../cescor_direct_probe_20261121/field-range-audit.md) · [Signed field profiles](axial-profiles.npz) · [Solver inputs and numerical checks](magnetic-audit.json)

| Same nominal TL12 winding, 19 turns at +10 A | Peak axial field (mT) | Peak z (mm) | Upper scan, 1387.5 mm | Lower scan, 1412.5 mm |
| --- | ---: | ---: | ---: | ---: |
| Single winding in vacuum, analytical | 2.690 | 1366.3 | 2.000 mT (74.35%) | 0.900 mT (33.46%) |
| Isolated TL12 with pole pieces, independent axisymmetric FEM | 11.821 | 1366.3 | 0.495 mT (4.19%) | 0.029 mT (0.25%) |
| Assembled 3-D FEM, field change when TL12 is energized | 11.936 | 1366.5 | 0.444 mT (3.72%) | 0.042 mT (0.35%) |

The assembled TL12 response falls through 10% at 1383.0 mm and 1% at 1395.3 mm on the downstream side. The half-maximum width is 18.0 mm. The pole-gap midpoint is 1366.3 mm and the 3-D profiles have 0.5 mm axial sampling. With all eight corrector channels at the audited drives, the **total corrector** axial field at the two scan centers is +1.163 and +0.774 mT. Independent integration of the same 3-D material moments gives +1.146 and +0.754 mT. Energized Mini Condenser and objective fields are outside these totals.

In the isolated axisymmetric calculation, the interval between the two scan centers contains 0.038% of the axial integral of Bz² for the complete TL12 magnetic circuit and 13.52% for its matched vacuum winding. Both integrals use the same 1083.6–1649.0 mm domain. This is a measure of where the paraxial round-lens focusing coefficient acts. A total field would also include cross terms with other magnets. No beam-width criterion enters this comparison. [Flux through the gap and returns, and signed axial field](tl12-field.png)

## Which windings share the same field

The main hexapole responses stay near the pole pins. HP2 is above 10% of its own peak from 1125.4 to 1163.2 mm, HP1 from 1281.7 to 1320.1 mm. At their colocated DPH2 and DPH1 dipoles the respective hexapole responses are 99% and nearly 100%. At the upper scan center both fall below 0.001% of their own peaks. Two radii (0.6 and 1.2 mm) give transverse sextupole profiles within 0.95% of the peak for each of the four hexapole channels. The shallow maximum of a main hexapole should not be used alone to set its optical midplane. [All eight excitation profiles](corrector-field-locations.png) · [Ranges and overlap at nineteen winding locations](field-ranges.md)

The strongest overlap in the modeled exit trims is between HPol and HP1. At HP1's center the HPol Y response is effectively at its own maximum. The HP1 response still reaches 57% at the HPol center, 16.35 mm downstream. The internal hosts of these trim windings are nominal. The computed coupling cannot establish the construction or tuning of the photographed instrument. DP22 and DP21 also sit in the transfer-lens fringes. At DP22, TL22 and TL21 reach 29% and 13% of their own peak responses. At DP21, TL21 reaches 43%. The published CESCOR diagram supports the registration of DPH with the main hexapoles and the position of the transfer-doublet deflectors. [Müller et al., corrector arrangement](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=3)

The static differences include each unpowered trim's magnetic sleeve and the fields of its neighbors. Driving an additional dipole or quadrupole would require its own current excitation and retention of the corresponding angular harmonic in the bore readout. The present bore continuation retains the threefold-symmetric harmonics appropriate to the active round and hexapole sources.

## Magnetic regions with high induction

At the archived TL22 and TL21 currents, the volume-weighted 95th percentiles of the pure-iron upper/lower return plates are 2.064/2.231 T for TL22 and 2.341/2.326 T for TL21. The full modeled volume of each of these four plates is above 1.8 T. For comparison, the two TL12 plates reach only 0.534/0.686 T at their 95th percentiles. The TL21 and TL22 plate sections and physical ampere-turns therefore matter when interpreting their broad fringe responses. Real return-plate thicknesses, operating currents and internal material assignments remain unmeasured. In the 3-D operating solution the maximum inductions over all elements are 2.351 T for ARMCO iron and 1.589 T for Ni–Fe. [Per-region induction and material-curve limits](magnetic-audit.json)

At the nominal 19 turns, the two-dimensional isolated-lens controls separate the effect of excitation from the rest of the assembly. The 10%-of-peak width of the static axial field for TL22 grows from 32.5 mm at −10 A to 108.5 mm at −145 A. For TL21 it grows from 32.5 mm at +10 A to 125.6 mm at +330 A. At the archived biases, an additional 1% current step gives 10% widths of about 138 mm (TL22) and 139 mm (TL21) in isolation. In the assembled three-dimensional model these incremental widths are 76.8 and 107.8 mm and both profiles become asymmetric. The static field and the incremental field have different widths at high induction.

## Where the return flux can go

The original joint three-dimensional solve included 40 ferromagnetic solids. The field used in the previous ray diagram combined that core with separate vacuum fields for the end lenses. The audit registers all 60 ferromagnetic solids identified in the installed ADL-through-TL12 CAD geometry. Gmsh/OpenCASCADE fragments the solids into distinct conforming volumes. Shared faces were measured geometrically before solving. [Material contacts and intended gaps](material-contacts.json)

| Station | Magnetic circuit in the installed design | Evidence and remaining uncertainty |
| --- | --- | --- |
| HP2 and HP1 | Each has six Ni–Fe pole pins, six pure-iron root sleeves and one annular return yoke. Each pin has contact both with its sleeve and directly with the yoke. The sleeve also contacts the yoke. | Both cartridges use a 152 mm yoke outside diameter, 8 mm bore and 30 mm axial pole length. The publication identifies the Ni–Fe alloy family. The specific PERMENORM grade, fits and winding turns in this assembly are modeling inputs. [Müller et al., pp. 443–444 and Figs. 2–3](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=3) |
| ADL, TL22, TL21, TL12 | At each end the pole seats on a sleeve, the sleeve on a radial return plate, and the plate on the outer yoke. All 24 intended interfaces have positive contact area. The opposing pole faces are separated by 14, 12, 12 and 14 mm respectively. | These are nominal CAD dimensions. At TL12 the two pole faces lie at 1359.3 and 1373.3 mm. The 0.4 mm thick radial plate has an 88.38 mm² seat on its sleeve and a 127.71 mm² seat on its yoke. The bore diameter is 20 mm. |
| QPH2, HPC, QPC, QPH1, HPol, QPol | Each has a separate magnetic sleeve around its winding. None of these six sleeves shares a material face with a neighboring magnetic stage. Magnetic coupling through air remains possible. | The installed geometry identifies their windings and sleeves. The physical host and detailed pole construction of each trim are unverified. Only HPol X/Y carry current in this audit. |
| DPH2, DP22, DP21, DPH1, DP11, AC scan | These are transverse windings. DPH2 and DPH1 are at the axial centers of their respective main hexapoles. The other stations sample the fringe fields of nearby magnets. | Deflection currents are zero in the audited bias. The two AC winding centers at 1387.5 and 1412.5 mm are observation locations. DP12's optical reference does not add another excitation. |

The native assembly has 48 ARMCO iron regions and 12 PERMENORM pole pins. Every material solve uses the stored operating point's isotropic, single-valued B–H tables. Their digests and all original knots were compared with the current part assignments. The ARMCO high-field extension comes from the manufacturer's Fig. 8. The Ni–Fe curve uses the archived interpolation to the manufacturer's 1.60 T value at 40 kA/m. Both solvers reject a final solution outside the tabulated domain. The external barrel photographs set scale but cannot locate an internal magnetic midplane. [ARMCO data](https://www.aksteel.nl/files/downloads/clf_productdata__armco_pure_iron_pdb_euro_final_072022_92.pdf) · [PERMENORM solid data](https://vacuumschmelze.com/03_Documents/Brochures/Datasheet%20PERMENORM%205000%20H2%20Solid.pdf)

All 92 installed winding-pack envelopes in the corrector were checked against the magnetic solids with exact CAD Boolean intersections. Their windows are hollow. All packs have positive clearance. The smallest is 0.10 mm between a DPH winding pack and its neighboring main hexapole pin. The nearest steel is 0.14 mm from DP22, DP21 and DP11, and 0.72 mm from the TL12 pack. These nominal clearances apply to pack envelopes including insulation. [Minimum clearance and nearest material for every pack](winding-clearances.json)

## Excitations and readout

The six archived excitations are HP2 +70 A, TL22 −145 A, TL21 +330 A, HP1 +16 A, HPol X −5 A and HPol Y −2.71705 A. ADL and TL12 use their installed single packs of 23 and 19 turns. Their physical operating currents are unknown. Each was switched from 0 to +10 A as an explicitly chosen test. ADL was switched with TL12 off, then TL12 with ADL on. Each difference comes from two nonlinear solutions with the other drives held fixed. The comparable vacuum control uses the same turns, radial section, axial winding interval and +10 A. The old three-group vacuum recipe is included solely to trace the origin of the previous field diagram.

For the six previously energized channels, increasing each current by 1% of its absolute bias gives a one-sided incremental response with all other currents fixed. The four round lenses use signed on-axis Bz. Main and auxiliary hexapoles use both phases of the transverse sextupole coefficient, sampled at 1.2 mm radius and checked at 0.6 mm. An ideal centered hexapole has zero transverse field on axis, so a plot of on-axis Bz alone would miss its action. Each band is the interpolated interval where the magnitude exceeds a fraction of that channel's own peak. The per-channel responses of this nonlinear assembly cannot simply be added to reproduce the DC field.

## Numerical checks

The control uses the existing reduced-scalar-potential P1 tetrahedral solver. Its standard material load samples the Biot–Savart field at each material tetrahedron's centroid. In permeable material, the winding field and reaction field can nearly cancel. An error in their difference can generate large erroneous material moments even when the nonlinear residual is small. The audit also integrates the unchanged physical winding field along three independent edges of each material tetrahedron and reconstructs the constant source field compatible with the scalar basis. Every current change is solved again with the nonlinear material laws. Only the impressed winding field is linear in current. [Analysis of reduced-potential cancellation](https://onlinelibrary.wiley.com/doi/abs/10.1002/jnm.638)

The beam-bore readout uses the scalar potential of the solved air tetrahedra, continued harmonically into the current-free bore and combined with the analytical winding field. All reported axial profiles lie in the native air part of the reconstruction. Additional readouts use the direct P1 gradient at each station and a four-point volume integral of the computed material moments. An independent axisymmetric vector-potential FEM uses the actual meridian sections, including the thin return plates and fitted sleeves, resolves the winding area, and checks against the analytic air-core result. A supplementary comparison places the passive Mini Condenser and objective poles and returns on the same axisymmetric mesh, with those external currents set to zero. The assembled 3-D results cover the corrector magnets. Full-column objective and condenser excitations and AC eddy currents require other solves.

| Isolated +190 ampere-turn TL12, all 60 steel solids present | Upper scan, bore potential / volume integral (mT) | Lower scan, bore potential / volume integral (mT) |
| --- | ---: | ---: |
| Centroid winding load, 5 mm nominal mesh and 170 mm air padding | +0.496 / +7.398 | +0.083 / −3.245 |
| Centroid winding load, 4 mm and 210 mm | +0.509 / +6.856 | +0.075 / −2.817 |
| Edge-integrated winding load, 5 mm and 170 mm | +0.441 / +0.408 | +0.040 / +0.033 |
| Edge-integrated winding load, 4 mm and 210 mm | +0.454 / +0.413 | +0.032 / +0.033 |

The two three-dimensional meshes contain 1,485,999 and 1,799,056 tetrahedra. For the weak lower-station field, the direct piecewise-constant air gradient on the finer mesh gives 0.066 mT. The smooth bore field and material integral are more consistent there. The last decimal in the table is useful for comparing numerical methods. In the independent 2-D solve, enlarging the exterior padding from 170 to 270 mm at 0.58 mm central spacing changes either scan-center value by less than 0.0002 mT. Adding the passive external Mini Condenser and objective steel on a matched mesh changes either value by less than 0.0006 mT. At +30 A the isolated TL12 scan-center amplitudes reach 6.91% and 1.73% of peak. A fixed spatial cutoff would miss this current dependence. [Meshes, excitations, alternate readouts and two-dimensional controls](magnetic-audit.json)

## Reproducing the study

Use the repository Python environment with Gmsh and the CUDA sparse solver. Run the scripts from the repository root in the following order. The default DC current and all current perturbations are in physical amperes per complete winding.

```powershell
$study = 'docs/development/probe_circuit_audit_20261122'
$python = '.\.venv\Scripts\python.exe'
& $python "$study/material_contacts.py"
& $python "$study/winding_clearances.py"
& $python "$study/round_section.py"
& $python "$study/circuit_audit.py"
& $python "$study/circuit_audit.py" --tl12-only
& $python "$study/circuit_audit.py" --mesh-mm 4 --padding-mm 210 --tl12-only
& $python "$study/field_profiles.py" --all-channels
& $python "$study/winding_discretization.py"
& $python "$study/winding_discretization.py" --mesh-mm 4 --padding-mm 210
& $python "$study/assembled_edge_fields.py"
& $python "$study/render_audit.py"
```

The compact signed profiles, complex sextupole coefficients and numerical receipts are stored with the figures. Meshes, DC solutions, edge loads and continuation checkpoints are under `outputs/probe-circuit-audit-20261122`. No beam particles or coherent image are needed to measure these field locations.

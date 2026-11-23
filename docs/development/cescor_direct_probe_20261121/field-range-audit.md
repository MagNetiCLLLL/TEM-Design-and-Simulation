# Corrector housing length and field overlap

The modeled HP2–TL22–TL21–HP1 core occupies 186.35 mm. The wider optical span from the entrance of ADL to the exit of TL12 is 309.0 mm, equal to the published height of a DCOR and 54 mm longer than a CESCOR. The core is compatible with both external envelopes. Assigning both complete end lenses to one DCOR-sized body uses its entire nominal height. A flange drawing or an identified mounting face at each end would establish which optics belong to that body. [CEOS CESCOR dimensions](https://www.ceos-gmbh.de/de/produkte/stemkorrektoren/cescor) · [CEOS DCOR dimensions](https://www.ceos-gmbh.de/de/produkte/stemkorrektoren/dcor-ascor)

## Mechanical comparison

The tape hook in each supplied photograph is near the lower barrel shoulder. The broad barrel extends roughly 21–22 cm in the first photograph and 23–24 cm in the second. The tape continues across the upper collars. The second housing carries a DDCOR A5 CP label. The mating faces and their relationship to the hook are partly obscured. The photographs establish the scale of the hardware. Product-family specifications give an independent comparison for its overall height. The internal winding centers have no visible tape references.

| Modeled assembly, first to last physical end | Column z (mm) | Length (mm) | Difference from 309 mm |
| --- | ---: | ---: | ---: |
| HP2 to HP1, including both 30 mm main elements | 1129.50–1315.85 | 186.35 | −122.65 |
| ADL to TL12 | 1070.00–1379.00 | 309.00 | 0.00 |
| Upstream beam shift/tilt assembly to TL12 | 1020.00–1379.00 | 359.00 | +50.00 |
| ADL to the downstream end of the AC scan assembly | 1070.00–1420.00 | 350.00 | +41.00 |

The CEOS dimension order is height, width, depth. The 360 mm dimension of the DCOR is transverse. The reported CESCOR hexapole has a 152 mm yoke **diameter**, an 8 mm bore and 30 mm axial pole length. The current HP2-to-TL22 and TL21-to-HP1 center spacings are 21.85 and 50.90 mm. The original design calls for optical double symmetry in the core. The product height alone cannot establish the field midplanes or transfer matrices. [Müller et al., Figures 2–3 and pp. 443–444](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=3)

## Which parts share fields in real instruments?

| Components | What the published design supports |
| --- | --- |
| Cold field emitter and extractor | The emitter tip is in the extraction electrode's electric field. The gun-exit ensemble already includes its upstream acceleration and transport. The gun is a separate assembly with an electrostatic field and does not set the axial length of the probe corrector. [JEOL extraction electrode](https://www.jeol.com/words/emterms/20121023.084657.php) |
| Condenser lenses, their deflectors and foil apertures | A passive aperture can intercept the beam inside a condenser fringe field. The exact foil and deflector positions relative to C1, C2 and C3 depend on the microscope. The corrector barrel photographs do not locate these parts. In this assembly C1 and C2 have sections of a common housing. Their calculated fields and the actual foil centers are tabulated separately below. |
| DPH2 and HP2, DPH1 and HP1 | The CESCOR drawing places the local steering dipole at the optical station of each main hexapole. A beam traversing the station experiences the hexapole and any excited steering field. The local hexapole strength is measured by its quadratic transverse field off axis. |
| BTlt, BSh and ADL | BTlt is drawn upstream. BSh is placed at the upstream edge of ADL. The drawing establishes axial registration. An axial field map is needed for field strength at either winding. |
| DP22 and TL22, DP21 and TL21 | The deflectors are drawn at opposite sides of the transfer doublet, near the downstream side of TL22 and the upstream side of TL21, respectively. These are candidates for exposure to lens fringe fields. |
| HPol, QPol, DP11 and TL12 | The drawings place these steering and stigmating elements together in the approach to TL12. Their exact winding boundaries, return flux and strength at neighboring stations are installation dependent. |
| DP12 and both AC scan stations | In the implementation described by Müller et al., appropriately weighted DC offsets on the physical upper/lower scan windings realize the DP12 alignment function. The two scan windings provide transverse deflection. The calibrated tilt pivot can be downstream of both windings. The source image of the corrector relay is intended to be near DP12. |
| TL11 and the microscope Mini Condenser | A compatible microscope can use its condenser-side minilens as TL11. Both names then refer to one physical winding. The ray subsequently traverses the illumination-side field of the objective. The front focus of this combination depends on its optical power. |
| Specimen, holder and objective | The specimen and the tip of the side-entry holder occupy the objective pole gap in the immersion configuration. Its incident field focuses the illumination and its outgoing field acts on the emerging electrons. A post-specimen objective aperture close to the back focal plane can lie in the outgoing field. The aperture and the illumination-side coma-free plane have different functions and positions. |
| Additional QPH1, QPH2, HPC and QPC trims | These have nominal geometry in this simulator. The photographs and published CESCOR schematic do not fix their individual hardware positions in the photographed DDCOR. |

The CESCOR schematic documents component registration and shared functions. No measured axial field profile or tolerance on fringe-field overlap for the photographed DDCOR was available in these sources. For DCOR/ASCOR, Hartel et al. specifically relate the residual six-fold astigmatism to the geometry and optics of the hexapoles and the transfer doublet between them. Their result supports checking these elements together. Transferring CESCOR field strengths or fine spacings to that instrument would require further evidence. [Müller et al., present design and ray diagram](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=3) · [Lu, CESCOR STEM optical bench, slide 2](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/MC2025_CEOS_PL_20250831.pdf#page=2) · [Hartel et al., DCOR/ASCOR](https://www.sciencedirect.com/science/article/pii/S0304399119302347) · [JEOL condenser-objective lens](https://www.jeol.com/words/emterms/20121023.034459.php) · [JEOL objective aperture](https://www.jeol.com/words/emterms/20121023.060458.php)

## Field ranges at the registered working point

The inventory follows the last archived 300 kV Nanoprobe Normal setting from the gun exit at z = 450 mm to the specimen upper surface at 1599.1999975 mm. It also reports the isolated objective field at the outgoing aperture, image/diffraction deflector pair and objective stigmator, along with their unit-current winding intervals. The fitted illumination uses the 100 µm C2 foil. [Per-component intervals, stations and field overlap](field-range-inventory.md) · [Axial field figure](illumination-field-ranges.png) · [Machine-readable intervals and numerical checks](evidence/field-footprints.json) · [Sampled incident profiles](evidence/field-footprint-profiles.npz)

The table uses 10% and 50% of each channel's peak amplitude, with 1% additionally recorded in the numerical data. All disconnected bands are retained. A spatially short field can have a large downstream optical effect. At fixed energy the leading round-lens focusing coefficient in the Larmor frame is proportional to the square of the **total signed** longitudinal field. The local field ranges show where the hardware can act. Ray transport determines where a beam waist forms. Apertures, specimen planes and virtual pivots have no separate winding.

For the two main hexapoles, HPol X/Y and TL21/TL22, twelve nonlinear solves perturb one current by ±1% around the original bias with the other currents fixed. A round-lens perturbation uses the on-axis change in longitudinal field per ampere. A sextupole perturbation uses both phases of the quadratic transverse field coefficient. These incremental profiles capture magnetic coupling in the shared material. They are not an additive decomposition of the static nonlinear field. The upper bound on the difference between forward and backward responses, divided by twice the peak centered slope, is 0.45%. Changing the sextupole sampling radius from 1.2 to 0.6 mm changes the profiles by at most 1.14% of peak. All twelve residuals are below 2 × 10⁻⁸ on the archived 3.8 mm material mesh.

The objective uses the archived axisymmetric material solution. C1, C2, C3, ADL, TL12 and Mini use signed vacuum winding groups in this particular ray calculation. Undriven steering and trim coils have zero applied field in the static trace. The inventory gives the vacuum field they would generate individually per ampere. Dipoles use on-axis transverse field and higher multipoles their leading transverse coefficient on a 1.2 mm ring. Coil quadrature was refined at stations within 50 mm of each winding center. A field percentage is relative to its own channel and never indicates the fraction of total focusing power or transmitted current.

Selected stations illustrate the field overlap in the model.

| Station | Current model, at the component center |
| --- | --- |
| C2 aperture, 765.0 mm | The C1 tail is 92.6 mT and the C2 contribution is 2.15 mT in magnitude. Both are below 10% of their own peaks. At the C3 aperture (1010.0 mm), C3 itself is at 74% of its peak. |
| Beam deflectors, 1027.5 and 1052.5 mm | The C3 field is at 51% and 28% of its peak. The ADL field is at 10% and 24%. |
| DPH2 and DPH1 | Both centers are within 1% of their respective primary sextupole response maxima. At DPH1, HPol X and Y also reach 51% and 85% of their own maxima. |
| DP22 and DP21 | At DP22 the TL22 and TL21 incremental responses reach 30% and 15%. At DP21 the TL21 response reaches 44%. The main TL22 response is at 1154.7–1228.5 mm with two shorter upstream intervals above 10%. |
| DP11 and the two AC stations | TL12 is at 26% of its peak at DP11, 94% at upper scan and 47% at lower scan. The upper/lower scan X unit profiles each exceed 10% over about 29 mm and the Y profiles over about 34 mm. The two X profiles overlap at 1398.0–1402.0 mm if driven. |
| Mini Condenser, sample and the outgoing side | The objective field at the Mini center has magnitude 75.9 mT, about 83% of its incident peak. At the sample it is 7.10 mT, about 7.8%. The outgoing image/diffraction upper and lower windings see 78.9 and 62.2 mT of objective field, respectively. |

The incident objective maximum occurs around 1530.5 mm on the 0.5 mm display grid, 68.7 mm upstream of the specimen. The independent archived objective solve gives 7.10 mT at the specimen and a 91.26 mT axial peak. The unusual distribution in and around an immersion gap warrants an independent check of the objective winding, return circuit, physical gap, axial field and calibration. The axis electric field in the ADL-to-AC region is below 1.4 × 10⁻⁶ V/m in this working point. [Objective solver receipt](evidence/normal-final-objective-field.json)

## The apparent focus at the upper and lower scan coils

The scanned region contains a strong **TL12** field in this working point. The three signed TL12 winding groups produce +0.471 T at the upper scan center (1387.5 mm) and +0.233 T at the lower center (1412.5 mm). The net axial fields there are +0.429 and +0.192 T. The scans and their DP12 offsets are all at zero for the static diagram. The two measured two-dimensional waists are at 1375.00–1375.25 mm and 1408.25–1408.52 mm. [Local ray and field comparison](normal-scan-waist-origin.png)

The causal control uses 32,768 original packets and compares the same 30,396 surviving source IDs with their absolute weights. Restarting from identical positions and slopes at the AC assembly entrance (1380 mm) and suppressing only the subsequent TL12 field removes the second waist. At z = 1408.254 mm the centered RMS radius changes from 0.443 µm to 416.24 µm. With the installed fields, independent ±1 µrad AC commands in X and Y shift the centroid by up to 5.07 nm at the sampled stations, while the largest centered RMS-radius change is below 0.00001 nm. Each signed command was compared with zero at the same stations. The closest common driven pivot is at 1496.72 mm and the isolated Mini plus upper-objective front focal plane is at 1490.38 mm. A static waist at 1408 mm does not satisfy that relay condition. [Scan controls and original-particle evidence](evidence/normal-scan-waist-origin.json) · [Calibrated planes](evidence/normal-objective-probe.json)

The current TL12 field uses vacuum windings even though its pole pieces, sleeves and annular return plates exist in the mechanical model. A useful next measurement is the signed axial TL12 field from its upstream pole across both scan stations at a known excitation. A simulation counterpart would include these TL12 materials, the surrounding return path, an exterior air domain with a boundary-distance check, and a finer bore and pole-gap mesh. The two scan fields, both specimen-proximal optical planes and the static waist should then be reevaluated at the same reference currents. Shortening the external assembly alone has no demonstrated connection to the second waist.

## Verification and reproduction

The corrected scan windings, solver input digests, centered width controls and the unit-current profiles were checked against the working-point data. At the saved AC field stations the four axial transverse magnitudes agree with the separately archived scan winding basis to better than 2 × 10⁻¹⁴ of peak. Each of the 40 auxiliary X and Y profiles was sampled again with finer winding quadrature. The largest difference was 2.3 × 10⁻⁶ of peak. The two existing deflector binding checks that require every insert to have a 20 mm bore fail for the 6.12 mm DPH inserts in both probe-corrector assemblies. The three DP inserts still have a 20 mm bore. Both input variants place DPH at their respective HP reference plane.

In the prepared study workspace, run the following from the repository root. The first two scripts use CUDA. The renderer needs only the versioned evidence.

```powershell
.\.venv\Scripts\python.exe docs/development/cescor_direct_probe_20261121/material_field_ranges.py
.\.venv\Scripts\python.exe docs/development/cescor_direct_probe_20261121/column_field_ranges.py
.\.venv\Scripts\python.exe docs/development/cescor_direct_probe_20261121/render_field_ranges.py
```

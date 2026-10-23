# Native TEM stigmators: structure, position, and control audit

Audit date: 2026-10-19. Main assembly: FEG + C3 + Probe Corrector + No Energy Filter.
This report distinguishes public manufacturer evidence, patent embodiments, and project configurations; OEM dimensions were not inferred from schematic proportions.
Corrector internals are examined only to establish control ownership; their multipoles are not all counted as additional TEM stigmators.

## Instrument models are not equivalent to project assembly names

| Reference family | Confirmed information | Applicability to this project |
| --- | --- | --- |
| Tecnai 20 / F20 / F30 | FEI Alignments manual p11–13 and p28 describe C1/C2; p24 places the mini-condenser downstream of beam deflection coils | Topological reference for a two-condenser system without a corrector; F30 does not imply a separate C3 |
| Titan | Manufacturer training Module 1 p1 describes the added C3; p8 compares Tecnai's two-condenser behavior | Optical-family reference for C3 without a corrector and C3 with a CEOS corrector |
| Spectra Ultra | DS0369 p3 describes three condensers, S-CORR/CEOS, and S-TWIN objective | Establishes the combination of system functions, but provides no internal stigmator installation coordinates |
| Iliad Ultra | DS0510 p3 places NanoPulser between the gun and condenser module | Reference for an upstream add-on module; it does not justify copying old CESCOR internals |

Sources: [original FEI Tecnai manual](https://www.dartmouth.edu/emlab/docs/fei_tecnai_f20_alignments_doc.pdf),
[Titan manufacturer training](https://emfacility.science.oregonstate.edu/sites/emfacility.science.oregonstate.edu/files/2025-09/Titan%20training%20manual_0.pdf),
[Spectra Ultra](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/ds0369-spectra-ultra-s-tem-semiconductor.pdf),
[Iliad Ultra](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/iliad-ultra-datasheet-ds0510.pdf).

## Original sections and existing three-dimensional representations

| Original source | Location inspected | What it supports | What it does not establish |
| --- | --- | --- | --- |
| Philips CM12/STEM technical paper | PDF p11 / printed p279, Fig.6, 7 | Eight stigmator coils distributed at 45°; two-stage BT in the upper-objective bore upstream of MC, and two-stage PT in the lower-objective bore; compact structural integration of MC and upper objective | Fig.7 does not label condenser/objective stigmators or establish that all devices share one removable cylinder; it is not a Titan section |
| Philips US4902940A | claim 4 | Two coplanar quadrupole groups with a 45° structural angular offset | Does not provide manufacturing dimensions for the current instrument |
| US5481164A | Fig.1, 2 and text | Normal/skew quadrupole groups normally overlap axially; axial separation in the drawing is illustrative | They cannot be drawn as two round lenses in axial series |
| JEOL US6548816B2 | Fig.1, 3 | Eight-coil and twelve-coil structures are different feasible implementations | Quadrupole action does not imply four physical coils or automatically establish eight ferromagnetic poles |
| Philips US3150258A | PDF p2, Fig.2 | Tilted miniature cored coils in nonmagnetic supports can be used near the objective | This patent's iron cores cannot simply be added to the existing Titan model |
| CEOS US20090079531A1 | Fig.10, 11 | Thin-layer wound PCB coils can be installed in lens bores or recesses | Does not establish a specific condenser-stigmator host in a particular Titan/Spectra |
| FEI US8569693B2 | PDF p4, 5, Fig.2, 3 | Embodiment evidence for separate physical objective stigmators near the objective and diffraction stigmators near the first imaging/diffraction lens | The patent's added LD stigmator is not a confirmed mandatory component in this project; no exact millimeter coordinates for the current system are supplied |

Sources: [Philips technical paper](https://www.worldradiohistory.com/Archive-Company-Publications/Philips-Technical-Review/80s/Philips-Technical-Review-1987-11.pdf),
[US4902940A](https://patents.google.com/patent/US4902940A/en),
[US5481164A](https://patents.google.com/patent/US5481164A/en),
[US6548816B2](https://patents.google.com/patent/US6548816B2/en),
[US3150258A](https://patents.google.com/patent/US3150258A/en),
[US20090079531A1](https://patents.google.com/patent/US20090079531A1/en),
[US8569693B2](https://patents.google.com/patent/US8569693B2/en).

The report's 2026-10-19 baseline uses eight radial copper coil blocks at the same axial position to represent two quadrupole winding groups.
These are not separate round lenses, and no iron cores, yokes, insulating sleeves, or eight-pole structures were invented.
Radial/axial coil proportions are engineering schematic allocations within existing envelopes, not OEM winding dimensions.
Gun stigmator uses a separate magnetic-gradient model and lacks declared winding construction, so it retains an unresolved envelope;
the column's eight-coil model was not applied to the electron gun without evidence.

For the 2026-10-20 follow-up structural additions, see [Condenser Deflector / Condenser Stigmator](condenser-deflector-stigmator-structure.md):
the default C3 + Probe Corrector Condenser Stigmator now retains the original eight copper windings while
explicitly adding non-OEM engineering formers, insulation covers, adhesive seats, and slotted supports. The preceding paragraph describes the baseline before these additions
and can no longer establish that the current component lacks passive support.

## Control aliases are not new physical devices

The Thermo S-CORR guide p2 maps `AT_A1fine` to `Condenser`; p5 also distinguishes initial coarse A1
from the microscope condenser stigmator used for final fine adjustment. This supports shared control,
but does not by itself establish that the condenser-stigmator coils have moved inside the corrector.
[Manufacturer guide](https://documents.thermofisher.com/TFS-Assets/MSD/manuals/nanoport-tuing-cs-probe-s-corr-user-guide.pdf)

In the project, Probe A1 fine already writes to the two drives of the same `condenser_stigmator`;
Image A1 fine corresponds to the same `objective_stigmator`. These aliases do not add another field.
Descan is an additional command explicitly bound to the Image/Diffraction deflector, sharing its two physical coil stages.
No separate copper coils or vacuum restriction are drawn for Descan.

QPH/QPC/QPOL are not currently complete independent normal/skew interfaces that can directly replace those dual-axis drives.
Accordingly, this round did not arbitrarily migrate A1 fine to a corrector multipole or delete the condenser stigmator.

## Actual positions in the selected assembly

The following are global axial coordinates in millimeters after parsing the configuration, without rigid-body poses. They are not manufacturer measurements.

| Device | Envelope Z range / mm | Reference position or two-stage coil centers / mm |
| --- | --- | --- |
| C2 | 550–750 | 650 |
| Condenser deflector | 790–830 | 800 / 820 |
| C3 | 830–1000 | 915 |
| Beam deflector | 1020–1060 | 1027.5 / 1052.5 |
| ADL | 1070–1106.4 | 1088.2 |
| Corrector TL12 | 1353.6–1379 | 1366.3 |
| AC scan | 1380–1420 | 1387.5 / 1412.5 |
| Condenser stigmator | 1436.5–1466.5 | 1451.5 |
| Mini-condenser | 1512.5–1532.5 | 1522.5 |
| Upper objective pole | 1534.5–1596.5 | 1565.5 |
| Specimen | — | 1599.2 |
| Lower objective pole | 1601.9–1663.9 | 1632.9 |
| Image/Diffraction deflector, also Descan | 1666.9–1706.9 | 1674.4 / 1699.4 |
| Objective stigmator | 1731.9–1761.9 | 1746.9 |
| SAA | 1814.4–1834.4 | 1824.4 |
| Diffraction stigmator | 1834.4–1844.4 | 1839.4 |
| Diffraction lens | 1846.9–1946.9 | 1896.9 |

The Condenser deflector and Beam deflector already follow the requested C2–C3 and C3–ADL segmentation.
The Objective stigmator occupies downstream bore space within the objective assembly, but its axial envelope is not within the lower pole's
1601.9–1663.9 mm solid extent. “Inside the objective assembly” must be distinguished from “within the axial extent of the pole-piece body.”
The 56 mm OD of the Condenser/objective stigmators leaves 2 mm radial envelope clearance relative to the configured 60 mm bore;
this is only a dimensional check, not mechanical installation, tolerance, thermal-design, or finite-element field validation.

All seven column variants are parsed individually by the audit script. In particular, the AC/CM
and Image/Objective-stig ordering of C3 + Probe Corrector differs from some image-corrector variants and cannot be unified in bulk.
The upstream Beam deflector in conventional C2/C3 configurations also does not yet represent the full bore integration shown in the Philips Twin section.
This round records these differences rather than overwriting the tuned C3+ProbeCorrector operating point with CM12 patent schematics.

## Completed code and calculation checks

1. Fixed three-dimensional windings ignoring `channel_x_angle_deg` and `channel_y_angle_deg`.
   Each X/Y group of four coils rotates with its own structural reference angle, retaining original IDs, axial positions, and default 0°/45° geometry.
   Independent bases separated by angles other than 45° narrow schematic coils according to the nearest angular spacing to prevent copper-block intersections; field strength does not vary with copper-block thickness.
2. Diffraction stigmator already had eight physical coils but was labeled `quadrupole_field_channel`.
   Both projection-module definitions now explicitly specify `magnetic_stigmator` / `stigmator_windings`
   and state that the FEI patent supports physical action only, not validation of configured dimensions. Field position, width, and excitation are unchanged.
3. Added read-only `Model Inspector → Stigmator hardware`, listing data directly from the current assembly,
   A1 fine aliases, and the Descan host. Aliases for absent correctors are not displayed.
   The audit also checks duplicate field objects and incorrect shared hosts to avoid counting logical channels as duplicate physical fields.
4. Performed CPU checks of existing propagation coefficients: a 1% X drive at the center gives
   `K = [[3,0],[0,-3]] m^-2`, while 1% Y gives `[[0,3],[3,0]] m^-2`;
   after rotating both bases by 15°, 1% X gives `[[2.598076,1.5],[1.5,-2.598076]] m^-2`.
   The principal axes rotate while eigenvalues remain ±3 m^-2, demonstrating that geometry changes did not alter the excitation scale.

These are numerical-consistency checks of the current analytical quadrupole model, not three-dimensional magnetostatic solutions or OEM current calibration.
The effective field is a coaxial Gaussian: `K=[[n,s],[s,-n]]`, `d(theta)/dz=-K r`;
angles enter the normal/skew coefficients through double angles. The existing scale of 300 m^-2 corresponds to 100% drive;
Condenser/Objective FWHM is 6 mm and Diffraction FWHM is 8 mm, all model parameters.
The structural-basis rotation added here is a separate input from the existing optical-axis tilt `rotation_*_mrad`; the latter's global semantics are unchanged.

## Reproducing figures and reports

Run in the project virtual environment:

```powershell
.venv\Scripts\python.exe scripts/audit_stigmator_hardware.py
.venv\Scripts\python.exe scripts/preview_stigmator_hardware.py
```

Outputs are under `outputs/stigmator-research-20261019/`:
`hardware-audit.json` records the seven assemblies and numerical checks; `configured-axial-order.png` plots actual positions;
`stigmator-field-bases.png` shows propagation slope responses, not magnetic flux density B.
The nine native PartModelView screenshots in `preview/` show default/rotated eight-coil states, an overview inside the objective,
upper/lower half-sections, and the exterior, overall section, and upper/lower local sections of the added sleeves.
Screenshot records retain selected/omitted components, saved dimensions, poses, source-file hashes, and temporary angle edits.
Rendering uses CPU and 32 angular segments; it does not run high-accuracy particle/coherent propagation or write back to the saved assembly.

Stigmator structure and readout regression: 264 passed and 1 real CUDA-device test skipped, approximately 41 seconds.
Coverage includes CAD coils, quadrupole tensors, readouts across seven assembly types, poses, Model Inspector,
corrector channels, and CI test registration; this does not constitute the full project suite or GPU validation.

Still requiring independent evidence: actual Titan/Spectra/Iliad stigmator hosts, shared insert-cylinder construction,
per-part internal dimensions, winding turns/currents, materials, and measured field maps. The current work does not claim to reproduce these OEM structures.

## Addendum: corrector use of TEM Beam Tilt and scan coils

Distinguish **software action names, the drive combinations used in calculations, and the installation positions of physical coils**.
A corrector can use native TEM devices; a control appearing in corrector software does not mean that its device is part of
the corrector's mechanical barrel or that every action called Beam Tilt uses the same coils.

The original paper by CEOS authors Müller et al. 2006, p443 and Fig.2, explicitly states that, where possible, the native TEM
mini lens replaces TL11 and weighted DC offsets of the upper/lower scan coils implement DP12. BSh/BTlt is shown
between the last condenser and ADL; the downstream scan coils lie between TL12 and the mini lens / objective.
These are two distinct hardware groups; DP12 is not a third coil that needs to be drawn separately.
[Original CEOS paper](https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf)

CEOS's original description of automatic B2 correction further states that it combines BTlt above the corrector
and DP12 below it to change direction through the hexapole system. This also shows that correction can span the corrector
and native TEM components, rather than concentrating all these controls in one physical deflector.
[CEOS B2 Auto Alignment](https://www.ceos-gmbh.de/de/produkte/residualsCEXCOR)

The original Titan UI manual p244 explicitly describes deflection controls being summed first and then converted through pivot-point relationships
into upper/lower coil excitations; beam shift, dark-field tilt, and rotation centre do not each have a separate coil set.
The CEOS 2025 tuning tutorial p4 uses AT BeamTilt for the objective rotation center; p14 also shows that channels
exported by the corrector, including Beam Tilt, can be operated through the microscope hand panel/software. Control origin and physical host should be recorded separately.
[Original Titan manual](https://chenj.hosted.uark.edu/ChenLab/TEM/HRTEM-Titan/UserInterface.pdf),
[CEOS tuning tutorial](https://www.ceos-gmbh.de/de/grundlagen/Workshop-Beitraege/03_Basics-of-daily-corrector-tuning.pdf)

| Selected C3 + Probe Corrector configuration | Physical/control relationship | Current global Z / mm |
| --- | --- | --- |
| BSh/BTlt | Two-stage `beam_deflector` upstream of the corrector | 1027.5 / 1052.5 |
| AC scan, static offset and wobble | Two-stage `ac_deflector` coils downstream of the corrector and upstream of the mini-condenser | 1387.5 / 1412.5 |
| DP12 | Virtual reference; implemented by weighted DC of the downstream scan pair in the public CESCOR design | No separate solid |

Regarding whether they remain “inside the upper-objective bore,” public sources support retaining downstream scanning/deflection components
within the upstream objective assembly; the original Philips Twin section explicitly shows bore integration.
However, the CESCOR optical diagram is not a Titan/Spectra/Iliad mechanical section and alone cannot establish a specific bore-installation Z.
The current model assigns downstream AC, Condenser Stigmator, and mini-condenser to the upper-objective assembly,
while upstream BSh/BTlt remains between C3 and ADL. The two groups are neither merged nor drawn twice.

**Software capability boundary:** current Hardware tuning Beam shift / Beam tilt, reference-plane steering,
and Beam centre/direction alignment all drive the upstream `beam_deflector`. AC static, wobble,
and raster inputs share the downstream scan-pair coupling. The Corrector panel currently implements only A1/A2 coarse/fine;
B2 / Beam Tilt correction commands involving native scan coils are not yet implemented. The upstream entry's name must not be taken
to mean that downstream native TEM Beam Tilt is implemented, nor can the current coupled AC static offset be treated directly as a four-drive
pure-tilt calibrator. This round corrected misleading descriptions without changing calculation bindings of either deflector group.

## Addendum: schematic plastic housings for upper/lower objective electronic components

At the user's request, two open-ended hollow cylindrical sleeves were added around the electronic components actually present in each assembly:
the upper sleeve contains AC, Condenser Stigmator, and mini-condenser; the lower sleeve contains the Image/Descan pair
and Objective Stigmator. Existing parent/child relationships, positions, and field parameters are retained.

The material category is an insulating polymer of unspecified grade, with color used only to distinguish materials; it is not identified as the actual instrument's PEEK,
PTFE, or any particular engineering plastic, and no unspecified permeability, permittivity, outgassing rate, or other calculation parameters were added.
The 59.2 mm ID, 59.8 mm OD, and 0.3 mm wall thickness are user-authorized schematic geometry choices:
they accommodate the 59 mm mini-condenser housing while remaining within the 60 mm objective bore.
The upper/lower sleeves terminate outside the pole pieces and do not pass through the pole pieces or specimen gap.

For the selected C3 + Probe Corrector, the upper sleeve is 153.5 mm long (global Z 1380–1533.5 mm),
and the lower sleeve is 97 mm long (global Z 1665.9–1762.9 mm). Both ends remain open; lengths enclose the components
while clearing neighboring pole pieces rather than extending the housings through objective pole pieces.

The sleeves are independently visible passive mechanical parts; they add no magnetic field and are not duplicate vacuum walls or electron-beam apertures.
Their lengths follow the electronic-component envelopes in each assembly rather than imposing one fixed Z range on all variants.
These housings represent the requested appearance and component grouping; they still do not claim support from OEM manufacturing drawings.

Clearance checks apply to saved nominal assembly dimensions. Sleeves do not rebind independently installed scanners and stigmators
to the objective pose; the mini-condenser retains its existing objective inheritance. Independently changing the objective or
internal-component pose may alter clearances; sleeves do not move other components automatically to conceal interference.

Enclosure and clearance relationships of the nominal assembly are checked in default-configuration tests, without preventing later user changes to
internal component positions or dimensions. Original sleeves retain schematic polymer materials; separate mechanical copies continue to support existing
material editing, but choosing a material does not turn a copy into a new field source.

Validation: 24 sleeve-specific tests passed (13.32 seconds); 183 supplementary integration tests passed
(30.03 seconds, no failures or skips), covering CI scope registration, stigmator hardware readout, Model Inspector,
and vacuum/camera endpoints in multiple assemblies. The integration record is saved in
`outputs/stigmator-research-20261019/sleeve-integration.xml`.
This round used CPU and did not run large particle or coherent propagation; generation of nine native 3D previews took approximately 5.1 seconds.

- [Sleeve exteriors](../../outputs/stigmator-research-20261019/preview/objective_sleeves_exterior.png)
- [Section through sleeves and internal components](../../outputs/stigmator-research-20261019/preview/objective_sleeves_section.png)
- [Local upper-sleeve section](../../outputs/stigmator-research-20261019/preview/upper_sleeve_section.png)
- [Local lower-sleeve section](../../outputs/stigmator-research-20261019/preview/lower_sleeve_section.png)

These outputs are generated by the script above and are not versioned in Git; they can be regenerated on other machines.

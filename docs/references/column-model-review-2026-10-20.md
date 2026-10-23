# 2026-10-20: Three-dimensional structure review from electron gun to camera

## Follow-up refinements: gun stigmator, C1/C2 cartridge, and receiver shapes

The original inspection records below retain their scope at the time; the unresolved gun stigmator and cartridge
details were added in a subsequent task on the same day. All three electron guns use eight coils, insulation,
formers, a slotted carrier, and adhesive seats with explicitly saved dimensions; the shared cartridge in all seven column variants gains two load-bearing sections and a central spacer ring.
The spacer material and adaptation of the gun supports remain engineering assumptions; public sources do not provide manufacturing dimensions for this instrument.
See the [gun structure evidence](cad-gun-filter-apertures.md) and [C1/C2 structure evidence](cad-round-lenses.md).

The main Camera receiving surface is square, HAADF/DF are annular, BF/the fluorescent screen are disks, and all six EDS heads
were made square at the user's request; EELS remains rectangular. A receiving surface and its separate circular vacuum chamber must not be treated as one part.
[Receiver shapes and model boundaries](cad-detectors.md) explains which parts of the previous receiver models are retained.
A component's local 20 mm bore is not the assembled clear beam diameter; the continuous tube ID of 5.76 mm is the narrower
particle, wave, and grounded electric-field boundary there. See the [vacuum-tube aperture contract](cad-vacuum-tube.md).

The updated default assembly contains 130 entries and 376 meshes, with no topology, construction, or material omission issues found.
The [seven-column audit](../../outputs/upper-detector-detail-20261020/variant-audit.json) also covers three electron guns
and two recording systems; the [backend structure preview](../../outputs/upper-detector-detail-20261020/detail-preview.png)
was generated from shared CAD/native meshes. Hidden parts and half-sections are viewing aids only and do not change the saved model.
The [final regression record](../../outputs/upper-detector-detail-20261020/verified-regression.xml) and
[round-trip record for 13 STEP files](../../outputs/upper-detector-detail-20261020/step-final/step-manifest.json)
are evidence from the follow-up task: the final 734 regression tests passed with zero failures or skips; 66 material solids passed STEP export and reimport.
Validation remains limited to backend software and configured geometry; it is not OEM or experimental validation.

## Original component-by-component inspection record

This review inspects the current default **FEG + C3 + Probe Corrector + No Energy Filter** assembly.
It uses `3dac7ba` as the code baseline and also checks the per-component field/solid overlays in `13255c9` and
the stigmator/objective sleeve changes in `0dce5d0`. The preceding Magnetic field interaction changes in the workspace were preserved.
No desktop interaction or Git commit was performed; inspection used actual production meshes, backend CAD, and offscreen figures.

The complete ordered inventory, every material region, and its provenance are recorded in the
[130-entry checklist](../../outputs/column-model-review-20261020/README.md) and
[structured inspection results](../../outputs/column-model-review-20261020/audit.json).
They describe the current actual assembly, not a microscope assembled by mixing different configurations from the directory.
The uninstalled image corrector and energy filter are outside this round's component inventory.

## Actual changes

### Electron-gun electrodes

Extractor and Gun Lens already had perforated-electrode recipes for CAD, but their native display previously used generic envelopes.
Both now use the same solid recipes and expose four selectable surface classes: upstream face, downstream face, beam-bore wall, and exterior.
The original configured end faces and inner/outer diameters are retained and checked against the electrode boundaries used by actual gun-field requests.
The shared implementation also covers the existing corresponding electrodes in FEG + Mono and Thermionic.
No unconfirmed electrode noses, fasteners, or gun-stigmator coils were added.

Evidence and limitations: [gun structure record](cad-gun-filter-apertures.md).

### Five DPH/DP inserts in the Probe Corrector

DPH2, DP22, DP21, DPH1, and DP11 each gain gap-insulation regions for the X/Y conductor layers, for 10 regions in total.
Each region subtracts the original pair of copper winding packs from the existing conductor-layer annulus, filling the windows and spaces between winding packs.
Existing dimensions are used; the copper winding packs, carriers, cover layers, adhesive layers, beam bores, and axial positions are unchanged.

The [CEOS multipole-coils patent US20090079531A1](https://patents.google.com/patent/US20090079531A1/en)
Fig. 4 supports the structural types of insulating covers and full-surface interlayer bonding. Choosing a filled equivalent bulk solid is an
**engineering inference within the structural family**; it does not establish the OEM internal construction of these named DPH/DP components.
The configuration explicitly records this scope; older files without the switch retain the old structure.
No new magnetic field, spherical aberration, or thermal performance is inferred from passive geometric refinements.

Detailed material boundaries: [Probe Corrector structure record](cad-probe-corrector.md).

### Lower recording planes

The Camera's particle receiving area is a 57.344 mm × 57.344 mm square, whereas its former generic 3D/CAD solid of revolution was a disk.
It is now consistently square, retaining the original center, receiving plane, and 0.5 mm layout thickness.
HAADF and DF retain annular openings; BF and the fluorescent screen remain disks.

All five recording planes gain selectable upstream receiving faces, rear faces, and edges; those with beam bores also gain bore rims,
and Camera distinguishes its four sides. Receiving faces remain at the saved upstream positions rather than moving to the plate-thickness centers.
Zero-thickness reference planes remain nonvolumetric guide surfaces. The 0.5 mm value is only the original layout thickness, not the real sensitive-layer thickness.
Insertion, retraction, readout, and particle-blocking semantics retain the existing implementation.

Evidence and limitations: [detector structure record](cad-detectors.md).

### Existing EDS solids omitted from the full-column display

Comparison of the figures showed that the EDS optical reference plane has zero thickness, but the declared detector-head materials have finite dimensions.
The old full-column filtering rule conflated the two and skipped the existing EDS internal solids.
EDS components with explicit material configurations now enter the existing strict validation and modeling workflow; old pure reference planes may still be omitted,
while invalid configurations report errors. No thickness was added to the optical reference plane, and the detector head was not redesigned.

The full-column and Magnetic field solid overlays share this fix. The inspector also gains an independent cross-check:
if a CAD definition contains material but the full-column assembly lacks a corresponding mesh, it reports `omitted_material`
rather than classifying the omission as normal solely under the old filtering rule.
The finite EDS material model does not mean that this detector's electromagnetic field or three-dimensional field-sampling region has been established.

### STEP colors for material regions with multiple solids

The new insulation regions consist of several disconnected volumes. STEP reimport revealed that the old exporter retained
names and materials when expanding these composite material regions but lost their colors. The same region color is now also written
to each solid that does not already have an explicit color, preserving colors individually set by the user;
reimport restores a region color only when every actual solid records the same color.
Missing or mixed colors are not replaced with expected inputs. New negative tests cover both cases.

## Inspection conclusions from top to bottom

The table below groups parts by column position; every assembly entry within each group appears in the 130-entry inventory above.
Z is the center coordinate of the currently saved assembly, in mm; receiving surfaces, thin-plate working faces, and optical centers are distinguished separately.

| Order / center Z | Component range | Inspection results and treatment in this round |
| --- | --- | --- |
| −0.5 | FEG Tip | Retained the spherical tip and truncated-cone structure; a macroscopic preview does not establish that the emission region is resolved. |
| 8, 18 | Extractor, Gun Lens | Fixed the native-display/CAD divergence and added four physical surface classes; saved field boundaries are unchanged. |
| 24 | Gun DPA | Retained the existing thin-plate/reference structure; an unspecified working opening cannot be inferred from the mechanical envelope. |
| 200 | Accelerator | Inspected 10 electrodes and 9 interstage insulation regions; electrode positions and separate insulation structures are retained. |
| 410, 433, 445 | Gun deflector, gun stigmator, C1 aperture | Deflector winding packs retained; gun stigmator still has an unresolved internal envelope, with no assumed coil or pole count. |
| 500–765 | C1/C2 pole pieces, coils, yokes, housings, shared cartridge, C2 aperture | Separate material subparts and shared ownership retained; no reliable boundary is available for mixed materials inside the cartridge, so hidden supports were not invented. |
| 810 | Condenser deflector | Retained existing two-stage orthogonal copper winding packs, carriers, insulation, and interlayer connections; no duplicate axial channels created. |
| 868–1010 | C3 pole pieces, coils, yoke, housing, C3 aperture | Inspected the open beam bore and actual pole-piece profiles; cooling plates, insulation layers, and fasteners without saved thicknesses were not added. |
| 1040 | Beam deflector | Retained its two-stage structure and existing supports; it is a different installed component from downstream AC scan. |
| 1075.6–1100.8 | ADL | Inspected upper/lower pole pieces, annular excitation winding pack, yoke, and housing, retaining the existing round-lens structure. |
| 1144.5, 1300.85 | HP2, HP1 | Each has six pole pins, six copper winding packs with windows, and a common yoke; mechanical pole length and effective optical length remain independent. |
| 1144.5–1346.3 | DPH2, DP22, DP21, DPH1, DP11 | Added two conductor-layer gap-insulation regions per group; the original 20 copper winding packs and five sets of radial supports are unchanged. |
| 1156.975–1376.15 | TL22, TL21, TL12 | Inspected material subparts of the three round lenses; optical signs were not used to invent another pole-piece structure. |
| 1128.15–1384 | QPH/HPC/QPC/HPol/QPol, DP12, and logical aliases | Control/reference identities retained; unknown hosts were not rendered as new hardware, and DP12 does not generate duplicate AC coils. |
| 1400, 1451.5, 1456.75 | AC scan, condenser stigmator, upper electronics sleeve | Retained two-stage scanning, existing stigmator coils with windows, and the open sleeve; separate “control axes” were not made into solids. |
| 1514.5–1632.9 | Mini-condenser, objective pole pieces/coils/yoke/housing | MC/objective nesting and the specimen gap retained; upper/lower objective optical actions do not generate duplicate mechanical magnetic circuits. |
| 1599.2–1600.7 | Specimen, holder, stage, EDS, objective aperture | Fixed existing EDS material solids being filtered by the zero-thickness reference-plane rule; retained existing recipes for specimen openings, holders/ports, and detector heads. |
| 1686.9–1746.9 | Image/Diffraction, Descan, lower sleeve, objective stigmator | Shared Image/Diffraction–Descan hardware retained and drawn only once; sleeve openings and coil structures retained. |
| 1824.4, 1839.4 | SA aperture, diffraction stigmator | Retained thin plates and existing stigmator windings; mechanical bores, vacuum bores, and unconfirmed working openings recorded separately. |
| 1878.9–2468.65 | D, I, P1, P2 | Inspected upper/lower pole pieces, coils, yokes, and housings in each group; no new coils or cooling structures inferred from focal lengths. |
| 2586.9 | Projection DPA | Retained the 0.2 mm plate thickness and current 12 mm design opening; the 0.2 mm reference aperture in the literature is a separate item of information. |
| 2594.4–2986.9 | HAADF, fluorescent screen, DF, BF, Camera | Added surface details and corrected the Camera's square outline; retained openings, upstream receiving faces, and original insertion-state logic. |
| 2750.65, 2950.775 | Post-projector chamber, camera housing | Retained separate engineering envelopes; no unsupported translation stages, fittings, screws, or sensitive-material layers added. |

One existing discrepancy in dimension definitions still needs clarification: the actual current CAD material inner diameter of the corrector's continuous vacuum tube is
5.76 mm, whereas the related configurations declare a 20 mm clear beam bore. They cannot be used as the same effective aperture.
The insulation regions added in this round are inside the original copper-layer volumes; they do not alter this tube or conceal the discrepancy behind an external envelope.
The exact tube ID and clear-beam definition require separate reconciliation; this round does not claim to resolve it.
The gun stigmator, internal material separation in the shared C1/C2 cartridge, and unspecified cooling/fastening structures
also remain areas with missing evidence or parameters.

## Validation and reproducible outputs

The production assembly increased from 314 to 342 meshes; 104 entries contain solids and 26 are intentionally omitted parent assemblies,
shared channels, or references. The 28 additional meshes comprise 10 insulation regions and 18 restored EDS material regions.
Vertex and triangle hashes of 307 original meshes are identical; another 7 were regenerated from the unified
electrode/recording-plane recipes, including the corrected Camera outline. No original meshes were removed,
and the axis-aligned bounds of every original mesh remain unchanged.

The inspection records finite coordinates, degenerate triangles, boundary edges, nonmanifold edges, signed volume, material identity, and provenance.
Open boundaries or nonpositive volumes of material solids are flagged as anomalies; zero-thickness guide surfaces are handled separately.
This is not collision, tolerance, magnetic-field, or OEM certification for arbitrary poses throughout the full column.

Nine offscreen contact sheets cover all solid entries; every panel uses its own equal-axis scale.
They use actual production triangle meshes, material colors, and the application's existing CPU depth renderer.
Lighting only helps identify surfaces; it does not represent magnetic fields or optical material properties. Images cannot replace exact collision checks.
From top to bottom: [1: Electron gun](../../outputs/column-model-review-20261020/contact-01.png),
[2: C1/C2/C3](../../outputs/column-model-review-20261020/contact-02.png),
[3: ADL/HP2](../../outputs/column-model-review-20261020/contact-03.png),
[4: Middle/lower Corrector](../../outputs/column-model-review-20261020/contact-04.png),
[5: TL12/AC/MC](../../outputs/column-model-review-20261020/contact-05.png),
[6: Objective/EDS/specimen](../../outputs/column-model-review-20261020/contact-06.png),
[7: Downstream of objective/D/I](../../outputs/column-model-review-20261020/contact-07.png),
[8: Projection system](../../outputs/column-model-review-20261020/contact-08.png),
[9: Recording system](../../outputs/column-model-review-20261020/contact-09.png).

Regenerate the complete inventory and all images:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:TEMSIM_CPU_THREADS = "1"
.venv/Scripts/python.exe scripts/audit_column_model_details.py --output outputs/column-model-review-20261020
```

Backend validation results for this round are listed below. Some tests are shared between groups, so the rows cannot simply be summed as a unique test count.

| Validation scope | Result |
| --- | --- |
| New gun tests, CAD auxiliary, part model | 95 passed |
| Recording planes, detectors, optical bindings, actual blocking planes, and vacuum endpoints | 129 passed |
| Corrector, hexapoles, DPH/DP, ordinary deflectors, sleeves, and stigmators | 260 passed |
| STEP and CAD kernel in the separate CAD environment | 6 + 9 passed, none skipped |
| Assembly, round lenses, optical bindings, datums, field-map geometry, and test-scope registration | 200 passed |
| Editing features and field-map solid overlays | 100 passed |
| Inventory inspector and updated test-scope registration | Final 145 passed, including depth rendering and material-omission cross-checks |
| Assembly, detectors, shared field-map geometry, and visibility after the EDS fix | 75 passed |
| Physical Layout and field-map solid display after the EDS fix | 70 passed |

The full-assembly audit found **0 mesh-topology issues, 0 CAD recipe errors, 0 assembly-construction errors,
and 0 omissions of defined material**;
the four automatically connected liner groups are also included. Exact BRep checks of the five DPH/DP inserts covered
275 material pairs; with a volume-overlap threshold of 1e-9 mm³, no interpenetration exceeded the threshold and 90 contacts were recorded.
This does not mean that exact collision checks were performed for every arbitrary pose of the entire microscope.

The five inserts and their combined assembly passed round-trip checks for **6 STEP files**, comparing solid counts, volumes,
bounds, names, materials, and RGBA. Before/after configuration comparison confirmed only five switches and their provenance notes were added;
existing optical values, positions, and effective lengths are unchanged.
Evidence: [exact material checks](../../outputs/corrector_detail_20261020/analytic-contacts.json),
[configuration comparison](../../outputs/corrector_detail_20261020/configuration-comparison.json),
[STEP round-trip records](../../outputs/corrector_detail_20261020/step/step-manifest.json).

Some initial CAD subprocess checks encountered Windows temporary-directory permission errors; reruns in a workspace temporary directory or
an approved backend environment passed. No assertions were removed or CAD checks skipped to suppress errors.
Existing Pydantic deprecation warnings remain and are unrelated to this round's geometry changes.

All generated evidence is under `outputs/`; code, configuration, tests, and this reference document remain in the project's normal directories.
This round did not rerun coherent propagation, full magnetic-field solutions, or particle convergence, and performed no hardware validation.

# Structures downstream of C3: C3 Aperture, Beam Deflector and Adapter Lens

This review starts from the workspace in which upstream detail revisions were already complete, and continues downstream along +Z. The default configuration is FEG / C3 + Probe Corrector / No Energy Filter; adding 450 mm to column-local coordinates gives global coordinates. Only background programs, offscreen mesh checks and an independent CAD kernel were used; no desktop application was operated.

## Structural evidence and changes in this review

| Component | Previous model | Refinement in this review | Sources and limits |
| --- | --- | --- | --- |
| C3 Aperture | A 0.2 mm plate and the currently selected working aperture | Added a separate connecting rod, fastening screw and matching mounting hole; 3 material regions in total | [JEOL JP2007329043A](https://patents.google.com/patent/JP2007329043A/en) distinguishes the aperture plate, holder, screw hole and screw; [JP2012146543A](https://patents.google.com/patent/JP2012146543A/en) supports a screwed connection between the aperture holder and moving mechanism. The single-rod topology follows the existing identification record from user photographs. Dimensions are explicitly saved engineering allocations; the photographs were not measured again. |
| Beam Deflector | Two axial stations, with 8 copper winding packs and 12 regions for insulating carriers, spacers, sleeves and pins | Circular locating holes in the outer carriers and spacers were changed to circumferential slots with rounded ends; inner circular holes, pin positions and copper shapes remain unchanged, still totaling 20 material regions | [Nikon US5994704A](https://patents.google.com/patent/US5994704A/en), Fig.1–4 and the text support the combination of outer slots, spacer slots and inner circular holes. The existing instrument configuration has Y outside, whereas the patent labels the outer layer X; this instrument's coil order is retained. The total 6° sweep is an engineering dimension, not an optical rotation or OEM measurement. |
| Adapter Lens | Two pole pieces, one excitation winding pack, one yoke and one housing | Added upper and lower groups of coil-end insulation and passive copper cooling plates beneath the housing; 9 material regions in total | [Hitachi JP2967966B2](https://patents.google.com/patent/JP2967966B2/en), conventional structure Fig.8 and paragraphs 0005/0015 support coil end plates and electrical insulation layers. They are adapted to the existing ADL envelope, without claiming that the patent is this instrument manufacturer's drawing. |

The new C3 Aperture and Beam Deflector details are explicitly enabled only in the C3 + Probe Corrector configuration reviewed here. The three existing ADL configurations save the same end-structure parameters. Old files without the new structural declarations continue to use the original geometry.

Native 3D, the Magnetic field component overlay and STEP all use the same material geometry definitions. No field sources, magnetic cores, complete turn counts, external mounting brackets, threads, vacuum seals or unconfirmed cooling pipes were added.

## Dimensions and clearances

| Item | Current nominal value / check result |
| --- | --- |
| C3 Aperture optical plane | Global Z = 1010 mm; the original 0.2 mm plate thickness, 75 mm mechanism outer diameter, working aperture and offset logic are retained |
| C3 Aperture actual axial material extent | Z = 1009.3–1012.1 mm; 9.3 mm clearance to C3 material above and 7.9 mm axial material clearance to the Beam Deflector below |
| Aperture side-entry slot | The vacuum tube opens on the +Y side; allocated 0.2 mm clearances are retained at the plate edges and axially, and the opposite tube wall remains; nominal radial clearance to the tube exterior is 1.9 mm for the connecting rod and 2.2 mm for the screw head |
| Beam Deflector | Z = 1020–1060 mm, with field centers at 1027.5 / 1052.5 mm; the actual winding-pack length is explicitly saved as 15 mm per station, while the original 15 mm effective thickness is unchanged |
| Beam Deflector to vacuum tube | The nearest actual insulation material has radius 12.5 mm; the tube outer radius is 9.6 mm, giving nominal radial clearance 2.9 mm; the declared 20 mm mechanical bore is not the nearest material diameter |
| Beam Deflector to ADL | Axial material clearance 10 mm |
| ADL | Z = 1070–1106.4 mm, center 1088.2 mm, original single gap 14 mm, pole-piece mechanical bore diameter 20 mm |
| ADL end layers | Each side has 0.1 mm insulation + 0.5 mm copper plate, using the original 1.82 mm end allowance and leaving 1.22 mm per end; the winding pack is not moved |
| ADL to vacuum tube | A 20 mm mechanical bore around a 19.2 mm tube outer diameter gives nominal radial clearance 0.4 mm |
| ADL to the next material groups | Axial clearance 29 mm to HP2 and 30.85 mm to DPH2 |

These are geometric clearances in the current configuration, excluding manufacturing tolerances, mounting eccentricity, thermal deformation and instrument inspection. The checks use the continuously varying-diameter vacuum tube with its actual aperture side slots, and the material solids of adjacent components; optical parent envelopes are not treated as material.

The continuous tube inner diameter remains 5.76 mm. C3 Aperture and Beam Deflector already had a separate local 5 mm beam-passage constraint; at ADL the effective beam-passage diameter is 5.76 mm, not the 20 mm pole-piece mechanical bore. Particle and wave boundary regressions check these distinctions together; working-aperture clipping remains a separate constraint.

## Parameters and physical limits

All new dimensions enter the existing edit, undo, save, reload, material and unified STEP paths. Invalid dimensions must not leave partially modified drafts. If the coil-end layers and winding-pack contact faces are inconsistent, the structural parameters must be synchronized; independent mechanical copies still do not become optical elements.

The new passive structures do not recalibrate analytical fields or change effective optical lengths, lens strength, aberrations, mode calibration or propagation algorithms. ADL geometry changes alter the geometry identity of imported field maps, requiring old maps to be rechecked; they do not automatically create new FEM material regions.

There is still insufficient evidence for the complete ADL yoke–pole-piece return-flux connection, external fastenings and cooling circuit; these remain unconfirmed. Displaying copper end plates does not mean heat conduction or eddy currents have been simulated; the insulation category does not specify a material grade or voltage-withstand performance.

## Verification record

The baseline, field-by-field configuration differences, whole-column closed-mesh checks, material Boolean intersections, STEP export/readback, propagation comparisons and regression reports are saved under `outputs/post-c3-details-20261020/`. Detailed component construction records are in [aperture structures](cad-gun-filter-apertures.md), [ordinary deflector structures](cad-multipoles.md) and [round-lens structures](cad-round-lenses.md).

- The default assembly retains 130 installation records; material meshes increased from 407 to 413. There are 6 new regions and no removed regions. 401 original meshes are identical; the aperture plate gained a mounting hole, 4 Beam regions gained slots, and only the ADL housing was retessellated, with its envelope and volume unchanged.
- Candidate screening covered 293 material solids in the same column. All 129 bounding-box-overlap pairs involving target materials underwent actual Boolean checks; material interpenetrations and unresolved envelope interpenetrations were both zero. All 7 representative assembly combinations had no mesh, CAD construction or missing-material errors.
- 33 target material regions, including the continuous vacuum tube, generated 5 STEP files. Strict readback checks passed for volumes, boundaries, material identities, colors and hierarchy. A STEP volume deviation caused by prefusing slot-cutting tools was fixed using equivalent separate cutting tools, without relaxing the original check tolerances.
- The full configuration comparison added only 50 structural fields and updated 1 source description; no existing numeric fields changed. Under STEM Nanoprobe Normal at 300 kV, 39 unique field sources, 13 finite deflection fields, 6080 valid field samples and 352 saved sections of 9 particle trajectories were elementwise identical before and after. A positive control actively changing upper Beam X from 0.012 to 0.014 mrad produced a maximum field-component difference of 2.80088e-7 T and maximum X-displacement difference of approximately 266.976 nm, confirming that the probe detects propagation responses.
- The final regressions, deduplicated by test identity, totaled **851 passed, with zero unresolved failures, errors or skips**. This combines 791 main regressions, targeted tests after review and independent CAD-kernel tests; it is not a single run. The raw reports retain 2 corrected obsolete shape assertions and records of 3 STEP files whose readback was denied inside the sandbox; the latter passed unchanged in the approved background test environment. All 9 CAD-kernel cases skipped in the main environment passed in the separate environment with CAD installed.

The final regression summaries are `verification.json` / `verified-regression.xml`; actual propagation evidence is in `transport-verification.json`, and STEP readback evidence is in `step/step-manifest.json`. These checks do not certify real-instrument structure, manufacturing tolerances, heat dissipation, voltage withstand or absolute magnetic-field accuracy.

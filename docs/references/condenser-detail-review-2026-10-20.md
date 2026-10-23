# Structural review of C2 Aperture, Condenser Deflector, Condenser Stigmator and C3

This review continues checking the saved FEG + C3 + Probe Corrector + No Energy Filter structure.
All checks were performed through background code. New materials continue to use the unified `SolidSpec`, native meshes,
STEP and material categories; 3D detail is not equated with a new field solution or OEM manufacturing drawing.

## Actual order and changes in this review

Global Z increases downward in the default assembly, in mm:

| Component | Center Z | Change in this review |
| --- | ---: | --- |
| C2 Aperture | 765 | Retained the strip plate and original working-aperture definition, adding one connecting rod, one simplified unthreaded screw and a matching mounting hole; 3 material regions in total |
| Condenser Deflector | 810 | Already had two stages with 8 copper winding packs and 12 insulation/locating regions; explicitly saved the 10 mm physical winding length to remove its default dependence on effective field width |
| C3 | 915 | Added two heat-dissipation plates and two insulating sheets in the existing end spaces, alongside the original two pole pieces, coil, yoke and housing; 9 material regions in total |
| Condenser Stigmator | 1451.5 | Located downstream of Probe Corrector, inside the objective assembly; retained the original 8 copper winding packs and added 8 insulation layers, 8 formers, 8 adhesive seats and 1 slotted support, totaling 33 regions |

Condenser Stigmator is not between C2 and C3. It was not moved to match its name or list order in this review.
The shared physical relationship between Descan and Image/Diffraction is unchanged.

## Supporting evidence and assumptions

- C2: Existing user-photo records support the strip-plate–single-rod–screw topology; the original photographs were not used for quantitative measurements.
  [JEOL JP2007329043A](https://patents.google.com/patent/JP2007329043A/en)
  and [JP2012146543A](https://patents.google.com/patent/JP2012146543A/en)
  support representing the aperture separately from its movable mounting mechanism, but do not establish this instrument's hole count, spacing or dimensions in millimeters.
- Deflector: [US5994704A](https://patents.google.com/patent/US5994704A/en)
  supports rigid insulating carriers, coaxial spacers, locating pins and insulating sleeves; the existing engineering implementation is retained.
- Stigmator: [CN102509689B](https://patents.google.com/patent/CN102509689B/en)
  describes an eight-slot cylindrical support, coil formers, insulated windings and adhesive fixation. This review adapts that structural principle;
  it does not identify this objective patent as a drawing of this instrument's Condenser Stigmator.
- C3: [Hitachi JP2967966B2](https://patents.google.com/patent/JP2967966B2/en),
  conventional structure Fig.8 and its text support coil-end heat-dissipation plates and thin electrical insulation layers.

All new dimensions are explicitly saved as `engineering_reconstruction_not_oem`.
The Stigmator equivalent insulation, former wall and adhesive layer are 0.1, 0.3 and 0.1 mm respectively;
C3 insulating sheets and heat-dissipation plates are 0.5 and 2.0 mm respectively. These are engineering allocations within saved envelopes,
not manufacturer dimensions or tolerances. Detailed parameters and comparisons are in the
[aperture record](cad-gun-filter-apertures.md),
[deflector and Stigmator record](condenser-deflector-stigmator-structure.md),
and [round-lens record](cad-round-lenses.md).

Host attachment, leads, vacuum seals, water circuits and the C3 radial return-flux connection remain unconfirmed.
Copper cooling plates cannot replace iron return-flux covers, nor support claims of a complete magnetic circuit or improved aberrations.

## Actual relationship to the vacuum tube

Checks use the same outer-diameter polyline and side-entry slot solids as Physical Layout.

| Component/interface | Default clearance |
| --- | ---: |
| C2 strip plate to side-slot edges | 0.2 mm per side |
| C2 connecting rod / screw head to tube exterior | 1.9 / 2.2 mm |
| Condenser Deflector actual support inner surface to tube exterior | Minimum radial clearance 2.9 mm |
| Condenser Stigmator support to tube exterior | Radial clearance 0.4 mm |
| C3 pole pieces to constant-diameter tube section | Radial clearance 0.4 mm |
| Near the C3 tube-diameter transition | Shortest 3D distance approximately 0.398015 mm |

The last two C3 entries differ: a 17.2 mm tube outer diameter and 18 mm pole-piece bore give a radial difference of 0.4 mm;
the adjacent tapered transition makes the shortest distance `0.4 / sqrt(1 + 0.1²)`.
Mesh distances at 32, 64 and 128 angular samples converge to this analytical result.
These are geometric clearances at the default positions, not assembly-tolerance certification.

The material tube's 5.76 mm inner diameter must also be distinguished from the solver's beam-passage domain: the C2 Aperture and Deflector sections remain constrained by local 5 mm
vacuum settings; the resolved diameter at C3 and Stigmator is 5.76 mm. The working aperture imposes its own independent clipping.
New mechanical details do not expand these beam-passage boundaries; particle and wave paths use the same resolved limits.

## Editing, propagation and evidence

Native display and STEP share material definitions. Invalid detail edits are rejected before replacing the draft and undo history;
copies retain their own dimensions, and local Z contact positions follow movement and length edits.
If C3 coil dimensions contradict contact with the new end plates, the housing recipe must be explicitly synchronized.
Material editing of the original housing `body` remains available; the new passive layers do not become additional FEM material regions.
Passive geometry changes do not recalibrate analytical fields; the C3 geometry fingerprint requires old imported field maps to be revalidated.
Nonuniform aperture scaling is applied first to the strip-plate dimensions, then the fixed 0.2 mm slot-edge clearance is added; the original instrument's optical tilt
still does not rotate the mechanical strip plate. Independent copies and replacement models that could enter the tube without a declared entry slot
are rejected by an explicit conservative envelope check, avoiding treating a template name as proof of safe insertion.

The default assembly increased from 376 to 407 meshes, adding 31 regions and removing none.
The C2 strip plate gained a mounting hole; the C3 housing is now tessellated by the shared CAD recipe, but its analytical annular-shell shape is unchanged.
The complete per-component audit found no non-closed meshes, missing materials or construction errors.

Background evidence:

- Per-component audits [before this review](../../outputs/condenser-details-20261020/before/audit.json)
  and [after this review](../../outputs/condenser-details-20261020/after/audit.json).
- [Actual assembly-clearance checks](../../outputs/condenser-details-20261020/assembly-clearance.json):
  checked 334 candidate bounding-box intersections involving this batch among 287 material regions in the same column; no positive-volume interpenetrations were found.
- [STEP readback](../../outputs/condenser-details-20261020/step/step-manifest.json):
  66 selected material regions, 5 grouped files and 1 complete assembly file, verifying solids, volumes, boundaries, names, materials and colors.
- [Seven assembly checks](../../outputs/condenser-details-20261020/variant-audit.json):
  covered three guns, seven column configurations and two recording systems, with no mesh, missing-material or CAD construction errors.
- [Relevant regression results](../../outputs/condenser-details-20261020/verified-regression.xml):
  comprehensive regressions plus targeted reruns after fixes, deduplicated by test identity: all 824 passed, 0 failures, 0 errors and 0 skips.
  This summarizes multiple actual test batches, not a single pytest run;
  the [batch and deduplication record](../../outputs/condenser-details-20261020/verification.json)
  retains every original report location. Initial electron-gun field caching and real STEP import were blocked by Windows sandbox file permissions;
  the former passed using a workspace-generated cache, and the latter passed after an approved background rerun.
  The 6 CAD tests skipped in the ordinary runtime environment all passed separately in the existing independent CAD environment.
- [Actual field and propagation comparison](../../outputs/passive-condenser-transport-20261020T212507Z/report.json):
  saved configuration copies were resolved again as the current Nanoprobe Normal assembly, separately changing the Stigmator's
  three support thicknesses, C3's two end thicknesses and their combinations. The 39 field sources, 6080 valid magnetic-field samples,
  trajectories of 9 particles at 352 saved sections and production propagation plan were identical, while material shapes did change.
  A positive control changing X excitation from 4% to 5% changed the field and trajectories as expected, with maximum Y difference approximately 0.963 nm;
  nonzero fields and this positive control ruled out false passes caused by empty fields or paths.

These are checks of code, saved engineering geometry and model consistency; no desktop interface was operated, and they are not real-instrument validation.
